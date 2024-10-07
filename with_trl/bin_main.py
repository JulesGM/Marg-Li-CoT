#!/usr/bin/env python

"""
The hydra config is in: 
    - ./config/config.yaml

For GSM8K, the dataset object is definted in 
    - ./libs_data/lib_gsm8k.py

The data collator is defined in:
    - ./lib_data.py -> data_item_collator

The training loop is here.
        
"""

from __future__ import annotations
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
# os.environ["DATASETS_VERBOSITY"] = "warning"
# os.environ["WANDB_SILENT"] = "true"
# os.environ["NCCL_DEBUG"] = "WARN"

DETERMINISTIC = False
if DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import dataclasses
import itertools
import logging
import pathlib
import random
import subprocess

import accelerate
import datasets
import hydra
import hydra.core.config_store
import omegaconf
import more_itertools as mit
import numpy as np
import peft
import rich
import rich.console
import rich.highlighter
import rich.logging
import rich.markup
import rich.status
import rich.table
import rich.traceback 
import torch
import torch.backends
import torch.backends.cuda
import torch.backends.cudnn
import torch.utils
import torch.utils.data
import torch.utils.data.sampler
import transformers
import trl
import typing
# import trl_fork
import accelerate.utils

import lib_data
import lib_eval
import lib_trl_utils
import lib_utils
import hydra_config

datasets.disable_caching()
rich.traceback.install(
    console=rich.console.Console(
        force_terminal=True
))


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOGGER = logging.getLogger(__name__)

if RANK == 0:
    import wandb


np.random.seed(0)
random.seed(1)
torch.manual_seed(2)
torch.cuda.manual_seed_all(3)
trl.set_seed(4)
# trl_fork.set_seed(4)

torch.backends.cuda.matmul.allow_tf32 = not DETERMINISTIC
torch.backends.cudnn.allow_tf32 = not DETERMINISTIC
torch.use_deterministic_algorithms(DETERMINISTIC)

CS = lib_trl_utils.CurriculumSchedule
CE = lib_trl_utils.CurriculumSchedule.CE

hydra_config.register_configs()

@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: hydra_config.BaseConfigHydra) -> None:

    args = omegaconf.OmegaConf.to_container(cfg, resolve=True) # Convert to dict for logging purposes
    hydra_config.BaseConfigHydra(**args)  # Check that the config is valid

    precision = lib_utils.ValidPrecisions(cfg.precision)  # type: ignore
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    no_training = cfg.no_training
    name = cfg.name
    use_curriculum = cfg.use_curriculum
    float32_precision_forward_backward = cfg.float32_precision_forward_backward
    use_few_shots = cfg.use_few_shots
    answer_only = cfg.answer_only
    model_generation_batch_size = int(cfg.generation_batch_size)
    float32_precision_generation = cfg.float32_precision_generation
    eval_every = int(cfg.eval_every)
    max_epochs = int(cfg.max_epochs) if cfg.max_epochs else None
    acc_maintain = cfg.acc_maintain
    
    assert not cfg.answer_only, "Needs to be re-examined"
    assert not cfg.value_pretrain_epochs, "Needs to be re-examined"

    if acc_maintain:
        assert WORLD_SIZE == 1, WORLD_SIZE
        acc_maintainer = lib_trl_utils.MAINTAINER_NAME_TO_CLASS[
            cfg.acc_maintain.class_name](
                cfg.acc_maintain.limit_to_respect
            )

    # Display command line args
    if RANK == 0:
        lib_utils.readable(args, "Command line args")

    # Check some command line args
    assert cfg.ppo_config.kl_penalty in {"kl", "abs", "mse", "full"}, cfg.ppo_config.kl_penalty
    batch_size = cfg.batch_size // WORLD_SIZE


    dataset_name = lib_data.DatasetChoices(cfg.dataset_name)
    inference_batch_size: transformers.GenerationConfig = cfg.inference_batch_size
    task_name = lib_utils.Task(cfg.task_name)

    # Make the output shorted if we use answer_only
    if cfg.answer_only:
        assert not task_name == lib_utils.Task.SENTIMENT, task_name
        cfg.generation_kwargs.max_new_tokens = 10
        cfg.inference_generation_kwargs.max_new_tokens = 10

    generation_kwargs = omegaconf.OmegaConf.to_object(cfg.generation_kwargs)
    inference_generation_kwargs: transformers.GenerationConfig = omegaconf.OmegaConf.to_object(
        cfg.inference_generation_kwargs)
    trl_library = trl

    if use_curriculum:
        # We could do progressively longer sequences.
        assert not task_name == lib_utils.Task.SENTIMENT, task_name

        curriculum_schedule = omegaconf.OmegaConf.to_object(
            cfg.curriculum_schedule
        )
        assert curriculum_schedule

        if not isinstance(curriculum_schedule, CS):
            curriculum_schedule = CS(literals=curriculum_schedule)
        else:
            # Only in else because CS.__init__ already calls check
            curriculum_schedule.check()

        # Just test a few
        curriculum_schedule(0)
        curriculum_schedule(1)
        all_levels = sorted(set(itertools.chain.from_iterable(
            entry.enabled_difficulties for entry in curriculum_schedule
        )))

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[rich.logging.RichHandler(markup=True)],
        format=f"[{RANK}/{WORLD_SIZE}] %(funcName)s:%(lineno)d - %(message)s",
    )
    logging.getLogger("transformers").setLevel(logging.ERROR)

    if RANK != 0:
        logging.getLogger("peft_lora").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)

    ###########################################################################
    # Find the type of model we are using
    ###########################################################################
    hf_config = transformers.AutoConfig.from_pretrained(cfg.model.model_name)
    assert not hf_config.is_encoder_decoder

    peft_config_dict: hydra_config.PeftConfigHydra = omegaconf.OmegaConf.to_object(cfg.peft_config)
    assert isinstance(peft_config_dict, hydra_config.PeftConfigHydra), (
        f"Expected peft_config_dict to be of type hydra_config.PeftConfigHydra, "
        f"but got {type(peft_config_dict)}"
    )
    
    assert peft_config_dict.task_type is None, (
        f"{peft_config_dict.task_type = }, {type(peft_config_dict.task_type) = }")
    ppo_config = omegaconf.OmegaConf.to_object(cfg.ppo_config)
    if not hf_config.is_encoder_decoder:
        peft_config_dict.task_type = peft.TaskType.CAUSAL_LM
    elif hf_config.is_encoder_decoder:
        peft_config_dict.task_type = peft.TaskType.SEQ_2_SEQ_LM
    else:
        raise ValueError(f"Unknown model type: {cfg.model.model_name}")
    assert not cfg.peft_do_all_lin_layers, cfg.peft_do_all_lin_layers

    accelerator_kwargs = dict(
        kwargs_handlers=[accelerate.utils.DistributedDataParallelKwargs(
                find_unused_parameters=False)])

    ppo_config_dict = dict(
        batch_size                  = batch_size,

        mini_batch_size             = cfg.mini_batch_size,
        model_name                  = cfg.model.model_name,
        
        accelerator_kwargs          = accelerator_kwargs,
        log_with                    = "wandb",

        **vars(ppo_config),
    )

    trl_config: trl.PPOConfig = lib_trl_utils.FixedPPOConfig(**ppo_config_dict)
    
    
    if RANK == 0:
        lib_utils.readable(vars(trl_config), title="ppo_config")

    if task_name == lib_utils.Task.MAIN:
        reward_type = lib_utils.RewardChoices(cfg.reward_type)
    else:
        assert task_name == lib_utils.Task.SENTIMENT, task_name
        reward_type = cfg.reward_type
        assert reward_type is None, reward_type


    if RANK == 0:
        wandb_dir = pathlib.Path(os.environ.get("TMPDIR", "/tmp"))
        assert wandb_dir.exists(), f"Wandb directory '{wandb_dir}' does not exist."

        wandb.init(
            dir       = wandb_dir,
            entity    = "julesgm",
            name      = f"{slurm_job_id}_{cfg.name}",
            project   = cfg.wandb_project,
            save_code = True,
            config    = dict(
                generation_kwargs = generation_kwargs,
                peft_config_dict  = peft_config_dict,
                ppo_config_args   = ppo_config_dict,
                script_args       = args,
                slurm_job_id      = slurm_job_id,
                gpus              = subprocess.check_output(
                        ["nvidia-smi", "-L"], universal_newlines=True,
                    ).strip(),
                all_env_vars      = dict(**os.environ),
                accelerate_env_vars = {
                    k: v 
                    for k, v in os.environ.items() 
                    if "accelerate" in k.lower()
                },
                deepspeed_env_vars = {
                    k: v 
                    for k, v in os.environ.items() 
                    if "deepspeed" in k.lower() or 
                    "ds" in k.lower() or 
                    "deep_speed" in k.lower()
                },
            )
        )

    assert isinstance(trl_config.model_name, str), type(trl_config.model_name)

    ###########################################################################
    # Load Model
    ###########################################################################
    assert cfg.use_peft
    assert not cfg.peft_do_all_lin_layers, cfg.peft_do_all_lin_layers
    # assert precision == torch.float32, precision

    assert float32_precision_generation in {"highest", "high", "medium"}, (
        float32_precision_generation
    )

    assert float32_precision_forward_backward in {"highest", "high", "medium"}, (
        float32_precision_forward_backward
    )
    
    with lib_utils.maybe_context_manager(
        lambda: rich.status.Status(
            f"[bold green]({RANK}/{WORLD_SIZE})Loading model: "
            f"[white]{rich.markup.escape(str(trl_config.model_name))} [green]...",
            spinner="weather",
        ),
        disable=True, # RANK != 0,
    ):
        output = lib_trl_utils.init_model(
            trust_remote_code      = "microsoft/phi-2" == trl_config.model_name.strip(),
            model_name             = trl_config.model_name,
            peft_config            = peft.LoraConfig(**vars(peft_config_dict)),
            peft_do_all_lin_layers = cfg.peft_do_all_lin_layers,
            precision              = precision,
            use_peft               = cfg.use_peft,
        )

        # Deal with fork vs non-fork
        forward_tokenizer    = output["forward_tokenizer"]
        prediction_tokenizer = output["prediction_tokenizer"]
        trainer_kwargs       = {}
        trainer_kwargs["model"] = output["model"]

        eos_token_id = forward_tokenizer.eos_token_id
        pad_token_id = forward_tokenizer.pad_token_id
        assert eos_token_id == prediction_tokenizer.eos_token_id
        assert pad_token_id == prediction_tokenizer.pad_token_id

    # This is a way to stop decoding.
    line_return_tok = prediction_tokenizer.encode("\n!")[-1]
    decoding_test_output = prediction_tokenizer.decode([line_return_tok])
    assert decoding_test_output == "!", (
        f"\"{decoding_test_output}\" != \"!\"")
    assert "eos_token_id" not in generation_kwargs, generation_kwargs
    assert "eos_token_id" not in inference_generation_kwargs, inference_generation_kwargs
    generation_kwargs          ["eos_token_id"] = line_return_tok
    inference_generation_kwargs["eos_token_id"] = line_return_tok

    ###########################################################################
    # Load Datasets
    ###########################################################################
    shared_dataset_arguments = dict(
        answer_only                        = answer_only,
        answer_only_path                   = cfg.answer_only_path,
        any_tokenizer                      = forward_tokenizer,
        arithmetic_dataset_root_folder_dir = cfg.arithmetic_dataset_root_folder_dir,
        dataset_name                       = dataset_name,
        extr_arith_ignore_one_line         = True,
        few_show_qty                       = cfg.few_shot_qty,
        question_prefix                    = None,
        question_suffix                    = None,
        tok_max_query_length               = cfg.tok_max_query_length,
        tok_max_answer_length              = cfg.tok_max_answer_length,
        tok_max_total_length               = cfg.tok_max_total_length,
        use_curriculum                     = use_curriculum,
        use_few_shots                      = use_few_shots,
    )
    
    dataset = lib_data.prep_dataset_rl(
        split=lib_utils.CVSets.TRAIN,
        **shared_dataset_arguments
    )

    eval_dataset = lib_data.prep_dataset_rl(
        split=lib_utils.CVSets.VALID,
        **shared_dataset_arguments,
    )
    
    ###########################################################################
    # Set model name specific flags
    ###########################################################################
    if not hf_config.is_encoder_decoder:
        if peft_config_dict:
            assert peft_config_dict.task_type == peft.TaskType.CAUSAL_LM, (
                peft_config_dict.task_type)

    ###########################################################################
    # Prep Training
    ###########################################################################
    data_collator = lambda x: lib_data.data_item_collator(
        batch_and_indices=x, 
        use_few_shots=use_few_shots, 
        prediction_tokenizer=prediction_tokenizer, 
        inspect_indices=cfg.inspect_indices,
    )

    # if use_curriculum:
    #     data_collator_arg = None
    #     dataset_arg       = None
    # else:
    data_collator_arg = data_collator
    dataset_arg       = dataset
        

    ds_plugin = accelerate.utils.DeepSpeedPlugin(gradient_accumulation_steps=1)
    ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = cfg.mini_batch_size
    ds_plugin.deepspeed_config["train_batch_size"] = cfg.mini_batch_size * WORLD_SIZE
    trl_config.accelerator_kwargs["deepspeed_plugin"] = ds_plugin

    ppo_trainer: trl.PPOTrainer = trl_library.PPOTrainer(
        config        = trl_config,
        ref_model     = None,
        tokenizer     = forward_tokenizer,
        data_collator = data_collator_arg,
        dataset       = dataset_arg,
        **trainer_kwargs,
    )
    
    metrics, reward_fn = lib_eval.make_metric_and_reward_fn(
        accelerator_device        = ppo_trainer.accelerator.device,
        accelerator_num_processes = ppo_trainer.accelerator.num_processes,
        dataset                   = dataset,
        dataset_name              = dataset_name,
        extractor                 = dataset.get_extractor(),
        pad_token                 = forward_tokenizer.pad_token,
        reward_type               = reward_type,
        task_name                 = task_name,
        use_peft                  = cfg.use_peft,
    )
    
    policy_model = ppo_trainer.model

    train_eval = lib_eval.EvalLoop(
        accelerated_model     = policy_model,
        accelerator           = ppo_trainer.accelerator,
        batch_size            = inference_batch_size,
        collator              = data_collator,
        dataset               = dataset,
        dataset_type          = dataset_name,
        eval_subset_size      = cfg.eval_subset_size,
        forward_tokenizer     = forward_tokenizer,
        generation_batch_size = model_generation_batch_size,
        inference_gen_kwargs  = inference_generation_kwargs,
        metrics               = metrics,
        prediction_tokenizer  = prediction_tokenizer,
        reward_fn             = reward_fn,
        split                 = lib_utils.CVSets.TRAIN,
        task_name             = task_name,
        use_few_shots         = use_few_shots,
    )

    eval_eval = lib_eval.EvalLoop(
        accelerated_model     = policy_model,
        accelerator           = ppo_trainer.accelerator,
        batch_size            = inference_batch_size,
        collator              = data_collator,
        dataset               = eval_dataset,
        dataset_type          = dataset_name,
        eval_subset_size      = cfg.eval_subset_size,
        forward_tokenizer     = forward_tokenizer,
        generation_batch_size = model_generation_batch_size,
        inference_gen_kwargs  = inference_generation_kwargs,
        metrics               = metrics,
        prediction_tokenizer  = prediction_tokenizer,
        reward_fn             = reward_fn,
        split                 = lib_utils.CVSets.VALID,
        task_name             = task_name,
        use_few_shots         = use_few_shots,
    )


    if cfg.just_metrics or cfg.start_eval:
        train_eval(0)
        eval_eval(0)

        if cfg.just_metrics:
            return

    ###########################################################################
    # Training Loop
    ###########################################################################
    epoch_count = -1

    # if use_curriculum:
    dataloader = torch.utils.data.DataLoader(
        batch_size  = batch_size,
        collate_fn  = data_collator,
        dataset     = dataset, 
        num_workers = 0,
    )
    # else:
        # dataloader = ppo_trainer.dataloader

    current_step = 0
    lib_utils.named_barrier(f"bin_main {lib_utils.get_linenumber()}")

    for epoch in range(max_epochs):
        epoch_count += 1

        #######################################################################
        # Check the difficulty so that the first batch is 
        # the first batch is of the right difficulty
        #######################################################################
        train_dataloader_iterator = iter(dataloader)
        if use_curriculum:
            train_dataloader_iterator.set_difficulties(
                curriculum_schedule(ppo_trainer.current_step).proportions
            )
    

        for batch_idx, batch in enumerate(lib_utils.progress(
            description=f"Epoch {epoch_count}, Global Epoch: {epoch}",
            disable=True,
            seq=train_dataloader_iterator,
        )):

            #######################################################################
            # Potentially change the difficulty level.
            # This only affects the next batch.
            #######################################################################
            lib_utils.named_barrier(f"bin_main {lib_utils.get_linenumber()}")
            if use_curriculum:
                train_dataloader_iterator.dataset.set_difficulties(
                    curriculum_schedule(ppo_trainer.current_step).proportions
                )
            
            ######################################################################
            # Keys of batch:
            ######################################################################
            #  - "query"
            #  - "input_ids"
            #  - "ref_answer" if in GSM8K
            #  - "ref_scratchpad"
            ######################################################################
            if eval_every and batch_idx % eval_every == 0:
                if RANK == 0: 
                    rich.print("[red bold]DOING EVAL: [white]TRAIN SET")
                
                train_eval(ppo_trainer.current_step)
                
                if RANK == 0: 
                    rich.print("[red bold]DOING EVAL: [white]EVAL SET")

                eval_eval(ppo_trainer.current_step)
                
                if RANK == 0: 
                    rich.print("[red bold]DONE WITH EVAL")

            lib_utils.named_barrier(f"bin_main {lib_utils.get_linenumber()}")
            torch.set_float32_matmul_precision(float32_precision_generation)

            # We use a format string to make sure that each
            # sample are place in the same way, including the 
            # few shot examples.
            #
            # The idea here is that the first part of the few-shot
            # format 
            # 
            # Hopefully this is good for both Arithmetic and GSM8K

            if task_name == lib_utils.Task.MAIN:
                post_process_gen_fewshots = dataset.post_process_gen_fewshots
            else:
                post_process_gen_fewshots = None

            outputs = lib_trl_utils.generate(
                answer_extractor             = dataset.get_extractor(),
                batch                        = batch, 
                batch_size                   = batch_size,
                generation_batch_size        = model_generation_batch_size,
                generation_kwargs            = generation_kwargs, 
                ppo_trainer                  = ppo_trainer,
                post_process_gen_fewshots_fn = post_process_gen_fewshots,
                prediction_tokenizer         = prediction_tokenizer,
                forward_tokenizer            = forward_tokenizer,
                task_name                    = task_name,
                use_few_shots                = use_few_shots,
                step_information             = dict(
                    trainer_step = ppo_trainer.current_step,
                    batch_idx    = batch_idx,
                    epoch_idx    = epoch, 
                ),
            )

            torch.set_float32_matmul_precision(float32_precision_forward_backward)
            lib_utils.named_barrier(f"bin_main {lib_utils.get_linenumber()}")

            reward_output = reward_fn(
                batch=batch,
                responses=outputs.response_text,
            )

            ###########################################################################
            # Checks & Step
            ###########################################################################
            # - For encoder decoders, the answers should start with the pad token
            # - For all models, the answers should not have any pad tokens in them
            # - For all models, the answers should have one or fewer eos token in them
            ###########################################################################
            if ppo_trainer.is_encoder_decoder:
                assert isinstance(pad_token_id, int), type(pad_token_id)
                lib_trl_utils.check_all_start_with_token_id(
                    outputs.response_tensors, pad_token_id,)

            # There should be no pad_token_ids, but the pad
            # token id might be the eos token id, so we can't
            # just blindly check for the pad token id
            if pad_token_id != eos_token_id:
                lib_trl_utils.check_qty_of_token_id(
                    list_of_sequences = outputs.response_tensors,
                    token_id          = pad_token_id,
                    qty               = 0,
                )

            if RANK == 0: 
                print(f"{RANK} ppo_trainer.step >>>")

            step_kwargs = {}
            lib_utils.named_barrier(f"{__file__}: {lib_utils.get_linenumber()}")
            
            # Log scores per difficulty level
            if use_curriculum:
                lib_utils.named_barrier(f"{__file__}: {lib_utils.get_linenumber()}")

                per_level = {level: [] for level in all_levels}                
                for score in mit.zip_equal(batch.difficulty_level, reward_output.values):
                    per_level[score[0]].append(score[1])  

                # Accumulate with accelerate
                per_level = {
                    k: torch.tensor(v, device=ppo_trainer.accelerator.device) 
                    for k, v in per_level.items()
                }
                all_per_level = ppo_trainer.accelerator.gather(per_level)

                ###################################################################
                # <log_per_level>
                ###################################################################
                if RANK == 0:
                    for k, v in all_per_level.items():
                        if v is not None and len(v) > 0:
                            wandb.log(
                                {
                                    f"reward_per_level/{k}": v.mean().item(),
                                    f"sample_count_per_level/{k}": 
                                    len(v) / (batch_size * ppo_trainer.accelerator.num_processes),
                                }, 
                                step=ppo_trainer.current_step,
                            )
                ###################################################################
                # </ log_per_level>
                ###################################################################

                ###################################################################
                # <acc_maintainer>: 
                #   - Pick indices to maintain a certain proportion of successful samples
                #   - Subselect indices in batches, outputs and rewards
                ###################################################################
                if acc_maintain:
                    indices_ok, stats = acc_maintainer(
                        batch.difficulty_level, reward_output.values)
                    
                    if not indices_ok:
                        continue
                    
                    rich.print(f"[red bold]NEW SIZE: [white] {indices_ok = }")
                    rich.print(f"[red bold]Batch Stats: [white] {stats = }")

                    ###################################################################
                    # <Filter_batch>
                    ###################################################################
                    for k, v in vars(batch).items():
                        vars(batch)[k] = [
                            v[i] for i in indices_ok]
                        
                    for k, v in vars(outputs).items():
                        vars(outputs)[k] = [
                            v[i] for i in indices_ok]
                        
                    reward_output_expected_keys = {
                        "extracted_ref", "extracted_gen", 
                        "logging_columns", "moving_averages", 
                        "name", "values",
                    }

                    assert vars(reward_output).keys() == reward_output_expected_keys, (
                        vars(reward_output).keys(), reward_output_expected_keys)

                    reward_output_keys_to_modify = {
                        "extracted_ref", 
                        "extracted_gen", 
                        "values",
                    }

                    for k in reward_output_keys_to_modify:
                        vars(reward_output)[k] = [
                            vars(reward_output)[k][i] for i in indices_ok
                        ]

                    for k, v in reward_output.logging_columns.items():
                        reward_output.logging_columns[k] = [
                            v[i] for i in indices_ok
                        ]
                        
                    ###################################################################
                    # </ Filter_batch>
                    ###################################################################
                        
                ###################################################################
                # </ acc_maintainer>
                ###################################################################

            if no_training:
                continue

            if isinstance(batch["tok_ref_query"], torch.Tensor):
                batch["tok_ref_query"] = list(batch["tok_ref_query"])

            stats = ppo_trainer.step(
                queries   = batch["tok_ref_query"],
                responses = outputs.response_tensors,
                scores    = reward_output.values,
                **step_kwargs,
            )

            if RANK == 0: 
                print(f"{RANK} ppo_trainer.step done <<<")
            lib_utils.named_barrier(f"{__file__}: {lib_utils.get_linenumber()}")

            # Log stats
            assert isinstance(reward_output.values, list), type(reward_output.values)
            assert isinstance(stats, dict), type(stats)

            batch_stats = dict(
                difficulty_level = batch["difficulty_level"],
                query            = batch["ref_qa_question"],
                ref_answer       = batch["ref_qa_answer"],
                ref_scratchpad   = batch["ref_qa_scratchpad"],
                response         = prediction_tokenizer.batch_decode(outputs.response_tensors),
                extracted_gen    = reward_output.extracted_gen,
                extracted_ref    = reward_output.extracted_ref,
                epoch            = [epoch],
                disabled_adapter = [0], #[int(should_disable_adapter)],
            )

            assert current_step == ppo_trainer.current_step, (
                current_step, ppo_trainer.current_step)
            
            ppo_trainer.log_stats(
                batch   = batch_stats,
                rewards = [x.to(torch.float32) for x in reward_output.values],
                stats   = stats,
                columns_to_log = [
                    "epoch",
                    "query", 
                    "response", 
                    "ref_answer", 
                    "ref_scratchpad", 
                    "difficulty_level",
                    "extracted_gen",
                    "extracted_ref",
                    "disabled_adapter",
                ],
            )
            lib_utils.named_barrier(f"{__file__}: {lib_utils.get_linenumber()}")
            
            current_step += 1
            ppo_trainer.current_step = current_step
            assert current_step == ppo_trainer.current_step, (
                current_step, ppo_trainer.current_step)
            

            lib_trl_utils.print_table(
                call_source       = "main_loop",
                difficulty_levels = batch["difficulty_level"],
                extra_columns     = reward_output.logging_columns,
                generation_kwargs = generation_kwargs,
                log_header        = f"(e{epoch}-b{batch_idx}) ",
                name              = str(name),
                qty               = None,
                queries           = batch["ref_qa_question"],
                responses         = outputs.response_text,
                rewards           = reward_output.values,
            )
            lib_utils.named_barrier(f"{__file__}: {lib_utils.get_linenumber()}")


if __name__ == "__main__":
    lib_utils.print_accelerate_envs()
    main()
