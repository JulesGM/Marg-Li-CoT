#!/usr/bin/env python
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
from typing import Any, Optional

import accelerate
import datasets
import fire
import hydra
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
import trl_fork
import wandb
import accelerate.utils

import lib_data
import lib_eval
import lib_trl_utils
import lib_utils

datasets.disable_caching()
rich.traceback.install(
    console=rich.console.Console(
        force_terminal=True
))

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "0"))
RANK = int(os.environ.get("RANK", "0"))
LOGGER = logging.getLogger(__name__)

np.random.seed(0)
random.seed(1)
torch.manual_seed(2)
torch.cuda.manual_seed_all(3)
trl.set_seed(4)
trl_fork.set_seed(4)

torch.backends.cuda.matmul.allow_tf32 = not DETERMINISTIC
torch.backends.cudnn.allow_tf32 = not DETERMINISTIC
torch.use_deterministic_algorithms(DETERMINISTIC)

CS = lib_trl_utils.CurriculumSchedule
CE = lib_trl_utils.CurriculumSchedule.CE


@dataclasses.dataclass
class Args:

    @dataclasses.dataclass
    class Model:
        batch_size:             int
        inference_batch_size:   int
        mini_batch_size:        int
        model_name:             str

    def __post_init__(self):
        if self.curriculum_schedule:
            self.curriculum_schedule = CS(
                [CE(x) for x in self.curriculum_schedule]
            )

        self.answer_only_path    = pathlib.Path(self.answer_only_path)
        self.dataset_name        = lib_data.DatasetChoices(self.dataset_name)
        self.generation_kwargs   = self.generation_kwargs
        self.model               = Args.Model(**self.model)
        self.precision           = lib_utils.ValidPrecisions(self.precision)
        self.peft_config         = peft.LoraConfig(**self.peft_config)
        self.task_name           = lib_utils.Task(self.task_name)
        self.wandb_dir           = pathlib.Path(self.wandb_dir)
        self.arithmetic_dataset_root_folder_dir = pathlib.Path(self.arithmetic_dataset_root_folder_dir)
        self.inference_generation_kwargs        = self.inference_generation_kwargs
    
    answer_only_path:    pathlib.Path
    curriculum_schedule: CS | None
    dataset_name:        str
    generation_kwargs:   transformers.GenerationConfig
    model:               Model
    peft_config:         peft.LoraConfig
    task_name:           lib_utils.Task
    wandb_dir:           pathlib.Path
    arithmetic_dataset_root_folder_dir: pathlib.Path
    inference_generation_kwargs:        transformers.GenerationConfig

    name:                        str
    learning_rate:               float
    answer_only:                 bool
    answer_only_max_length:      int
    input_max_length:            int
    eval_every:                  int
    eval_subset_size:            int
    gradient_accumulation_steps: int
    just_metrics:                bool
    kl_penalty_mode:             str
    peft_do_all_lin_layers:      bool
    use_curriculum:              bool
    use_peft:                    bool
    use_few_shots:               bool
    precision:                   torch.dtype | str
    
    reward_type:                 Optional[str]
    wandb_project:               str


@hydra.main(config_path="config", config_name="arithmetic", version_base="1.3")
def main(cfg):

    args = omegaconf.OmegaConf.to_object(cfg)
    precision = lib_utils.ValidPrecisions(cfg.precision)  # type: ignore

    # Display command line args
    if RANK == 0:
        lib_utils.readable(args, "Command line args")

    # Check some command line args
    assert cfg.kl_penalty_mode in {"kl", "abs", "mse", "full"}, cfg.kl_penalty_mode
    batch_size = cfg.model.batch_size
    dataset_name = lib_data.DatasetChoices(cfg.dataset_name)
    inference_batch_size: transformers.GenerationConfig = cfg.model.inference_batch_size
    task_name = lib_utils.Task(cfg.task_name)

    # Make the output shorted if we use answer_only
    if cfg.answer_only:
        assert not task_name == lib_utils.Task.SENTIMENT, task_name
        cfg.generation_kwargs.max_new_tokens = 10
        cfg.inference_generation_kwargs.max_new_tokens = 10

    generation_kwargs = omegaconf.OmegaConf.to_object(cfg.generation_kwargs)
    inference_generation_kwargs: transformers.GenerationConfig = omegaconf.OmegaConf.to_object(cfg.inference_generation_kwargs)
    trl_library = trl

    if cfg.use_curriculum:
        # We could do progressively longer sequences.
        assert not task_name == lib_utils.Task.SENTIMENT, task_name

        curriculum_schedule = omegaconf.OmegaConf.to_object(
            cfg.curriculum_schedule)
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
            entry.proportions.keys() for entry in curriculum_schedule
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

    peft_config_dict = omegaconf.OmegaConf.to_object(cfg.peft_config)
    assert "task_type" not in peft_config_dict, peft_config_dict

    if not hf_config.is_encoder_decoder:
        peft_config_dict["task_type"] = peft.TaskType.CAUSAL_LM
    elif hf_config.is_encoder_decoder:
        peft_config_dict["task_type"] = peft.TaskType.SEQ_2_SEQ_LM
    else:
        raise ValueError(f"Unknown model type: {cfg.model.model_name}")

    if not cfg.peft_do_all_lin_layers:
        assert "target_modules" not in peft_config_dict, peft_config_dict
        peft_config_dict["target_modules"] = (
            peft.utils.other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[hf_config.model_type]
        )

    accelerator_kwargs = dict(
        kwargs_handlers=[accelerate.utils.DistributedDataParallelKwargs(
                find_unused_parameters=False
        )])

    ppo_config_dict = dict(
        accelerator_kwargs          = accelerator_kwargs,
        batch_size                  = batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        learning_rate               = cfg.learning_rate,
        log_with                    = "wandb",
        mini_batch_size             = cfg.model.mini_batch_size,
        model_name                  = cfg.model.model_name,
        kl_penalty                  = cfg.kl_penalty_mode,
        
    )

    trl_config: trl.PPOConfig = trl_library.PPOConfig(**ppo_config_dict)

    if RANK == 0:
        lib_utils.readable(vars(trl_config), title="ppo_config")

    if task_name == lib_utils.Task.MAIN:
        reward_type = lib_utils.RewardChoices(cfg.reward_type)
    else:
        assert task_name == lib_utils.Task.SENTIMENT, task_name
        reward_type = cfg.reward_type
        assert reward_type is None, reward_type

    if RANK == 0:
        if cfg.wandb_dir is None or cfg.wandb_dir == "":
            wandb_dir = pathlib.Path(os.environ["SLURM_TMPDIR"]) / "wandb"
        wandb_dir = pathlib.Path(cfg.wandb_dir)

        if not wandb_dir.exists():
            wandb_dir.mkdir(parents=True)
            assert wandb_dir.exists(), wandb_dir
            
        wandb.init(
            save_code=True,
            project=cfg.wandb_project,
            entity="julesgm",
            name=cfg.name,
            config=dict(
                generation_kwargs = generation_kwargs,
                peft_config_dict  = peft_config_dict,
                ppo_config_args   = ppo_config_dict,
                script_args       = args,
            ),
            dir=wandb_dir,
        )

    assert isinstance(trl_config.model_name, str), type(trl_config.model_name)

    ###########################################################################
    # Load Model
    ###########################################################################
    with lib_utils.maybe_context_manager(
        lambda: rich.status.Status(
            f"[bold green]({RANK}/{WORLD_SIZE})Loading model: "
            f"[white]{rich.markup.escape(str(trl_config.model_name))} [green]...",
            spinner="weather",
        ),
        disable=RANK != 0,
    ):
        output = lib_trl_utils.init_model(
            model_name             = trl_config.model_name,
            peft_config_dict       = peft_config_dict,
            peft_do_all_lin_layers = cfg.peft_do_all_lin_layers,
            precision              = precision,
            trl_library_mode       = lib_utils.TrlLibraryMode.TRL,
            use_peft               = cfg.use_peft,
        )

        # Deal with fork vs non-fork
        forward_tokenizer    = output["forward_tokenizer"]
        prediction_tokenizer = output["prediction_tokenizer"]
        trainer_kwargs       = {}
        trainer_kwargs["model"] = output["model"]

        # if trl_library_mode == lib_utils.TrlLibraryMode.TRL:
        # elif trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
        #     trainer_kwargs["policy_model"] = output["policy_model"]
        #     trainer_kwargs[ "value_model"] = output[ "value_model"]
        # else:
        #     raise ValueError(f"Unknown trl_library_mode: {trl_library_mode}")

        eos_token_id = forward_tokenizer.eos_token_id
        pad_token_id = forward_tokenizer.pad_token_id
        assert eos_token_id == prediction_tokenizer.eos_token_id
        assert pad_token_id == prediction_tokenizer.pad_token_id

    # This is a way to stop decoding.
    line_return_tok = prediction_tokenizer.encode("\n!")[-1]
    decoding_test_output = prediction_tokenizer.decode([line_return_tok])
    assert decoding_test_output == "!", (
        f"\"{decoding_test_output}\" != \"!\""
    )

    assert "eos_token_id" not in generation_kwargs, generation_kwargs
    assert "eos_token_id" not in inference_generation_kwargs, inference_generation_kwargs
    generation_kwargs   ["eos_token_id"] = line_return_tok
    inference_generation_kwargs["eos_token_id"] = line_return_tok

    ###########################################################################
    # Load Datasets
    ###########################################################################
    dataset = lib_data.prep_dataset_rl(
        any_tokenizer=forward_tokenizer,
        answer_only=cfg.answer_only,
        answer_only_path=cfg.answer_only_path,
        dataset_name=dataset_name,
        input_max_length=cfg.input_max_length,
        question_prefix=None,
        question_suffix=None,
        split=lib_utils.CVSets.TRAIN,
        use_few_shots=cfg.use_few_shots,
        arithmetic_dataset_root_folder_dir=cfg.arithmetic_dataset_root_folder_dir,
        extr_arith_ignore_one_line=True,
        use_curriculum=cfg.use_curriculum,
    )

    eval_dataset = lib_data.prep_dataset_rl(
        any_tokenizer=forward_tokenizer,
        answer_only=cfg.answer_only,
        answer_only_path=cfg.answer_only_path,
        dataset_name=dataset_name,
        input_max_length=cfg.input_max_length,
        question_prefix=None,
        question_suffix=None,
        split=lib_utils.CVSets.VALID,
        use_few_shots=cfg.use_few_shots,
        arithmetic_dataset_root_folder_dir=cfg.arithmetic_dataset_root_folder_dir,
        extr_arith_ignore_one_line=True,
        use_curriculum=False,
    )
    
    ###########################################################################
    # Set model name specific flags
    ###########################################################################
    if not hf_config.is_encoder_decoder:
        if peft_config_dict:
            assert peft_config_dict["task_type"] == peft.TaskType.CAUSAL_LM, (
                peft_config_dict["task_type"])

    ###########################################################################
    # Prep Training
    ###########################################################################
    data_collator = lib_data.data_item_collator

    if cfg.use_curriculum:
        data_collator_arg = None
        dataset_arg       = None
    else:
        data_collator_arg = data_collator   
        dataset_arg       = dataset
        
    ppo_trainer: trl.PPOTrainer = trl_library.PPOTrainer(
        config        = trl_config,
        data_collator = data_collator_arg,
        dataset       = dataset_arg,
        ref_model     = None,
        tokenizer     = forward_tokenizer,
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
    
    policy_model = (
        ppo_trainer.model 
        # if trl_library_mode == lib_utils.TrlLibraryMode.TRL else 
        # ppo_trainer.policy_model
    )

    train_eval = lib_eval.EvalLoop(
        accelerated_model    = policy_model,
        accelerator          = ppo_trainer.accelerator,
        batch_size           = inference_batch_size,
        dataset              = dataset,
        dataset_type         = dataset_name,
        eval_subset_size     = cfg.eval_subset_size,
        forward_tokenizer    = forward_tokenizer,
        inference_gen_kwargs = inference_generation_kwargs,
        metrics              = metrics,
        prediction_tokenizer = prediction_tokenizer,
        reward_fn            = reward_fn,
        split                = lib_utils.CVSets.TRAIN,
        task_name            = task_name,
        use_few_shots        = cfg.use_few_shots,
    )

    eval_eval = lib_eval.EvalLoop(
        accelerated_model    = policy_model,
        accelerator          = ppo_trainer.accelerator,
        batch_size           = inference_batch_size,
        dataset              = eval_dataset,
        dataset_type         = dataset_name,
        forward_tokenizer    = forward_tokenizer,
        eval_subset_size     = cfg.eval_subset_size,
        inference_gen_kwargs = inference_generation_kwargs,
        metrics              = metrics,
        prediction_tokenizer = prediction_tokenizer,
        reward_fn            = reward_fn,
        split                = lib_utils.CVSets.VALID,
        task_name            = task_name,
        use_few_shots        = cfg.use_few_shots,
    )

    if cfg.just_metrics:
        train_eval(0)
        eval_eval(0)
        return

    ###########################################################################
    # Training Loop
    ###########################################################################
    epoch_count = -1
    if cfg.use_curriculum:
        dataloader = torch.utils.data.DataLoader(
            batch_size = batch_size,
            dataset    = dataset, 
            collate_fn = data_collator,
        )
        dataloader.dataset.set_proportion_difficulties(
            curriculum_schedule(ppo_trainer.current_step).proportions
        )
    else:
        dataloader = ppo_trainer.dataloader

    current_step = 0
    for epoch in itertools.count():
        epoch_count += 1
        for batch_idx, batch in enumerate(
            lib_utils.progress(
                description=(
                    f"Epoch {epoch_count}, "
                    f"Global Epoch: {epoch}"
                    ),
                disable=True,
                seq=dataloader,
            )
        ):
            if cfg.use_curriculum:
                dataloader.dataset.set_proportion_difficulties(
                    curriculum_schedule(ppo_trainer.current_step).proportions
                )
            
            ############################################################
            # Keys of batch:
            #   - "query"
            #   - "input_ids"
            #   - "ref_answer" if in GSM8K
            #   - "ref_scratchpad"
            ############################################################

            if cfg.eval_every and batch_idx % cfg.eval_every == 0:
                if RANK == 0: rich.print("[red bold]DOING EVAL: [white]TRAIN SET")
                train_eval(ppo_trainer.current_step)
                if RANK == 0: rich.print("[red bold]DOING EVAL: [white]EVAL SET")
                eval_eval(ppo_trainer.current_step)
                if RANK == 0: rich.print("[red bold]DONE WITH EVAL")

            outputs = lib_trl_utils.generate(
                accelerator                  = ppo_trainer.accelerator, 
                answer_extractor             = dataset.get_extractor(),
                batch                        = batch, 
                batch_size                   = batch_size,
                dataset_name                 = dataset_name,
                generation_kwargs            = generation_kwargs, 
                policy_model                 = policy_model, 
                post_process_gen_fewshots_fn = 
                    dataset.post_process_gen_fewshots 
                    if task_name == lib_utils.Task.MAIN 
                    else None,
                prediction_tokenizer         = prediction_tokenizer,
                task_name                    = task_name,
                use_few_shots                = cfg.use_few_shots,
                step_information             = dict(
                    trainer_step = ppo_trainer.current_step,
                    batch_idx    = batch_idx,
                    epoch_idx    = epoch, 
                ),
            )

            reward_output = reward_fn(
                batch=batch,
                responses=outputs.response_text,
            )

            ###########################################################################
            # Checks & Step
            #################
            # - For encoder decoders, the answers should start with the pad token
            # - For all models, the answers should not have any pad tokens in them
            # - For all models, the answers should have one or fewer eos token in them
            ###########################################################################
            if ppo_trainer.is_encoder_decoder:
                assert isinstance(pad_token_id, int), type(pad_token_id)
                lib_trl_utils.check_all_start_with_token_id(
                    outputs.response_tensors, pad_token_id)

            # There should be no pad_token_ids, but the pad
            # token id might be the eos token id, so we can't
            # just blindly check for the pad token id
            if pad_token_id != eos_token_id:
                lib_trl_utils.check_qty_of_token_id(
                    list_of_sequences=outputs.response_tensors,
                    qty=0,
                    token_id=pad_token_id,
                )

            if RANK == 0: print(f"{RANK} ppo_trainer.step >>>")

            step_kwargs = {}
            # if trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
            #     step_kwargs["answers"] = batch.tok_ref_answer                    


            # Log scores per difficulty level
            if cfg.use_curriculum:
                per_level = {level: [] for level in all_levels}
                
                for score in mit.zip_equal(batch.difficulty_level, reward_output.values):
                    per_level[score[0]].append(score[1])

                # Accumulate with accelerate
                per_level = {
                    k: torch.tensor(v, device=ppo_trainer.accelerator.device) 
                    for k, v in per_level.items()
                }
                per_level = ppo_trainer.accelerator.gather(per_level)

                if RANK == 0:
                    for k, v in per_level.items():
                        if v is not None and len(v) > 0:
                            wandb.log(
                                {
                                    f"reward_per_level/{k}": v.mean().item(),
                                    f"sample_count_per_level/{k}": 
                                    len(v) / (batch_size * ppo_trainer.accelerator.num_processes),
                                }, 
                                step=ppo_trainer.current_step,
                            )

            stats = ppo_trainer.step(
                queries   = batch.tok_ref_query,
                responses = outputs.response_tensors,
                scores    = reward_output.values,
                **step_kwargs,
            )
            

            if RANK == 0: 
                print(f"{RANK} ppo_trainer.step done <<<")

            # Log stats
            assert isinstance(reward_output.values, list), type(reward_output.values)
            assert isinstance(stats, dict), type(stats)

            batch_stats = dict(
                response         = prediction_tokenizer.batch_decode(
                    outputs.response_tensors),
                query            = batch.detok_ref_query,
                ref_answer       = batch.detok_ref_answer,
                ref_scratchpad   = batch.detok_ref_scratchpad,
                difficulty_level = batch.difficulty_level,
            )

            assert current_step == ppo_trainer.current_step, (
                current_step, ppo_trainer.current_step)
            
            ppo_trainer.log_stats(
                batch   = batch_stats,
                rewards = [x.to(torch.float32) for x in reward_output.values],
                stats   = stats,
                columns_to_log = [
                    "query", 
                    "response", 
                    "ref_answer", 
                    "ref_scratchpad", 
                    "difficulty_level",
                ],
            )
            
            current_step += 1
            ppo_trainer.current_step = current_step
            assert current_step == ppo_trainer.current_step, (
                current_step, ppo_trainer.current_step)
            
            lib_trl_utils.print_table(
                call_source       = "main_loop",
                extra_columns     = reward_output.logging_columns,
                generation_kwargs = generation_kwargs,
                log_header        = f"(e{epoch}-b{batch_idx}) ",
                name              = str(cfg.name),
                qty               = 5,
                queries           = batch.detok_ref_query,
                responses         = outputs.response_text,
                rewards           = reward_output.values,
                difficulty_levels = batch.difficulty_level,
            )



if __name__ == "__main__":
    lib_utils.print_accelerate_envs()
    main()
