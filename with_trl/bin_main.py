#!/usr/bin/env python
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["DATASETS_VERBOSITY"] = "warning"
os.environ["WANDB_SILENT"] = "true"
os.environ["NCCL_DEBUG"] = "WARN"

DETERMINISTIC = False
if DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import collections
import enum
import itertools
import logging
import random
import typing
from typing import Any, Optional, Union

import accelerate
import datasets
import fire
import numpy as np
import peft
import rich
import rich.console
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
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm

import lib_base_classes
import lib_data
import lib_eval
import lib_trl_utils
import lib_utils

rich.traceback.install()
datasets.disable_caching()

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "0"))
RANK = int(os.environ.get("RANK", "0"))
LOGGER = logging.getLogger(__name__)

datasets.logging.set_verbosity_warning()
transformers.logging.set_verbosity_warning()  # type: ignore
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("deepspeed").setLevel(logging.WARNING)

torch.autograd.set_detect_anomaly(True)

np.random.seed(0)
random.seed(1)
torch.manual_seed(2)
torch.cuda.manual_seed_all(3)
trl.set_seed(4)
trl_fork.set_seed(4)

torch.backends.cuda.matmul.allow_tf32 = not DETERMINISTIC
torch.backends.cudnn.allow_tf32 = not DETERMINISTIC
torch.use_deterministic_algorithms(DETERMINISTIC)

DEFAULT_LOG_STATS_VERBOSE = True
DEFAULT_REWARD_VERBOSE = False

##############################################################################
##############################################################################
DEFAULT_USE_FEW_SHOTS = True
DEFAULT_GEN_KWARGS = dict(
    repetition_penalty=1,
    min_new_tokens=5,
    early_stopping=True,
    synced_gpus=True,
    temperature=1.,
    top_k=0.0,
    top_p=1.0,
    use_cache=True,
    num_return_sequences=1,
)

DEFAULT_TRL_LIBRARY_MODE = lib_utils.TrlLibraryMode.TRL
DEFAULT_TASK_NAME: str = lib_utils.Task.MAIN
DEFAULT_EVAL_EVERY: int = 0
DEFAULT_VALUE_MODEL_PRETRAIN_AMOUNT: int = 1


if DEFAULT_TASK_NAME == lib_utils.Task.MAIN:
    DEFAULT_WANDB_PROJECT: str = "commonsense"
    DEFAULT_REWARD_TYPE = lib_utils.RewardChoices.EXACT_MATCH
    DEFAULT_DATASET_NAME = lib_data.DatasetChoices.COMMONSENSEQA_MC

    # -------------------------------------------------------
    #########################################################
    DEFAULT_GEN_KWARGS["do_sample"] = False
    # DEFAULT_MODEL_NAME = "EleutherAI/gpt-j-6B"; DEFAULT_BATCH_SIZE: int = 24
    DEFAULT_MODEL_NAME = "EleutherAI/pythia-410m-v0"; DEFAULT_BATCH_SIZE: int = 24
    DEFAULT_INFERENCE_BATCH_SIZE: int = DEFAULT_BATCH_SIZE
    DEFAULT_MINI_BATCH_SIZE: int = 1
    DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 1

    DEFAULT_CAUSAL_QUESTION_PREFIX: str = None
    DEFAULT_CAUSAL_QUESTION_SUFFIX: str = None

    DEFAULT_PEFT_QLORA_MODE = False
    DEFAULT_PRECISION = lib_utils.ValidPrecisions.bfloat16

    #########################################################
    # -------------------------------------------------------

    DEFAULT_GEN_KWARGS["max_new_tokens"] = 200

    assert DEFAULT_EVAL_EVERY == 0 or DEFAULT_EVAL_EVERY >= (
        DEFAULT_GRADIENT_ACCUMULATION_STEPS // WORLD_SIZE
    ), (
        f"DEFAULT_EVAL_EVERY ({DEFAULT_EVAL_EVERY}) must be >= "
        f"DEFAULT_GRADIENT_ACCUMULATION_STEPS "
        f"({DEFAULT_GRADIENT_ACCUMULATION_STEPS})"
    )

    DEFAULT_INFERENCE_GEN_KWARGS = DEFAULT_GEN_KWARGS.copy()
    DEFAULT_INFERENCE_GEN_KWARGS["num_beams"] = 1
    DEFAULT_INFERENCE_GEN_KWARGS["do_sample"] = False
    DEFAULT_INFERENCE_GEN_KWARGS["num_return_sequences"] = 1
    # We could use a custom batch size too.


elif DEFAULT_TASK_NAME == lib_utils.Task.SENTIMENT:
    DEFAULT_WANDB_PROJECT: str = "sentiment"
    DEFAULT_GEN_KWARGS["max_new_tokens"] = 20
    DEFAULT_GEN_KWARGS["min_new_tokens"] = 4
    DEFAULT_GEN_KWARGS["num_return_sequences"] = 1
    DEFAULT_GEN_KWARGS["do_sample"] = True
    DEFAULT_EVAL_BATCH_SIZE: int = 24
    DEFAULT_MINI_BATCH_SIZE: int = 24
    DEFAULT_BATCH_SIZE: int = 24
    DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 1

    DEFAULT_PEFT_QLORA_MODE = True
    DEFAULT_REWARD_TYPE: typing.Optional[str] = None

    DEFAULT_PRECISION = lib_utils.ValidPrecisions.bfloat16
    # DEFAULT_MODEL_NAME: str                   = "huggyllama/llama-65b"
    DEFAULT_MODEL_NAME: str = "tiiuae/falcon-40b-instruct"
    DEFAULT_CAUSAL_QUESTION_PREFIX: str = ""
    DEFAULT_CAUSAL_QUESTION_SUFFIX: str = ""

    DEFAULT_DATASET_NAME = lib_data.DatasetChoices.SENTIMENT
    DEFAULT_INFERENCE_BATCH_SIZE: int = 96
    DEFAULT_INFERENCE_GEN_KWARGS = DEFAULT_GEN_KWARGS.copy()

else:
    raise ValueError(f"Unknown task name: {DEFAULT_TASK_NAME}")


##############################################################################
##############################################################################
DEFAULT_EVAL_QTY: int = 100
DEFAULT_NUM_EPOCHS: int = 10
DEFAULT_USE_PEFT: bool = True

DEFAULT_LEARNING_RATE: float = 1.41e-5

DEFAULT_PEFT_CONFIG = dict(
    inference_mode=False,
    lora_dropout=0.,
    lora_alpha=16,
    bias="none",
    r=16,
)

def main(
    *,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    inference_gen_kwargs: dict[str, Any] = DEFAULT_INFERENCE_GEN_KWARGS,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    generation_kwargs: dict[str, Any] = DEFAULT_GEN_KWARGS,
    peft_config_dict: Optional[dict[str, Any]] = DEFAULT_PEFT_CONFIG,
    input_max_length: int = 115,
    eval_subset_size: int = DEFAULT_EVAL_QTY,
    mini_batch_size: int = DEFAULT_MINI_BATCH_SIZE,
    causal_question_prefix: str = DEFAULT_CAUSAL_QUESTION_PREFIX,
    causal_question_suffix: str = DEFAULT_CAUSAL_QUESTION_SUFFIX,
    peft_qlora_mode: bool = DEFAULT_PEFT_QLORA_MODE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    dataset_name: str = DEFAULT_DATASET_NAME,
    just_metrics: bool = False,
    reward_type: Union[None, str, lib_utils.RewardChoices] = DEFAULT_REWARD_TYPE,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_every: int = DEFAULT_EVAL_EVERY,
    precision: Union[str, torch.dtype, lib_utils.ValidPrecisions] = DEFAULT_PRECISION,
    task_name: lib_utils.Task = lib_utils.Task(DEFAULT_TASK_NAME),
    use_peft: bool = DEFAULT_USE_PEFT,
    name: Optional[str] = None,
    use_few_shots: int = DEFAULT_USE_FEW_SHOTS,
    trl_library_mode: lib_utils.TrlLibraryMode = DEFAULT_TRL_LIBRARY_MODE,
    value_model_pretrain_amount: int = DEFAULT_VALUE_MODEL_PRETRAIN_AMOUNT,
):
    precision = lib_utils.ValidPrecisions(precision)  # type: ignore
    args = locals().copy()
    
    if RANK == 0:
        table = rich.table.Table("Key", "Value", title="Command Line Arguments", show_lines=True)
        for key, value in sorted(args.items(), key=lambda x: x[0]):
            table.add_row("[bold]" + rich.markup.escape(str(key)), rich.markup.escape(str(value)))
        rich.print(table)

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

    task_name = lib_utils.Task(task_name)
    dataset_name = lib_data.DatasetChoices(dataset_name)
    trl_library_mode = lib_utils.TrlLibraryMode(trl_library_mode)
    trl_library = lib_utils.TRL_LIBRARIES[trl_library_mode]

    ###########################################################################
    # Find the type of model we are using
    ###########################################################################
    hf_config = transformers.AutoConfig.from_pretrained(  # type: ignore
        model_name, 
    )
    
    if hf_config.is_encoder_decoder:
        assert causal_question_prefix == "", (
            causal_question_prefix,
            causal_question_suffix,
        )
        assert causal_question_suffix == "", (
            causal_question_suffix,
            causal_question_suffix,
        )

    if not peft_qlora_mode:
        assert peft_config_dict is not None
        assert "task_type" not in peft_config_dict

        if not hf_config.is_encoder_decoder:
            peft_config_dict["task_type"] = peft.TaskType.CAUSAL_LM
        elif hf_config.is_encoder_decoder:
            peft_config_dict["task_type"] = peft.TaskType.SEQ_2_SEQ_LM
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        assert "target_modules" not in peft_config_dict, peft_config_dict
        peft_config_dict[
            "target_modules"
        ] = peft.utils.other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
            hf_config.model_type
        ]
    else:
        peft_config_dict = None

    ppo_config_dict = dict(
        gradient_accumulation_steps=gradient_accumulation_steps,
        accelerator_kwargs=dict(
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=False)
            ]
        ), 
        mini_batch_size=mini_batch_size,
        learning_rate=learning_rate,
        model_name=model_name,
        batch_size=batch_size,
        log_with="wandb",
    )

    trl_config = trl_library.PPOConfig(
        **ppo_config_dict,
    )

    if task_name == lib_utils.Task.MAIN:
        reward_type = lib_utils.RewardChoices(reward_type)

    if RANK == 0:
        wandb.init(
            save_code=True,
            project=wandb_project,
            entity="julesgm",
            name=name,
            config=dict(
                generation_kwargs=generation_kwargs,
                peft_config_dict=peft_config_dict,
                ppo_config_args=ppo_config_dict,
                script_args=args,
            ),
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
            peft_config_dict=peft_config_dict,
            peft_qlora_mode=peft_qlora_mode,
            model_name=trl_config.model_name,
            precision=precision,
            use_peft=use_peft,
            trl_library_mode=trl_library_mode,
        )

        # Deal with fork vs non-fork
        forward_tokenizer = output["forward_tokenizer"]
        prediction_tokenizer = output["prediction_tokenizer"]
        trainer_kwargs = {}
        if trl_library_mode == lib_utils.TrlLibraryMode.TRL:
            trainer_kwargs["model"] = output["model"]
        elif trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
            trainer_kwargs["policy_model"] = output["policy_model"]
            trainer_kwargs["value_model"] = output["value_model"]
        else:
            raise ValueError(f"Unknown trl_library_mode: {trl_library_mode}")

        eos_token_id = forward_tokenizer.eos_token_id
        assert eos_token_id == prediction_tokenizer.eos_token_id
        pad_token_id = forward_tokenizer.pad_token_id
        assert pad_token_id == prediction_tokenizer.pad_token_id

    ###########################################################################
    # Load Datasets
    ###########################################################################
    dataset = lib_data.prep_dataset_rl(
        input_max_length=input_max_length,
        question_prefix=causal_question_prefix,
        question_suffix=causal_question_suffix,
        any_tokenizer=forward_tokenizer,
        use_few_shots=use_few_shots,
        dataset_name=dataset_name,
        split="train",
    )

    eval_dataset = lib_data.prep_dataset_rl(
        input_max_length=input_max_length,
        question_prefix=causal_question_prefix,
        question_suffix=causal_question_suffix,
        any_tokenizer=forward_tokenizer,
        use_few_shots=use_few_shots,
        dataset_name=dataset_name,
        split="eval",
    )
    
    ###########################################################################
    # Set model name specific flags
    ###########################################################################
    if not hf_config.is_encoder_decoder:
        if peft_config_dict:
            assert peft_config_dict["task_type"] == peft.TaskType.CAUSAL_LM
        generation_kwargs["pad_token_id"] = eos_token_id

    ###########################################################################
    # Prep Training
    ###########################################################################
    ppo_trainer: trl.PPOTrainer = trl_library.PPOTrainer(
        data_collator=(
            lib_utils.collator if task_name == lib_utils.Task.SENTIMENT 
            else lib_data.data_item_collator
        ),
        ref_model=None,
        tokenizer=forward_tokenizer,
        dataset=dataset,
        config=trl_config,
        **trainer_kwargs,
    )

    metric_accuracy, reward_fn = lib_eval.make_metric_and_reward_fn(
        ppo_trainer=ppo_trainer,
        reward_type=reward_type,
        task_name=task_name,
        extractor=dataset.get_extractor(),
        use_peft=use_peft,
    )

    policy_model = ppo_trainer.model if trl_library_mode == lib_utils.TrlLibraryMode.TRL else ppo_trainer.policy_model

    train_eval = lib_eval.EvalLoop(
        inference_gen_kwargs=inference_gen_kwargs,
        prediction_tokenizer=prediction_tokenizer,
        forward_tokenizer=forward_tokenizer,
        accelerated_model=policy_model,
        eval_subset_size=eval_subset_size,
        metric_accuracy=metric_accuracy,
        use_few_shots=use_few_shots,
        dataset_type=dataset_name,
        accelerator=ppo_trainer.accelerator,
        batch_size=inference_batch_size,
        reward_fn=reward_fn,
        task_name=task_name,
        dataset=dataset,
        split="train",
    )

    eval_eval = lib_eval.EvalLoop(
        inference_gen_kwargs=inference_gen_kwargs,
        prediction_tokenizer=prediction_tokenizer,
        forward_tokenizer=forward_tokenizer,
        accelerated_model=policy_model,
        eval_subset_size=eval_subset_size,
        metric_accuracy=metric_accuracy,
        use_few_shots=use_few_shots,
        dataset_type=dataset_name,
        accelerator=ppo_trainer.accelerator,
        batch_size=inference_batch_size,
        reward_fn=reward_fn,
        task_name=task_name,
        dataset=eval_dataset,
        split="eval",
    )

    if just_metrics:
        train_eval()
        eval_eval()
        return

    ###########################################################################
    # Training Loop
    ###########################################################################
    class TrainingState(enum.Enum):
        REGULAR_TRAINING = "regular_training"
        VALUE_MODEL_PRETRAINING = "value_model_pretraining"

    epoch_counts = collections.defaultdict(lambda: -1)

    if trl_library_mode == lib_utils.TrlLibraryMode.TRL:
        needs_params = [
            param for param in ppo_trainer.model.parameters() 
            if param.requires_grad
        ]
    elif trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
        needs_params = [
            param for param in ppo_trainer.policy_model.parameters() 
            if param.requires_grad
        ] + [
            param for param in ppo_trainer.value_model.parameters() 
            if param.requires_grad
        ]
    else:
        raise ValueError(trl_library_mode)

    for epoch in itertools.count():
        if epoch < value_model_pretrain_amount:
            training_state = TrainingState.VALUE_MODEL_PRETRAINING
            for x in needs_params:
                x.requires_grad = False
        else:
            training_state = TrainingState.REGULAR_TRAINING
            for x in needs_params:
                x.requires_grad = True

        epoch_counts[training_state] += 1

        for batch_idx, batch in enumerate(
            lib_utils.progress(
                description=(
                    f"{training_state} "
                    f"Epoch {epoch_counts[training_state]}, "
                    f"Global Epoch: {epoch}"
                    ),
                disable=True,
                seq=ppo_trainer.dataloader,
            )
        ):
            ############################################################
            # Keys of batch:
            #   - "query"
            #   - "input_ids"
            #   - "ref_answer" if in GSM8K
            #   - "ref_scratchpad"
            ############################################################

            if eval_every and batch_idx % eval_every == 0:
                rich.print("[red bold]DOING EVAL: [white]TRAIN SET")
                train_eval()
                rich.print("[red bold]DOING EVAL: [white]EVAL SET")
                eval_eval()
                rich.print("[red bold]DONE WITH EVAL")

            print(f"{RANK} lib_trl_utils.batched_unroll >>>")
            raw_gen_outputs = lib_trl_utils.batched_unroll(
                accelerated_model=policy_model,
                generation_kwargs=generation_kwargs,
                query_tensors=batch.tok_ref_query,
                accelerator=ppo_trainer.accelerator,
                prediction_tokenizer=prediction_tokenizer,
                dataset_name=dataset_name, 
                use_few_shots=use_few_shots,
                dataset_obj=dataset,
                task_name=task_name, 
            )
            print(f"{RANK} lib_trl_utils.batched_unroll <<<")

            if task_name == lib_utils.Task.MAIN:
                outputs = lib_trl_utils.keep_good_one_generation(
                    num_return_seq=generation_kwargs["num_return_sequences"],
                    other_rewards=None,
                    generations=raw_gen_outputs,
                    ref_answers=batch.detok_ref_answer,
                    batch_size=batch_size,
                    prediction_tokenizer=prediction_tokenizer,
                    answer_extractor=dataset.get_extractor()
                )
            else:
                assert task_name == lib_utils.Task.SENTIMENT, task_name

                assert (
                    generation_kwargs["num_return_sequences"] == 1
                ), generation_kwargs["num_return_sequences"]
                outputs = raw_gen_outputs
                assert len(outputs.response_tensors) == batch_size, (
                    len(outputs.response_tensors),
                    batch_size,
                )

            outputs = lib_base_classes.BatchedUnrollReturn(
                response_tensors=lib_trl_utils.unpad(
                    responses=outputs.response_tensors,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                ),
                raw_response_tensors=lib_trl_utils.unpad(
                    outputs.raw_response_tensors,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                ),
                any_tokenizer=prediction_tokenizer,
            )

            reward_output = reward_fn(
                batch=batch,
                responses=outputs.response_text,
            )

            metric_output = metric_accuracy(
                batch=batch,
                responses=outputs.response_text,
            )

            ###########################################################################
            # Print Rewards
            ###########################################################################
            lib_trl_utils.log_reward(
                reward_output=reward_output,
                metric_output=metric_output,
                ppo_trainer=ppo_trainer,
                batch_idx=batch_idx,
                epoch=epoch,
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

            lib_trl_utils.check_max_qty_of_token_id(
                list_of_sequences=outputs.response_tensors,
                max_qty=1,
                token_id=eos_token_id,
            )

            # There should be no pad_token_ids, but the pad
            # token id might be the eos token id, so we can't
            # just blindly check for the pad token id
            if pad_token_id != eos_token_id:
                lib_trl_utils.check_qty_of_token_id(
                    list_of_sequences=outputs.response_tensors,
                    qty=0,
                    token_id=pad_token_id,
                )

            print(f"{RANK} ppo_trainer.step >>>")

            step_kwargs = {}
            if trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
                step_kwargs["answers"] = batch.tok_ref_answer

            stats = ppo_trainer.step(
                queries=batch.tok_ref_query,
                responses=outputs.response_tensors,
                scores=reward_output.values,
                **step_kwargs,
            )

            import ipdb; ipdb.set_trace()
            print(f"{RANK} ppo_trainer.step done <<<")

            # Log stats
            assert isinstance(reward_output.values, list), type(reward_output.values)
            assert isinstance(stats, dict), type(stats)

            batch_stats = dict(
                response = prediction_tokenizer.batch_decode(outputs.response_tensors),
                query    = batch.detok_ref_query,
            )

            ppo_trainer.log_stats(
                rewards=[x.to(torch.float32) for x in reward_output.values],
                batch=batch_stats,
                stats=stats,
            )

            lib_trl_utils.print_table(
                extra_columns=reward_output.logging_columns,
                log_header=f"(e{epoch}-b{batch_idx}) ",
                responses=outputs.response_text,
                queries=batch.detok_ref_query,
                rewards=reward_output.values,
                name=str(name),
                qty=5,
                generation_kwargs=generation_kwargs,
            )


if __name__ == "__main__":
    lib_utils.print_accelerate_envs()
    fire.Fire(main)
