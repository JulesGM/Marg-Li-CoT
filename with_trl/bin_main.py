#!/usr/bin/env python
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["DATASETS_VERBOSITY"] = "warning"
os.environ["WANDB_SILENT"] = "true"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.sampler
import transformers
import trl
import trl.core
import wandb
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm

import lib_base_classes
import lib_data
import lib_eval
import lib_metric
import lib_reward_exact_match
import lib_reward_ppl
import lib_sentiment_specific
import lib_trl_utils
import lib_utils


def rich_escape(value):
    return rich.markup.escape(str(value))


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "0"))
RANK = int(os.environ.get("RANK", "0"))


LOGGER = logging.getLogger(__name__)

datasets.logging.set_verbosity_warning()
transformers.logging.set_verbosity_warning()  # type: ignore
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("deepspeed").setLevel(logging.WARNING)

np.random.seed(0)
random.seed(1)
torch.manual_seed(2)
torch.cuda.manual_seed_all(3)
trl.set_seed(4)

torch.use_deterministic_algorithms(True)

DEFAULT_LOG_STATS_VERBOSE = True
DEFAULT_REWARD_VERBOSE = False

##############################################################################
##############################################################################

DEFAULT_GEN_KWARGS = dict(
    repetition_penalty=5.0,
    min_new_tokens=5,
    top_k=0.0,
    top_p=1.0,
    early_stopping=True,
    synced_gpus=True,
)

DEFAULT_TASK_NAME: str = lib_utils.Task.GSM8K
DEFAULT_EVAL_EVERY: int = 0

if DEFAULT_TASK_NAME == lib_utils.Task.GSM8K:
    DEFAULT_WANDB_PROJECT: str = "gsm8k"
    DEFAULT_REWARD_TYPE = lib_utils.GSM8KRewardChoices.EXACT_MATCH

    # -------------------------------------------------------
    # DEFAULT_GEN_KWARGS["temperature"] = 1
    DEFAULT_GEN_KWARGS["do_sample"] = False
    # -------------------------------------------------------
    #########################################################
    DEFAULT_GEN_KWARGS["num_beams"] = 20
    DEFAULT_GEN_KWARGS["num_return_sequences"] = 20

    DEFAULT_MODEL_NAME: str = "ausboss/llama-30b-supercot"
    DEFAULT_PEFT_QLORA_MODE = True
    DEFAULT_PRECISION = lib_utils.ValidPrecisions.bfloat16

    # DEFAULT_MODEL_NAME: str = "tiiuae/falcon-40b-instruct"
    # DEFAULT_GEN_KWARGS["beam_search_sub_batch_size"] = 32
    #########################################################
    # -------------------------------------------------------

    DEFAULT_GEN_KWARGS["max_new_tokens"] = 200
    DEFAULT_MINI_BATCH_SIZE: int = 1
    DEFAULT_BATCH_SIZE: int = 1
    DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 16

    assert DEFAULT_EVAL_EVERY == 0 or DEFAULT_EVAL_EVERY >= (
        DEFAULT_GRADIENT_ACCUMULATION_STEPS // WORLD_SIZE
    ), (
        f"DEFAULT_EVAL_EVERY ({DEFAULT_EVAL_EVERY}) must be >= "
        f"DEFAULT_GRADIENT_ACCUMULATION_STEPS "
        f"({DEFAULT_GRADIENT_ACCUMULATION_STEPS})"
    )

    DEFAULT_CAUSAL_QUESTION_PREFIX: str = "### Instructions: "
    DEFAULT_CAUSAL_QUESTION_SUFFIX: str = "\nBe concise.\n### Response:\n"
    DEFAULT_INFERENCE_BATCH_SIZE: int = 4
    DEFAULT_INFERENCE_GEN_KWARGS = DEFAULT_GEN_KWARGS.copy()
    DEFAULT_INFERENCE_GEN_KWARGS["num_beams"] = 4
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

    DEFAULT_INFERENCE_BATCH_SIZE: int = 96
    DEFAULT_INFERENCE_GEN_KWARGS = DEFAULT_GEN_KWARGS.copy()

else:
    raise ValueError(f"Unknown task name: {DEFAULT_TASK_NAME}")


##############################################################################
##############################################################################
DEFAULT_EVAL_QTY: int = 50
DEFAULT_NUM_EPOCHS: int = 10
DEFAULT_USE_PEFT: bool = True

DEFAULT_LEARNING_RATE: float = 1.41e-5

DEFAULT_PEFT_CONFIG = dict(
    inference_mode=False,
    lora_dropout=0.0,
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
    learning_rate: float = DEFAULT_LEARNING_RATE,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    just_metrics: bool = False,
    reward_type: Union[None, str, lib_utils.GSM8KRewardChoices] = DEFAULT_REWARD_TYPE,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_every: int = DEFAULT_EVAL_EVERY,
    precision: Union[str, torch.dtype, lib_utils.ValidPrecisions] = DEFAULT_PRECISION,
    task_name: lib_utils.Task = lib_utils.Task(DEFAULT_TASK_NAME),
    use_peft: bool = DEFAULT_USE_PEFT,
    name: Optional[str] = None,
    peft_qlora_mode: bool = DEFAULT_PEFT_QLORA_MODE,
):
    precision = lib_utils.ValidPrecisions(precision)  # type: ignore
    args = locals().copy()

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

    ###########################################################################
    # Find the type of model we are using
    ###########################################################################
    config = transformers.AutoConfig.from_pretrained(  # type: ignore
        model_name, trust_remote_code=True
    )

    if config.is_encoder_decoder:
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

        if not config.is_encoder_decoder:
            peft_config_dict["task_type"] = peft.TaskType.CAUSAL_LM
        elif config.is_encoder_decoder:
            peft_config_dict["task_type"] = peft.TaskType.SEQ_2_SEQ_LM
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        assert "target_modules" not in peft_config_dict, peft_config_dict
        peft_config_dict[
            "target_modules"
        ] = peft.utils.other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
            config.model_type
        ]
    else:
        peft_config_dict = None

    ppo_config_dict = dict(
        gradient_accumulation_steps=gradient_accumulation_steps,
        accelerator_kwargs=dict(
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        ),  # type: ignore
        mini_batch_size=mini_batch_size,
        learning_rate=learning_rate,
        model_name=model_name,
        batch_size=batch_size,
        log_with="wandb",
    )

    config = trl.PPOConfig(
        **ppo_config_dict,
    )

    if task_name == lib_utils.Task.GSM8K:
        reward_type = lib_utils.GSM8KRewardChoices(reward_type)

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

    assert isinstance(config.model_name, str), type(config.model_name)

    with lib_utils.maybe_context_manager(
        lambda: rich.status.Status(
            f"[bold green]Loading model: "
            f"[white]{rich_escape(config.model_name)} [green]...",
            spinner="weather",
        ),
        disable=True,  # RANK != 0,
    ):
        model, tokenizer = lib_trl_utils.init_model(
            peft_config_dict=peft_config_dict,
            model_name=config.model_name,
            precision=precision,
            use_peft=use_peft,
            peft_qlora_mode=peft_qlora_mode,
        )

    dataset = lib_data.prep_dataset(
        input_max_length=input_max_length,
        task_name=task_name,
        tokenizer=tokenizer,
        split="train",
        question_prefix=causal_question_prefix,
        question_suffix=causal_question_suffix,
    )

    eval_dataset = lib_data.prep_dataset(
        input_max_length=input_max_length,
        task_name=task_name,
        tokenizer=tokenizer,
        split="test",
        question_prefix=causal_question_prefix,
        question_suffix=causal_question_suffix,
    )

    ###########################################################################
    # Set model name specific flags
    ###########################################################################
    if not model.config.is_encoder_decoder:
        if peft_config_dict:
            assert peft_config_dict["task_type"] == peft.TaskType.CAUSAL_LM
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    ###########################################################################
    # Prep Training
    ###########################################################################
    ppo_trainer = trl.PPOTrainer(
        data_collator=lib_utils.collator,
        ref_model=None,
        tokenizer=typing.cast(
            typing.Union[
                transformers.PreTrainedTokenizer,  # type: ignore
                transformers.PreTrainedTokenizerFast,
            ],  # type: ignore
            tokenizer,
        ),
        dataset=dataset,
        config=config,
        model=model,
    )

    metric_accuracy, reward_fn = lib_eval.make_metric_and_reward_fn(
        ppo_trainer=ppo_trainer,
        reward_type=reward_type,
        task_name=task_name,
        tokenizer=tokenizer,
        use_peft=use_peft,
    )

    train_eval = lib_eval.EvalLoop(
        inference_gen_kwargs=inference_gen_kwargs,
        batch_size=inference_batch_size,
        eval_subset_size=eval_subset_size,
        metric_accuracy=metric_accuracy,
        ppo_trainer=ppo_trainer,
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        task_name=task_name,
        dataset=dataset,
        split="train",
    )

    eval_eval = lib_eval.EvalLoop(
        inference_gen_kwargs=inference_gen_kwargs,
        batch_size=inference_batch_size,
        eval_subset_size=eval_subset_size,
        metric_accuracy=metric_accuracy,
        ppo_trainer=ppo_trainer,
        reward_fn=reward_fn,
        tokenizer=tokenizer,
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
    def answer_extractor(sample):
        return metric_accuracy._make_comparable(metric_accuracy._extract_answer(sample))

    for epoch in itertools.count():
        for batch_idx, batch in enumerate(
            lib_utils.progress(
                description=f"Training - Epoch {epoch}",
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

            raw_gen_outputs = lib_trl_utils.batched_unroll(
                generation_kwargs=generation_kwargs,
                query_tensors=batch["input_ids"],
                ppo_trainer=ppo_trainer,
                tokenizer=tokenizer,
            )

            if task_name == lib_utils.Task.GSM8K:
                outputs = lib_trl_utils.keep_good_one_generation(
                    num_return_seq=generation_kwargs["num_return_sequences"],
                    other_rewards=None,
                    generations=raw_gen_outputs,
                    ref_answers=batch["ref_answer"],
                    extract_fn=answer_extractor,
                    batch_size=batch_size,
                    tokenizer=tokenizer,
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

            outputs = lib_trl_utils.BatchedUnrollReturn(
                response_tensors=lib_trl_utils.unpad(
                    responses=outputs.response_tensors,
                    tokenizer=tokenizer,
                ),
                tokenizer=tokenizer,
            )

            # We need a "ref_answer" iff we are doing GSM8K.
            # We shouldn't have a "ref_answer" iff we are doing sentiment.
            assert ("ref_answer" in batch) is (
                task_name == lib_utils.Task.GSM8K
            ), task_name
            assert (not ("ref_answer" in batch)) is (
                task_name == lib_utils.Task.SENTIMENT
            ), task_name

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
                assert isinstance(tokenizer.pad_token_id, int), type(
                    tokenizer.pad_token_id
                )
                lib_trl_utils.check_all_start_with_token_id(
                    outputs.response_tensors, tokenizer.pad_token_id
                )

            lib_trl_utils.check_max_qty_of_token_id(
                list_of_sequences=outputs.response_tensors,
                max_qty=1,
                token_id=tokenizer.eos_token_id,
            )

            # There should be no pad_token_ids, but the pad
            # token id might be the eos token id, so we can't
            # just blindly check for the pad token id
            if tokenizer.pad_token_id != tokenizer.eos_token_id:
                lib_trl_utils.check_qty_of_token_id(
                    list_of_sequences=outputs.response_tensors,
                    qty=0,
                    token_id=tokenizer.pad_token_id,
                )

            stats = ppo_trainer.step(
                queries=batch["input_ids"],
                responses=typing.cast(list[torch.LongTensor], outputs.response_tensors),
                scores=reward_output.values,
            )

            # Log stats
            assert isinstance(reward_output.values, list), type(reward_output.values)
            assert isinstance(stats, dict), type(stats)
            assert isinstance(batch, dict), type(batch)

            batch["response"] = outputs.response_tensors

            ppo_trainer.log_stats(
                rewards=[x.to(torch.float32) for x in reward_output.values],
                batch=batch,
                stats=stats,
            )

            lib_trl_utils.print_table(
                extra_columns=reward_output.logging_columns,
                log_header=f"(b{batch_idx}e{epoch}) ",
                responses=outputs.response_text,
                queries=batch["query"],
                rewards=reward_output.values,
                name=str(name),
                qty=5,
            )


if __name__ == "__main__":
    lib_utils.print_accelerate_envs()
    fire.Fire(main)
