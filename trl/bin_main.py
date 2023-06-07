#!/usr/bin/env python
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["DATASETS_VERBOSITY"]     = "warning"
os.environ["WANDB_SILENT"]           = "true"
os.environ["NCCL_DEBUG"]             = "WARN"


import enum
import itertools
import logging
import random
import typing

import accelerate
import datasets
import fire
import numpy as np
import peft
import rich
import rich.console
import rich.logging
import rich.status
import rich.table
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.sampler
import transformers
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm

import lib_base_classes
import lib_data
import lib_eval
import lib_utils
import lib_metric
import lib_reward_exact_match
import lib_reward_ppl
import lib_sentiment_specific
import lib_trl_utils
import trl
import trl.core
import wandb


LOGGER = logging.getLogger(__name__)

datasets    .logging.set_verbosity_warning()
transformers.logging.set_verbosity_warning()
logging.getLogger("datasets"    ).setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("deepspeed"   ).setLevel(logging.WARNING)

np.random            .seed(0)
random               .seed(1)
torch         .manual_seed(2)
torch.cuda.manual_seed_all(3)
trl              .set_seed(4)

DEFAULT_LOG_STATS_VERBOSE = True
DEFAULT_REWARD_VERBOSE    = False

##############################################################################
##############################################################################

DEFAULT_GEN_KWARGS = dict(
    repetition_penalty = 5.0,
    min_length         = 4,
    top_k              = 0.0,
    top_p              = 1.0,
    early_stopping     = True,
    synced_gpus        = True,
)

DEFAULT_TASK_NAME: str = "gsm8k"
DEFAULT_EVAL_EVERY: int = 16

if DEFAULT_TASK_NAME == "gsm8k":
    DEFAULT_WANDB_PROJECT: str                 = "gsm8k"
    DEFAULT_TASK_NAME:                     str = "gsm8k"
    DEFAULT_REWARD_TYPE:  typing.Optional[str] = "exact_match"

    # -------------------------------------------------------
    DEFAULT_GEN_KWARGS["temperature"]          = 0.1
    DEFAULT_GEN_KWARGS["do_sample"]            = True
    # -------------------------------------------------------
    # DEFAULT_GEN_KWARGS["num_beams"]:       int = 32
    DEFAULT_GEN_KWARGS["num_return_sequences"] = 32
    # -------------------------------------------------------

    DEFAULT_GEN_KWARGS["max_new_tokens"]       = 192
    DEFAULT_MINI_BATCH_SIZE:               int = 1
    DEFAULT_BATCH_SIZE:                    int = 1
    DEFAULT_GRADIENT_ACCUMULATION_STEPS:   int = 16

    DEFAULT_PRECISION                          = lib_utils.ValidPrecisions.bfloat16
    DEFAULT_MODEL_NAME:                    str = "openaccess-ai-collective/wizard-mega-13b"

    DEFAULT_CAUSAL_QUESTION_PREFIX:        str = "### Instruction: "
    DEFAULT_CAUSAL_QUESTION_SUFFIX:        str = "\n\n### Assistant:\n"

    DEFAULT_INFERENCE_BATCH_SIZE:          int = 1
    DEFAULT_INFERENCE_GEN_KWARGS = DEFAULT_GEN_KWARGS.copy()
    DEFAULT_INFERENCE_GEN_KWARGS["num_beams"] = 8
    DEFAULT_INFERENCE_GEN_KWARGS["do_sample"] = False
    DEFAULT_INFERENCE_GEN_KWARGS["num_return_sequences"] = 1
    # We could use a custom batch size too.


elif DEFAULT_TASK_NAME == "sentiment":
    DEFAULT_WANDB_PROJECT: str                = "sentiment"
    DEFAULT_GEN_KWARGS["max_new_tokens"]      = 20
    DEFAULT_GEN_KWARGS["do_sample"]           = True
    DEFAULT_EVAL_BATCH_SIZE:              int = 16
    DEFAULT_MINI_BATCH_SIZE:              int = 16
    DEFAULT_BATCH_SIZE:                   int = 16
    DEFAULT_GRADIENT_ACCUMULATION_STEPS: int  = 1

    DEFAULT_REWARD_TYPE: typing.Optional[str] = None

    DEFAULT_PRECISION                         = lib_utils.ValidPrecisions.int8
    DEFAULT_MODEL_NAME:                   str = "edbeeching/gpt-neo-125M-imdb-lora-adapter-merged"
    DEFAULT_CAUSAL_ANSWER_PREFIX:                str = ""
    DEFAULT_CAUSAL_QUESTION_PREFIX:              str = ""

    DEFAULT_INFERENCE_BATCH_SIZE:          int = 4
    # DEFAULT_MODEL_NAME:            str = "edbeeching/gpt-neo-2.7B-imdb"
    # DEFAULT_MODEL_NAME:            str = "edbeeching/gpt-neox-20b-imdb-lora-lr5e-5-adapter-merged"
    DEFAULT_INFERENCE_GEN_KWARGS = DEFAULT_GEN_KWARGS.copy()

else:
    raise ValueError(f"Unknown task name: {DEFAULT_TASK_NAME}")


##############################################################################
##############################################################################
DEFAULT_EVAL_QTY:                    int   = 200
DEFAULT_NUM_EPOCHS:                  int   = 10
DEFAULT_USE_PEFT:                    bool  = True

DEFAULT_LEARNING_RATE:               float = 1.41e-5

DEFAULT_PEFT_CONFIG = dict(
    inference_mode = False,
    lora_dropout   = 0.05,
    lora_alpha     = 4,
    bias           = "none",
    r              = 4,
)

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "0"))
RANK       = int(os.environ.get("RANK",       "0"))


def prep_dataset(
    *,
    input_max_length: int, 
    question_prefix: str,
    question_suffix: str,
    task_name: str, 
    tokenizer: transformers.PreTrainedTokenizerBase,
    split: str,
) -> torch.utils.data.Dataset:
    
    if task_name == lib_utils.Task.GSM8K:
        assert isinstance(LOCAL_RANK, int), type(LOCAL_RANK)
        dataset = lib_data.GSM8K(
            max_length = input_max_length, 
            tokenizer  = tokenizer,
            device     = torch.device(LOCAL_RANK),
            ds         = datasets.load_dataset(
                split = split,
                path  = "gsm8k", 
                name  = "main", 
            ),
            question_prefix = question_prefix,
            question_suffix = question_suffix,
        )
        
    elif task_name == "asdiv":
        assert split is None, "split must be None for ASDiv"
        dataset = lib_data.ASDiv(
            input_max_length, 
            tokenizer, 
            datasets.load_dataset("asdiv"),
        )

    elif task_name == lib_utils.Task.SENTIMENT:
        assert split == "train", "split must be None for sentiment"
        dataset = lib_sentiment_specific.prep_dataset(
            txt_in_len = 5,
            tokenizer  = tokenizer,
        )

    else:
        raise ValueError(f"Unknown task: {task_name}")

    return dataset


def main(
    *, 
    gradient_accumulation_steps: int          = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    inference_gen_kwargs: dict[str, typing.Any] = DEFAULT_INFERENCE_GEN_KWARGS,
    inference_batch_size: int                 = DEFAULT_INFERENCE_BATCH_SIZE,
    generation_kwargs: dict[str, typing.Any]  = DEFAULT_GEN_KWARGS,
    peft_config_dict:  dict[str, typing.Any]  = DEFAULT_PEFT_CONFIG, 
    input_max_length: int                     = 115,
    eval_subset_size: int                     = DEFAULT_EVAL_QTY,
    mini_batch_size: int                      = DEFAULT_MINI_BATCH_SIZE,
    causal_question_prefix: str                      = DEFAULT_CAUSAL_QUESTION_PREFIX,
    causal_question_suffix: str                      = DEFAULT_CAUSAL_QUESTION_SUFFIX,
    learning_rate: float                      = DEFAULT_LEARNING_RATE,
    wandb_project: str                        = DEFAULT_WANDB_PROJECT,
    just_metrics: bool                        = False,
    reward_type: typing.Union[None, str, lib_utils.GSM8KRewardChoices] = DEFAULT_REWARD_TYPE,
    model_name: str                           = DEFAULT_MODEL_NAME,
    batch_size: int                           = DEFAULT_BATCH_SIZE,
    eval_every: int                           = DEFAULT_EVAL_EVERY,
    precision: typing.Union[str, torch.dtype, lib_utils.ValidPrecisions
                                            ] = DEFAULT_PRECISION,
    task_name: lib_utils.Task                 = lib_utils.Task(DEFAULT_TASK_NAME),
    use_peft: bool                            = DEFAULT_USE_PEFT,
    name: typing.Optional[str]                = None,
):
    precision = lib_utils.ValidPrecisions(precision)  # type: ignore
    args = locals().copy()
    
  

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%H:%M:%S",  
        handlers=[rich.logging.RichHandler(markup=True)],
        format=f"[{RANK}/{WORLD_SIZE}]%(funcName)s:%(lineno)d - %(message)s",
    )
  
    logging.getLogger("transformers").setLevel(logging.ERROR)
    # logging.getLogger("lib_metric").setLevel(logging.ERROR)

    task_name = lib_utils.Task(task_name)

    ###########################################################################
    # Find the type of model we are using
    ###########################################################################
    config = transformers.AutoConfig.from_pretrained(model_name)
    assert "task_type" not in peft_config_dict
    if not config.is_encoder_decoder:
        peft_config_dict["task_type"] = peft.TaskType.CAUSAL_LM
    
    elif config.is_encoder_decoder:
        causal_question_prefix = ""
        causal_question_suffix = ""
        peft_config_dict["task_type"] = peft.TaskType.SEQ_2_SEQ_LM
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    assert "target_modules" not in peft_config_dict, peft_config_dict
    peft_config_dict["target_modules"] = (
        peft.utils.other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
            config.model_type])

    ppo_config_dict = dict(
        gradient_accumulation_steps = gradient_accumulation_steps,
        accelerator_kwargs          = dict(kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)]), # type: ignore
        mini_batch_size             = mini_batch_size,
        learning_rate               = learning_rate,
        model_name                  = model_name,
        batch_size                  = batch_size,
        log_with                    = "wandb",
    )

    config = trl.PPOConfig(
        **ppo_config_dict,
    )

    if task_name == lib_utils.Task.GSM8K:
        reward_type = lib_utils.GSM8KRewardChoices(reward_type)

    wandb.init(
        save_code = True,
        project   = wandb_project,
        entity    = "julesgm",
        name      = name,
        config    = dict(
            generation_kwargs = generation_kwargs,
            peft_config_dict  = peft_config_dict,
            ppo_config_args   = ppo_config_dict,
            script_args       = args,
        ),
    )

    assert isinstance(config.model_name, str), (
        type(config.model_name))
    
    model, tokenizer = lib_trl_utils.init_model(
        peft_config_dict = peft_config_dict,
        model_name       = config.model_name,
        precision        = precision,
        use_peft         = use_peft,
    )

    dataset = prep_dataset(
        input_max_length = input_max_length, 
        task_name        = task_name, 
        tokenizer        = tokenizer,
        split            = "train",
        question_prefix  = causal_question_prefix,
        question_suffix  = causal_question_suffix,
    )

    eval_dataset =  prep_dataset(
        input_max_length = input_max_length, 
        task_name        = task_name, 
        tokenizer        = tokenizer,
        split            = "test",
        question_prefix  = causal_question_prefix,
        question_suffix  = causal_question_suffix,
    )

    ###########################################################################
    # Set model name specific flags
    ###########################################################################
    if not model.config.is_encoder_decoder:
        assert peft_config_dict["task_type"] == peft.TaskType.CAUSAL_LM
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
        generation_kwargs["eos_token_id"] = -1
        generation_kwargs["min_length"]   = -1
    
    ###########################################################################
    # Prep Training
    ###########################################################################
    ppo_trainer = trl.PPOTrainer(
        data_collator = lib_utils.collator,
        ref_model     = None,
        tokenizer     = typing.cast(
            typing.Union[
                transformers.PreTrainedTokenizer, 
                transformers.PreTrainedTokenizerFast,],
            tokenizer,
        ),
        dataset       = dataset,
        config        = config, 
        model         = model,
    )

    metric_accuracy, reward_fn = lib_eval.make_metric_and_reward_fn(
        ppo_trainer = ppo_trainer,
        reward_type = reward_type,
        task_name   = task_name,
        tokenizer   = tokenizer,
        use_peft    = use_peft,
    )

    train_eval = lib_eval.EvalLoop(
        inference_gen_kwargs = inference_gen_kwargs,
        batch_size           = inference_batch_size,
        eval_subset_size     = eval_subset_size,
        metric_accuracy      = metric_accuracy,
        ppo_trainer          = ppo_trainer,
        reward_fn            = reward_fn,
        tokenizer            = tokenizer,
        task_name            = task_name,
        dataset              = dataset,
        split                = "train",
    )

    eval_eval = lib_eval.EvalLoop(
        inference_gen_kwargs = inference_gen_kwargs,
        batch_size           = inference_batch_size,
        eval_subset_size     = eval_subset_size,
        metric_accuracy      = metric_accuracy,
        ppo_trainer          = ppo_trainer,
        reward_fn            = reward_fn,
        tokenizer            = tokenizer,
        task_name            = task_name,
        dataset              = eval_dataset,
        split                = "eval",
    )    

    if just_metrics:
        train_eval()
        eval_eval()
        return

    ###########################################################################
    # Training Loop
    ###########################################################################
    answer_extractor = lambda sample: (
        metric_accuracy._make_comparable(
            metric_accuracy._extract_answer(sample)
        ))

    for epoch in itertools.count():
        for batch_idx, batch in enumerate(lib_utils.progress(
            description = f"Training - Epoch {epoch}",
            disable     = True,
            seq         = ppo_trainer.dataloader, 
        )):
            ############################################################
            # Keys of batch:
            #   - "query"
            #   - "input_ids"
            #   - "ref_answer"
            #   - "ref_scratchpad"
            ############################################################
            
            if batch_idx % eval_every == 0: 
                rich.print("[red bold]DOING EVAL: [white]TRAIN SET")
                train_eval()
                rich.print("[red bold]DOING EVAL: [white]EVAL SET")
                eval_eval()
                rich.print("[red bold]DONE WITH EVAL")

            raw_gen_outputs = lib_trl_utils.batched_unroll(
                generation_kwargs = generation_kwargs, 
                query_tensors     = batch["input_ids"], 
                ppo_trainer       = ppo_trainer, 
                tokenizer         = tokenizer,
            )

            if task_name == lib_utils.Task.GSM8K:
                outputs = lib_trl_utils.keep_good_one_generation(
                    num_return_seq = generation_kwargs["num_return_sequences"],
                    other_rewards  = None, 
                    generations    = raw_gen_outputs, 
                    ref_answers    = batch["ref_answer"], 
                    extract_fn     = answer_extractor,
                    batch_size     = batch_size,
                    tokenizer      = tokenizer,
                )

            else:
                assert generation_kwargs["num_return_sequences"] == 1, (
                    generation_kwargs["num_return_sequences"])
                outputs = raw_gen_outputs
                assert len(outputs.response_tensors) == batch_size, (
                    len(outputs.response_tensors), batch_size)

            if task_name == lib_utils.Task.SENTIMENT:
                ref_answers = None
            else:
                ref_answers = batch["ref_answer"]

            reward_output = reward_fn(
                queries     = batch["query"],
                responses   = outputs.response_text,
                ref_answers = ref_answers,
            )

            ###########################################################################
            # Print Rewards
            ###########################################################################
            all_rewards = typing.cast(
                torch.Tensor, 
                ppo_trainer.accelerator.gather_for_metrics(
                    torch.tensor(reward_output.values).to(
                        ppo_trainer.accelerator.device
                    )
                )
            )

            rich.print(
                f"[bold blue]" +
                f"({lib_trl_utils.get_rank()}/{lib_trl_utils.get_world_size()}) " +
                f"({epoch = } {batch_idx = }) " +
                f"[/][white bold]" +
                f"Average rewards: " +
                f"{all_rewards.mean().item():0.4} " +
                f"+- {all_rewards.std().item():0.1}"
            )

            if lib_trl_utils.get_rank() == 0:
                wandb.log({"avg_all_rewards": all_rewards.mean().item()})

            ###########################################################################
            # Checks & Step
            ###########################################################################
            # PPO Step
            if ppo_trainer.is_encoder_decoder:
                assert isinstance(tokenizer.pad_token_id, int), type(tokenizer.pad_token_id)
                lib_trl_utils.check_all_start_with_token_id(
                    outputs.response_tensors, tokenizer.pad_token_id)
            
            stats = ppo_trainer.step(
                queries   = batch["input_ids"],
                responses = typing.cast(list[torch.LongTensor], outputs.response_tensors),
                scores    = reward_output.values,
            )

            # Log stats
            assert isinstance(reward_output.values, list), type(reward_output.values)
            assert isinstance(stats,   dict), type(stats)
            assert isinstance(batch,   dict), type(batch)

            batch["response"] = outputs.response_tensors

            ppo_trainer.log_stats(
                rewards = [x.to(torch.float32) for x in reward_output.values],
                batch   = batch,
                stats   = stats,
            )

            lib_trl_utils.print_table(
                extra_columns = reward_output.logging_columns,
                log_header    = f"(b{batch_idx}e{epoch}) ",
                responses     = outputs.response_text,
                queries       = batch["query"],
                rewards       = reward_output.values,
                name          = str(name), 
                qty           = 5,
            )

    

if __name__ == "__main__":
    fire.Fire(main)