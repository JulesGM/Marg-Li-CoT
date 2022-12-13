#!/usr/bin/env python
# coding: utf-8

import datetime
import os
import json
import logging
from pathlib import Path
import pickle
import re
import sys
import types
from typing import *
import yaml

import datasets
import numpy as np
import rich
import rich.console
from text2digits import text2digits
import torch
from   torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import transformers


datasets    .logging.set_verbosity_error()
transformers.logging.set_verbosity_error()


import policy
import metric
import reward
import gsm8k_dataset
import asdiv_dataset
import general_utils as utils


sys.path.append("/home/mila/g/gagnonju/RL4LMs")
import rl4lms.envs.text_generation.training_utils as rl4lms_training_utils
import rl4lms.envs.text_generation.logging_utils  as rl4lms_logging_utils
import rl4lms.envs.text_generation.registry       as rl4lms_registry
CONSOLE = rich.console.Console(width=80)
LOGGER = logging.getLogger(__name__)
NUM_PAT = re.compile(r"\d+(?:[\,\.]\d+)?")
TEXT2DIGITS = text2digits.Text2Digits()


torch.backends.cuda.matmul.allow_tf32 = True


REWARD_MODEL_DTYPE   = torch.bfloat16
REWARD_MODEL_HF_NAME = "google/flan-t5-xxl"
POLICY_TYPE          = "custom"
DO_PROFILE           = True
N_ENVS_TRAIN         = 1    
EVAL_BATCH_SIZE      = 1
POLICY_DTYPE         = REWARD_MODEL_DTYPE
POLICY_MODEL_HF_NAME = REWARD_MODEL_HF_NAME
LOG_LEVEL           = logging.WARNING


###############################################################################
###############################################################################


SETTINGS = dict(
    DO_PROFILE           = DO_PROFILE,
    POLICY_TYPE          = POLICY_TYPE,
    POLICY_DTYPE         = POLICY_DTYPE,
    N_ENVS_TRAIN         = N_ENVS_TRAIN,
    EVAL_BATCH_SIZE      = EVAL_BATCH_SIZE,
    REWARD_MODEL_DTYPE   = REWARD_MODEL_DTYPE,
    REWARD_MODEL_HF_NAME = REWARD_MODEL_HF_NAME,
    POLICY_MODEL_HF_NAME = POLICY_MODEL_HF_NAME,
)
utils.print_dict(SETTINGS)


_shared_pol_args = {
    "prompt_truncation_side": "right",
    "apply_model_parallel"  : True,
    "model_name"            : POLICY_MODEL_HF_NAME,
    "generation_kwargs"     : {
        "max_new_tokens": 200,
        "min_length"    : 15,
        "do_sample"     : True,
        "top_k"         : 50,
    }
}

POL_BASIC = lambda _: {
    "id": "seq2seq_lm_actor_critic_policy",
    "args": _shared_pol_args,
}

POL_CUSTOM = lambda reward_model_inst: {
    "id"  : "precision_control_seq2seq_lm_actor_critic_policy",
    "args": {
        "from_pretrained_kwargs": {"torch_dtype": POLICY_DTYPE},
        "head_kwargs"           : {"dtype": POLICY_DTYPE},
        "same_model_for_value"  : True,
        "ref_model"             : reward_model_inst, 
        } | _shared_pol_args
}



if POLICY_TYPE == "custom":
    POLICY = POL_CUSTOM
elif POLICY_TYPE == "naive":
    POLICY = POL_BASIC
else:
    raise ValueError(f"Invalid policy type: {POLICY_TYPE}")



torch.set_default_dtype(REWARD_MODEL_DTYPE)


def deal_with_words(text: str) -> Optional[float]:
    converted = TEXT2DIGITS.convert(text)
    output = NUM_PAT.findall(converted)

    if not output:
        return None

    LOGGER.info("[bold blue]" + "#" * 80)
    LOGGER.info(
        f"[bold blue]# text2digits[/]:\n"
        f" \t -> [green]source:[/]    {text}\n"
        f" \t -> [green]converted:[/] {converted}\n"
        f" \t -> [green]final:[/]     {output}"
    )
    LOGGER.info("[bold blue]" + "#" * 80)

    return output 


def split_fn(generated_text: str) -> Optional[str]:

    results = NUM_PAT.findall(generated_text)
    
    if results:
        # Numbers found
        output = results[-1]
    else:
        # No numbers found
        try:
            output = deal_with_words(generated_text)
        except ValueError:
            output = None

        if output is not None:
            output = output[-1]
        else:
            LOGGER.info(
                f"[red]split_fn: no numbers found. \n"
                f"\t-> Received:[/] `{generated_text}`"
            )
            output = None

    return output


def clean_config_for_wandb(config_node: Any) -> Any:
    """    
    
    Creates a copy of the config file for the purposes of logging, 
    changing the objects that aren't json serializable to their 
    names in string form.

    Simple depth first traversal.
    
    """

    if isinstance(config_node, dict):
        transformed_node = {}
        for k, v in config_node.items():
            transformed_node[k] = clean_config_for_wandb(v)
    elif isinstance(config_node, list):
        transformed_node = []
        for v in config_node:
            transformed_node.append(clean_config_for_wandb(v))
    elif isinstance(config_node, (str, int, float)):
        transformed_node = config_node
    else:
        if isinstance(config_node, types.FunctionType):
            transformed_node = f"Function called `{config_node.__name__}`"
            LOGGER.info(f"[red]{transformed_node}")
        else:
            transformed_node = (
                f"Instance of type "
                f"`{type(config_node).__name__}`"
            )
            LOGGER.info(f"[red]{transformed_node}")
    return transformed_node


def log_rank_0(level, message):
    if os.getenv("LOCAL_RANK", "0") == "0":
        LOGGER.log(level, message)


def info_rank_0(message):
    log_rank_0(logging.INFO, message)


def debug_rank_0(message):
    log_rank_0(logging.DEBUG, message)


def main():
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    logging.basicConfig(
        level=logging.INFO, 
        format=f"[{local_rank + 1} / {world_size}]:\t%(message)s", 
        datefmt="[%X]", 
        handlers=[rich.logging.RichHandler(markup=True, rich_tracebacks=True)]
    )

    LOGGER.info("[bold bright_yellow]#" * 80)
    LOGGER.info("[bold bright_yellow]# >>> Loading REWARD model")
    LOGGER.info("[bold bright_yellow]#" * 80)

    reward_model_inst = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        REWARD_MODEL_HF_NAME, torch_dtype=REWARD_MODEL_DTYPE).eval()
    reward_model_tok  = transformers.AutoTokenizer        .from_pretrained(
        REWARD_MODEL_HF_NAME)
    
    for param in reward_model_inst.parameters():
        param.requires_grad = False

    LOGGER.info("[bold bright_yellow]#" * 80)
    LOGGER.info("[bold bright_yellow]# <<< Done loading reward model")
    LOGGER.info("[bold bright_yellow]#" * 80)

    alg_config = {
        "id": "ppo",
        "args": {
            "learning_rate": 2e-06,
            "batch_size"   : 64,
            "ent_coef"     : 0.0,
            "n_epochs"     : 5,
            "n_steps"      : 512,
            "verbose"      : 1,
        },
        "kl_div": {
            "target_kl": 0.2,
            "coeff"    : 0.001, 
        },
        "policy": POLICY(reward_model_inst)
    }

    datapool_config = {
        "args"   : {},
        "id"     : "zero_shot_gsm8k_text_gen_pool",
    }

    env_config = {
        "n_envs": N_ENVS_TRAIN,  
        "args"  : {
            "prompt_truncation_side": "right",
            "context_start_token"   : 0,
            "max_episode_length"    : 100,
            "max_prompt_length"     : 200,
            "terminate_on_eos"      : True,
        }
    }

    reward_config = {
        "id"  : "scratchpad_answer_reward",
        "args": {
            "reward_tokenizer_kwargs": {"padding": True},
            "generation_splitter_fn" : reward.flan_t5_answer_removal,
            "sp_answer_joiner_fn"    : reward.flan_t5_answer_joiner,
            "reward_tokenizer"       : reward_model_tok,
            "reward_model"           : reward_model_inst,
        },
    }

    tokenizer_config = {
        "pad_token_as_eos_token": False,
        "truncation_side"       : "left", 
        "padding_side"          : "left", 
        "model_name"            : POLICY_MODEL_HF_NAME, 
    }

    train_evaluation_config = {
        "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_every"     : 10,
        "save_every"     : 1,
        "n_iters"        : 1,
        "metrics"        : [
            {
                "id": "scratchpad_answer_accuracy",
                "args": {
                    "extract_answer_fn" : split_fn,
                    "make_comparable_fn": metric.convert_to_int,
                },
            }
        ],

        "generation_kwargs": {
            "max_new_tokens": 200,
            "temperature"   : 0.7,
            "min_length"    : 5,
            "do_sample"     : True,
            "top_k"         : 0,
        }
    }

    config = {
        "train_evaluation": train_evaluation_config,
        "tokenizer"       : tokenizer_config,
        "datapool"        : datapool_config,
        "reward"          : reward_config,
        "env"             : env_config,
        "alg"             : alg_config,
    }
    
    tracker = rl4lms_logging_utils.Tracker(
        base_path_to_store_results = "/home/mila/g/gagnonju/Marg-Li-CoT/rl4lms/results/",
        experiment_name            = "first_experiments",
        project_name               = "rl_scratchpad",
        entity_name                = "julesgm",
        run_config                 = clean_config_for_wandb(config),
        wandb_log                  = True,
        log_level                  = LOG_LEVEL,
    )

    trainer = rl4lms_training_utils.OnPolicyTrainer( 
        on_policy_alg_config = alg_config,
        train_eval_config    = train_evaluation_config,
        tokenizer_config     = tokenizer_config,
        datapool_config      = datapool_config,
        reward_config        = reward_config,
        env_config           = env_config,
        tracker              = tracker,
    )


    if DO_PROFILE:
        prof = profile(
            profile_memory = True, 
            record_shapes  = True,
            activities     = [ProfilerActivity.CUDA],
        ).__enter__()

    transformers.logging.set_verbosity_error()
    datasets    .logging.set_verbosity_error()
    trainer.train_and_eval()

    if DO_PROFILE:
        LOGGER.info("[blue bold]Our code: [white bold]prof.__exit__")
        prof.__exit__(None, None, None)
        LOGGER.info("[blue bold]Our code: [white bold]Done with prof.__exit__")
        LOGGER.info("[blue bold]Our code: [white bold]preparing prof output")
        
        prof_obj = prof.key_averages()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = f"/home/mila/g/gagnonju/Marg-Li-CoT/rl4lms/profiling/profiling_{timestamp}_{POLICY_TYPE}"

        with Path(path + ".pkl").open("wb") as f:
            pickle.dump(prof_obj, f)
        with Path(path + ".json").open("w") as f:
            json.dump(SETTINGS, f, indent=4, default=str)

        output = prof_obj.table(sort_by="self_cuda_memory_usage", row_limit=100)
        "[blue bold]Our code: [white bold]Done preparing prof output"
        LOGGER.info(output)


if __name__ == "__main__":
    main()