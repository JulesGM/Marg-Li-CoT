#!/usr/bin/env python
# coding: utf-8

import datetime
import enum
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

import pretty_traceback
import numpy as np
import rich
import rich.console
from text2digits import text2digits
import torch
from   torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import transformers

pretty_traceback.install()
datasets    .logging.set_verbosity_error()
transformers.logging.set_verbosity_error()


import general_utils as utils
import policy
import metric
import reward
import dataset_gsm8k
import dataset_asdiv
import general_utils
import algorithm_deepspeed_ppo

from our_scratchpad.bin_deepspeed_experim import make_deepspeed_config


sys.path.append("/home/mila/g/gagnonju/RL4LMs")
import rl4lms.envs.text_generation.training_utils as rl4lms_training_utils
import rl4lms.envs.text_generation.logging_utils  as rl4lms_logging_utils
import rl4lms.envs.text_generation.registry       as rl4lms_registry
CONSOLE = rich.console.Console(width=80)
LOGGER = logging.getLogger(__name__)
NUM_PAT = re.compile(r"\d+(?:[\,\.]\d+)?")


class DatasetChoices(str, enum.Enum):
    gsm8k = "gsm8k"
    asdiv = "asdiv"

class PolicyTypes(str, enum.Enum):
    custom = "custom"
    naive  = "naive"
    deepspeed = "deepspeed"


MODEL_PARALLEL       = True
REWARD_MODEL_HF_NAME = "google/flan-t5-small"
POLICY_TYPE          = PolicyTypes.naive
N_ENVS_TRAIN         = 128
POLICY_MODEL_HF_NAME = REWARD_MODEL_HF_NAME
LOG_LEVEL            = logging.WARNING

# I believe N_STEPS is the number of generation steps per environment.
# N_ENVS is really the batch size. 
# TRAIN_BATCH_SIZE relates to the rollouts.. 
N_STEPS              = 24 * 8
TRAIN_BATCH_SIZE     = N_ENVS_TRAIN

# EVAL_BATCH_SIZE is the regular batch size 
EVAL_BATCH_SIZE      = N_ENVS_TRAIN
N_ITERS              = 1000
LOG_LEVEL            = logging.INFO

###############################################################################
# Doesn't change
###############################################################################
DATASET_CHOICE       = DatasetChoices.asdiv
REWARD_MODEL_DTYPE   = torch.bfloat16
POLICY_DTYPE         = REWARD_MODEL_DTYPE
MAX_PROMPT_LENGTH    = 107
MAX_EPISODE_LENGTH   = N_STEPS
torch.backends.cuda.matmul.allow_tf32 = True
DO_PROFILE           = False

POLICY_KWARGS = {
    "max_new_tokens": 200,
    "min_length"    : 5,
    "do_sample"     : True,
    "top_k"         : 50,
}

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
utils.info_rank_0(LOGGER, utils.print_dict(SETTINGS, return_str=True))


if MODEL_PARALLEL:
    assert not POLICY_TYPE == PolicyTypes.deepspeed


_shared_pol_args = {
    "prompt_truncation_side": "right",
    "apply_model_parallel"  : MODEL_PARALLEL,
    "model_name"            : POLICY_MODEL_HF_NAME,
    "generation_kwargs"     : POLICY_KWARGS,
}


POL_BASIC = lambda _: {
    "id": "seq2seq_lm_actor_critic_policy",
    "args": _shared_pol_args,
}


reward_parallelism_mode = None
if MODEL_PARALLEL:
    reward_parallelism_mode = reward.ParallelizeMode.parallelize
elif POLICY_TYPE == PolicyTypes.deepspeed:
    reward_parallelism_mode = reward.ParallelizeMode.nothing
elif POLICY_TYPE == PolicyTypes.naive:
    reward_parallelism_mode = reward.ParallelizeMode.data_parallel
else:
    raise NotImplementedError


POL_CUSTOM = lambda reward_model_inst: {
    "id"  : "precision_control_seq2seq_lm_actor_critic_policy",
    "args": {
        "from_pretrained_kwargs": {"torch_dtype": POLICY_DTYPE},
        "head_kwargs"           : {"dtype": POLICY_DTYPE},
        "same_model_for_value"  : True,
        "ref_model"             : reward_model_inst, 
        "parallelize_mode"      : reward_parallelism_mode,
    } | _shared_pol_args
}


POL_DEEPSPEED = lambda _: {
    "id"  : "deepspeed_experimentation_policy",
    "args": {
        "ds_configs": make_deepspeed_config(
            batch_size=N_ENVS_TRAIN,
            wandb_config={
                "enabled": True,
                "team"   : "julesgm", 
                "project": "rl4lms-deepspeed",
            }
        )
    } | _shared_pol_args
}

POLICY_MAP = {
    PolicyTypes.custom: POL_CUSTOM,
    PolicyTypes.naive: POL_BASIC,
    PolicyTypes.deepspeed: POL_DEEPSPEED,
}

assert POLICY_TYPE in POLICY_MAP, (
    f"Unknown policy type: {POLICY_TYPE}, "
    f"must be one of {list(POLICY_MAP.keys())}"
)
POLICY = POLICY_MAP[POLICY_TYPE]



torch.set_default_dtype(REWARD_MODEL_DTYPE)



def deal_with_words(text: str) -> Optional[float]:
    converted = TEXT2DIGITS.convert(text)
    output = NUM_PAT.findall(converted)

    if not output:
        return None

    utils.debug_rank_0(LOGGER, 
        "[bold blue]" + "#" * 80
    )
    utils.debug_rank_0(LOGGER, 
        f"[bold blue]# text2digits[/]:\n"
        f" \t -> [green]source:[/]    {text}\n"
        f" \t -> [green]converted:[/] {converted}\n"
        f" \t -> [green]final:[/]     {output}"
    )
    utils.debug_rank_0(
        LOGGER, 
        "[bold blue]" + "#" * 80
    )

    return output


def deal_with_words(text: str) -> Optional[float]:
    converted = text2digits.Text2Digits().convert(text)
    output = NUM_PAT.findall(converted)

    if not output:
        return None

    utils.debug_rank_0(LOGGER, 
        "[bold blue]" + "#" * 80
    )
    utils.debug_rank_0(LOGGER, 
        f"[bold blue]# text2digits[/]:\n"
        f" \t -> [green]source:[/]    {text}\n"
        f" \t -> [green]converted:[/] {converted}\n"
        f" \t -> [green]final:[/]     {output}"
    )
    utils.debug_rank_0(
        LOGGER, 
        "[bold blue]" + "#" * 80
    )

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
            utils.debug_rank_0(LOGGER, 
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
            utils.info_rank_0(LOGGER, f"[red]{transformed_node}")
        else:
            transformed_node = (
                f"Instance of type "
                f"`{type(config_node).__name__}`"
            )
            utils.info_rank_0(LOGGER, f"[red]{transformed_node}")
    return transformed_node


def main():
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    logging.basicConfig(
        level=LOG_LEVEL, 
        format=f"[{local_rank + 1} / {world_size}][bold]\[%(name)s]:[/]  %(message)s", 
        datefmt="[%X]", 
        handlers=[rich.logging.RichHandler(markup=True, rich_tracebacks=True)]
    )
    reward.LOGGER.setLevel(logging.WARNING)
    logging.getLogger("rl4lms.envs.text_generation.policy.base_policy").setLevel(logging.WARNING)

    utils.info_rank_0(LOGGER, "[bold bright_yellow]#" * 80)
    utils.info_rank_0(LOGGER, "[bold bright_yellow]# >>> Loading REWARD model")
    utils.info_rank_0(LOGGER, "[bold bright_yellow]#" * 80)

    reward_model_inst = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        REWARD_MODEL_HF_NAME, torch_dtype=REWARD_MODEL_DTYPE).eval()
    reward_model_tok  = transformers.AutoTokenizer.from_pretrained(
        REWARD_MODEL_HF_NAME)
    
    for param in reward_model_inst.parameters():
        param.requires_grad = False

    utils.info_rank_0(LOGGER, "[bold bright_yellow]#" * 80)
    utils.info_rank_0(LOGGER, "[bold bright_yellow]# <<< Done loading reward model")
    utils.info_rank_0(LOGGER, "[bold bright_yellow]#" * 80)

    alg_config = {
        "id": "deepspeed_ppo" if POLICY_TYPE == PolicyTypes.deepspeed else "ppo",
        "args": {
            "learning_rate": 2e-06,
            "batch_size"   : TRAIN_BATCH_SIZE,
            "ent_coef"     : 0.0,
            "n_epochs"     : 5,
            "n_steps"      : N_STEPS,
            "verbose"      : 1,
        },
        "kl_div": {
            "target_kl": 0.2,
            "coeff"    : 0.001, 
        },
        "policy": POLICY(reward_model_inst)
    }

    # assert alg_config["args"]["batch_size"] % alg_config["args"]["n_steps"] == 0, (
    #     alg_config["args"]["batch_size"] % alg_config["args"]["n_steps"], 
    #     alg_config["args"]["batch_size"],
    #     alg_config["args"]["n_steps"],
    # )

    gsm8k_config = {
        "args"   : {
            "max_sum_squares" : 41957,
            "tokenizer": reward_model_tok,
        },
        "id"     : "zero_shot_gsm8k_text_gen_pool",
    }

    asdiv_config = {
        "args"   : {},
        "id"     : "zero_shot_asdiv_text_gen_pool",
    }

    dataset_configs = {"gsm8k": gsm8k_config, "asdiv": asdiv_config}

    datapool_config = dataset_configs[DATASET_CHOICE]

    env_config = {
        "n_envs": N_ENVS_TRAIN,  
        "args"  : {
            "prompt_truncation_side": "right",
            "context_start_token"   : 0,
            "max_episode_length"    : MAX_EPISODE_LENGTH,
            "max_prompt_length"     : MAX_PROMPT_LENGTH,
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
        "n_iters"        : N_ITERS,
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
        base_path_to_store_results = os.path.join(
            os.environ["SLURM_TMPDIR"], "results"
        ),
        experiment_name            = "first_experiments",
        project_name               = "rl_scratchpad",
        entity_name                = "julesgm",
        run_config                 = clean_config_for_wandb(config),
        wandb_log                  = os.getenv("RANK", "0") == "0",
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
        utils.info_rank_0(LOGGER, "[blue bold]Our code: [/blue bold][bold]prof.__exit__")
        prof.__exit__(None, None, None)
        utils.info_rank_0(LOGGER, "[blue bold]Our code: [/blue bold][bold]Done with prof.__exit__")
        utils.info_rank_0(LOGGER, "[blue bold]Our code: [/blue bold][bold]preparing prof output")
        
        prof_obj = prof.key_averages()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = f"/home/mila/g/gagnonju/Marg-Li-CoT/rl4lms/profiling/profiling_{timestamp}_{POLICY_TYPE}"

        with Path(path + ".pkl").open("wb") as f:
            pickle.dump(prof_obj, f)
        with Path(path + ".json").open("w") as f:
            json.dump(SETTINGS, f, indent=4, default=str)

        output = prof_obj.table(sort_by="self_cuda_memory_usage", row_limit=100)
        "[blue bold]Our code: [/blue bold][bold]Done preparing prof output"
        utils.info_rank_0(LOGGER, output)


if __name__ == "__main__":
    main()
