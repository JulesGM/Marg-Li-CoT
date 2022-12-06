#!/usr/bin/env python
# coding: utf-8

import re
import types

import datasets
import numpy as np
import rich
import rich.console
import sys
from text2digits import text2digits
import torch
from   torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import transformers
from typing import *
import yaml

datasets    .logging.set_verbosity_error()
transformers.logging.set_verbosity_error()

import policy
import metric
import reward
import gsm8k_dataset
import asdiv_dataset

sys.path.append("/home/mila/g/gagnonju/RL4LMs")
import rl4lms.envs.text_generation.training_utils as rl4lms_training_utils
import rl4lms.envs.text_generation.logging_utils  as rl4lms_logging_utils
import rl4lms.envs.text_generation.registry       as rl4lms_registry

CONSOLE     = rich.console.Console(width=81)
NUM_PAT     = re.compile(r"\d+(?:[\,\.]\d+)?")
TEXT2DIGITS = text2digits.Text2Digits()

REWARD_MODEL_DTYPE   = torch.float32
REWARD_MODEL_HF_NAME = "google/flan-t5-small"
POLICY_MODEL_HF_NAME = REWARD_MODEL_HF_NAME
POLICY_DTYPE         = REWARD_MODEL_DTYPE

torch.set_default_dtype(REWARD_MODEL_DTYPE)


def deal_with_words(text: str) -> Optional[float]:
    converted = TEXT2DIGITS.convert(text)
    output = NUM_PAT.findall(converted)

    if not output:
        return None

    CONSOLE.print("[bold blue]" + "#" * 80)
    CONSOLE.print(
        f"[bold blue]# text2digits[/]:\n"
        f" \t -> [green]source:[/]    {text}\n"
        f" \t -> [green]converted:[/] {converted}\n"
        f" \t -> [green]final:[/]     {output}"
    )
    CONSOLE.print("[bold blue]" + "#" * 80)

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
            CONSOLE.print(
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
            CONSOLE.print(f"[red]{transformed_node}")
        else:
            transformed_node = f"Instance of type `{type(config_node).__name__}`"
            CONSOLE.print(f"[red]{transformed_node}")
    return transformed_node



def main():
    CONSOLE.print("[bold bright_yellow]#" * 80)
    CONSOLE.print("[bold bright_yellow]# >>> Loading REWARD model")
    CONSOLE.print("[bold bright_yellow]#" * 80)

    reward_model_inst = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        REWARD_MODEL_HF_NAME, torch_dtype=REWARD_MODEL_DTYPE)
    reward_model_tok  = transformers.AutoTokenizer        .from_pretrained(
        REWARD_MODEL_HF_NAME)
    
    CONSOLE.print("[bold bright_yellow]#" * 80)
    CONSOLE.print("[bold bright_yellow]# <<< Done loading reward model")
    CONSOLE.print("[bold bright_yellow]#" * 80)

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
        "policy": {
            # "id"  : "precision_control_seq2seq_lm_actor_critic_policy",
            "id"  : "seq2seq_lm_actor_critic_policy",
            
            "args": {
                # "from_pretrained_kwargs": {"torch_dtype": POLICY_DTYPE},
                # "head_kwargs"           : {"dtype": POLICY_DTYPE},

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
        }
    }

    datapool_config = {
        "args"   : {},
        "id"     : "zero_shot_gsm8k_text_gen_pool",
    }

    env_config = {
        "n_envs": 1,  # was 10
        "args"  : {
            "prompt_truncation_side": "right",
            "context_start_token"   : 0,
            "max_episode_length"    : 200,
            "max_prompt_length"     : 512,
            "terminate_on_eos"      : True,
        }
    }

    reward_config = {
        "id"  : "scratchpad_answer_reward",
        "args": {
            "reward_tokenizer_kwargs": {"padding": True},
            "answer_remover_fn"      : reward.flan_t5_answer_removal,
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
        "eval_batch_size": 100,
        "eval_every"     : 10,
        "save_every"     : 1,
        "n_iters"        : 100,
        "metrics"        : [
            {
                "id": "scratchpad_answer_accuracy",
                "args": {
                    "extract_answer_fn"      : split_fn,
                    "make_answ_comparable_fn": metric.convert_to_int,
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


    # with profile(
    #     profile_memory = True, 
    #     record_shapes  = True,
    #     activities     = [ProfilerActivity.CUDA],
    # ) as prof:
    #     try:
    #         transformers.logging.set_verbosity_error()
    #         datasets    .logging.set_verbosity_error()
    trainer.train_and_eval()


    # rich.print("[green bold]Preparing profile results...")
    # output = prof.key_averages().table(sort_by="self_gpu_memory_usage", row_limit=10)
    # rich.print(output)


if __name__ == "__main__":
    main()