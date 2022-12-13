#!/usr/bin/env python
# coding: utf-8



import collections
import copy
import logging
import os
import time

import datasets
import deepspeed
import rich
import rich.console
import rich.logging
import torch
from tqdm import tqdm
import transformers
import transformers.deepspeed

import general_utils as utils

# Required for deepspeed
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOGGER = logging.getLogger(__file__)
CONSOLE = rich.console.Console(force_terminal=True, width=80)
NO_DEEPSPEED_MODE = False
INFERENCE_MODE = False
DTYPE = torch.float32

MODEL_NAME = "google/flan-t5-small"
APPROX_BATCH_SIZE = int(os.environ["WORLD_SIZE"])
ZERO_LEVEL = 3
ZERO_LEVEL_3_CPU_OFFLOAD = False


def log_rank_0(level, message):
    if os.getenv("LOCAL_RANK", "0") == "0":
        LOGGER.log(level, "[white bold]\[log-zero]:[/] " + message)


def info_rank_0(message):
    log_rank_0(logging.INFO, message)


def debug_rank_0(message):
    log_rank_0(logging.DEBUG, message)


class OptimizerMerger:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()


def main():
    if NO_DEEPSPEED_MODE:
        local_rank = 0
        world_size = 1
        assert "LOCAL_RANK" not in os.environ, (
            "LOCAL_RANK should not be set in NO_DEEPSPEED_MODE"
        )
        assert "WORLD_SIZE" not in os.environ, (
            "WORLD_SIZE should not be set in NO_DEEPSPEED_MODE"
        )
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # Round the batch size to the closest multiple of the world size
    batch_size = world_size * (APPROX_BATCH_SIZE // world_size)

    logging.basicConfig(
        level=logging.INFO, 
        format=f"[{local_rank + 1} / {world_size}]:\t%(message)s", 
        datefmt="[%X]", 
        handlers=[rich.logging.RichHandler(
            markup=True, rich_tracebacks=True
        )]
    )

    info_dict = dict(
        BATCH_SIZE=batch_size,
        INFERENCE_MODE=INFERENCE_MODE,
        NO_DEEPSPEED_MODE=NO_DEEPSPEED_MODE,
        ZERO_LEVEL=ZERO_LEVEL,
        ZERO_LEVEL_3_CPU_OFFLOAD=ZERO_LEVEL_3_CPU_OFFLOAD,
        DTYPE=DTYPE,
    )
    
    
    info_rank_0("\n" + utils.print_dict(info_dict, return_str=True))
    info_rank_0(f"[green]Starting main. zero_level: {ZERO_LEVEL}[/green]")

    ####################################################################################
    # Instantiate dataset
    ####################################################################################
    info_rank_0(f"[green bold]LOADING DATA GSM8K")
    dataset = datasets.load_dataset("gsm8k", "main", split="train")


    ####################################################################################
    # Instantiate models
    ####################################################################################
    info_rank_0(f"[green bold]LOADING MODEL {MODEL_NAME}")
    policy_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME) # .to(local_rank)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    info_rank_0(f"[green bold]DONE LOADING MODEL {MODEL_NAME} :)")
    

    # value_model  = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # ref_model    = copy.deepcopy(policy_model).eval()

    # for parameter in ref_model.parameters():
        # parameter.requires_grad = False 

    # value_head   = torch.nn.Linear(value_model.config.hidden_size, 1, bias=False)


    ####################################################################################
    # Instantiate engines
    ####################################################################################
    zero_3_optimization = {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": True,
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "stage3_max_live_parameters": 1e9, # Same as default. I wonder if we could switch to auto?
        "stage3_max_reuse_distance": 1e9, # Same as default. I wonder if we could switch to auto?
        "sub_group_size": 1e9, # ... No info
    }

    if ZERO_LEVEL_3_CPU_OFFLOAD:
        zero_3_optimization["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }

    zero_2_optimization = {
        "stage": 2,
        
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },

        "overlap_comm": True, # Default is False
        "contiguous_gradients": True, # Default is True
        "reduce_scatter": True, # Default is True
        "allgather_partitions": True, # Default is True
        # "allgather_bucket_size": 2e8, # Default is 5e8
        # "reduce_bucket_size": 2e8, # Default is 5e8
    }

    per_zero_level = {
        2: zero_2_optimization,
        3: zero_3_optimization,
    }
    
    ds_config_train = {
        "micro_batch_size_per_gpu": "auto",
        "micro_batch_per_gpu": "auto",
        "train_batch_size": batch_size,
        "steps_per_print": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.0001,
                "betas": [
                    0.9,
                    0.999,
                ]
            }
        },
        "zero_optimization": per_zero_level[ZERO_LEVEL],
    }

    assert isinstance(per_zero_level[ZERO_LEVEL], dict), (
        f"per_zero_level[ZERO_LEVEL] should be a dict. "
        f"It is a {type(per_zero_level[ZERO_LEVEL])}"
    )

    ds_config_inference = {
        "micro_batch_size_per_gpu": "auto",
        "zero": per_zero_level[ZERO_LEVEL],
    }


    if DTYPE == torch.bfloat16:
        ds_config_inference["bfloat16"] = {
            "enabled": True
        }
        ds_config_train["grad_accum_dtype"] = "bfloat16"
        ds_config_train["optimizer"]["grad_accum_dtype"] = "bfloat16"
        ds_config_train["bfloat16"] = {
            "enabled": True
        }
    elif DTYPE == torch.float16:
        ds_config_inference["fp16"] = {
            "enabled": True
        }
        ds_config_train["grad_accum_dtype"] = "fp16"
        ds_config_train["fp16"] = {
            "enabled": True
        }
    elif DTYPE == torch.float32:
        pass
    else:
        raise ValueError(f"Invalid DTYPE: {DTYPE}")

    if not NO_DEEPSPEED_MODE:
        if INFERENCE_MODE:
            transformers.deepspeed.HfDeepSpeedConfig(ds_config_inference)
            # deepspeed.init_distributed()
        else:
            transformers.deepspeed.HfDeepSpeedConfig(ds_config_train)

    models = {
        "policy_model":   policy_model, 
        # "value_model":  value_model, 
        # "ref_model":    ref_model, 
        # "value_head":   value_head,
    }


    engines = {}
    info_rank_0(f"[red bold]Starting LOOP")
    if NO_DEEPSPEED_MODE:
        engines = {k: v.cuda() for k, v in models.items()}
    else:
        for idx, (model_name, model) in enumerate(models.items()):
            info_rank_0(f"[red bold]LOOP {idx}")
            if INFERENCE_MODE:
                engines[model_name] = deepspeed.init_inference(
                    model=model, 
                    mp_size=world_size,
                    replace_method="auto",
                    replace_with_kernel_inject=True,
                    config=ds_config_inference,
                )
            else:
                engines[model_name] = deepspeed.initialize(
                    model=model, 
                    # mp_size=world_size,
                    # replace_method="auto",
                    # replace_with_kernel_inject=True,
                    config_params=ds_config_train,
                    dist_init_required=idx == 0,
                )[0]

    LOGGER.info(f"[red bold]Exiting LOOP")

    # optimizer = OptimizerMerger(list(models.values()))    

    for idx in tqdm(range(0, len(dataset), batch_size)):

        input_ids = tokenizer(
            dataset["question"][idx:idx + batch_size], 
            return_tensors="pt",
            padding=True,
        )
        label_ids = tokenizer(
            dataset["answer"][idx:idx + batch_size],
            return_tensors="pt",
            padding=True,
        )

        assert input_ids["input_ids"].shape[0], input_ids["input_ids"].shape
        LOGGER.info(input_ids["input_ids"].shape)

        if INFERENCE_MODE:
            ################################################################
            # Inference Test
            ################################################################
            assert False
            engines["policy_model"].eval()
            start = time.perf_counter()
            output_ids = engines["policy_model"].generate(
                input_ids["input_ids"].to(local_rank), 
                max_new_tokens=200
            ).cpu().numpy()
            delta = time.perf_counter() - start
            LOGGER.info(f"[green bold]\[batch {idx}]Took {delta} seconds.")
            LOGGER.info(f"[green bold]\[batch {idx}]This is {delta / batch_size} seconds per sample.")
        else:
            ################################################################
            # Training Test
            ################################################################
            LOGGER.info(f"[red bold]\[batch {idx}] Training.")
            start = time.perf_counter()
            inputs = {k: v.to(local_rank) for k, v in input_ids.items()}
            outputs = engines["policy_model"](
                **inputs,
                decoder_input_ids=label_ids["input_ids"].to(local_rank),
                decoder_attention_mask=
                    label_ids["attention_mask"].to(local_rank),
                labels=label_ids["input_ids"].to(local_rank),
            )
            outputs.loss.backward()
            engines["policy_model"].step()

            delta = time.perf_counter() - start
            LOGGER.info(f"[green bold]\[batch {idx}] Took {delta} seconds.")
            LOGGER.info(f"[green bold]\[batch {idx}] This is {delta / len(inputs['input_ids'])} seconds per sample.")
            info_rank_0("\n" + utils.print_dict(info_dict, return_str=True))



if __name__ == "__main__":
    main()