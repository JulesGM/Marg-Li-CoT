#!/usr/bin/env python
# coding: utf-8

import datetime
import enum
import json
import logging
import os
import pickle
import re
import sys
import types
from pathlib import Path
from typing import *

import mlc_datasets
import general_utils
import general_utils as utils
import numpy as np
import pretty_traceback
import rich
import rich.console
import torch
import transformers
import yaml
from text2digits import text2digits
from tqdm import tqdm

import libs_compute_accuracy.dataset_asdiv as dataset_asdiv
import libs_compute_accuracy.dataset_gsm8k as dataset_gsm8k

pretty_traceback.install()
mlc_datasets.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()

CONSOLE = rich.console.Console(width=80)
LOGGER = logging.getLogger(__name__)

class DatasetChoices(str, enum.Enum):
    gsm8k = "gsm8k"
    gsm8k_silver = "gsm8k_silver"
    asdiv = "asdiv"


MODEL_PARALLEL = True
MODEL_HF_NAME = "google/flan-t5-xxl"
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 3
LOG_LEVEL = logging.WARNING


###############################################################################
# Doesn't change
###############################################################################
DATASET_CHOICE = DatasetChoices.gsm8k
MAX_PROMPT_LENGTH = 107
MAX_EPISODE_LENGTH = 200
torch.backends.cuda.matmul.allow_tf32 = True
GENERATION_KWARGS = {
    "max_new_tokens": MAX_EPISODE_LENGTH,
    "min_length": 5,
    "do_sample": True,
    "top_k": 50,
}

###############################################################################
#
###############################################################################


def main(
    precision = "int8",
    hf_model_name = MODEL_HF_NAME,
):

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    logging.basicConfig(
        level=LOG_LEVEL,
        format=(
            f"[{local_rank + 1} / {world_size}]"
            f"[bold]\[%(name)s]:[/]  %(message)s"
        ),
        datefmt="[%X]",
        handlers=[rich.logging.RichHandler(markup=True, rich_tracebacks=True)],
    )
    
    general_utils.check_contained(precision, ["int8", "fp16", "bf16", "fp32", None])

    if precision == "int8":
        accelerator_device = os.environ["LOCAL_RANK"]
        dmap_keys = ["encoder", "lm_head", "shared", "decoder"]
        device_map = {k: accelerator_device for k in dmap_keys}
        model_inst = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            hf_model_name, 
            device_map=device_map,
            load_in_8bits=True,
            torch_dtype=torch.bfloat16, 
        )
    elif precision == "fp16":
        model_inst = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            hf_model_name, torch_dtype=torch.float16
        )
    elif precision == "bf16":
        model_inst = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            hf_model_name, torch_dtype=torch.bfloat16
        )
    
    model_tok = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    gsm8k_config = {
        "args": {"max_sum_squares": 41957, "tokenizer": model_tok},
        "id": "zero_shot_gsm8k_text_gen_pool",
    }

    asdiv_config = {
        "args": {},
        "id": "zero_shot_asdiv_text_gen_pool",
    }

    trainer = transformers.Trainer(
        model=model_inst,
        train_dataset=dataset_gsm8k.SupervisedGSM8K.prepare(
            "train", 
            hf_model_name,
        ),
        eval_dataset=
    )

    # transformers.logging.set_verbosity_error()
    # datasets.logging.set_verbosity_error()
    


if __name__ == "__main__":
    main()
