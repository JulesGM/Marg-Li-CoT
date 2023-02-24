#!/usr/bin/env python
# coding: utf-8


"""
Training script for the RL part of the project.

There are wrappers for GSM8K and for the ASDiv datasets.

By default, we support the GPT2 model.


"""


TOKENIZER_MODEL = "google/flan-t5-small"
DATASET_TO_USE  = "gsm8k"
REWARD_MODEL    = "google/flan-t5-small"
MAIN_MODEL      = "google/flan-t5-small"


import collections
import contextlib
import enum
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import itertools
import random
import re
import sys
import time
from pathlib import Path
from typing import *

import accelerate
import datasets
import fire
import general_utils
import numpy as np
import rich
import rich.logging
import torch
import transformers
import trlx
from trlx.data.configs import TRLConfig
import yaml
from tqdm import tqdm

import lib_data
import reward
from general_utils import parallel_guard


print("Done with imports")


LOGGER = logging.getLogger(__name__)

PPO_CONFIG_PATH = str(Path(__file__).parent / "ppo_config.yml")
assert Path(PPO_CONFIG_PATH).exists(), f"{PPO_CONFIG_PATH = }"
print(f"{PPO_CONFIG_PATH = }")


def check_tokenizer(tokenizer):
    assert (
        tokenizer.pad_token != tokenizer.eos_token
    ), f"{tokenizer.pad_token = }, {tokenizer.eos_token = }"
    assert (
        tokenizer.pad_token != tokenizer.cls_token
    ), f"{tokenizer.pad_token = }, {tokenizer.cls_token = }"
    assert (
        tokenizer.eos_token != tokenizer.cls_token
    ), f"{tokenizer.eos_token = }, {tokenizer.cls_token = }"

    assert (
        tokenizer.pad_token_id != tokenizer.eos_token_id
    ), f"{tokenizer.pad_token_id = }, {tokenizer.eos_token_id = }"
    assert (
        tokenizer.pad_token_id != tokenizer.cls_token_id
    ), f"{tokenizer.pad_token_id = }, {tokenizer.cls_token_id = }"
    assert (
        tokenizer.eos_token_id != tokenizer.cls_token_id
    ), f"{tokenizer.eos_token_id = }, {tokenizer.cls_token_id = }"


@contextlib.contextmanager
def setup(
    *,
    model: Optional[transformers.PreTrainedModel],
    reward_model: Optional[transformers.PreTrainedModel],
    tokenizer: Optional[transformers.PreTrainedTokenizer],
    main_model_hf_name_or_path: Optional[Union[str, Path]],
    reward_model_hf_name_or_path: Optional[Union[str, Path]],
    tokenizer_hf_name_or_path: Optional[Union[str, Path]],
    model_class: Optional[Type[transformers.PreTrainedModel]],
):

    assert reward_model is None, f"{reward_model = }"
    assert model is None, f"{type(model    ) = }"
    assert tokenizer is None, f"{type(tokenizer) = }"

    LOGGER.info("[bold red]Loading from HF name or path")
    LOGGER.info(f"[bold red]{main_model_hf_name_or_path   = }")
    LOGGER.info(f"[bold red]{reward_model_hf_name_or_path = }")
    LOGGER.info(f"[bold red]{tokenizer_hf_name_or_path    = }")

    reward_model = model_class.from_pretrained(reward_model_hf_name_or_path)
    assert reward_model.config.model_type == "t5"
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_hf_name_or_path
    )

    return (
        reward_model_hf_name_or_path,
        tokenizer_hf_name_or_path,
        reward_model,
        reward_tokenizer,
    )


def stats_for_key(
    ds: lib_data.GSM8KLMDataset, 
    field: str, 
    reward_tokenizer: transformers.PreTrainedTokenizer,
):
    """
    Evaluate stats on the number of tokens per sample
    """
    stuff = collections.Counter()
    shortest = []
    field_options = ("inputs", "labels")

    for entry in ds:
        # 1. Extract the text of the inputs or of the labels

        # assert field in field_options, (
        #     f"inputs_or_outputs should be in {field_options}, "
        #     f"got `{field}`"
        # )
        target = entry[field]

        # 2. Tokenize the text
        # assert (
        #   target.endswith(reward_tokenizer.cls_token) or
        #   target.endswith(reward_tokenizer.eos_token)
        # ), f"{target = }"
        target = target.removesuffix(reward_tokenizer.cls_token).removesuffix(
            reward_tokenizer.eos_token
        )

        input_ids = reward_tokenizer(target)["input_ids"]
        if len(input_ids) <= 7:
            shortest.append((target, input_ids))

        stuff.update([len(input_ids)])

    keys = np.fromiter(stuff.keys(), dtype=float)
    values = np.fromiter(stuff.values(), dtype=float)

    mean = np.average(keys, weights=values)
    std = np.sqrt(np.average((keys - mean) ** 2, weights=values))
    max_ = np.max(keys)
    min_ = np.min(keys)

    LOGGER.info(f"\n[bold blue]{field}:")
    LOGGER.info(f"input max  = {int(max_)}")
    LOGGER.info(f"input min  = {int(min_)}")
    LOGGER.info(f"input mean = {mean:0.3}")
    LOGGER.info(f"input std  = {std :0.3}")

    # plt.title(field)
    # plt.hist(keys, bins=10, weights=values)
    # plt.gca().xaxis.set_major_locator(
    #     ticker.MaxNLocator(integer=True))
    # plt.show()


class ModelClassChoices(str, enum.Enum):
    seq2seq = "seq2seq"
    causal_lm = "causal_lm"


MODEL_CLASS_CHOICES = {
    ModelClassChoices.causal_lm: transformers.AutoModelForCausalLM,
    ModelClassChoices.seq2seq: transformers.AutoModelForSeq2SeqLM,
}

MODEL_TYPE_CHECKS = {
    ModelClassChoices.causal_lm: {"gpt2"},
    ModelClassChoices.seq2seq: {"bart", "t5"},
}


def sanity_check_model_type(model_class_name: str, hf_name_or_path: str):
    """
    Check that the model type is compatible with the model class.
    Basically checks that we're not trying to instantiate a seq2seq gpt2 model or something like that.
    """
    config = transformers.AutoConfig.from_pretrained(hf_name_or_path)
    assert (
        config.model_type in MODEL_TYPE_CHECKS[model_class_name]
    ), f"Model type {config.model_type} is not compatible with model class {model_class_name}. "


def train(
    main_model_hf_name_or_path: Optional[str] = MAIN_MODEL,
    reward_model_hf_name_or_path: Optional[str] = REWARD_MODEL,
    tokenizer_hf_name_or_path: Optional[str] = TOKENIZER_MODEL,
    model_class_name: str = ModelClassChoices.seq2seq,  # One of ModelClassChoices
    # ds_eval: Optional[torch.data.Dataset] = None,
    # ds_train: Optional[torch.data.Dataset] = None,
    trlx_config_path: Union[Path, str] = PPO_CONFIG_PATH,
    dataset_to_use: str = DATASET_TO_USE,
    log_level: str = "INFO",
):

    args = locals().copy()
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    logging.basicConfig(
        level=log_level,
        format=f"[{rank}/{world_size}] %(name)s - %(message)s ",
        handlers=[
            rich.logging.RichHandler(markup=True),
        ],
    )

    if rank == 0:
        LOGGER.info("[bold blue]Arguments:")
        general_utils.print_dict(args, logger=LOGGER, log_level="INFO")
        print("")

    assert dataset_to_use in list(
        lib_data.DatasetChoices
    ), f"{dataset_to_use = } not in {list(lib_data.DatasetChoices)}"

    sanity_check_model_type(model_class_name, main_model_hf_name_or_path)
    sanity_check_model_type(model_class_name, reward_model_hf_name_or_path)

    # The setup makes use of a tempfile.TemporaryDirectory to go around the peculiarities of TRLX.
    # This is why setup is a context manager.
    hf_path = main_model_hf_name_or_path
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_hf_name_or_path)
    config_dict = yaml.safe_load(Path(trlx_config_path).read_text())

    max_input_length = (
        config_dict["train"]["seq_length"] - 
        config_dict["method"]["gen_kwargs"]["max_new_tokens"]
    )
    ds_train_obj = lib_data.GSM8KLMDataset(
        datasets.load_dataset("gsm8k", "main", split="train"),
        tokenizer=reward_tokenizer,
        max_length=max_input_length,
    )
    ds_eval_obj = lib_data.GSM8KLMDataset(
        datasets.load_dataset("gsm8k", "main", split="train"),
        tokenizer=reward_tokenizer,
        max_length=max_input_length,
    )


    assert "model_path" not in config_dict["model"]
    config_dict["model"]["model_path"] = hf_path
    assert "tokenizer_path" not in config_dict["tokenizer"]
    config_dict["tokenizer"]["tokenizer_path"] = tokenizer_hf_name_or_path

    config_dict["method"]["gen_kwargs"][
        "eos_token_id"
    ] = reward_tokenizer.cls_token_id

    if rank == 0:
        LOGGER.info("[bold blue]Config:")
        rich.print(config_dict)

    config = TRLConfig.from_dict(config_dict)

    # stats_for_key(ds_train_obj, "input", reward_tokenizer)
    # stats_for_key(ds_train_obj, "value", reward_tokenizer)
    # stats_for_key(ds_eval_obj , "input", reward_tokenizer)
    # stats_for_key(ds_eval_obj , "value", reward_tokenizer)

    scratchpad_reward_fn = reward.ScratchpadRewardFn(
        reward_model_hf_name_or_path=reward_model_hf_name_or_path,
        reward_tokenizer=reward_tokenizer,
        ds_train_obj=ds_train_obj,
        batch_size=config_dict["train"]["batch_size"],
    )

    model = trlx.train(
        model_path=hf_path,
        config=config,
        prompts=list(ds_train_obj),
        eval_prompts=list(ds_eval_obj),
        reward_fn=scratchpad_reward_fn,
    )


if __name__ == "__main__":
    
    if int(os.getenv("RANK", "0")) == 0:
        for k, v in os.environ.items():
            if "deepspeed" in k.lower():
                print(f"{k} = {v}")

        for k, v in os.environ.items():
            if "accelerate" in k.lower() and "deepspeed" not in k.lower():
                print(f"{k} = {v}")

    fire.Fire(train)
