""" Datasets parsing and loading. """

import collections
import collections.abc
import dataclasses
import enum
import logging
import math
import os
import re
import time
import typing
import xml
import xml.etree
from pathlib import Path
from typing import Any, Optional, Union

import datasets
import more_itertools
import numpy as np
import rich
import rich.box
import rich.table
import torch
import torch.utils
import torch.utils.data
import transformers
import wget
from text2digits import text2digits

import lib_base_classes
import lib_sentiment_specific
import lib_utils
import libs_data
import libs_extraction

LOGGER = logging.getLogger(__name__)
RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "0"))


class DatasetChoices(str, enum.Enum):
    ASDIV = "asdiv"
    GSM8K = "gsm8k"
    COMMONSENSEQA_MC = "commonsenseqa_mc"
    SENTIMENT = "sentiment"


DATASET_KEY_TO_CLASS = {
    DatasetChoices.ASDIV: libs_data.lib_asdiv.ASDiv,
    DatasetChoices.GSM8K: libs_data.lib_gsm8k.GSM8K,
    DatasetChoices.COMMONSENSEQA_MC: libs_data.lib_commonsense_qa.CommonSenseQAMC,
    DatasetChoices.SENTIMENT: libs_data.lib_sentiment.SentimentData
}

DATASET_KEY_TO_ANSWER_EXTRACTOR = {
    DatasetChoices.ASDIV: libs_extraction.lib_numerical,
    DatasetChoices.GSM8K: libs_extraction.lib_numerical,
    DatasetChoices.COMMONSENSEQA_MC: libs_extraction.lib_multiple_choice,
    DatasetChoices.SENTIMENT: None
}


def data_item_collator(
    batch: list[lib_base_classes.DataItemContainer]
) -> lib_base_classes.DataListContainer:
    
    new_batch = lib_base_classes.DataListContainer()
    for item in batch:
        for k, v in vars(item).items():
            vars(new_batch)[k].append(v)

    return new_batch

    
def prep_dataset_rl(
    *,
    input_max_length: int,
    question_prefix: str,
    question_suffix: str,
    dataset_name: str,
    any_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    split: str,
) -> libs_data.lib_base.Dataset:
    if dataset_name == DatasetChoices.GSM8K:
        assert isinstance(LOCAL_RANK, int), type(LOCAL_RANK)
        dataset = libs_data.lib_gsm8k.GSM8K(
            tok_max_query_length=input_max_length,
            any_tokenizer=any_tokenizer,
            device=torch.device(LOCAL_RANK),
            ds=datasets.load_dataset(  # type: ignore
                split=split,
                path="gsm8k",
                name="main",
            ),
            question_prefix=question_prefix,
            question_suffix=question_suffix,
        )

    elif dataset_name == DatasetChoices.COMMONSENSEQA_MC:
        dataset = libs_data.lib_commonsense_qa.CommonSenseQAMC(
            any_tokenizer=any_tokenizer,
            split=split,
            question_prefix=question_prefix,
            question_suffix=question_suffix,
        )

    elif dataset_name == DatasetChoices.ASDIV:
        raise NotImplemented
        assert split is None, "split must be None for ASDiv"
        dataset = libs_data.lib_asdiv.ASDiv(
            tokenizer=any_tokenizer,
            cache_path="/tmp/asdiv",
        )

    elif dataset_name == DatasetChoices.SENTIMENT:
        dataset = libs_data.lib_sentiment.SentimentData(
            any_tokenizer=any_tokenizer, 
            split=split,
        )
    else:
        raise ValueError(f"Unsupported task: {dataset_name}")

    return dataset


def prep_dataset_sft(
    *,
    max_total_length_tok: typing.Optional[int],
    question_prefix: str,
    question_suffix: str,
    task_name: str,
    any_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    split: str,
) -> libs_data.lib_base.Dataset:
    assert split in ["train", "eval",], split
    
    if task_name == DatasetChoices.GSM8K:
        assert isinstance(LOCAL_RANK, int), type(LOCAL_RANK)
        
        dataset = libs_data.lib_gsm8k.GSM8K(
            tok_max_total_length=max_total_length_tok,
            question_prefix=question_prefix,
            question_suffix=question_suffix,
            any_tokenizer=any_tokenizer,
            device=torch.device(LOCAL_RANK),
            ds=datasets.load_dataset(  # type: ignore
                split="train" if split == "train" else "test",
                path="gsm8k",
                name="main",
            ),
        )

    elif task_name == DatasetChoices.ASDIV:
        raise NotImplemented("ASDiv not implemented for SFT")
        assert split is None, "split must be None for ASDiv"
        dataset = ASDiv(
            tokenizer=tokenizer,
            cache_path="/tmp/asdiv",
        )

    elif task_name == DatasetChoices.SENTIMENT:
        
        dataset = lib_sentiment_specific.prep_dataset_sft(
            any_tokenizer, split, 
            maxlen_tok=1024, 
            minlen_tok=100,
            maxlen_char=None, 
            minlen_char=None,
        )

    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    return dataset