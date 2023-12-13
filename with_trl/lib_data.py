""" Datasets parsing and loading. """

import collections.abc
import enum
import logging
import os
from pathlib import Path
import typing
from typing import Any, Optional, Union

import datasets
import more_itertools as mit
import numpy as np
import rich
import rich.box
import rich.table
import torch
import torch.utils
import torch.utils.data
import transformers

import lib_base_classes
import libs_data
import libs_extraction
import lib_utils

LOGGER = logging.getLogger(__name__)
RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "0"))


class DatasetChoices(str, enum.Enum):
    ASDIV = "asdiv"
    GSM8K = "gsm8k"
    COMMONSENSEQA_MC = "commonsenseqa_mc"
    SENTIMENT = "sentiment"
    ARITHMETIC = "arithmetic"


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
        for k, v in item.items():
            vars(new_batch)[k].append(v)

    return new_batch

    
def prep_dataset_rl(
    *,
    answer_only: bool,
    answer_only_path: Union[str, Path],
    any_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    dataset_name: str,
    input_max_length: int,
    question_prefix: str,
    question_suffix: str,
    split: str,
    use_few_shots: int,
    arithmetic_dataset_root_folder_dir: Optional[str],
    extr_arith_ignore_one_line,
    use_curriculum,
) -> libs_data.lib_base.Dataset:
    split = lib_utils.CVSets(split)

    if answer_only and not dataset_name in {
        DatasetChoices.COMMONSENSEQA_MC, DatasetChoices.ARITHMETIC
    }:
        raise NotImplementedError()

    if dataset_name == DatasetChoices.GSM8K:
        assert not use_few_shots, "n_few_shots must be 0 for GSM8K"
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
            answer_only=answer_only,
            answer_only_path=answer_only_path,
            any_tokenizer=any_tokenizer,
            split=split,
            question_prefix=question_prefix,
            question_suffix=question_suffix,
            use_few_shots=use_few_shots,
        )

    elif dataset_name == DatasetChoices.ASDIV:
        raise NotImplemented
        assert split is None, "split must be None for ASDiv"
        dataset = libs_data.lib_asdiv.ASDiv(
            tokenizer=any_tokenizer,
            cache_path="/tmp/asdiv",
        )

    elif dataset_name == DatasetChoices.SENTIMENT:
        assert False
        dataset = libs_data.lib_sentiment.SentimentData(
            any_tokenizer=any_tokenizer, 
            split=split,
        )

    elif dataset_name == DatasetChoices.ARITHMETIC:
        dataset = libs_data.lib_arithmetic.Arithmetic(
            dataset_root_folder_dir=arithmetic_dataset_root_folder_dir,
            answer_only     = answer_only,
            any_tokenizer   = any_tokenizer,
            eos_token       = any_tokenizer.eos_token,
            pad_token       = any_tokenizer.pad_token,
            question_prefix = None,
            question_suffix = None,
            sft_mode = False,
            shuffle_once    = False,
            split           = split,
            use_few_shots   = True,
            use_curriculum  = use_curriculum,
            extractor_ignore_one_line = extr_arith_ignore_one_line,
        )

    else:
        raise ValueError(f"Unsupported task: {dataset_name}")

    return dataset

