import collections
import enum
import pathlib
import os
import sys

import itertools as it
import logging

import more_itertools as mi
import numpy as np
import rich
import rich.markup
import rich.table
import torch
import torch.backends
import torch.backends.cuda
import torch.utils
import torch.utils.data
import transformers
import transformers.utils
from tqdm import tqdm

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR.parent))

import lib_data
import lib_utils

import approach_sft.lib_sft_collators as lib_sft_collators
import approach_sft.lib_sft_constants as lib_sft_constants
import approach_sft.lib_sft_dataset as lib_sft_dataset


RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
LOGGER = logging.getLogger(__name__)






def get_dataloaders(
    *,
    output_type: lib_sft_constants.OutputTypes,
    lm_mode: lib_sft_constants.LMModes,
    forward_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    prediction_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    train_batch_size: int,
    eval_batch_size: int,
    qty_eval_small: int,
    data_directory: pathlib.Path,
):
    ###########################################################################
    # Datasets
    ###########################################################################
    datasets = lib_sft_dataset.openai_commonsense_qa_output(data_directory)

    ###########################################################################
    # Collator
    ###########################################################################
    assert forward_tokenizer is not prediction_tokenizer
    if lm_mode == lib_sft_constants.LMModes.CAUSAL_FULL:
        data_collator = lib_sft_collators.CausalFullCollator(
            output_type=output_type,
            forward_tokenizer=forward_tokenizer,
            prediction_tokenizer=prediction_tokenizer,
        )
    else:
        raise NotImplementedError(lm_mode)

    ###########################################################################
    # Dataloaders
    ###########################################################################
    dataloaders = {}
    for k, v in datasets.items():
        k = lib_sft_constants.CVSet(k)

        dataloaders[k] = torch.utils.data.DataLoader(
            v,
            batch_size=train_batch_size if k == "train" else eval_batch_size,
            collate_fn=data_collator,
            shuffle=k == lib_sft_constants.CVSet.TRAIN,
        )

    small_dataloader_eval = torch.utils.data.DataLoader(
        torch.utils.data.Subset(datasets[lib_sft_constants.CVSet.VALIDATION], range(qty_eval_small)),
        batch_size=eval_batch_size,
            collate_fn=data_collator,
            shuffle=False,
    )

    return dataloaders, small_dataloader_eval