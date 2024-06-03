import pathlib
import os
import sys

import logging

import torch
import torch.backends
import torch.backends.cuda
import torch.utils
import torch.utils.data
import transformers
import transformers.utils

from with_trl import libs_data
from with_trl import lib_utils
from with_trl.libs_data import lib_arithmetic

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
    answer_only: bool,
    data_directory: pathlib.Path,
    dataset_choice,
    eval_batch_size: int,
    extractor_ignore_one_line,
    filter_bads: bool,
    forward_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    prediction_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    lm_mode: lib_sft_constants.LMModes,
    output_type: lib_sft_constants.OutputTypes,
    qty_eval_small: int,
    train_batch_size: int,
):
    ###########################################################################
    # Datasets
    ###########################################################################
    assert forward_tokenizer.pad_token == prediction_tokenizer.pad_token
    assert forward_tokenizer.eos_token == prediction_tokenizer.eos_token

    if dataset_choice == lib_utils.Datasets.COMMONSENSE_QA:
        datasets = lib_sft_dataset.openai_commonsense_qa_output(
            root_path=data_directory, 
            filter_bads=filter_bads,
        )
    elif dataset_choice == lib_utils.Datasets.ARITHMETIC:

        datasets = {cv_set: lib_arithmetic.Arithmetic(
            answer_only               = answer_only,
            dataset_root_folder_dir   = data_directory,
            eos_token                 = forward_tokenizer.eos_token,
            extractor_ignore_one_line = extractor_ignore_one_line,
            pad_token                 = forward_tokenizer.pad_token,
            split                     = cv_set,
            shuffle_once              = cv_set != lib_utils.CVSets.TRAIN,
            
            sft_mode                  = True,
            use_few_shots             = False,
            use_curriculum            = False,
            use_cached_dataset        = True,

        ) for cv_set in lib_utils.CVSets
    }

    ###########################################################################
    # Collator
    ###########################################################################
    assert forward_tokenizer is not prediction_tokenizer
    if lm_mode == lib_sft_constants.LMModes.CAUSAL_FULL:
        assert dataset_choice == lib_utils.Datasets.ARITHMETIC
        data_collator = lib_sft_collators.CausalMaskedCollator(
            output_type          = output_type,
            forward_tokenizer    = forward_tokenizer,
            prediction_tokenizer = prediction_tokenizer,
            has_choices          = False
        )
    else:
        raise NotImplementedError(lm_mode)

    ###########################################################################
    # Dataloaders
    ###########################################################################
    dataloaders = {}
    for cv_set, cv_set_dataset in datasets.items():
        cv_set = lib_utils.CVSets(cv_set)

        if cv_set == lib_utils.CVSets.TRAIN:
            batch_size = train_batch_size
        else: 
            assert cv_set == lib_utils.CVSets.VALID
            batch_size = eval_batch_size

        dataloaders[cv_set] = torch.utils.data.DataLoader(
            cv_set_dataset,
            batch_size = batch_size,
            collate_fn = data_collator,
            # shuffle    = cv_set == lib_utils.CVSets.TRAIN,
        )

    small_eval_subset_obj = datasets[lib_utils.CVSets.VALID]

    small_dataloader_eval = torch.utils.data.DataLoader(
        small_eval_subset_obj,
        batch_size = eval_batch_size,
        collate_fn = data_collator,
    )

    return dataloaders, small_dataloader_eval