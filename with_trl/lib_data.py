""" 
Prep data & collator for RL.

- Contains `prep_dataset_rl`, the function that picks form the different dataset classes
and calls them.

- Contains `data_item_collator`, the main collator used for the RL dataloader.

"""
import collections
import enum
import logging
import os
from pathlib import Path
from typing import Optional, Union

import more_itertools as mit
import datasets
import torch
import torch.utils
import torch.utils.data
import transformers

from with_trl import lib_base_classes
from with_trl import libs_data
from with_trl import libs_extraction
from with_trl import lib_utils

import with_trl.libs_extraction.lib_numerical
import with_trl.libs_extraction.lib_multiple_choice
import with_trl.libs_data.lib_arithmetic
import with_trl.libs_data.lib_commonsense_qa
import with_trl.libs_data.lib_gsm8k
import with_trl.libs_data.lib_sentiment
import with_trl.libs_data.lib_asdiv
import with_trl.libs_data.lib_base


LOGGER = logging.getLogger(__name__)
RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "0"))


class DatasetChoices(str, enum.Enum):
    ARITHMETIC       = "arithmetic"
    ASDIV            = "asdiv"
    COMMONSENSEQA_MC = "cqa"
    GSM8K            = "gsm8k"
    SENTIMENT        = "sentiment"


DATASET_KEY_TO_CLASS = {
    DatasetChoices.ASDIV:            with_trl.libs_data.lib_asdiv.ASDiv,
    DatasetChoices.GSM8K:            with_trl.libs_data.lib_gsm8k.GSM8K,
    DatasetChoices.SENTIMENT:        with_trl.libs_data.lib_sentiment.SentimentData,
    DatasetChoices.COMMONSENSEQA_MC: with_trl.libs_data.lib_commonsense_qa.CommonSenseQAMC,
}


DATASET_KEY_TO_ANSWER_EXTRACTOR = {
    DatasetChoices.ASDIV:            with_trl.libs_extraction.lib_numerical,
    DatasetChoices.GSM8K:            with_trl.libs_extraction.lib_numerical,
    DatasetChoices.SENTIMENT:        None,
    DatasetChoices.COMMONSENSEQA_MC: with_trl.libs_extraction.lib_multiple_choice,
}


def data_item_collator(
    batch_and_indices: list[lib_base_classes.DataItemContainer], 
    use_few_shots: bool,
    prediction_tokenizer: transformers.PreTrainedTokenizerBase,
    inspect_indices: bool = False,
):
    """
    Main collator used for the RL dataloader.
    """
    
    batch, indices = zip(*batch_and_indices)
    assert isinstance(indices, (list, tuple)), f"{indices = }"
    assert isinstance(indices[0], int), f"{indices[0] = }"

    # Prepare indices if we want to inspect them.
    if inspect_indices:
        for i in range(WORLD_SIZE):
            if i == RANK:
                print(f"{RANK}: {sorted(indices)}")
            torch.distributed.barrier()

    # Create a dict of lists from a list of dicts
    new_batch = collections.defaultdict(list)
    
    # 1. Make sure that all of the keys are the same,
    # 2. unzip the batch into a dict of lists
    keys_first = batch[0].keys()
    for item in batch:
        assert keys_first == item.keys(), (
            keys_first, item.keys(),
        )
        
        for k, v in item.items():
            new_batch[k].append(v)

    assert isinstance(new_batch, dict), f"{new_batch = }"
    batch = dict(new_batch)
    del new_batch 

    # Convert to samples.
    samples = []
    batch_size = len(batch["ref_qa_question"])
    first_part_format = "Question: {query}\nReasoning: "
    few_shot_format = first_part_format + "{scratchpad}\nAnswer: {answer}\n\n"

    # 1. Format the sample according to the format we just defined
    # 2. Add the few-shot examples
    for i in range(batch_size):
        sample = first_part_format.format(query=batch["ref_qa_question"][i])
        # We potentially need to move all of this to the collator
        if use_few_shots:
            few_shot_str = ""
            extra_info = batch["extra_information"][i]
            few_shot_examples = extra_info["few_shot_examples"]
            
            for question, scratchpad, answer in mit.zip_equal(
                few_shot_examples["ref_qa_question"],
                few_shot_examples["ref_qa_scratchpad"],
                few_shot_examples["ref_qa_answer"],
            ):
                few_shot_str += few_shot_format.format(
                    query=question, scratchpad=scratchpad, answer=answer)
            sample = few_shot_str + sample

        samples.append(sample.strip())

    batch["tok_ref_query"] = (
        prediction_tokenizer(
            samples, 
            padding=True, 
            return_tensors="pt",
        )["input_ids"]
    )

    batch["text_samples"] = samples
    del batch["extra_information"]

    return batch

    
def prep_dataset_rl(
    *,
    answer_only: bool,
    answer_only_path: Union[str, Path],
    any_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    arithmetic_dataset_root_folder_dir: Optional[str],
    dataset_name: str,
    extr_arith_ignore_one_line,
    few_show_qty: int, 
    question_prefix: str,
    question_suffix: str,
    split: str,
    use_few_shots: int,
    use_curriculum,
    tok_max_query_length,
    tok_max_answer_length,
    tok_max_total_length,
) -> with_trl.libs_data.lib_base.Dataset:
    
    """
    Call the dataset class based on the dataset name.
    """

    split = lib_utils.CVSets(split)
    if answer_only and dataset_name not in {
        DatasetChoices.COMMONSENSEQA_MC, 
        DatasetChoices.ARITHMETIC,
    }:
        raise NotImplementedError()

    if dataset_name == DatasetChoices.GSM8K:
        assert isinstance(LOCAL_RANK, int), type(LOCAL_RANK)
        full_ds = datasets.load_dataset(  # type: ignore
            split="train",
            path="gsm8k",
            name="main",
        )
        split_limit = int(0.9 * len(full_ds))
        if split == with_trl.lib_utils.CVSets.TRAIN:
            ds = full_ds[:split_limit]
        elif split == with_trl.lib_utils.CVSets.VALID:
            ds = full_ds[split_limit:]
        else:
            raise ValueError(split)

        assert few_show_qty >= 1, f"few_show_qty: {few_show_qty}"

        dataset = with_trl.libs_data.lib_gsm8k.GSM8K(
            any_tokenizer=any_tokenizer,
            cv_set=split,
            device=torch.device(LOCAL_RANK),
            ds=ds,
            few_show_qty=few_show_qty,
            tok_max_query_length=tok_max_query_length,
            tok_max_answer_length=tok_max_answer_length,
            tok_max_total_length=tok_max_total_length,
            use_curriculum=use_curriculum,
            use_few_shots=use_few_shots,
        )

    elif dataset_name == DatasetChoices.COMMONSENSEQA_MC:
        raise NotImplementedError
        dataset = with_trl.libs_data.lib_commonsense_qa.CommonSenseQAMC(
            answer_only=answer_only,
            answer_only_path=answer_only_path,
            any_tokenizer=any_tokenizer,
            split=split,
            question_prefix=question_prefix,
            question_suffix=question_suffix,
            use_few_shots=use_few_shots,
        )

    elif dataset_name == DatasetChoices.ASDIV:
        raise NotImplementedError
        assert split is None, "split must be None for ASDiv"
        dataset = with_trl.libs_data.lib_asdiv.ASDiv(
            tokenizer=any_tokenizer,
            cache_path="/tmp/asdiv",
        )

    elif dataset_name == DatasetChoices.SENTIMENT:
        raise NotImplementedError
        dataset = with_trl.libs_data.lib_sentiment.SentimentData(
            any_tokenizer=any_tokenizer, 
            split=split,
        )
            
    elif dataset_name == DatasetChoices.ARITHMETIC:
        dataset = with_trl.libs_data.lib_arithmetic.Arithmetic(
            split                     = split,
            sft_mode                  = False,
            eos_token                 = any_tokenizer.eos_token,
            pad_token                 = any_tokenizer.pad_token,
            return_idx                = False,
            answer_only               = answer_only,
            shuffle_once              = False,
            any_tokenizer             = any_tokenizer,
            use_few_shots             = True,
            use_curriculum            = use_curriculum,
            use_cached_dataset        = False,
            dataset_root_folder_dir   = arithmetic_dataset_root_folder_dir,
            extractor_ignore_one_line = extr_arith_ignore_one_line,
        )

    else:
        raise ValueError(f"Unsupported task: {dataset_name}")

    return dataset

