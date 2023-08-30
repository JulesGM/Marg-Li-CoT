import itertools as it
import os
import pathlib
import sys
import typing
from typing import Any, Optional, Union

import datasets
import fire
import jsonlines as jsonl
import more_itertools as mit
import numpy as np
import rich
import rich.box
import rich.table
import torch
import torch.utils
import torch.utils.data
import transformers

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR.parent))

import lib_utils

datasets.disable_caching()


def openai_commonsense_qa_output(root_path):
    path = pathlib.Path(root_path)
    assert path.exists(), f"{path} does not exist"
    assert path.is_dir(), f"{path} is not a directory"

    paths = dict(
        train=path/"commonsenseqa.chatgpt.train.jsonl",
        eval=path/"commonsenseqa.chatgpt.validation.jsonl",
    )

    for split, path in paths.items():
        assert path.exists(), f"{split}: {path} does not exist"
        assert path.is_file(), f"{split}: {path} is not a file"

    # Read the data
    data = {}
    for split, path in paths.items():
        with jsonl.open(path) as f:
            data[split] = list(f)

    # Invert the data
    for split, split_data in data.items():
        keys = list(split_data[0].keys())
        data[split] = {key: [d[key] for d in split_data] for key in keys}

    # Make into dataset objects
    output_data = {}
    for split, split_data in data.items():
        output_data[split] = lib_utils.DictDataset(split_data)

    return output_data


def main(
):
    
    import lib_base_classes
    import approach_sft.lib_sft_constants as lib_sft_constants
    import approach_sft.lib_sft_collators as lib_sft_collators
    import lib_trl_utils
    import transformers
    
    data_directory = "/network/scratch/g/gagnonju/saved_scratchpad_gen_outputs/chatgpt-3.5-commonsenseqa-scratchpads/cond-on-answers"
    model_name_or_path = "EleutherAI/pythia-410m"
    output_type = lib_sft_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER

    tmp_tokenizers = lib_trl_utils.load_tokenizers(
        model_name=model_name_or_path, 
        config=transformers.AutoConfig.from_pretrained(model_name_or_path),
    )

    forward_tokenizer = tmp_tokenizers["forward_tokenizer"]
    prediction_tokenizer = tmp_tokenizers["prediction_tokenizer"]
    del tmp_tokenizers

    datasets = openai_commonsense_qa_output(data_directory)
    data_collator = lib_sft_collators.CausalFullCollator(
        output_type=output_type,
        forward_tokenizer=forward_tokenizer,
        prediction_tokenizer=prediction_tokenizer,
    )


if __name__ == "__main__":
    fire.Fire(main)