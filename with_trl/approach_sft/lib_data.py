import itertools as it
import os
import pathlib
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

datasets.disable_caching()


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict):
        self._data = data
        self._keys = list(data.keys())
        self._length = len(data[self.keys[0]])


    def __getitem__(self, *args, **kwargs):
        return {
            key: self._data[key].__getitem__(*args, **kwargs) 
            for key in self._keys
        }

    def __len__(self):
        return self._length

    def __iter__(self):
        return (self[i] for i in range(self._length))


def openai_commonsense_qa_output(root_path):
    path = pathlib.Path(root_path)
    assert path.exists(), f"{path} does not exist"
    assert path.is_dir(), f"{path} is not a directory"

    paths = dict(
        train=path/"commonsenseqa.chatgpt.train.jsonl",
        validation=path/"commonsenseqa.chatgpt.validation.jsonl",
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
        for key in keys:
            data[split] = {key: [d[key] for d in split_data]}

    # Make into dataset objects
    for split, split_data in data.items():
        data[split] = DictDataset(split_data)

    return data


def main():
    pass


if __name__ == "__main__":
    fire.Fire(main)