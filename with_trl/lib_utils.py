import contextlib
import enum
import os
import typing
from typing import Any, Optional, Union

import accelerate
import numpy as np
import rich
import rich.table
import torch
import transformers
from tqdm import tqdm
import trl
import trl_fork


RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

class TrlLibraryMode(enum.Enum):
    TRL = "trl"
    TRL_FORK = "trl_fork"
    
TRL_LIBRARIES = {
    TrlLibraryMode.TRL: trl,
    TrlLibraryMode.TRL_FORK: trl_fork,
}


class MovingAverage:
    def __init__(self, window_size: int):
        self._window_size = window_size
        self._window = np.zeros(window_size)
        self._pointer = 0
        self._size = 0

    @property
    def window_size(self):
        return self._window_size

    @property
    def size(self):
        return self._size

    def update(self, value: float):
        self._window[self._pointer] = value
        self._pointer = (self._pointer + 1) % self._window_size
        self._size = min(self._size + 1, self._window_size)
        return self.get()

    def get(self) -> tuple[float, tuple[float, int]]:
        if self._size == 0:
            raise ValueError("No data in the moving average window. " "self._size == 0")

        window_sum = self._window.sum()
        return window_sum / self._size, (window_sum, self._size)


class RewardChoices(str, enum.Enum):
    EXACT_MATCH = "exact_match"
    REF_PPL = "ref_ppl"


class Task(str, enum.Enum):
    SENTIMENT = "sentiment"
    MAIN = "main"


class ValidPrecisions(enum.Enum):
    _4bit = "int4"
    _8bit = "int8"
    bfloat16 = torch.bfloat16
    float16 = torch.float16
    float32 = torch.float32

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ValidPrecisions):
            raise TypeError(f"Cannot compare {type(self)} with {type(value)}")

        return super().__eq__(value)


def not_first_token(*, tensor, forward_tokenizer):
    assert len(tensor.shape) == 2
    assert not (tensor[:, 0] == forward_tokenizer.pad_token_id).any()
    assert forward_tokenizer.padding_side == "right"


def not_last_token(*, tensor, predict_tokenizer):
    assert len(tensor.shape) == 2
    assert not (tensor[:, -1] == predict_tokenizer.pad_token_id).any()
    assert predict_tokenizer.padding_side == "left"


def progress(seq, description, total=None, disable=False):
    yield from tqdm(seq, desc=description, total=total, disable=disable)


def child_names(pt_module):
    return set(name for name, _ in pt_module.named_children())


def print_accelerate_envs():
    if RANK == 0:
        keys = [k for k in sorted(os.environ) if "accelerate" in k.lower()]

        table = rich.table.Table(
            "Key", "Value", title="Accelerate Environment Variables"
        )
        for k in keys:
            if "accelerate" in k.lower():
                form_k = k.replace(
                    "DEEPSPEED",
                    "[green]DEEPSPEED[/]",
                )
                table.add_row(form_k, os.environ[k])
        table.caption = str(len(table.rows))
        rich.print(table)


@contextlib.contextmanager
def maybe_context_manager(caller, disable):
    if disable:
        yield
    else:
        with caller():
            yield

class DictDataset(torch.utils.data.Dataset):
    # Object Pandas without the fluff

    def __init__(self, data=None, keys=None):
        assert data or keys
        assert isinstance(data, dict) or data is None

        if data and keys:
            assert data.keys() == keys, (data.keys(), keys)

        if data is None:
            self._dataset = {k: [] for k in keys}
        else:
            self._dataset = data

    def __getitem__(self, key: typing.Union[str, int]):
        if isinstance(key, (int, slice)):
            return {k: v[key] for k, v in self._dataset.items()}
        elif isinstance(key, str):
            return self._dataset[key]
        else:
            raise TypeError(type(key))

    def __len__(self) -> int:
        one_len = len(next(iter(self._dataset.values())))
        return one_len

    def check_lens(self):
        lengths = []
        for v in self._dataset.values():
            assert v is not self
            lengths.append(len(v))

        assert all(lengths[0] == l for l in lengths[1:]), lengths
        return tuple(lengths)

    def append(self, dict_) -> None:
        assert dict_.keys() == self._dataset.keys(), (
            dict_.keys(),
            self._dataset.keys(),
        )

        for k, v in dict_.items():
            self._dataset[k].append(v)

    def __iter__(self):
        len_ = len(self)

        # We make a copy of the dataset to avoid
        # the case when the dataset is modified
        per_item_copy = [self[i] for i in range(len_)]
        assert len(per_item_copy) == len_, (len(per_item_copy), len_)

        return iter(per_item_copy)

    def items(self):
        return self._dataset.items()

    def keys(self):
        return self._dataset.keys()

    def values(self):
        return self._dataset.values()

    def get_dict(self):
        return self._dataset
    