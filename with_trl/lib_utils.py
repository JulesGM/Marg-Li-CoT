import contextlib
import enum
import os
import pathlib
import typing
from typing import Any, Optional, Union

import accelerate
import more_itertools as mit
import numpy as np
import rich
import rich.table
import torch
import transformers
from tqdm import tqdm
import trl
import trl_fork
import wandb

import lib_base_classes

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


class Datasets(str, enum.Enum):
    COMMONSENSE_QA = "commonsense_qa"
    ARITHMETIC = "arithmetic"


class TrlLibraryMode(enum.Enum):
    TRL = "trl"
    TRL_FORK = "trl_fork"


TRL_LIBRARIES = {
    TrlLibraryMode.TRL: trl,
    TrlLibraryMode.TRL_FORK: trl_fork,
}


class CVSets(str, enum.Enum):
    TRAIN = "train"
    VALID = "validation"


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


def line_return_token(any_tokenizer):
    candidate = mit.one(any_tokenizer("\n").input_ids)
    assert isinstance(candidate, int), type(candidate)
    decoded = any_tokenizer.decode([candidate])
    assert decoded == "\n", (decoded, candidate)
    a_tok = mit.one(any_tokenizer("a").input_ids)
    decoded_test = any_tokenizer("a\na").input_ids
    assert decoded_test == [a_tok, candidate, a_tok], decoded_test
    return candidate
    

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

def all_equal(iterable):
    
    it = iter(iterable)
    try:
        first = next(it)
    except StopIteration:
        return True
    
    return all(first == rest for rest in it)


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
            first = None
            
            assert isinstance(data, dict), type(data).mro()
            for k, v in self._dataset.items():
                assert isinstance(k, str), type(k).mro()
                if first is None:
                    first = len(v)
                else:
                    assert isinstance(v, list), type(v).mro()
                    assert first == len(v), (first, len(v))

    def __getitem__(self, key: typing.Union[str, int]):
        if isinstance(key, (int, slice)):
            if isinstance(key, int):
                assert len(self) > key, (len(self), key)
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

    def extend(self, dict_) -> None:
        assert dict_.keys() == self._dataset.keys(), (
            dict_.keys(),
            self._dataset.keys(),
        )

        for k, v in dict_.items():
            self._dataset[k].extend(v)

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

    def shuffle(self):
        indices = np.random.permutation(len(self))
        for k, v in self._dataset.items():
            self._dataset[k] = [v[i] for i in indices]


def get_tmp_dir() -> pathlib.Path:
    if "SLURM_TMPDIR" not in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
        tmp_dir = pathlib.Path(f"/Tmp/slurm.{job_id}.0")
        assert tmp_dir.exists(), f"{tmp_dir} does not exist."
    else:
        tmp_dir = pathlib.Path(os.environ["SLURM_TMPDIR"])

    return tmp_dir


class WandbTableRepair:
    def __init__(self, *, wandb_args=None, wandb_kwargs=None, new_table_mode=False):
        if wandb_args is None:
            wandb_args = []
        if wandb_kwargs is None:
            wandb_kwargs = {}
        self._creation_args = wandb_args
        self._creation_kwargs = wandb_kwargs
        self._new_table_mode = new_table_mode
        self._table = wandb.Table(*wandb_args, **wandb_kwargs)

    def add_data(self, *args, **kwargs):
        self._table.add_data(*args, **kwargs)

    def get_loggable_object(self):
        if self._new_table_mode:
            table_to_return = self._table
            self._table = wandb.Table(
                *self._creation_args, 
                **self._creation_kwargs, 
                data=self._table.data,
            )
        else:
            table_to_return = self._table

        return table_to_return
    

class WandbAndRichTable:
    def __init__(self, columns, rich_kwargs=None, wandb_kwargs=None):
        
        if rich_kwargs is None:
            rich_kwargs = {}

        if wandb_kwargs is None:
            wandb_kwargs = {}

        self._columns      = columns
        self._rich_kwargs  = rich_kwargs
        self._wandb_kwargs = wandb_kwargs
        self._rich_table   = rich.table.Table(
            *self._columns, 
            **self._rich_kwargs,
        )
        self._wandb_table = WandbTableRepair(
            wandb_kwargs = dict(columns=columns, **wandb_kwargs)
        )

    def add_row(self, *args, **kwargs):
        self._rich_table .add_row (*args, **kwargs)
        self._wandb_table.add_data(*args, **kwargs)
    
    def get_loggable_object(self):
        rich.print(self._rich_table)
        self._rich_table = rich.table.Table(*self._columns, **self._rich_kwargs)
        return self._wandb_table.get_loggable_object()
    

def compute_and_gather_metrics(
        *, 
        accelerator:   accelerate.Accelerator,
        batch:         lib_base_classes.DataListContainer,
        metrics:       dict[str, lib_base_classes.Metric],
        response_text: list[str],
    ):
    
    ###########################################################################
    # Entrance checks.
    ###########################################################################
    assert isinstance(response_text, list), type(response_text)
    assert isinstance(response_text[0], str), type(response_text[0])
    assert isinstance(batch, lib_base_classes.DataListContainer), type(batch)
    assert isinstance(accelerator, accelerate.Accelerator), type(accelerator)

    assert isinstance(metrics, dict), type(metrics)
    first_metric = next(iter(metrics.values()), None)
    assert not metrics or isinstance(first_metric, lib_base_classes.Metric), metrics
    del first_metric

    ###########################################################################
    # Action.
    ###########################################################################
    gathered_values = {}
    local_metric_values = {}
    for metric_name, metric_callable in metrics.items():
        local_metric = metric_callable(
            responses = response_text,
            batch     = batch,
        )
        assert len(local_metric.values) == len(batch), (
            len(local_metric.values), 
            len(batch),
        )
        local_metric_values[metric_name] = local_metric
        
        not_none_values = [x for x in local_metric.values if x is not None]
        if not_none_values:
            gathered_values[metric_name] = accelerator.gather_for_metrics(
                torch.tensor(not_none_values).to(accelerator.device))

    ###########################################################################
    # Exit checks.
    ###########################################################################
    assert isinstance(    gathered_values, dict), type(    gathered_values)
    assert isinstance(local_metric_values, dict), type(local_metric_values)
    
    first_gathered_values = next(iter(gathered_values.values()), None)
    assert not gathered_values or isinstance(first_gathered_values, 
        torch.Tensor), type(first_gathered_values)
    del first_gathered_values

    first_local_metric_values = next(iter(local_metric_values.values()), None)
    assert not local_metric_values or isinstance(first_local_metric_values, 
        lib_base_classes.MetricOutput), type(first_local_metric_values)
    del first_local_metric_values

    return gathered_values, local_metric_values