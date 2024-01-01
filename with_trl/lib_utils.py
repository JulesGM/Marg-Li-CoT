from __future__ import annotations

import contextlib
import math
import enum
import jsonlines as jl
import os
import pathlib
import typing
from typing import Any, Optional

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


def ValidPrecisions(precision):
    return dict(
        _4bit="int4",
        _8bit="int8",
        bfloat16=torch.bfloat16,
        float16=torch.float16,
        float32=torch.float32,
    )[precision]


def readable(obj, title="", outer_kwargs=None, is_inner=False):
    assert isinstance(obj, (dict, list, tuple)), type(obj)

    if outer_kwargs is None:
        outer_kwargs = {}

    if is_inner:
        kwargs = dict(
            show_edge   = True,
            show_footer = False,
            show_header = False,
            # show_lines = True,
        )
    else:
        kwargs = dict(
            show_header = True,
            show_edge   = True,
            show_footer = False,
            # show_lines = True,
            
        )

    dict_mode = isinstance(obj, dict)
    
    if dict_mode:
        fields = ["Keys", "Values"]
    else:
        assert isinstance(obj, (tuple, list))
        fields = ["Values"]

    if title:
        title = f"{title}: "

    table = rich.table.Table(
        *fields,
        highlight=True, 
        expand=True,
        title=title + (
            "Dict" 
            if dict_mode 
            else type(obj).__name__
        ),
        title_justify="left",
        **kwargs,
    )

    def should_nest(v):
        is_collection = isinstance(v, (dict, list, tuple))
        return is_collection and not len(v) == 0

    if dict_mode:
        for k, v in sorted(obj.items(), key=lambda kv: kv[0]):
            if should_nest(v):
                table.add_row(str(k), readable(v, title=k, is_inner=True))
            else:
                table.add_row(str(k), rich.markup.escape(str(v)))
    else:
        for v in obj:
            if should_nest(v):
                table.add_row(readable(v, is_inner=True))
            else:
                table.add_row(rich.markup.escape(str(v)))

    if not is_inner:
        rich.print(table)

    return table


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
    yield from tqdm(
        seq, 
        desc    = description, 
        total   = total, 
        disable = disable
    )


def child_names(pt_module):
    return set(name for name, _ in pt_module.named_children())


def print_accelerate_envs():
    if RANK == 0:
        keys = [k for k in sorted(os.environ) if "accelerate" in k.lower()]

        table = rich.table.Table(
            "Key", 
            "Value", 
            highlight     = True,
            show_header   = False,
            title         = " Accelerate Environment Variables:",
            title_justify = "left",
        )

        for k in keys:
            if "accelerate" in k.lower():
                form_k = k.replace(
                    "DEEPSPEED",
                    "[green]DEEPSPEED[/]",
                )

                table.add_row(
                    rich.markup.escape(str(form_k)), 
                    rich.markup.escape(str(os.environ[k]))
                )

        print()
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


def is_valid_simple_filename(filename):
    assert isinstance(filename, str), type(filename)
    
    for char in filename:
        if not char.isalnum() and not char in "_-":
            return False
        
    return True


class WandbTableRepair:
    def __init__(
            self, *, 
            columns:         list[str],
            new_table_mode:  bool                     = False,
            run_output_path: str | pathlib.Path       = None,
            save_to_file:    bool                     = False,
            table_name:      str,
            wandb_on:        bool                     = False,
            wandb_kwargs:    Optional[dict[str, Any]] = None, 
        ):

        #######################################################################
        # Checks
        #######################################################################
        assert isinstance(wandb_on, bool), type(wandb_on)
        if wandb_kwargs is None:
            wandb_kwargs = {}

        if save_to_file:
            if not is_valid_simple_filename(table_name):
                raise ValueError(
                    "table_name must be a valid simple filename, " +
                    f"so alphanum or _-. Got {table_name}"
                )
            self._file_path = pathlib.Path(run_output_path) / f"{table_name}.jsonl"
        else:
            self._file_path = None

        #######################################################################
        # Action
        #######################################################################
        self._columns_list    = list(columns)
        self._columns_keys    = set (columns)
        self._creation_kwargs = wandb_kwargs
        self._new_table_mode  = new_table_mode
        self._save_to_file    = save_to_file
        
        assert not wandb_on
        self._wandb_on        = wandb_on
        
        if wandb_on:
            self._table = wandb.Table(columns=columns, **wandb_kwargs)
        else:
            self._table = None

    def add_data(self, *args, **kwargs):
        assert bool(args) ^ bool(kwargs), (
            args, kwargs)

        if self._wandb_on:
            if not args:
                args = self._dict_to_list(kwargs)
            self._table.add_data(*args)

        if not kwargs:
            kwargs = self._list_to_dict(args)

        if self._save_to_file:
            self._write_to_file(kwargs)
        
    def get_loggable_object(self):
        if self._wandb_on:
            if self._new_table_mode:
                table_to_return = self._table
                self._recreate_table(self)
            else:
                table_to_return = self._table
        else:
            table_to_return = None

        return table_to_return
    
    def _list_to_dict(self, values):
        return {
            column_name: value 
            for column_name, value 
            in mit.zip_equal(self._columns_list, values)
        }
    
    def _dict_to_list(self, values):
        return [
            values[column_name] 
            for column_name 
            in self._columns_list
        ]

    def _write_to_file(self, dict_values):
        # We don't want multiple processes trying to write to the file.
        assert isinstance(dict_values, dict), type(dict_values)

        assert RANK == 0, RANK 
        with jl.open(self._file_path, "a") as f:
            f.write(dict_values)

    def _recreate_table(self):
        assert self._wandb_on, self._wandb_on
        self._table = wandb.Table(
            columns = self._columns_list,
            data    = self._table.data,
            **self._creation_kwargs,
        )

class WandbAndRichTable:
    def __init__(
            self, *, 
            columns, 
            new_table_mode,
            table_name,
            run_output_path,
            rich_kwargs  = None, 
            wandb_kwargs = None,
        ):
        
        if rich_kwargs is None:
            rich_kwargs = {}

        if wandb_kwargs is None:
            wandb_kwargs = {}

        self._columns      = columns
        self._rich_kwargs  = rich_kwargs
        self._wandb_kwargs = wandb_kwargs
        
        if "title" not in self._rich_kwargs:
            self._rich_kwargs["title"] = table_name

        self._rich_table = rich.table.Table(
            *self._columns, 
            **self._rich_kwargs,
        )
        
        self._run_output_path = run_output_path

        self._wandb_table = WandbTableRepair(
            columns         = columns,
            new_table_mode  = new_table_mode,
            run_output_path = run_output_path,
            table_name      = table_name, 
            wandb_kwargs    = wandb_kwargs,
            wandb_on        = True,
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


def check_curriculum_schedule(curriculum):
    # Check that distributions sum to 1
    for num_steps, difficulty_distribution in curriculum:
        if not math.isclose(sum(difficulty_distribution.values()), 1.):
            raise ValueError(
                "Difficulty distributions must sum to 1. "
                f"{num_steps = }, {difficulty_distribution =}"
            )
        
    # Check that the number of steps are integers and  strictly increasing
    if not curriculum[0][0] == 0:
        raise ValueError(
            f"First step must be 0. {curriculum[0][0] = }"
        )
    
    for i in range(1, len(curriculum)):
        current_curriculum_step = curriculum[i][0]
        previous_curriculum_step = curriculum[i - 1][0]

        # Check that the number of steps are integers
        if not isinstance(current_curriculum_step, int):
            raise ValueError(
                f"Number of steps must be integers. {curriculum[i][0] = }"
            )
        # Check that the number of steps are strictly increasing
        if not current_curriculum_step > previous_curriculum_step:
            raise ValueError(
                f"Number of steps must be strictly increasing. {curriculum[i][0] = }"
            )

    return curriculum