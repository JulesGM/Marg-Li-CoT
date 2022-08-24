from beartype import beartype
import collections
import h5py  # type: ignore[import]
import functools
import inspect
import itertools
import math
from pathlib import Path
import subprocess
import types; 
from typing import *

import natsort
import numpy as np
import rich

SCRIPT_DIR = Path(__file__).absolute().parent


def check_not_equal(a, b):
    assert a != b, f"{a} == {b}"
    
    
def check_equal(a, b):
    assert a == b, f"{a} != {b}"


def check_and_print_args(all_arguments, function):
    check_args(all_arguments, function)
    rich.print("[bold]Arguments:")
    print_dict(all_arguments)
    print()


def check_args(all_arguments, function):
    # We get the arguments by calling `locals`. This makes sure that we
    # really called locals at the very beginning of the function, otherwise
    # we have supplementary keys.
    assert all_arguments.keys() == inspect.signature(function).parameters.keys(), (
        f"\n{sorted(all_arguments.keys())} != "
        f"{sorted(inspect.signature(function).parameters.keys())}"
    )

@beartype
def shorten_path(path: Union[Path, str]) -> str:
    
    if isinstance(path, str):
        path = Path(path.strip())

    if path.is_relative_to(SCRIPT_DIR):
        path = "<pwd> /" + str(path.relative_to(SCRIPT_DIR))
    else:
        path = str(path)
    return path

def print_list(_list):
    at_least_one = False
    for line in _list:
        at_least_one = True
        if isinstance(line, (str, Path)):
            line = shorten_path(line)
        rich.print(f"\t- {line}")

    if not at_least_one:
        rich.print("\t[bright_black]<empty list>")

def print_dict(_dict: dict[str, Any]) -> None:
    # Pad by key length
    max_len = len(max(_dict, key=lambda key: len(str(key)))) + 1
    at_least_one = False
    for k, value in _dict.items():
        at_least_one = True
        if isinstance(value, (Path, str)):
            value = shorten_path(value)

        rich.print(f"\t- {k} =" + (max_len - len(k)) * " " + f" {value}")

    if not at_least_one:
        rich.print("\t[bright_black]<empty dict>")


def dict_zip(dict_of_lists: dict[Any, Iterable[Any]], check_lengths=True) -> Generator[dict[Any, Any], None, None]:
    """
    Takes a dict of iterables and returns a generator of dicts with the same keys and a value of the lists at each iteration.
    """

    if check_lengths:
        # Requires all lists to have the same length.
        # Could do some iterator stuff to keep track of the length. Haven't done that yet.
        l0 = len(next(iter(dict_of_lists.values())))
        assert all(l0 == len(v) for v in dict_of_lists.values()), {k: len(v) for k, v in dict_of_lists.items()}

    iters = {k: iter(v) for k, v in dict_of_lists.items()}
    
    while True:
        try:
            yield {k: next(iters[k]) for k in dict_of_lists.keys()}

        except StopIteration:
            break


def dict_unzip(list_of_dicts: list[dict[Any, Any]], key_subset=None) -> dict[Any, list[Any]]:
    """
    Unzips a list of dicts with the same keys into a dict of lists.
    """
    if key_subset is None:
        keys = list_of_dicts[0].keys()
    else:
        keys = key_subset

    dict_of_lists = collections.defaultdict(list)
    
    for ld in list_of_dicts:
        assert ld.keys() == keys, f"{ld.keys()} != {keys}"
        for k in keys:
            dict_of_lists[k].append(ld[k])

    return dict_of_lists



def clean_locals(locals_, no_modules=True, no_classes=True, no_functions=True, no_caps=True):
    print([
        k for k, v in locals_.items() if 
        not k.startswith("_") and 
        (not no_modules or not isinstance(v, types.ModuleType)) and 
        (not no_classes or not isinstance(v, type)) and 
        (not no_functions or not callable(v)) and 
        (not no_caps or k == k.lower())
    ])


def concat_lists(lists):
    assert all(isinstance(l, list) for l in lists)
    return sum(lists, [])


def concat_tuples(tuples):
    assert all(isinstance(l, tuple) for l in tuples)
    return sum(tuples, ())


def concat_iters(iters):
    return list(itertools.chain.from_iterable(iters))


def sort_iterable_text(list_text):
    return natsort.natsorted(list_text)


def find_last(seq: Sequence[Any], item: Any) -> int:
    return len(seq) - seq[::-1].index(item) - 1


def cmd(command: list[str]) -> list[str]:
    return subprocess.check_output(command).decode("utf-8").strip().split("\n")


def only_one(it: Iterable):
    iterated = iter(it)
    good = next(iterated)
    for bad in iterated:
        raise ValueError("Expected only one item, got more than one.")
    return good


def count_lines(path: Path) -> int:
    return int(check_len(only_one(cmd(["wc", "-l", str(path)])).split(), 2)[0])

def check_len(seq: Sequence, expected_len: int) -> Sequence:
    if not len(seq) == expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(seq)}.")
    return seq

SIZE_HUMAN_NAMES = {
        0: "B",
        1: "KB",
        2: "MB",
        3: "GB",
    }

def to_human_size(size: int) -> str:
    if size == 0:
        return "0 B"

    exponent = int(math.log(size, 1000))
    mantissa = size / 1000 ** exponent
    return f"{mantissa:.2f} {SIZE_HUMAN_NAMES[exponent]}"

def print_structure_h5(file_object: h5py.File):
    work_stack = [(file_object, "")]
    POSSIBLE_TYPES = (h5py.Dataset, h5py.Group, h5py.File)

    all_text = []

    while work_stack:

        obj, parent_name = work_stack.pop()
        assert isinstance(obj, (h5py.Group, h5py.Dataset, h5py.File))

        if obj.name == "":
            obj_name = "<root>"
        else:
            obj_name = obj.name
        
        if parent_name:    
            name = parent_name + "/" + obj_name
        else:
            name = obj_name

        message = f"\"{name}\": {type(obj).__name__}"
        if isinstance(obj, h5py.Dataset):
            message += f" {obj.shape} {obj.dtype}"
        all_text.append(message)

        if obj.attrs:
            for k, v in obj.attrs.items():
                message_attr = f"\t- {k}: {type(v).__name__}"
                if isinstance(v, (str, int, float)):
                    message_attr += f" value=\"{v}\""
                elif isinstance(v, np.ndarray):
                    message_attr += f" shape={v.shape} dtype={v.dtype}"
                elif isinstance(v, (tuple, list, dict)):
                    message_attr += f" {type(v)} {len(v)}"
                all_text.append(message_attr)

        if hasattr(obj, "items"):
            assert isinstance(obj, (h5py.Group, h5py.File))
            for key, value in obj.items():
                work_stack.append((value, parent_name))

    rich.print("\n".join(all_text))

def check_shape(shape: Sequence[int], expected_shape: Sequence[int]):
    if not shape == expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {shape}.")