#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import enum
import multiprocessing.pool as mp_pool
from pathlib import Path
import shutil
import time
import hashlib

from beartype import beartype
import fire
import numpy as np
import h5py
import jsonlines as jsonl
import rich
from tqdm import tqdm
import transformers

import general_utils as gu

DATA_URI = "./data"
INT_DTYPE = np.int64
CHAINER = " => "

# In[4]:


def md5sum(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def pad_and_mask(arr: list[int], pad_token_id: int):
    assert isinstance(arr[0][0], int), type(arr[0][0])
    max_len = max(len(x) for x in arr)
    padded = np.array([sequence + [pad_token_id] * (max_len - len(sequence)) for sequence in arr], dtype=INT_DTYPE)
    attention_mask = np.array([len(sequence) * [1] + [0] * (max_len - len(sequence)) for sequence in arr], dtype=INT_DTYPE)
    return padded, attention_mask


def convert_jsonl_data_to_h5py(
    *, 
    tokenizer: transformers.PreTrainedTokenizer, 
    tokenizer_name: str, 
    path_h5: Path, 
    path_jsonl: Path, 
    md5_jsonl: str, 
    verbose=True
    ):

    with h5py.File(path_h5, "w") as h5:
        h5.attrs["tokenizer_name"] = tokenizer_name

        if verbose:
            rich.print(f"Reading h5py from {path_jsonl}")

        with jsonl.open(path_jsonl) as reader:
            h5.attrs["md5"] = md5_jsonl
            raw_data = list(reader)

        if verbose:
            rich.print(f"Converting {len(raw_data)} lines")
        text = {
            "input": [x["input"] + CHAINER for x in raw_data],
            "value": [x["value"] for x in raw_data],
            "scratchpad": [x["scratchpad"] for x in raw_data],
        }

        if verbose:
            text_items = tqdm(text.items(), desc="Padding, tokenizing, masking then saving to h5py.")
        else:
            text_items = text.items()

        for key, text_v in text_items:
            assert isinstance(text_v[0], str), (type(text_v[0]), text_v[0])
            tokenized = tokenizer(text_v, add_special_tokens=False, return_tensors=None)["input_ids"]
            padded, attention_mask = pad_and_mask(tokenized, tokenizer.pad_token_id)
            h5.create_dataset(key, data=padded)
            h5.create_dataset(key + "_text", data=text_v)
            h5.create_dataset(key + "_attention_mask", data=attention_mask)
            
    return path_h5

class FakePool:
    def __init__(self, n_procs=None):
        pass

    class _FakeAsyncResult:
        def __init__(self, result):
            self._result = result

        def ready(self):
            return True

        def get(self):
            return self._result
    
    def apply_async(self, func, args, kwargs):
        rich.print(f"\n[bold]FakePool.[/bold]")
        rich.print(f"func: {func}")
        rich.print(f"args:")
        gu.print_list(args)
        rich.print(f"kwargs:")
        gu.print_dict(kwargs)
        return self._FakeAsyncResult(func(*args, **kwargs))

    def close(self):
        pass


class PoolType(str, enum.Enum):
    FAKE = "fake"
    PROCESS = "process"
    THREAD = "thread"


POOL_TYPE_MAPPING = {
    PoolType.FAKE: FakePool,
    PoolType.PROCESS: mp_pool.Pool,
    PoolType.THREAD: mp_pool.ThreadPool,
}


@beartype
def main(
    hf_name: str = "gpt2-medium", 
    pool_type: str = PoolType.FAKE,
    verbose: bool = False,
):

    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_name)
    if "gpt2" in hf_name:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pool = POOL_TYPE_MAPPING[pool_type]()
    jobs = collections.deque()

    for path_jsonl in [p for p in Path(DATA_URI).glob("**/*.jsonl")]:
        assert path_jsonl.exists(), path_jsonl
        md5_jsonl = md5sum(path_jsonl)
        path_h5 = path_jsonl.with_suffix(".h5")
        shared_kwargs = dict(
            tokenizer=tokenizer, 
            path_h5=path_h5, 
            path_jsonl=path_jsonl, 
            md5_jsonl=md5_jsonl, 
            verbose=verbose,
            tokenizer_name=hf_name,
        )

        rich.print(f"Working with \"{path_h5}\".")
        if not path_h5.exists():
            rich.print(f"{path_h5} does not exist, creating it")
            jobs.append(pool.apply_async(convert_jsonl_data_to_h5py, (), shared_kwargs))
        else:

            with h5py.File(path_h5, "r") as h5:
                md5_h5 = h5.attrs.get("md5", None)
                h5_hf_name = h5.attrs.get("tokenizer_name", None)

            if md5_h5 and md5_jsonl != md5_h5:
                rich.print(f"\"{path_h5}\" exists but md5 does not match, creating it")
                jobs.append(pool.apply_async(convert_jsonl_data_to_h5py, (), shared_kwargs))
            elif h5_hf_name is None or h5_hf_name != hf_name:
                rich.print(f"\"{path_h5}\" exists but tokenizer name does not match. `{hf_name}` in args, `{h5_hf_name}` in current h5 file.")
                decision = input(f"Do you want to recreate it? [y/n]")
                if decision.lower() == "y":
                    jobs.append(pool.apply_async(convert_jsonl_data_to_h5py, (), shared_kwargs))
            else:
                rich.print(f"\"{path_h5}\" exists, do you want to overwrite it? [y/n]")
                if input() == "y":
                    jobs.append(pool.apply_async(convert_jsonl_data_to_h5py, (), shared_kwargs))
                else:
                    rich.print("skipping")

    progress_bar = tqdm(total=len(jobs))
    while jobs:
        top = jobs.pop()
        if not top.ready():
            jobs.appendleft(top)
            time.sleep(0.1)
        else:
            progress_bar.update(1)
            rich.print(f"\nDone with \"{top.get()}\"")

    rich.print("Done.")
    pool.close()
    rich.print("Closed.")


if __name__ == "__main__":
    fire.Fire(main)
            


# In[ ]:




