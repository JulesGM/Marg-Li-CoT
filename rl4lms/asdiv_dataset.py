
import collections

import itertools
import math
import more_itertools
from pathlib import Path
import random
import re
import xml

import matplotlib.pyplot as plt
import numpy as np
import rich
import rich.table
import torch
import wget

import rl4lms.data_pools.text_generation_pool as rl4lms_pool
import rl4lms.envs.text_generation.registry as rl4lms_registry



class ASDivRaw(torch.utils.data.Dataset):
    def __init__(
        self, 
        cache_path, 
        url="https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml", 
        quiet=False):
        super().__init__()

        self._cache_path = Path(cache_path)
        self._url = url

        if not self._cache_path.exists():
            if not quiet:
                print("Downloading dataset...")
            wget.download(self._url, out=str(self._cache_path), bar=None)
            if not quiet:
                print("Download complete.")
        
        if not quiet:
            print("Parsing dataset...")

        with self._cache_path.open() as fp:
            root = xml.etree.ElementTree.parse(fp).getroot()[0]
            self._data = [
                {element.tag: element.text for element in x} | 
                dict(x.items()) for x in root
            ]
        
        if not quiet:
            print("Parsing complete.")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


class ASDiv(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._inner_ds = ASDivRaw(*args, **kwargs)

        for inner_item in self._inner_ds:
            new_keys = {"question", "answer"}
            assert not any(k in inner_item for k in new_keys), new_keys - (new_keys & set(inner_item))
    
    def __len__(self):
        return len(self._inner_ds)
    
    def _get_indiv_item(self, index):
        inner_item = self._inner_ds[index]

        return {
            "question"  : inner_item["Body"] + " " + inner_item["Question"],
            "answer"    : inner_item["Answer"],
            "scratchpad": inner_item["Formula"],
        } | inner_item

    def __getitem__(self, index_or_slice):
        if isinstance(index_or_slice, int): 
            return self._get_indiv_item(index_or_slice)
            
        elif isinstance(index_or_slice, slice):
            s = index_or_slice
            return [
                self._get_indiv_item(i) for i in 
                range(s.start if s.start else 0, s.stop, s.step if s.step else 1)
            ]
            

class ASDivInteger(torch.utils.data.Dataset):
    def __init__(self, *args, quiet, **kwargs):
        super().__init__()
        
        inner_ds = ASDiv(*args, **kwargs, quiet=quiet)
        self._inner_ds = [x for x in inner_ds if self._is_integer(x["answer"])]
        assert self._inner_ds
        assert len(self._inner_ds) / len(inner_ds) > 0.85

        if not quiet:
            print(f"kept {len(self._inner_ds) / len(inner_ds):0.1%} of data points")
    
    def __len__(self):
        return len(self._inner_ds)

    def _get_indiv_item(self, index):
        output_pre = self._inner_ds[index]
        output_pre["answer"] = output_pre["answer"].split(" ")[0]
        return output_pre
    
    def __getitem__(self, index_or_slice):
        if isinstance(index_or_slice, int): 
            return self._get_indiv_item(index_or_slice)
            
        elif isinstance(index_or_slice, slice):
            s = index_or_slice
            return [
                self._get_indiv_item(i) for i in 
                range(s.start if s.start else 0, s.stop, s.step if s.step else 1)
            ]
        
    @classmethod
    def _is_integer(cls, text):
        clean_text = re.sub(r"\s+", " ", text).replace(",", "").strip()
        return re.match(r"^\-?\d+(?: \([\w\s\-\./\\]+\))?$", clean_text) is not None


def _build_dataset_integer(split):
    return ASDivInteger(
        cache_path=f"asdiv.xml",
        quiet=True,
    )


class ZeroShotASDivTextGenPool(rl4lms_pool.TextGenPool):
    @classmethod
    def prepare(cls, split: str):
        dataset = _build_dataset_integer(split)

        samples = []
        for idx, item in enumerate(dataset):
            sample = rl4lms_pool.Sample(
                id                   = f"{split}_{idx}",
                meta_data            = {"ref_scratchpad": item["scratchpad"],},
                references           = [item["answer"]],
                prompt_or_input_text = item["question"],
            )
            samples.append(sample)
        pool_instance = cls(samples)
        
        return pool_instance


rl4lms_registry.DataPoolRegistry.add(
    "zero_shot_asdiv_text_gen_pool",
    ZeroShotASDivTextGenPool,
)


if __name__ == "__main__":
    pool = ZeroShotASDivTextGenPool.prepare("train")
    rich.print(pool[3])
