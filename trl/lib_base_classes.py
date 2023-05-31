import enum
import typing
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data
from beartype import beartype


class DictDataset(torch.utils.data.Dataset):
    # Object Pandas without the fluff

    def __init__(self, keys):
        self._dataset = {k: [] for k in keys}

    def __getitem__(self, key: typing.Union[str, int]) -> list[typing.Any]:
        if isinstance(key, int):
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
        return lengths

    def append(self, dict_) -> None:
        assert dict_.keys() == self._dataset.keys(), (
            dict_.keys(), self._dataset.keys())
        
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


FloatSequence = typing.Union[list[float], torch.Tensor, np.ndarray]

# @beartype
@dataclass
class MetricOutput:
    name: str
    values: FloatSequence
    logging_columns: DictDataset
    moving_averages: typing.Optional[dict[str, float]] = None


# @beartype
@dataclass
class RewardOutput:
    name: str
    values: list[torch.Tensor]
    logging_columns: DictDataset
    moving_averages: typing.Optional[dict[str, float]] = None


class Reward:
    def __call__(
        self,
        *,
        queries:     list[str],
        responses:   list[str],
        ref_answers: list[str],
    ) -> RewardOutput:
        
        raise NotImplementedError()

class Metric:
    def __call__(
        self,
        *,
        queries:     list[str],
        responses:   list[str],
        ref_answers: list[str],
    ) -> MetricOutput:
        
        raise NotImplementedError()