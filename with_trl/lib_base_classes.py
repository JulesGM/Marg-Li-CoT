"""Base classes for metrics and rewards."""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from beartype import beartype


@dataclass
class IndivualReturn:
    response_tensor: torch.Tensor
    response_text: str

        
# @beartype
class BatchedUnrollReturn:
    def __init__(
            self, *, 
            response_tensors, 
            any_tokenizer,
        ):

        """
        The exctractors expect there to be no padding tokens in the response.

        Raw means, the response has not been post-processes by the dataset's few-shot post-processor.
        The goal of this is to remove the possible subsequent generations. This is probably not used
        anymore.
        
        """
        print("BatchedUnrollReturn.__init__")
        self._response_tensors = response_tensors
        # self._raw_response_tensors = raw_response_tensors

        self._response_text = any_tokenizer.batch_decode(
            self.response_tensors, 
            skip_special_tokens=True,
        )

    @property
    def response_tensors(self):
        return self._response_tensors

    @property
    def response_text(self):
        return self._response_text

    @response_text.setter
    def response_text(self, value):
        raise RuntimeError("Cannot set response_text")

    @response_tensors.setter
    def response_tensors(self, value):
        raise RuntimeError("Cannot set response_tensors")

    def __len__(self):
        assert len(self.response_tensors) == len(self.response_text), (
            f"{len(self.response_tensors) = } " f"{len(self.response_text)    = } "
        )
        return len(self.response_tensors)


    def __iter__(self):
        for i in range(len(self)):
            yield IndivualReturn(
                response_tensor = self.response_tensors[i],
                response_text   = self.response_text[i],
            )    
        

@dataclasses.dataclass
class DataListContainer:
    # tok_ref_query:        list = dataclasses.field(default_factory=list)
    # tok_ref_answer:       list = dataclasses.field(default_factory=list)
    # tok_ref_scratchpad:   list = dataclasses.field(default_factory=list)

    detok_ref_query:      list = dataclasses.field(default_factory=list)
    detok_ref_answer:     list = dataclasses.field(default_factory=list)
    detok_ref_scratchpad: list = dataclasses.field(default_factory=list)
    difficulty_level:     list = dataclasses.field(default_factory=list)
    extra_information:    list = dataclasses.field(default_factory=list)

    def __len__(self):
        lengths = {k: len(getattr(self, k)) for k in vars(self).keys()}
        iterator = iter(lengths.values())
        one_len = next(iterator)

        # `all` defaults to True if the iterator is empty, 
        # which works in this case.
        assert all(v == one_len for v in iterator), lengths
        
        return len(self.detok_ref_query)

    @classmethod
    def from_list_of_items(cls, list_items):
        list_container = DataListContainer()
        for item, _ in list_items:
            if not isinstance(item, dict):
                raise RuntimeError(type(item))
            assert isinstance(item, dict), type(item).mro()
            for k, v in item.items():
                list_container[k]  = v
        return list_container

    collate = from_list_of_items
    
    def __getitem__(self, idx_or_slice: int | slice):
        return DataItemContainer(
            **{k: getattr(self, k)[idx_or_slice] for k in vars(self)})
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def shuffle(self):
        indices = np.random.permutation(len(self))
        for k, v in vars(self).items():
            self.to_dict()[k] = [v[i] for i in indices]

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()
    
    def to_dict(self):
        return vars(self)
    
    def append(self, item_container=None, **kwargs):

        assert (item_container is None) ^ isinstance(item_container, DataItemContainer), (
            item_container is None, 
            isinstance(item_container, DataItemContainer),
        )
        if item_container is not None:
            assert isinstance(item_container, DataItemContainer)
            kwargs = vars(item_container)

        assert len(kwargs) == len(vars(self)), (
            f"{len(kwargs) = }, {len(vars(self)) = } "
        )

        for k, v in kwargs.items():
            getattr(self, k).append(v)
    

@dataclasses.dataclass
class DataItemContainer:
    # detok are str, tok are torch.Tensor

    # tok_ref_query:         torch.Tensor
    # tok_ref_answer:        Optional[torch.Tensor]
    # tok_ref_scratchpad:    Optional[torch.Tensor]
    
    detok_ref_query:       str
    detok_ref_answer:      Optional[str]
    detok_ref_scratchpad:  Optional[str]

    difficulty_level:      Optional[int]
    extra_information:     Optional[list]

    def items(self):
        return self.to_dict().items()

    def keys(self):
        return self.to_dict().keys()
    
    def values(self):
        return self.to_dict().values()

    def to_dict(self):
        return vars(self)

    def __post_init__(self):
        for k, v in vars(self).items():
            if k.startswith("tok_ref_"):
                if v is not None and not isinstance(v, torch.Tensor):
                    setattr(self, k, torch.tensor(v))



FloatSequence = typing.Union[list[float], torch.Tensor, np.ndarray]


# @beartype
@dataclass
class MetricOutput:
    name: str
    values: FloatSequence
    logging_columns: "lib_utils.DictDataset"
    extracted_gen: str
    extracted_ref: str
    moving_averages: typing.Optional[dict[str, float]] = None


# @beartype
@dataclass
class RewardOutput:
    name: str
    values: list[torch.Tensor]
    logging_columns: "lib_utils.DictDataset"
    extracted_gen: str
    extracted_ref: str
    moving_averages: typing.Optional[dict[str, float]] = None
    

class Reward:
    def __call__(
        self,
        *,
        batch: list[str],
        responses: list[str],
    ) -> RewardOutput:
        raise NotImplementedError()


class Metric:
    def __call__(
        self,
        *,
        responses: list[BatchedUnrollReturn],
        batch: DataListContainer,
    ) -> MetricOutput:
        raise NotImplementedError()
    
if __name__ == "__main__":
    BatchedUnrollReturn(
        response_tensors="response_tensors", 
        any_tokenizer="any_tokenizer",
    )
