"""Base classes for metrics and rewards."""

import abc
import dataclasses
import enum
import typing
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from beartype import beartype



@beartype
class BatchedUnrollReturn:
    def __init__(self, *, response_tensors, raw_response_tensors, any_tokenizer):
        self._response_tensors = response_tensors
        self._raw_response_tensors = raw_response_tensors
        self._response_text = any_tokenizer.batch_decode(
            self.response_tensors,
        )
        if (self._raw_response_tensors is None or 
            self._raw_response_tensors is response_tensors):
            self._raw_response_text = self._response_text
            self._raw_response_tensors = self._response_tensors
        else:
            self._raw_response_text = any_tokenizer.batch_decode(
                self._raw_response_tensors,
            )


    @property
    def response_tensors(self):
        return self._response_tensors

    @property
    def response_text(self):
        return self._response_text

    @property
    def raw_response_tensors(self):
        return self._raw_response_tensors
    
    @property
    def raw_response_text(self):
        return self._raw_response_text
    
    @raw_response_tensors.setter
    def raw_response_tensors(self, value):
        raise RuntimeError("Cannot set raw_response_tensors")
    
    @raw_response_text.setter
    def raw_response_text(self, value):
        raise RuntimeError("Cannot set raw_response_text")

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
    
    @beartype
    @dataclass
    class IndivualReturn:
        response_tensor: torch.Tensor
        response_text: str
        raw_response_tensor: torch.Tensor
        raw_response_text: str

    def __iter__(self):
        response_text = self.response_text

        for i in range(len(self)):
            yield self.IndivualReturn(
                response_tensor=self.response_tensors[i],
                response_text=response_text[i],
                raw_response_tensor=self.raw_response_tensors[i],
                raw_response_text=self.raw_response_text[i],
            )


@dataclasses.dataclass
class DataListContainer:
    tok_ref_query:        list = dataclasses.field(default_factory=list)
    tok_ref_answer:       list = dataclasses.field(default_factory=list)
    tok_ref_scratchpad:   list = dataclasses.field(default_factory=list)
    detok_ref_query:      list = dataclasses.field(default_factory=list)
    detok_ref_answer:     list = dataclasses.field(default_factory=list)
    detok_ref_scratchpad: list = dataclasses.field(default_factory=list)
    obj_ref_equations:    list = dataclasses.field(default_factory=list)

    @classmethod
    def from_list_of_items(cls, list_items):
        list_container = DataListContainer()
        for item in list_items:
            assert isinstance(item, DataItemContainer)
            for k, v in vars(item).items():
                getattr(list_container, k).append(v)
        return list_container

    collate = from_list_of_items


@dataclasses.dataclass
class DataItemContainer:
    # detok are str, tok are torch.Tensor
    tok_ref_query:        torch.Tensor
    tok_ref_answer:       Optional[torch.Tensor]
    tok_ref_scratchpad:   Optional[torch.Tensor]
    detok_ref_query:      str
    detok_ref_answer:     Optional[str]
    detok_ref_scratchpad: Optional[str]
    obj_ref_equations:    Optional[list]




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
        queries: list[str],
        responses: list[str],
        ref_answers: list[str],
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
