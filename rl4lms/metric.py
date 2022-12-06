import re
from typing import *

from beartype import beartype
import datasets
import tqdm
import rich
import transformers
import torch
import numpy as np
import pandas as pd

import rl4lms.envs.text_generation.metric as rl4lms_metric
import rl4lms.envs.text_generation.registry as rl4lms_registry


TO_INT_PAT = re.compile(r"((?:- ?)?\d+)(\.0+)?)")


@beartype
def convert_to_int(num_str: str) -> Optional[int]:
    output = TO_INT_PAT.fullmatch(num_str)
    if output is None:
        return None

    return int(output.group(1))



class ScratchpadAnswerAccuracy(rl4lms_metric.BaseMetric):
    def __init__(
        self, *, make_comparable_fn, extract_answer_fn: Callable[[str], str]
    ):
        super().__init__()
        self._make_comparable = make_comparable_fn
        self._extract_answer = extract_answer_fn

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]]= None,
        model: transformers.PreTrainedModel = None,
        split_name: str= None,
    ):

        gen_answers = [
            self._extract_answer(gen) 
            for gen in generated_texts
        ]
        
        em_value = []
        for raw_gen, raw_ref in zip(gen_answers, reference_texts):
            assert isinstance(ref, list), type(ref)
            assert len(ref) == 1, len(ref)
            assert isinstance(ref[0], str), type(ref[0])
            ref = self._make_comparable(raw_ref[1])

            if raw_gen is not None and ref is not None:
                gen = self._make_comparable(raw_gen)
                em_value.append(1. if gen == ref else 0.)
            else:
                em_value.append(0.)

        return dict(
            em_accuracy=(em_value, np.mean(em_value),),
        )


rl4lms_registry.MetricRegistry.add(
    "scratchpad_answer_accuracy",
    ScratchpadAnswerAccuracy,
)