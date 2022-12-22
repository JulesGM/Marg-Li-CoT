import logging
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


import general_utils as utils

TO_INT_PAT = re.compile(r"((?:- ?)?\d+)(\.0*)?")
LOGGER = logging.getLogger(__name__)

@beartype
def convert_to_int(num_str: str) -> Optional[int]:
    num_str = num_str.replace(",", "")
    output = TO_INT_PAT.fullmatch(num_str)
    if output is None:
        return None

    return int(output.group(1).replace(" ", ""))



class ScratchpadAnswerAccuracy(rl4lms_metric.BaseMetric):
    def __init__(
        self, *, make_comparable_fn, extract_answer_fn: Callable[[str], str]
    ):
        super().__init__()
        self._make_comparable = make_comparable_fn
        self._extract_answer = extract_answer_fn

    def compute(
        self,
        prompt_texts   : List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos     : List[Dict[str, Any]]= None,
        model          : transformers.PreTrainedModel = None,
        split_name     : str= None,
    ):

        gen_answers = [
            self._extract_answer(gen) 
            for gen in generated_texts
        ]
        
        em_value = []
        parsed = []

        for raw_gen, raw_ref in zip(gen_answers, reference_texts):
            assert isinstance(raw_ref, list), type(raw_ref)
            assert len(raw_ref) == 1, len(raw_ref)
            assert isinstance(raw_ref[0], str), type(raw_ref[0])
            ref = self._make_comparable(raw_ref[0])
            assert ref is not None

            if raw_gen is not None :
                gen = self._make_comparable(raw_gen)
                parsed.append((gen, ref))
                if gen is not None:
                    em_value.append(1. if gen == ref else 0.)
                else:
                    em_value.append(0.)
            else:
                parsed.append((None, ref))
                em_value.append(0.)

        output = dict(
            em_accuracy=(em_value, np.mean(em_value),),
        )
        LOGGER.info(f"[bold green]EM Result: [bold white]{np.mean(em_value):0.1%}")
        return output


rl4lms_registry.MetricRegistry.add(
    "scratchpad_answer_accuracy",
    ScratchpadAnswerAccuracy,
)


def test():
    utils.check_equal(convert_to_int("1"     ),  1)
    utils.check_equal(convert_to_int("- 1"   ), -1)
    utils.check_equal(convert_to_int("1."    ),  1)
    utils.check_equal(convert_to_int("-1."   ), -1)
    utils.check_equal(convert_to_int("- 1."  ), -1)
    utils.check_equal(convert_to_int("1.0"   ),  1)
    utils.check_equal(convert_to_int("-1.0"  ), -1)
    utils.check_equal(convert_to_int("- 1.0" ), -1)
    utils.check_equal(convert_to_int("1.00"  ),  1)
    utils.check_equal(convert_to_int("-1.00" ), -1)
    utils.check_equal(convert_to_int("- 1.00"), -1)

    utils.check_equal(convert_to_int("1.4123"    ), None)
    utils.check_equal(convert_to_int("-1.1"      ), None)
    utils.check_equal(convert_to_int("- 1.000234"), None)
    


if __name__ == "__main__":
    test()