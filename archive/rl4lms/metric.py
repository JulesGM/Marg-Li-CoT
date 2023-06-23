import logging
import re
from typing import *

import datasets
import general_utils as utils
import numpy as np
import pandas as pd
import rich
import torch
import tqdm
import transformers
from beartype import beartype

import rl4lms.envs.text_generation.metric as rl4lms_metric
import rl4lms.envs.text_generation.registry as rl4lms_registry

TO_INT_PAT = re.compile(r"((?:- ?)?\d+)(\.0*)?")
LOGGER = logging.getLogger(__name__)


@beartype
def convert_to_int(num_str: Optional[str]) -> Optional[int]:
    if num_str is None:
        return None
        
    num_str = num_str.replace(",", "")
    output = TO_INT_PAT.fullmatch(num_str)
    if output is None:
        return None

    return int(output.group(1).replace(" ", ""))


class ScratchpadAnswerAccuracy(rl4lms_metric.BaseMetric):
    def __init__(
        self, *, 
        make_comparable_fn: Callable, 
        extract_answer_fn: Callable[[str], str],
    ):
        super().__init__()
        self._make_comparable = make_comparable_fn
        self._extract_answer = extract_answer_fn

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: transformers.PreTrainedModel = None,
        split_name: str = None,
    ):
        assert len(generated_texts) == len(reference_texts), (
            len(generated_texts),
            len(reference_texts),
        )

        parsed = []
        em_value = []
        qty_skipped_bc_ref = 0

        for raw_gen, raw_ref in zip(generated_texts, reference_texts):
            assert isinstance(raw_ref, list), type(raw_ref)
            assert len(raw_ref) == 1, len(raw_ref)
            assert isinstance(raw_ref[0], str), type(raw_ref[0])

            extracted_ref = self._extract_answer(raw_ref[0])
            ref = self._make_comparable(extracted_ref)

            if ref is None:
                raise ValueError(f"[{split_name}] - Reference is None: {raw_ref = } {extracted_ref = }")

            assert ref is not None, raw_ref
            assert raw_gen is not None

            if raw_gen is not None:
                extracted_gen = self._extract_answer(raw_gen)
                if extracted_gen is not None:
                    gen = self._make_comparable(extracted_gen)
                else:
                    gen = None

                parsed.append((gen, ref))
                if gen is not None:
                    em_value.append(1.0 if gen == ref else 0.0)
                else:
                    LOGGER.debug(
                        f"[bold yellow]gen is None:[/] "
                        f"{raw_gen = } {extracted_gen = }"
                    )
                    em_value.append(0.0)
            else:
                parsed.append((None, ref))
                em_value.append(0.0)

        output = dict(em_accuracy=(em_value, np.mean(em_value)))
        LOGGER.info(
            f"[bold green]EM Result: [bold white]"
            f"{np.mean(em_value):0.1%}\n"
            f"[bold red]Fraction skipped because of ref: [white]"
            f"{qty_skipped_bc_ref / len(generated_texts):0.0%}"
        )
        
        assert len(em_value) == len(generated_texts) == len(reference_texts), (
            f"\n{len(generated_texts)  = }\n"
            f"{len(reference_texts)  = }\n"
            f"{len(em_value)         = }"
        )
        
        return output


rl4lms_registry.MetricRegistry.add(
    "scratchpad_answer_accuracy", ScratchpadAnswerAccuracy,
)


def test():
    utils.check_equal(convert_to_int("1"), 1)
    utils.check_equal(convert_to_int("- 1"), -1)
    utils.check_equal(convert_to_int("1."), 1)
    utils.check_equal(convert_to_int("-1."), -1)
    utils.check_equal(convert_to_int("- 1."), -1)
    utils.check_equal(convert_to_int("1.0"), 1)
    utils.check_equal(convert_to_int("-1.0"), -1)
    utils.check_equal(convert_to_int("- 1.0"), -1)
    utils.check_equal(convert_to_int("1.00"), 1)
    utils.check_equal(convert_to_int("-1.00"), -1)
    utils.check_equal(convert_to_int("- 1.00"), -1)

    utils.check_equal(convert_to_int("1.4123"), None)
    utils.check_equal(convert_to_int("-1.1"), None)
    utils.check_equal(convert_to_int("- 1.000234"), None)


if __name__ == "__main__":
    test()
