import logging
import re
from typing import *
import typing

import datasets
import general_utils as utils
import math
import numpy as np
import pandas as pd
import rich
import torch
import tqdm
import transformers
from beartype import beartype

import metric
import lib_data

LOGGER = logging.getLogger(__name__)



class ScratchpadAnswerAccuracy:
    def __init__(
        self, *, 
        extra_info_engine,
    ):
        super().__init__()
        self._extra_info_fn = extra_info_engine
        self._make_comparable   = self._make_comparable
        self._num_conv_instance = lib_data.ConvToNum()
        self._extract_answer    = self._num_conv_instance.extract_answer

    def _make_comparable(self, match: re.Match, original_text: typing.Optional[str] = None) -> Optional[float]:
        if match is None:
            return None
        
        assert isinstance(match, re.Match), type(match).mro()
        try:
            converted = float(match.group(0))
        except ValueError:
            try:
                converted = float(match.group(0).replace(",", ""))
            except ValueError:
                LOGGER.info(
                    f"[red bold]ValueError: [white]"
                    f"`{match.group(0).replace(',', '') = }` `{original_text = }`"
                )
                return None
        
        return converted

    def __call__(
        self,
        prompts: List[str],
        samples: List[str],
        outputs: List[str],
    ):
        assert prompts
        
        extra_info = self._extra_info_fn(prompts)
        generated_texts = outputs
        reference_texts = [x["answer"] for x in extra_info]
        assert len(reference_texts) == len(generated_texts), (
            len(reference_texts),
            len(generated_texts),
        )

        assert len(generated_texts) == len(reference_texts), (
            len(generated_texts),
            len(reference_texts),
        )

        parsed = [] # Only used for debugging. Convert to a dataframe as needed.
        em_value = []

        for raw_gen, raw_ref in zip(generated_texts, reference_texts):
            if isinstance(raw_ref, str):
                raw_ref = [raw_ref]

            assert isinstance(raw_ref, list), type(raw_ref)
            assert len(raw_ref) == 1, len(raw_ref)
            assert isinstance(raw_ref[0], str), type(raw_ref[0])

            extracted_ref = self._extract_answer(raw_ref[0])
            ref = self._make_comparable(extracted_ref, raw_ref[0])

            if ref is None:
                rich.print(
                    f"[bold red on white]REF IS NONE: "
                    f"\"{raw_ref = }\" \"{extracted_ref = }\""
                )

            assert ref is not None, raw_ref
            assert raw_gen is not None

            if raw_gen is not None:
                extracted_gen = self._extract_answer(raw_gen)
                if extracted_gen is not None:
                    gen = self._make_comparable(extracted_gen, raw_gen)
                else:
                    gen = None
                    
                parsed.append(dict(
                    gen=gen, ref=ref, gen_text=raw_gen, ref_text=raw_ref[0]
                ))
                if gen is not None:
                    em_value.append(1.0 if math.isclose(gen, ref) else 0.0)
                else:
                    LOGGER.debug(
                        f"[bold yellow]gen is None:[/] "
                        f"{raw_gen = } {extracted_gen = }"
                    )
                    em_value.append(0.0)
            else:
                assert False, "This should never happen."
                parsed.append(dict(
                    gen=None, ref=ref, gen_text=raw_gen, ref_text=raw_ref[0]
                ))
                em_value.append(0.0)

        num_nones_parsed = sum(x["gen"] is None for x in parsed)
        output = dict(em_accuracy=em_value)
        assert parsed

        LOGGER.info(
            f"[bold green]EM Result: [bold white]"
            f"{np.mean(em_value):0.1%}\n"
            f"[bold red on white]Fraction of no answer found: "
            f"{num_nones_parsed / len(generated_texts):0.1%}\n"
        )
        

        assert len(em_value) == len(generated_texts) == len(reference_texts), (
            f"\n{len(generated_texts)  = }\n"
            f"{len(reference_texts)  = }\n"
            f"{len(em_value)         = }"
        )
        
        return output


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
