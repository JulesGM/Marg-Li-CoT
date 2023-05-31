import collections
import logging
import math
import os
import re
import typing
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

import lib_base_classes
import lib_data
import lib_trl_utils

LOGGER = logging.getLogger(__name__)
RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "0"))


class ScratchpadAnswerAccuracy(lib_base_classes.Metric):
    def __init__(self):
        self._num_conv_instance = lib_data.ConvToNum()
        self._extract_answer = self._num_conv_instance.extract_answer

    def _make_comparable(
        self, 
        match: re.Match, 
        original_text: typing.Optional[str] = None
    ) -> Optional[float]:
        
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
                    f"`{match.group(0).replace(',', '') = }` "
                    f"`{original_text = }` "
                )
                return None
        
        return converted

    def _compute(
            self, 
            generated_texts: list[lib_trl_utils.BatchedUnrollReturn], 
            reference_texts: list[str],
        ):
        parsed = lib_base_classes.DictDataset(["ref", "gen", "ref_text"]) 
        em_values = []

        for ith_sample, (raw_gen, raw_ref) in enumerate(
            zip(generated_texts, reference_texts)
        ):
            # -----------------------------------------------------------------
            # Prepare the ref
            # -----------------------------------------------------------------
            if isinstance(raw_ref, str):
                raw_ref = [raw_ref]
            assert isinstance(raw_ref, list)  , type(raw_ref)
            assert len(raw_ref) == 1          ,  len(raw_ref)
            assert isinstance(raw_ref[0], str), type(raw_ref[0])
            extracted_ref = self._extract_answer (raw_ref[0])
            ref           = self._make_comparable(extracted_ref, raw_ref[0])

            
            if ref is None:
                rich.print(
                    f"[bold red on white]REF IS NONE: "
                    f"\"{raw_ref = }\" \"{extracted_ref = }\"")

            assert ref is not None, raw_ref

            # -----------------------------------------------------------------
            # Prepare Gen
            # -----------------------------------------------------------------
            assert raw_gen is not None, raw_gen
            extracted_gen = self._extract_answer(raw_gen)
            
            if extracted_gen is not None:
                gen = self._make_comparable(
                    extracted_gen, raw_gen)
            else:
                gen = None
            
            parsed.append(dict(
                gen=gen, 
                ref=ref, 
                # gen_text=raw_gen, 
                ref_text=raw_ref[0],
            ))

            if gen is not None:
                em_values.append(1.0 if math.isclose(gen, ref) else 0.0)
            else:
                LOGGER.debug(
                    f"[bold yellow]gen is None:[/] "
                    f"{raw_gen = } {extracted_gen = }")
                em_values.append(0.0)

        return em_values, parsed

    def __call__(
        self,
        *,
        queries: list[str],
        responses: list[lib_trl_utils.BatchedUnrollReturn],
        ref_answers: list[str],
    ) -> lib_base_classes.MetricOutput:
        
        #######################################################################
        # Make checks
        #######################################################################
        assert queries
        assert responses

        generated_texts = responses
        reference_texts = ref_answers

        assert len(reference_texts) == len(generated_texts), (
            len(reference_texts),
            len(generated_texts),)
        assert len(generated_texts) == len(reference_texts), (
            len(generated_texts),
            len(reference_texts),)

        #######################################################################
        # Compare each sample one by one
        #######################################################################
        em_values, parsed = self._compute(
            generated_texts=generated_texts,
            reference_texts=reference_texts,
        )

        #######################################################################
        # Compute stats, do checks and log
        #######################################################################
        num_nones_parsed = sum(x["gen"] is None for x in parsed)
        assert parsed

        LOGGER.info(
            f"[bold green]EM Result: [bold white]"
            f"{np.mean(em_values):0.2%}\n"
            f"[bold red on white]Fraction of no answer found: "
            f"{num_nones_parsed / len(generated_texts):0.1%}\n")

        assert (
            len(em_values) == 
            len(generated_texts) == 
            len(reference_texts)
        ), (
            f"\n" + 
            f"{len(generated_texts)   = }\n" + 
            f"{len(reference_texts)   = }\n" + 
            f"{len(em_values)          = }")

        return lib_base_classes.MetricOutput(
            values=em_values,
            logging_columns=parsed,
            name="exact_match",
            moving_averages=None,
        )