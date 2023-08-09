import logging
import math
import os
import re
import typing
from typing import Any, Optional, Union

import general_utils as utils
import more_itertools
import multiset
import numpy as np

import lib_base_classes
import libs_extraction
import lib_utils

LOGGER = logging.getLogger(__name__)
RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "0"))


class ScratchpadAnswerAccuracy(lib_base_classes.Metric):
    """Exact match answer accuracy.

    Takes
    """

    def __init__(self, extractor):
        self._extractor = extractor

    def _make_comparable(
        self, match: re.Match, original_text: typing.Optional[str] = None
    ) -> Optional[float]:
        """Takes a regex match of a number and tries to convert it to a float."""

        if match is None:
            return None

        assert isinstance(match, re.Match), type(match).mro()

        try:
            # We try directly
            converted = float(match.group(0))

        except ValueError:
            try:
                # We try to remove eventual commas.
                # We only do this if the first attempt failed,
                # because the comma could be a decimal separator.
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
        generated_texts: list[lib_base_classes.BatchedUnrollReturn],
        reference_answer_texts: list[str],
    ) -> tuple[list[float], lib_utils.DictDataset]:
        """For each answer, extract the output number, then compare."""

        parsed = lib_utils.DictDataset(["ref", "gen", "ref_text"])
        em_values = []

        for ith_sample, (raw_gen, raw_ref) in enumerate(
            more_itertools.zip_equal(generated_texts, reference_answer_texts)
        ):
            # -----------------------------------------------------------------
            # Prepare the ref
            # -----------------------------------------------------------------
            if isinstance(raw_ref, str):
                raw_ref = [raw_ref]
            assert isinstance(raw_ref, list), type(raw_ref)
            assert len(raw_ref) == 1, len(raw_ref)
            assert isinstance(raw_ref[0], str), type(raw_ref[0])
            extracted_ref = self._extractor(raw_ref[0])

            # -----------------------------------------------------------------
            # Prepare Gen
            # -----------------------------------------------------------------
            assert raw_gen is not None, raw_gen
            extracted_gen = self._extractor(raw_gen)


            parsed.append(
                dict(
                    gen=extracted_gen,
                    ref=extracted_ref,
                    ref_text=raw_ref[0],
                )
            )
            
            # -----------------------------------------------------------------
            # Compare
            # -----------------------------------------------------------------
            if extracted_gen is not None:
                em_values.append(float(self._extractor.compare(
                    extracted_gen, extracted_ref)))
            else:
                LOGGER.debug(
                    f"[bold yellow]gen is None:[/] " +
                    f"{raw_gen = } {extracted_gen = }")
                em_values.append(0.0)
        return em_values, parsed

    def __call__(
        self,
        *,
        batch: lib_base_classes.DataListContainer,
        responses: list[lib_base_classes.BatchedUnrollReturn],
    ) -> lib_base_classes.MetricOutput:
        
        #######################################################################
        # Make checks
        #######################################################################
        generated_texts = responses
        reference_answer_texts = batch.detok_ref_answer

        assert len(reference_answer_texts) == len(generated_texts), (
            len(reference_answer_texts),
            len(generated_texts),
        )
        assert len(generated_texts) == len(reference_answer_texts), (
            len(generated_texts),
            len(reference_answer_texts),
        )

        #######################################################################
        # Compare each sample one by one
        #######################################################################
        em_values, parsed = self._compute(
            generated_texts=generated_texts,
            reference_answer_texts=reference_answer_texts,
        )

        #######################################################################
        # Compute stats, do checks and log
        #######################################################################
        num_nones_parsed = sum(
            x["gen"] is None for x in parsed) # type: ignore
        assert parsed

        LOGGER.debug(
            f"[bold green]EM Result:[bold white] {np.mean(em_values):0.2%}\n"
            f"[bold red on white]Fraction of no answer found: "
            f"{num_nones_parsed / len(generated_texts):0.1%}\n"
        )

        assert len(em_values) == len(generated_texts) == len(reference_answer_texts), (
            f"\n"
            + f"{len(generated_texts)   = }\n"
            + f"{len(reference_answer_texts)   = }\n"
            + f"{len(em_values)          = }"
        )

        return lib_base_classes.MetricOutput(
            logging_columns=parsed,
            moving_averages=None,
            values=em_values,
            name="exact_match",
            extracted_gen=[x["gen"] for x in parsed],
            extracted_ref=[x["ref"] for x in parsed],
        )


class ScratchpadNumericalSubStepAccuracy(lib_base_classes.Metric):
    def __init__(self):
        self._extractor = libs_extraction.lib_numerical.ConvToNum()

    def __call__(
        self,
        *,
        responses: list[lib_base_classes.BatchedUnrollReturn],
        batch: lib_base_classes.DataListContainer,
    ):
        ref_scratchpads = batch.detok_ref_scratchpad
        ref_substeps = batch.obj_ref_equations

        outputs = lib_utils.DictDataset(
            ["gen_ms", "ref_ms", "intermediate_results", "gen_numbers", "metric"]
        )

        #######################################################################
        # We iterate per sample
        #######################################################################
        for response, ref_substep, ref_scratchpad in more_itertools.zip_equal(
            responses,
            ref_substeps,
            ref_scratchpads
        ):
            # 1. Extract numbers from responses.
            intermediate_results = []
            for substep_dict in ref_substep[:-1]: # type: ignore
                intermediate_results.append(substep_dict["answer"])  # type: ignore
            
            ref_ms = multiset.Multiset(intermediate_results)

            # 2. Extract numbers from generated answers.
            extracted_gen_numbers = self._extractor.extract_numbers(
                response.response_text)
            extracted_gen_numbers_not_last = []
            if extracted_gen_numbers:
                extracted_gen_numbers_not_last = extracted_gen_numbers[:-1]  # type: ignore
            gen_ms = multiset.Multiset(extracted_gen_numbers_not_last)

            # 3. Compare
            outputs.append(
                dict(
                    gen_ms=gen_ms, 
                    ref_ms=ref_ms, 
                    intermediate_results=intermediate_results, 
                    gen_numbers=extracted_gen_numbers_not_last,
                    metric=len(gen_ms & ref_ms) / len(ref_ms) if ref_ms else None,  # type: ignore
                ))
            
        return lib_base_classes.MetricOutput(
            logging_columns=outputs,
            moving_averages=None,
            values=outputs["metric"], # type: ignore
            name="substeps",
        )
            

if __name__ == "__main__":
    import transformers
    
    prediction_tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-12b-deduped")  # type: ignore
    prediction_tokenizer.add_special_tokens(dict(pad_token="<|pad|>"))
    prediction_tokenizer.padding_side = "left"

    metric_em = ScratchpadAnswerAccuracy() # type: ignore
    metric_substep = ScratchpadNumericalSubStepAccuracy()
    ref_queries = ["give me a zero"]
    ref_answers = ["0"]
    ref_scratchpads = ["Blah blah blah 0"]

    response_tensors = prediction_tokenizer(
        "give me a zero: 1 2 3 0",
        return_tensors="pt"
    ).to(LOCAL_RANK).input_ids # type: ignore

    batch = lib_base_classes.DataListContainer(
        tok_ref_query=prediction_tokenizer(ref_queries), # type: ignore
        tok_ref_answer=prediction_tokenizer(ref_answers), # type: ignore
        tok_ref_scratchpad=prediction_tokenizer(ref_scratchpads), # type: ignore
        detok_ref_query=ref_queries,
        detok_ref_answer=ref_answers,
        detok_ref_scratchpad=ref_scratchpads,
        obj_ref_equations=[[(None, 1), (None, 2), (None, 3)]],
    )
    
    responses = [lib_base_classes.BatchedUnrollReturn(
        response_tensors=response_tensors, 
        any_tokenizer=prediction_tokenizer,
    )]
    
    print("Calling metric")
    metric_em_output = metric_em(batch=batch, responses=responses)
    metric_ss_output = metric_substep(batch=batch, responses=responses)
    print(metric_ss_output)