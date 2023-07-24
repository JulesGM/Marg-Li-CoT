
import collections
import collections.abc
import logging
import re
import typing
from typing import Any, Optional, Union

import more_itertools
import rich
import torch
import torch.utils.data
import transformers

import lib_base_classes
import libs_extraction.lib_numerical
import libs_data.lib_base

LOGGER = logging.getLogger(__name__)


class GSM8K(libs_data.lib_base.Dataset):
    _int_patt = re.compile(r"\-?\d+")
    _eqn_patt = re.compile(r"<<[\(\)0-9\+\-/\*=\.]+>>")

    def __init__(
        self,
        *,
        tok_max_query_length: Optional[int] = None,
        tok_max_answer_length: Optional[int] = None,
        tok_max_total_length: Optional[int] = None,
        any_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
        device: torch.device,
        ds: collections.abc.Sequence[str],
        question_prefix: str,
        question_suffix: str,
    ):
        self._extractor = libs_extraction.lib_numerical.ConvToNum()
        self._question_prefix = question_prefix
        self._question_suffix = question_suffix
        self._output_container: Optional[lib_base_classes.DataListContainer] = None
        self._tok_max_query_length = tok_max_query_length
        self._tok_max_answer_length = tok_max_answer_length
        self._tok_max_total_length = tok_max_total_length
        self._any_tokenizer = any_tokenizer
        self._device = device
        self._populate_ds(ds)

    def get_extractor(self):
        return self._extractor

    def _populate_ds(self, ds):
        text_queries = []
        text_scratchpads = []
        text_answers = []

        ######################################################################
        # Parse the original Hugging Face dataset object.
        ######################################################################
        for idx in range(len(ds)):
            sample = ds[idx]["question"].strip()
            scratchpad, answer = ds[idx]["answer"].split("####")

            scratchpad = scratchpad.strip()
            answer = answer.strip().replace(",", "")

            if str(int(answer)) != answer.strip():
                assert False, f"{answer = }"

            text_queries.append(
                self._question_prefix + sample + self._question_suffix
            )
            text_scratchpads.append(scratchpad)
            text_answers.append(answer)

        ######################################################################
        # Tokenize and Detokenize.
        ######################################################################
        LOGGER.info("> Tokenizing.")

        tokenized_ref_queries = [
            torch.tensor(x, dtype=torch.long)
            for x in self._any_tokenizer(text_queries)["input_ids"] # type: ignore
        ]

        tokenized_ref_answers = [
            torch.tensor(x, dtype=torch.long)
            for x in self._any_tokenizer(text_answers)["input_ids"] # type: ignore
        ]

        tokenized_ref_scratchpads = [
            torch.tensor(x, dtype=torch.long)
            for x in self._any_tokenizer(text_scratchpads)["input_ids"] # type: ignore
        ]

        detokenized_ref_queries    = self._any_tokenizer.batch_decode(tokenized_ref_queries)
        detokenized_ref_answers    = self._any_tokenizer.batch_decode(tokenized_ref_answers)
        detokenized_ref_scratchpad = self._any_tokenizer.batch_decode(tokenized_ref_scratchpads)
        LOGGER.info("< Done Tokenizing.")
        
        ######################################################################
        # Filter out samples on length criterias
        ######################################################################
        self._output_container = lib_base_classes.DataListContainer()

        qty_failed_equations = 0
        qty_total = 0
        for t_q, t_a, t_s, dt_q, dt_a, dt_s in more_itertools.zip_equal(
            tokenized_ref_queries,
            tokenized_ref_answers,
            tokenized_ref_scratchpads,
            detokenized_ref_queries,
            detokenized_ref_answers,
            detokenized_ref_scratchpad,
        ):
            
            # Filter conditions
            query_length_ok = (
                not self._tok_max_query_length or
                len(t_q) <= self._tok_max_query_length)

            scratchpad_length_ok = (
                not self._tok_max_answer_length or
                len(t_s) <= self._tok_max_answer_length)

            total_length_ok = (
                not self._tok_max_total_length or
                len(t_q) + len(t_s) <= self._tok_max_total_length)

            # Apply the filter
            if (query_length_ok and scratchpad_length_ok and total_length_ok):
                
                # Extract the insides of the equation annotations.
                equations = []
                splitted = self._eqn_patt.findall(dt_s) # type: ignore
                count_left_side = dt_s.count("<<") # type: ignore
                assert len(splitted) == count_left_side, (
                    len(splitted),
                    count_left_side,
                    dt_s,
                    splitted,
                )

                # Modify each equation.
                for eqn_str in splitted:
                    assert eqn_str[ :2] == "<<", f"`{eqn_str}`"
                    assert eqn_str[-2:] == ">>", f"`{eqn_str}`"
                    eqn_str = eqn_str[2:-2]
                    left, answer = eqn_str.split("=")
                    equations.append(dict(left=left, answer=answer))
                
                if len(equations) <= 1:
                    qty_failed_equations += 1
                qty_total += 1

                # Only keep the filtered.
                self._output_container.obj_ref_equations   .append(equations)
                self._output_container.tok_ref_query       .append(t_q)
                self._output_container.tok_ref_answer      .append(t_a)
                self._output_container.tok_ref_scratchpad  .append(t_s)
                self._output_container.detok_ref_query     .append(dt_q)
                self._output_container.detok_ref_answer    .append(dt_a)
                self._output_container.detok_ref_scratchpad.append(dt_s)

        rich.print(f"[red bold] {qty_failed_equations} / {qty_total} = {qty_failed_equations / qty_total: 0.1%} had one or fewer eqns.")

        final_len = len(self._output_container.tok_ref_query)
        init_len  = len(detokenized_ref_queries)

        LOGGER.info(
            f"[red bold] Kept {final_len / init_len:0.1%} samples, "
            f"{final_len} / {init_len}"
        )

    def __len__(self):
        return len(self._output_container.tok_ref_query)  # type: ignore
        
    def __getitem__(
        self, idx_or_slice: typing.Union[int, slice]
    ) -> lib_base_classes.DataItemContainer:
        
        return lib_base_classes.DataItemContainer(
            **{
                k: v[idx_or_slice] 
                for k, v in vars(self._output_container).items()
            }
        )

