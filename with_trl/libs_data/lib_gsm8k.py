"""

August 4th 2024, Jules:

    Prepares answer-only and answer-with-reasoning datasets for GSM8K.

    Was mainly used recently for the SFT experiments.
    
    Doesn't currently profive zero shot support, which we will definitely need to add.

    Also provides the difficulty level as well as the equations.
    The interface of the samples is:
    {
        "ref_qa_question"  : ...,
        "ref_qa_answer"    : ...,
        "ref_qa_scratchpad": ...,
        "difficulty_level" : ..., # len of equations
        "extra_information": {"equations": ...},
    }

    
    # Reflexions on few-shots:
    The samples are composed in the data collator. It would make sense if the eventual 
    few-shots were also composed in the data collator.

    For now, we will create few shots by just adding samples to extra_information, & 
    disabling few-shot for those so we don't create an infinite recursion tree.

"""

import collections
import collections.abc
import logging
import re
import typing
from typing import Any, Optional, Union

import datasets
import more_itertools
import rich
import torch
import torch.utils.data
import transformers

import lib_base_classes
import libs_extraction.lib_numerical
import libs_data.lib_base

LOGGER = logging.getLogger(__name__)


def dict_unzip(list_of_dict):
    keys = list_of_dict[0].keys()

    for list_obj in list_of_dict:
        assert list_obj.keys() == keys, (list_obj.keys(), keys)
    
    return {k: [d[k] for d in list_of_dict] for k in keys}


class GSM8K:
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
        use_few_shots: bool,
        few_show_qty: int,
        ds: Optional[collections.abc.Sequence[str]] = None,
    ):
        self._extractor = libs_extraction.lib_numerical.ConvToNum()
        self._output_container: Optional[dict[str, list[Any]]] = None
        
        self._tok_max_query_length  = tok_max_query_length
        self._tok_max_answer_length = tok_max_answer_length
        self._tok_max_total_length  = tok_max_total_length

        self._any_tokenizer = any_tokenizer
        self._device = device
        
        self._use_few_shots = use_few_shots
        self._few_show_qty = few_show_qty
        self._few_shot_examples = None

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

            text_queries.append(sample)
            text_scratchpads.append(scratchpad)
            text_answers.append(answer)

        ######################################################################
        # Tokenize and Detokenize.
        ######################################################################
        LOGGER.info("> Tokenizing.")

        tokenized_ref_queries = [
            torch.tensor(x, dtype=torch.long)
            for x 
            in self._any_tokenizer(text_queries)["input_ids"] # type: ignore
        ]

        tokenized_ref_scratchpads = [
            torch.tensor(x, dtype=torch.long)
            for x in self._any_tokenizer(
                text_scratchpads)["input_ids"] # type: ignore
        ]

        tokenized_ref_answers = [
            torch.tensor(x, dtype=torch.long)
            for x in self._any_tokenizer(text_answers)["input_ids"] # type: ignore
        ]

        detokenized_ref_queries    = self._any_tokenizer.batch_decode(tokenized_ref_queries    , skip_special_tokens=True)
        detokenized_ref_answers    = self._any_tokenizer.batch_decode(tokenized_ref_answers    , skip_special_tokens=True)
        detokenized_ref_scratchpad = self._any_tokenizer.batch_decode(tokenized_ref_scratchpads, skip_special_tokens=True)
        
        LOGGER.info("< Done Tokenizing.")
        
        ######################################################################
        # Filter out samples on length criterias
        ######################################################################
        self._output_container = dict(
            ref_qa_question   = [],
            ref_qa_answer     = [],
            ref_qa_scratchpad = [],
            difficulty_level  = [],
            extra_information = [],
        )

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
                len(t_q) <= self._tok_max_query_length
            )

            scratchpad_length_ok = (
                not self._tok_max_answer_length or
                len(t_s) <= self._tok_max_answer_length)

            total_length_ok = (
                not self._tok_max_total_length or
                len(t_q) + len(t_s) <= self._tok_max_total_length
            )

            # Apply the filter
            if (
                query_length_ok and 
                scratchpad_length_ok and 
                total_length_ok
            ):
                ###############################################################
                # Extract the equation annotations.
                ###############################################################
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


                ###############################################################
                # Accumulate things
                ###############################################################
                qty_total += 1
                # Only keep the filtered.
                self._output_container["ref_qa_question"  ].append(dt_q)
                self._output_container["ref_qa_answer"    ].append(dt_a)
                self._output_container["ref_qa_scratchpad"].append(dt_s)

                self._output_container["difficulty_level" ].append(len(equations))
                self._output_container["extra_information"].append(dict(equations=equations))

        # Check lengths in self._output_container
        first = len(self._output_container["ref_qa_question"])
        for k, v in self._output_container.items():
            assert len(v) == first, (k, len(v), first)

        rich.print(
            f"[red bold] {qty_failed_equations} / {qty_total} = " +
            f"{qty_failed_equations / qty_total: 0.1%} " +
            f"had one or fewer eqns."
        )

        final_len = len(self._output_container["ref_qa_question"])
        init_len  = len(detokenized_ref_queries)

        LOGGER.info(
            f"[red bold] Kept {final_len / init_len:0.1%} samples, "
            f"{final_len} / {init_len}"
        )

    def __len__(self):
        return len(self._output_container["ref_qa_question"])  # type: ignore

    def _get(self, idx_or_slice, use_few_shots):
        """
        We unzip the dict, then, if use_few_shots is True, we add the few-shot examples.
        The few shot examples could vary by example. They don't this time, but this is
        why we have a reference for each element of the batch.
        """

        examples =  {k: v[idx_or_slice] for k, v in self._output_container.items()}

        if use_few_shots:
            examples["extra_information"] = dict(
                **examples["extra_information"], 
                few_shot_examples=[
                    self._few_shot_examples 
                    for _ in range(len(examples["ref_qa_question"]))
                ]
            )

        return examples
        

    def __getitem__(
        self, idx_or_slice: typing.Union[int, slice]
    ) -> lib_base_classes.DataItemContainer:

        # We use a secondary method, in order to have both a toggelable use_few_shots argument.
        # The "without" version is used to build the few-shot examples in the same way that
        # the "with" examples are built.

        return self._get(idx_or_slice, self._use_few_shots)




if __name__ == "__main__":
    gsm8k = GSM8K(
        tok_max_query_length=64,
        tok_max_answer_length=64,
        tok_max_total_length=128,
        any_tokenizer=transformers.AutoTokenizer.from_pretrained("gpt2"),
        device=torch.device("cpu"),
        use_few_shots=True,
        few_show_qty=1,
    )

    import rich
    rich.print(
        next(iter(gsm8k))
    )