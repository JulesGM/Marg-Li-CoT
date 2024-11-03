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
import numpy as np
import rich
import torch
import torch.utils.data
import transformers

import lib_utils
import lib_base_classes
import libs_extraction.lib_numerical

LOGGER = logging.getLogger(__name__)


def dict_unzip(list_of_dict):
    keys = list_of_dict[0].keys()

    for list_obj in list_of_dict:
        assert list_obj.keys() == keys, (list_obj.keys(), keys)
    
    return {k: [d[k] for d in list_of_dict] for k in keys}


class GSM8K(torch.utils.data.Dataset):
    _int_patt = re.compile(r"\-?\d+")
    _eqn_patt = re.compile(r"<<[\(\)0-9\+\-/\*=\.]+>>")

    def __init__(
        self,
        *,
        any_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
        device: torch.device,
        ds: Optional[collections.abc.Sequence[str]],
        few_show_qty: int,
        
        tok_max_query_length: Optional[int] = None,
        tok_max_answer_length: Optional[int] = None,
        tok_max_total_length: Optional[int] = None,

        use_few_shots: bool,
        
        cv_set: lib_utils.CVSets,
        use_curriculum: bool,
    ):
        self._extractor = libs_extraction.lib_numerical.ConvToNum()
        self._output_container: Optional[dict[str, list[Any]]] = None
        
        self._tok_max_query_length = tok_max_query_length
        self._tok_max_answer_length = tok_max_answer_length
        self._tok_max_total_length = tok_max_total_length

        self._any_tokenizer = any_tokenizer
        self._device = device
        
        self._use_few_shots = use_few_shots
        self._few_show_qty = few_show_qty
        self._few_shot_examples = None
        self._use_curriculum = use_curriculum
        self._cv_set = cv_set

        self._populate_ds(ds)

        if self._use_few_shots:
            self._select_few_shots()

    @property
    def use_few_shots(self):
        return self._use_few_shots

    def get_extractor(self):
        return self._extractor

    def _populate_ds(self, ds):
        text_queries = []
        text_scratchpads = []
        text_answers = []

        ######################################################################
        # Parse the original Hugging Face dataset object.
        ######################################################################
        ds_len = None

        if isinstance(ds, datasets.Dataset):
            ds = ds.to_dict()
        
        for k, v in ds.items():
            if ds_len is None:
                ds_len = len(v)
            else:
                assert len(v), {k: len(v) for k, v in ds.items()}

        for idx in range(ds_len):
            sample = ds["question"][idx].strip()
            scratchpad, answer = ds["answer"][idx].split("####")

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

        detokenized_ref_queries = self._any_tokenizer.batch_decode(
            tokenized_ref_queries, skip_special_tokens=True)
        detokenized_ref_answers = self._any_tokenizer.batch_decode(
            tokenized_ref_answers, skip_special_tokens=True)
        detokenized_ref_scratchpad = self._any_tokenizer.batch_decode(
            tokenized_ref_scratchpads, skip_special_tokens=True)
        
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

        for k, v in self._output_container.items():
            assert isinstance(v, list), (k, type(v))

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
        
        lens = [(k, len(v)) for k, v in self._output_container.items()]
        first = lens[0][1]
        assert all(l == first for _, l in lens), lens

        return first

    def _get(self, idx, use_few_shots):
        """
        Separate function from __get__ to be able to pass `use_few_shots`.

        We unzip the dict, then, if use_few_shots is True, we add the few-shot examples.

        The few shot examples could vary by example. They don't this time, but this is
        why we have a reference for each element of the batch.
        """

        examples =  {k: v[idx] for k, v in self._output_container.items()}

        if use_few_shots:
            assert self._few_shot_examples is not None, (
                f"self._few_shot_examples is None, {self._few_shot_examples = }")
            assert len(self._few_shot_examples), (
                f"self._few_shot_examples is len(0), {self._few_shot_examples = }")

            # Add to the example.
            examples["extra_information"] = dict(
                **examples["extra_information"], 
                few_shot_examples=self._few_shot_examples
            )

        return examples
        
    def __getitem__(
        self, idx: typing.Union[int, slice]
    ) -> lib_base_classes.DataItemContainer:

        # We use a secondary method, in order to have both a toggelable use_few_shots argument.
        # The "without" version is used to build the few-shot examples in the same way that
        # the "with" examples are built.

        return self._get(idx, self._use_few_shots), idx

    def _select_few_shots(self):
        """
        We select a few shots for each example. The collator 
        will actually format the few shots in the sample.
        """
        assert self._output_container is not None, (
            f"{self._output_container = }")

        if self._use_curriculum:
            raise NotImplementedError()
        
        # We have to build the few-shot examples.
        # Collect `self._few_show_qty` examples for each key.
        
        output = collections.defaultdict(list)
        for k, output_container_v in self._output_container.items():
            output_list = output[k]
            for i in range(self._few_show_qty):
                output_list.append(output_container_v[i])
        
        # Freeze the default dict.
        self._few_shot_examples = dict(output.items())

    def post_process_gen_fewshots(
        self, 
        input_ids,
        raw_gen_outputs: np.ndarray,
        forward_tokenizer: transformers.PreTrainedTokenizerBase,
    ):

        input_text = forward_tokenizer.batch_decode(input_ids)
        for line in input_text:
            assert "Question:" in line, line

        decoded = forward_tokenizer.batch_decode(
            raw_gen_outputs, 
            skip_special_tokens=True,
        )
        

        # If we generate new questions, remove them.
        outputs = []
        for line in decoded:
            final = line.find("Question:")
            if final == -1:
                final = len(line)

            outputs.append(line[:final].strip())

        # Re-tokenize the outputs.
        # Should be padded right.
        assert forward_tokenizer.padding_side == "right", (
            f"{forward_tokenizer.padding_side = }")
        
        tokenized = forward_tokenizer(
            outputs,
            return_tensors="pt",
            padding=True,
        )["input_ids"]
        

        return tokenized


if __name__ == "__main__":
    import sys
    import pathlib
    SCRIPT_DIR = pathlib.Path(__file__).parent
    sys.path.append(str(SCRIPT_DIR))
    sys.path.append(str(SCRIPT_DIR.parent))
    import lib_data
    import accelerate
    
    gsm8k = GSM8K(
        any_tokenizer         = transformers.AutoTokenizer.from_pretrained("gpt2"),
        cv_set                = lib_utils.CVSets.TRAIN,
        device                = torch.device("cpu"),
        ds                    = datasets.load_dataset(
            split="train",
            path="gsm8k",
            name="main",
        ),
        few_show_qty          = 10,
        tok_max_query_length  = 64,
        tok_max_answer_length = 64,
        tok_max_total_length  = 128,
        use_few_shots         = True,
        use_curriculum        = False,
    )

    dataloader = torch.utils.data.DataLoader(
        gsm8k,
        collate_fn=lib_data.data_item_collator,
        batch_size=1,
    )

    normal_iterator = iter(dataloader)
    print(f"{type(normal_iterator) = }")

    batch = next(normal_iterator)
    print(f"{type(batch) = }")