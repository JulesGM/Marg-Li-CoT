import collections
import logging
import os
import pathlib
import re
import string
import sys
import typing
from typing import Any, Optional, Union

import datasets
import fire
import more_itertools
import rich
import rich.logging
import torch
import torch.utils.data
import tqdm
import transformers

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR.parent))
import lib_base_classes
import lib_utils
import libs_extraction.lib_multiple_choice
import libs_data.lib_base


RANK = int(os.getenv("RANK", 0))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOGGER = logging.getLogger(__name__)


def _tok_detok(samples, any_tokenizer, prefix, suffix):
    output = {}
    assert "question" in samples, samples.keys()
    question_seen = False

    for k, v in samples.items():
        if k == "question":
            preped_text = [prefix + x.strip() + suffix for x in v]
            question_seen = True
        else:
            preped_text = [x.strip() for x in v]

        tok = any_tokenizer(preped_text)
        detok = any_tokenizer.batch_decode(tok["input_ids"])
        output[k + "_tok"] = tok.input_ids
        output[k + "_detok"] = detok

    assert question_seen, samples.keys() # Redundant.
    return output


class CommonSenseQAMC(libs_data.lib_base.Dataset):
    def __init__(
            self, 
            *, 
            any_tokenizer, 
            split: str, 
            question_prefix: str, 
            question_suffix: str,
        ):
        self.answer_mode = "multiple_choices"
        self._extractor = libs_extraction.lib_multiple_choice.MultipleChoiceRegexExtractor(
            ["A", "B", "C", "D", "E"])

        self._hf_ds = datasets.load_dataset("commonsense_qa", split=split)
        self._ds = self._hf_ds.map(
            self._prep_hf_ds
        ).map(
            lambda batch: _tok_detok(batch, any_tokenizer, question_prefix, question_suffix),
            batched=True,
        )

        self._output_container = lib_base_classes.DataListContainer()
        if question_prefix is None:
            question_prefix = ""
        if question_suffix is None:
            question_suffix = ""
        self._question_prefix = question_prefix
        self._question_suffix = question_suffix

        for sample in tqdm.tqdm(self._ds, desc="Building DataListContainer"):
            self._output_container.tok_ref_query       .append(torch.tensor(sample["question_tok"]))
            self._output_container.tok_ref_answer      .append(torch.tensor(sample["answer_tok"]))
            self._output_container.tok_ref_scratchpad  .append(None)
            self._output_container.detok_ref_query     .append(sample["question_detok"])
            self._output_container.detok_ref_answer    .append(sample["answer_detok"])
            self._output_container.detok_ref_scratchpad.append(None)
            self._output_container.obj_ref_equations   .append(None)

    @classmethod
    def _prep_hf_ds(cls, sample):
        sample["question"] += " " + ", ".join(
            f"{label.strip()}) {text.strip()}"
            for label, text in 
            more_itertools.zip_equal(
                sample["choices"]["label"], 
                sample["choices"]["text"],
            )
        )
        sample["answer"] = sample["answerKey"].strip()

        del sample["id"]
        del sample["question_concept"]
        del sample["choices"]
        del sample["answerKey"]

        return sample

    def __len__(self):
        return len(self._ds) # type: ignore

    def __getitem__(
        self, idx_or_slice: typing.Union[int, slice]
    ) -> lib_base_classes.DataItemContainer:
        
        return lib_base_classes.DataItemContainer(
            **{
                k: v[idx_or_slice] 
                for k, v in vars(self._output_container).items()
            }
        )

    @property
    def question_prefix(self):
        return self._question_prefix

    @property
    def question_suffix(self):
        return self._question_suffix

    def get_extractor(self):
        return self._extractor

def tests():

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[rich.logging.RichHandler(markup=True)],
        format=f"[{RANK}/{WORLD_SIZE}] %(funcName)s:%(lineno)d - %(message)s",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained("ausboss/llama-30b-supercot")
    ds = CommonSenseQAMC(any_tokenizer=tokenizer, split="train")
    
    import ipdb
    ipdb.set_trace()


if __name__ == "__main__":
    fire.Fire(tests)