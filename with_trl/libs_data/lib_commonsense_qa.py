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
import libs_data.data_commonsense_qa_few_shot

RANK = int(os.getenv("RANK", 0))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOGGER = logging.getLogger(__name__)


def _tok_detok(*, batch, any_tokenizer):
    output = {}
    assert "question" in batch, batch.keys()

    for k, v in batch.items():
        tok = any_tokenizer(v)
        detok = any_tokenizer.batch_decode(tok.input_ids, skip_special_tokens=False)
        output[k + "_tok"] = tok.input_ids
        output[k + "_detok"] = detok

    return output


class CommonSenseQAMC(libs_data.lib_base.FewShotMixin, libs_data.lib_base.Dataset):
    def __init__(
            self, 
            *, 
            any_tokenizer, 
            split: str,
            question_prefix: str, 
            question_suffix: Optional[str],
            use_few_shots: bool,
        ):
        assert question_prefix is None, f"question_prefix: `{question_prefix}`"
        assert question_suffix is None, f"question_suffix: `{question_suffix}`"

        self._use_few_shots = use_few_shots

        if use_few_shots:
            assert not question_prefix
            assert not question_suffix
            few_shot_data = libs_data.data_commonsense_qa_few_shot.FEW_SHOT
            question_prefix = few_shot_data.strip() + "\n\n"
            question_suffix = " "

        if question_prefix is None:
            question_prefix = ""
        if question_suffix is None:
            question_suffix = ""
        self._question_prefix = question_prefix
        self._question_suffix = question_suffix

        self._extractor = libs_extraction.lib_multiple_choice.MultipleChoiceRfindExtractor(
            ["(A)", "(B)", "(C)", "(D)", "(E)"])

        self._hf_ds = datasets.load_dataset("commonsense_qa", split=split)
        self._ds = self._hf_ds.map(
            lambda sample:
                self._prep_hf_ds(
                    sample=sample,
                    question_prefix=question_prefix, 
                    question_suffix=question_suffix,
                )
        ).map(
            lambda batch: _tok_detok(
                batch=batch, 
                any_tokenizer=any_tokenizer, 
            ),
            batched=True,
        )

        self._output_container = lib_base_classes.DataListContainer()
        

        for sample in tqdm.tqdm(self._ds, desc="Building DataListContainer"):
            self._output_container.tok_ref_query       .append(torch.tensor(sample["question_tok"]))
            self._output_container.tok_ref_answer      .append(torch.tensor(sample["answer_tok"]))
            self._output_container.tok_ref_scratchpad  .append(None)
            self._output_container.detok_ref_query     .append(sample["question_detok"])
            self._output_container.detok_ref_answer    .append(sample["answer_detok"])
            self._output_container.detok_ref_scratchpad.append(None)
            self._output_container.obj_ref_equations   .append(None)

    
    @classmethod
    def _prep_hf_ds(cls, *, sample, question_prefix, question_suffix):
        choices_text = []
        for label, text in more_itertools.zip_equal(
            sample["choices"]["label"], 
            sample["choices"]["text"]
        ):
            choices_text.append(f"({label.strip()}) {text.strip()}")

        choices = "\n".join(choices_text)

        # This format matches the few-shot examples
        sample["question"] = (
            f"{question_prefix}Q: {sample['question']}\nAnswer Choices:\n{choices}\nA:{question_suffix}"
        )
        sample["answer"] = "(" + sample["answerKey"].strip() + ")"

        del sample["id"]
        del sample["question_concept"]
        del sample["choices"]
        del sample["answerKey"]

        return sample


    @property
    def use_few_shots(self):
        return self._use_few_shots

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

    @classmethod
    def post_process_gen_fewshots(cls, *, any_tokenizer, raw_gen_outputs):
        # With few-shots, the model tends to generate other questions, lol. 
        # We don't want that.
        decoded = any_tokenizer.batch_decode(
            raw_gen_outputs,
            skip_special_tokens=True,
        )
        decoded = [
            x.strip().split("\n", 1)[0] + any_tokenizer.eos_token
            for x in decoded
        ]

        return [torch.tensor(x, dtype=torch.long, device=torch.device(LOCAL_RANK)) for x in any_tokenizer(decoded)["input_ids"]]
         

def tests():

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[rich.logging.RichHandler(markup=True)],
        format=f"[{RANK}/{WORLD_SIZE}] %(funcName)s:%(lineno)d - %(message)s",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained("ausboss/llama-30b-supercot")
    ds = CommonSenseQAMC(
        any_tokenizer=tokenizer, 
        split="test",
        question_prefix="",
        question_suffix="",
        use_few_shots=False,
    )
    
    print(ds[2].detok_ref_query)

    import ipdb
    ipdb.set_trace()

if __name__ == "__main__":
    fire.Fire(tests)



