import os
import pathlib
import sys
from typing import Any, Optional, Union

import datasets
import more_itertools
import torch
import tqdm

import libs_data
import libs_extraction
import lib_base_classes
import lib_utils

_SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(_SCRIPT_DIR.parent))

from approach_answer import data_few_shot_commonsense_qa

LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))


def _tok_detok(*, batch, any_tokenizer, ignore_keys=None):
    """
    For each key in batch, tokenize and detokenize the values.

    Values associated to the `ignore_keys` are kept as-is.
    """
    output = {}

    for k, v in batch.items():
        if ignore_keys is not None and k in ignore_keys:
            output[k] = v
        else:
            preped_text = [x.strip() for x in v]
            tok = any_tokenizer(preped_text)
            detok = any_tokenizer.batch_decode(tok["input_ids"], skip_special_tokens=False)
            output[k + "_tok"] = tok.input_ids
            output[k + "_detok"] = detok

    return output


class CommonSenseScratchpadGenMC(libs_data.lib_base.FewShotMixin, libs_data.lib_base.Dataset):
    def __init__(
            self, 
            *, 
            any_tokenizer, 
            split: str,
        ):

        self._few_shot_data = data_few_shot_commonsense_qa.FEW_SHOT + "\n\n"

        self._extractor = libs_extraction.lib_multiple_choice.MultipleChoiceRfindExtractor(
            ["(A)", "(B)", "(C)", "(D)", "(E)"])

        _hf_ds = datasets.load_dataset("commonsense_qa", split=split)
        self._ds = _hf_ds.map(
            lambda sample: self._prep_hf_ds(
                sample=sample,
                few_shots_text=self._few_shot_data,
            ),
        ).remove_columns(["id", "question", "question_concept", "choices", "answerKey"]).map(
            lambda batch: _tok_detok(
                batch=batch,
                any_tokenizer=any_tokenizer,
            ),
            batched=True,
        )
        assert all(k.startswith("ref_qa") or k.startswith("ref_fs") for k in self._ds.features.keys()), [
            k for k in self._ds.features.keys() if not (k.startswith("ref_qa") or k.startswith("ref_fs"))
        ]

    @classmethod
    def _prep_hf_ds(cls, *, sample, few_shots_text):
        # Build Choices
        choices_text = []
        for label, text in more_itertools.zip_equal(
            sample["choices"]["label"], 
            sample["choices"]["text"]
        ):
            choices_text.append(f"({label.strip()}) {text.strip()}")
        choices = "\n".join(choices_text)

        ref_answer = "(" + sample["answerKey"].strip() + ")"
        few_shots_query = (
            f"{few_shots_text}Q: {sample['question']}\nAnswer Choices:\n{choices}\nA: {ref_answer}\nReasoning: ")

        # Build the output sample
        output_sample = {}
        # This format matches the few-shot examples
        output_sample["ref_fs_scratchpad_gen_query"] = few_shots_query
        output_sample["ref_qa_question"] = sample["question"]
        output_sample["ref_qa_id"] = sample["id"]
        output_sample["ref_qa_choices"] = choices
        output_sample["ref_qa_answer"] = ref_answer

        return output_sample

    @property
    def use_few_shots(self):
        return self._use_few_shots

    def __len__(self):
        return len(self._ds) # type: ignore

    def __getitem__(
        self, idx_or_slice: Union[int, slice]
    ):   
        return self._ds[idx_or_slice] 

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
         