import logging
import os
import pathlib
import sys
from typing import Any, Optional, Union

import datasets
import more_itertools
import rich.traceback
import transformers
import tqdm

rich.traceback.install()

LOGGER = logging.getLogger(__name__)
_SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(_SCRIPT_DIR.parent))
from approach_answer import data_few_shot_commonsense_qa_scratchpads
from libs_data import data_commonsense_qa_few_shot
import libs_data
import libs_extraction


LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
datasets.disable_caching()


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


def _prep_hf_ds(*, sample, few_shots_text, give_model_answer):
    # Build Choices
    choices_text = []
    for label, text in more_itertools.zip_equal(
        sample["choices"]["label"], 
        sample["choices"]["text"]
    ):
        choices_text.append(f"({label.strip()}) {text.strip()}")

    choices = "\n".join(choices_text)
    ref_answer = "(" + sample["answerKey"].strip() + ")"

    if give_model_answer:
        few_shots_query = (
            f"{few_shots_text}\n\n" +
            f"Q: {sample['question']}\n" +
            f"Answer Choices:\n" +
            f"{choices}\n" +
            f"A: {ref_answer}\nReasoning: "
        )
    else:
        few_shots_query = (
        f"{few_shots_text}\n\n" +
        f"Q: {sample['question']}\n" +
        f"Answer Choices:\n" +
        f"{choices}\n" +
        f"A:"
    )

    # Build the output sample
    output_sample = {}
    # This format matches the few-shot examples
    output_sample["ref_fs_scratchpad_gen_query"] = few_shots_query
    output_sample["ref_qa_question"] = sample["question"]
    output_sample["ref_qa_id"] = sample["id"]
    output_sample["ref_qa_choices"] = choices
    output_sample["ref_qa_answer"] = ref_answer

    return output_sample


class CommonSenseScratchpadGenMC(libs_data.lib_base.Dataset):
    def __init__(
            self, 
            *, 
            any_tokenizer: Optional[transformers.PreTrainedTokenizerBase],
            split: str,
            give_model_answers: bool,
            text_only: bool = False,
        ):
        
        if give_model_answers:
            self._few_shots_str = (
                data_few_shot_commonsense_qa_scratchpads.FEW_SHOT.strip()
            )
        else:
            self._few_shots_str = data_commonsense_qa_few_shot.FEW_SHOT.strip()

        self._give_model_answers = give_model_answers
        self._text_only = text_only
        self._split = split

        if any_tokenizer is None:
            assert text_only, "Must provide a tokenizer if `text_only` is False"

        self._extractor = libs_extraction.lib_multiple_choice.MultipleChoiceRfindExtractor(
            ["(A)", "(B)", "(C)", "(D)", "(E)"])

        if LOCAL_RANK == 0:
            print(f"Loading dataset: {split}")
        
        self._prepare_dataset(any_tokenizer)

        if LOCAL_RANK == 0:
            print(f"Dataset loaded: {split}")
            
        assert all(
            k.startswith("ref_qa") or k.startswith("ref_fs") 
            for k in self._ds.features.keys()
        ), [
            k for k in self._ds.features.keys() 
            if not (k.startswith("ref_qa") or k.startswith("ref_fs"))
        ]


    def _prepare_dataset(self, any_tokenizer):
        hf_ds = datasets.load_dataset("commonsense_qa", split=self.split)
        tmp_ds = hf_ds.map(
            lambda sample: _prep_hf_ds(
                sample=sample,
                few_shots_text=self._few_shots_str,
                give_model_answer=self._give_model_answers,
            ),
            num_proc=1,
        ).remove_columns(
            ["id", "question", "question_concept", "choices", "answerKey"]
        )
        
        if not self._text_only:
            tmp_ds = tmp_ds.map(
                lambda batch: _tok_detok(
                    batch=batch,
                    any_tokenizer=any_tokenizer,
                ),
                batched=True,
            )

            self._ds = tmp_ds.remove_columns(
                [key for key in tmp_ds.features 
                 if not (key.endswith("_tok") or key.endswith("_detok"))
                ]
            )
        else:
            self._ds = tmp_ds
    
    @property
    def split(self):
        return self._split

    @property
    def text_only(self):
        return self._text_only

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