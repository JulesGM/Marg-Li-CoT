"""

Implements support for the "aqua_rat" dataset.

4th of August, 2024 - Jules: 
    Not planned for either of SFT or RL.
    Loads the Huggingface Dataset aqua_rat.

"""

import pathlib
import re
import string
import sys
import typing

import datasets
import fire
import torch
import torch.utils.data
import tqdm
import transformers

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR.parent))
import lib_base_classes

LETTER_MAP = {
    v: i 
    for i, v in 
    enumerate(string.ascii_uppercase)
}
PATS_STR = [
    r"^\w\.?$",
    r"^answer is \w\.?$",
    r"^option[=\:\s\-]+\w+\.?$",
    r"^answer[=\:\s\-]+\w+\.?$",
    r"^answer: option \w+\.?$",
    r"^choice[=\:\s\-]+\w+\.?$",
]
PATS = [re.compile(pat) for pat in PATS_STR]


def one_of(str_):
    str_ = str_.strip().lower()
    matches = [pat.match(str_) for pat in PATS]
    return any(matches)


def extract_answer(example):
    letter = example["correct"].strip()
    good_answer_idx = LETTER_MAP[letter]
    answer_text = example["options"][good_answer_idx].replace(",", "")
    assert answer_text.startswith(letter + ")")
    
    while answer_text.upper().startswith(letter + ")"):
        answer_text = answer_text[len(letter) + 1:].strip()

    while answer_text.upper().startswith(letter + "."):
        answer_text = answer_text[len(letter) + 1:].strip()

    while answer_text.upper().startswith("[" + letter + "]"):
        answer_text = answer_text[1 + len(letter) + 1:].strip()

    while answer_text.upper().startswith(letter + " "):
        answer_text = answer_text[len(letter) + 1:].strip()

    rationale = " ".join(example["rationale"].split("\n")[:-1])
    rationale = rationale.replace("\n", " ")

    return {
        "answer": answer_text, 
        "rationale": rationale.strip() + ". The answer is " + answer_text
    }


def only_one_int(sample):
    matches = re.findall(r"\d+", sample["answer"])
    return matches and len(matches) == 1


def make_dataset():
    rat = datasets.load_dataset("aqua_rat")
    # Fields: 'question', 'rationale', 'answer'

    return (
        rat["train"] # type: ignore
    ).filter(lambda x: one_of(x["rationale"].split("\n")[-1]) # type: ignore
    ).map(extract_answer, batched=False
    ).filter(only_one_int, batched=False
    ).remove_columns(["options", "correct"])


def tok_detok(samples, any_tokenizer):
    output = {}
    for k, v in samples.items():
        tok = any_tokenizer(v)
        detok = any_tokenizer.batch_decode(tok["input_ids"])
        output[k + "_tok"] = tok.input_ids
        output[k + "_detok"] = detok

    print({k: type(v) for k, v in output.items()})
    return output


class RAT(torch.utils.data.Dataset):
    def __init__(self, any_tokenizer):
        ds = make_dataset()
        self._ds = ds.map(
            lambda batch: tok_detok(batch, any_tokenizer),
            batched=True, 
            batch_size=len(ds),
        )

        self._output_container = lib_base_classes.DataListContainer()

        for sample in tqdm.tqdm(self._ds, desc="Building DataListContainer"):
            self._output_container.tok_ref_query       .append(sample["question_tok"]) # type: ignore
            self._output_container.tok_ref_answer      .append(sample["answer_tok"]) # type: ignore
            self._output_container.tok_ref_scratchpad  .append(sample["rationale_tok"]) # type: ignore
            self._output_container.detok_ref_query     .append(sample["question_detok"]) # type: ignore
            self._output_container.detok_ref_answer    .append(sample["answer_detok"]) # type: ignore
            self._output_container.detok_ref_scratchpad.append(sample["rationale_detok"]) # type: ignore
            self._output_container.obj_ref_equations   .append(None)

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
    

def _tests():
    MODEL_HF_NAME = "ausboss/llama-30b-supercot"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_HF_NAME)
    RAT(tokenizer)


if __name__ == "__main__":
    fire.Fire(_tests)