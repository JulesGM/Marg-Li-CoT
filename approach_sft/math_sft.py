import os
import pathlib
import sys
from typing import Optional

import datasets
import fire
import rich
import rich.traceback
import torch
import transformers

import lib_sft_constants

import tqdm
import more_itertools as mit

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

#####################################
trl_path = SCRIPT_DIR.parent / "with_trl"
assert trl_path.exists(), trl_path
sys.path.append(str(trl_path))
import lib_utils
#####################################
general_path = SCRIPT_DIR.parent / "general"
assert general_path.exists(), general_path
sys.path.append(str(general_path))
import hendrycks_math_utils
#####################################

rich.traceback.install()


SPLIT_CONVERSION_MAP = {
    lib_utils.CVSets.TRAIN: "train",
    lib_utils.CVSets.VALID: "test",
    "small_eval": "test",
}


class MATHDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            split: lib_utils.CVSets, 
            max_num_tokens: Optional[int],
            forward_tokenizer, 
            prediction_tokenizer,
            output_type: lib_sft_constants.OutputTypes,
            use_chat_templates: bool,
        ):

        try:
            hf_dataset = datasets.load_dataset(
                "hendrycks/competition_math",
                trust_remote_code=True,
            )
        except datasets.exceptions.DatasetNotFoundError:
            hf_dataset = datasets.load_from_disk(
                os.environ.get(
                    "MATH_DATASET_PATH", 
                    str(pathlib.Path("~/hendrycks_math").expanduser())
                ),
            )

        hf_dataset = hf_dataset[SPLIT_CONVERSION_MAP[split]]

        assert split in [
            lib_utils.CVSets.TRAIN, 
            lib_utils.CVSets.VALID, 
            "small_eval"
        ], split

        if split == lib_utils.CVSets.VALID or split == "small_eval":
            # Using half as validation, half as test
            hf_dataset = datasets.Dataset.from_dict(hf_dataset[:int(len(hf_dataset) * 0.5)])

        self._output_type = output_type
        self._hf_dataset = hf_dataset
        self._forward_format = "Question:\n{problem}\n\nAnswer:\n{solution}"
        self._prediction_format = "Question:\n{problem}\n\nAnswer:\n"

        collator = MATHCollator(
            forward_tokenizer=forward_tokenizer,
            prediction_tokenizer=prediction_tokenizer,
            use_chat_templates=use_chat_templates,
            answer_only=output_type == lib_sft_constants.OutputTypes.ANSWER_ONLY,
        )

        outputs = []
        assert len(self) == len(self._hf_dataset), (
            len(self), len(self._hf_dataset))

        for preped_sample, raw_sample in tqdm.tqdm(
            mit.zip_equal(self, self._hf_dataset), 
            total=len(self._hf_dataset),
            desc=f"Filtering for {split}"
        ):
            answer = hendrycks_math_utils.last_boxed_only_string(
                preped_sample["ref_qa_answer"]
            )

            if not answer:
                rich.print("[red bold]Skipping sample with no answer[/]")
                continue

            collated = collator([preped_sample])
            
            if max_num_tokens:
                forward = len(mit.one(collated["forward"]["input_ids"])) > max_num_tokens
                prediction = len(mit.one(collated["predict"]["input_ids"])) > max_num_tokens
                if not (forward or prediction):
                    outputs.append(raw_sample)
            else:
                outputs.append(raw_sample)

        rich.print(
            f"Filtered out {len(self._hf_dataset) - len(outputs)} samples, "
            f"{len(outputs) / len(self._hf_dataset):0.1%} kept"
        )

        self._hf_dataset = outputs
        assert isinstance(self._hf_dataset, list), type(self._hf_dataset)


    def _apply_format(self, hf_dict, forward_or_prediction):
        assert isinstance(hf_dict, dict), type(hf_dict)
        
        if forward_or_prediction == "forward":
            if self._output_type == lib_sft_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER:
                return self._forward_format.format(**hf_dict)

            elif self._output_type == lib_sft_constants.OutputTypes.ANSWER_ONLY: 
                final_answer = hendrycks_math_utils.last_boxed_only_string(
                    hf_dict["solution"]
                )
                
                return self._forward_format.format(
                    problem=hf_dict["problem"],
                    solution=final_answer,
                )
            
        elif forward_or_prediction == "prediction":
            return self._prediction_format.format(problem=hf_dict["problem"])
        
        else:
            raise NotImplementedError(forward_or_prediction)

        return NotImplementedError(self._output_type)

    def __getitem__(self, idx: int):
        a = dict(
            formatted_forward=self._apply_format(self._hf_dataset[idx], "forward"),
            formatted_prediction=self._apply_format(self._hf_dataset[idx], "prediction"),
            ref_qa_answer=self._hf_dataset[idx]["solution"],
            ref_qa_question=self._hf_dataset[idx]["problem"],
        )
        return a

    def __len__(self):
        assert isinstance(self._hf_dataset, (
            list, datasets.Dataset)), type(self._hf_dataset)
        
        return len(self._hf_dataset)


LIGHTEVAL_PROMPT_MATH = dict(
    question="{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n", 
    answer="{answer}\n\n"
)

class MATHCollator:
    def __init__(
            self, 
            *,
            forward_tokenizer: transformers.PreTrainedTokenizer, 
            prediction_tokenizer: transformers.PreTrainedTokenizer,
            use_chat_templates: bool,
            answer_only: bool,
        ):

        self._answer_only = answer_only
        self._use_chat_templates = use_chat_templates
        self._forward_tokenizer = forward_tokenizer
        self._prediction_tokenizer = prediction_tokenizer

    def __call__(self, batch):

        formatted_forward = [x["formatted_forward"] for x in batch]
        formatted_prediction = [x["formatted_prediction"] for x in batch]
        ref_qa_answer_batch = [x["ref_qa_answer"] for x in batch]
        ref_qa_question_batch = [x["ref_qa_question"] for x in batch]


        if self._use_chat_templates:
            messages_forward = []
            messages_prediction = []
            for question, answer in mit.zip_equal(ref_qa_question_batch, ref_qa_answer_batch):
                if self._answer_only:
                    # Extract the last boxed only string from the answer
                    last_boxed_only_string = hendrycks_math_utils.last_boxed_only_string(answer)

                    messages_forward.append(
                        [{"role": "user", "content": question},
                         {"role": "assistant", "content": last_boxed_only_string},
                        ]
                    )
                    messages_prediction.append(
                        [{"role": "user", "content": question},
                        ]
                    )
                else:
                    messages_forward.append(
                        [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer},
                        ]
                    )
                    messages_prediction.append(
                        [{"role": "user", "content": question}]
                    )

            forward_tokenized_not_padded = [
                self._forward_tokenizer.apply_chat_template(
                    forward_messages, tokenize=True, add_generation_prompt=False
                ) for forward_messages in messages_forward
            ]

            tokenized_forward = self._forward_tokenizer.pad(
                dict(input_ids=forward_tokenized_not_padded), 
                padding=True, 
                return_tensors="pt",
            )

            prediction_tokenized_not_padded = [
                self._prediction_tokenizer.apply_chat_template(
                    prediction_messages, tokenize=True, add_generation_prompt=True
                ) for prediction_messages in messages_prediction
            ]

            tokenized_prediction = self._prediction_tokenizer.pad(
                dict(input_ids=prediction_tokenized_not_padded), 
                padding=True, 
                return_tensors="pt",
            )
                
        else:
            tokenized_forward = self._forward_tokenizer(
                formatted_forward,
                padding=True,
                return_tensors="pt",
            )

            tokenized_prediction = self._prediction_tokenizer(
                formatted_prediction,
                padding=True,
                return_tensors="pt",
            )

        return dict(
            forward=tokenized_forward,
            predict=tokenized_prediction,
            extra_info=dict(
                ref_qa_answer=ref_qa_answer_batch,
                ref_qa_question=ref_qa_question_batch,
            ),
        )


def test(hf_name="gpt2", split=lib_utils.CVSets.TRAIN, n=1):
    split = lib_utils.CVSets(split)
    dataset = MATHDataset(split)
    
    forward_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_name)
    prediction_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_name)

    for tokenizer in [forward_tokenizer, prediction_tokenizer]:
        tokenizer.pad_token = tokenizer.eos_token


    import collections
    import more_itertools as mit
    import numpy as np

    lengths = {}
    for output_type in [
        lib_sft_constants.OutputTypes.ANSWER_ONLY, 
        lib_sft_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER
    ]:

        collator = MATHCollator(
            forward_tokenizer=forward_tokenizer,
            prediction_tokenizer=prediction_tokenizer,
            output_type=output_type,
        )

        individual_lengths = []
        for sample in dataset:
            c_sample = collator([sample])
            individual_lengths.append(len(mit.one(c_sample["forward"]["input_ids"])))
        
        individual_lengths = np.array(individual_lengths, dtype=np.int64)
        individual_lengths.sort()
        lengths[output_type] = collections.Counter(individual_lengths)

    print(lengths)



if __name__ == "__main__":
    fire.Fire(test)
    
