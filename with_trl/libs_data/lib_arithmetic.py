from __future__ import annotations
import collections
import logging
import os
import pathlib
import re
import string
import subprocess
import sys
import time
import typing
from typing import Any, Optional, Union

import datasets
import fire
import jsonlines as jsonl
import more_itertools as mit
import rich
import rich.logging
import torch
import torch.utils.data
import tqdm
import transformers

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR.parent))
import lib_base_classes
import lib_metric
import lib_utils
import libs_extraction.lib_multiple_choice
import libs_data.lib_base
import libs_data.data_commonsense_qa_few_shot
import libs_data.arithmetic.arithmetic_10_shot

datasets.disable_caching()


RANK = int(os.getenv("RANK", 0))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOGGER = logging.getLogger(__name__)


def _tok_detok(*, batch, any_tokenizer, batched, exclusion_set=None):
    if exclusion_set is None:
        exclusion_set = set()

    output = {}
    assert "question" in batch, batch.keys()


    for k, v in batch.items():
        if k in exclusion_set:
            output[k] = v
            continue

        tok = any_tokenizer(v)
        if batched:
            detok_skip = any_tokenizer.batch_decode(tok.input_ids, skip_special_tokens=True)
            # detok_not_skip = any_tokenizer.batch_decode(tok.input_ids, skip_special_tokens=False)
        else:
            detok_skip = any_tokenizer.decode(tok.input_ids, skip_special_tokens=True)
            # detok_not_skip = any_tokenizer.decode(tok.input_ids, skip_special_tokens=False)

        output[k + "_tok"  ] = tok.input_ids
        # output[k + "_detok"] = detok_skip
        output[k + "_detok_skip"] = detok_skip
        # output[k + "_detok_not_skip"] = detok_not_skip

    return output

def _count_lines(file):
    return int(subprocess.check_output(
            ["wc", "-l", str(file)], 
            universal_newlines=True
        ).split()[0])

class Arithmetic(
    libs_data.lib_base.FewShotMixin,
    libs_data.lib_base.Dataset,
):
    def __init__(
            self, 
            *, 
            any_tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
            dataset_root_folder_dir: str | pathlib.Path,
            pad_token: str,
            eos_token: str,
            question_prefix: str, 
            question_suffix: Optional[str],
            sft_mode: bool,
            shuffle_once: bool,
            split: lib_utils.CVSets,
            use_few_shots: bool,
            extractor_ignore_one_line: bool,
            use_cached_dataset: bool = False,
        ):

        dataset_root_folder_dir = pathlib.Path(dataset_root_folder_dir)
        self._dataset_root_folder_dir = dataset_root_folder_dir
        assert dataset_root_folder_dir.exists(), dataset_root_folder_dir

        assert question_prefix is None, f"question_prefix: `{question_prefix}`"
        assert question_suffix is None, f"question_suffix: `{question_suffix}`"
        self._question_prefix = question_prefix
        self._question_suffix = question_suffix
        self._eos_token = eos_token

        self._use_few_shots = use_few_shots
        if use_few_shots:
            self._question_prefix = self._prepare_few_shots()
            self._question_suffix = "<scratch>"

        self._extractor = libs_extraction.lib_final_line.FinalLineExtractor(
            pad_token=pad_token,
            ignore_one_line=extractor_ignore_one_line,
        )

        split = lib_utils.CVSets(split)
        folder = dataset_root_folder_dir / f"{split.value}_scratch"
        assert folder.exists(), folder
        assert folder.is_dir(), folder
        glob_pattern = f"*.jsonl"
        target_files = list(folder.glob(glob_pattern))
        assert target_files, (glob_pattern, folder, list(folder.iterdir()))

        if sft_mode:
            self._setup_sft_mode(target_files)
        else:
            tok_name = any_tokenizer.name_or_path.replace("/", "_")
            hf_dataset_path = folder / f"{split}_{tok_name}.hf_dataset"

            if not use_cached_dataset or not hf_dataset_path.exists():
                self._prepare_rl_mode_data(target_files, any_tokenizer)
                if use_cached_dataset:
                    self._save_rl_mode_dataset(hf_dataset_path)
            else:
                self._load_cached_rl_mode_dataset(hf_dataset_path)        
            
        if shuffle_once:
            self._core.shuffle()

    def _load_cached_rl_mode_dataset(self, hf_dataset_path):
        rich.print(f"[green bold]Loading hf_dataset from disk:[/]  {hf_dataset_path}")
        self._core = datasets.load_from_disk(hf_dataset_path)

    def _save_rl_mode_dataset(self, hf_dataset_path):
        rich.print(f"[green bold]Saving hf_dataset to disk:[/] {hf_dataset_path}")
        hf_dataset_obj = datasets.Dataset.from_dict(
            dict(self._core.items()))
        hf_dataset_obj.save_to_disk(hf_dataset_path)
        self._core = hf_dataset_obj

    def _prepare_rl_mode_data(self, target_files, any_tokenizer):
        self._core = lib_base_classes.DataListContainer()
        target_files.sort(key=lambda x: x.name)

        for file in tqdm.tqdm(target_files, desc="Loading files"):
            num = _count_lines(file)

            with jsonl.open(file) as f:
                for sample in tqdm.tqdm(
                    f, 
                    desc=f"{file.name}: Building DataListContainer, including Tok-detok.", 
                    total=num
                ):
                    sample["question"] = (
                        self._question_prefix + "\n" +
                        "Q: " + sample["input"] + "\n" +
                        self._question_suffix + "\n"
                    )

                    self._core.extra_information.append(
                        dict(num_digits=int(sample["num_digits"]))
                    )
                    
                    sample = _tok_detok(
                        any_tokenizer = any_tokenizer, 
                        batch         = sample, 
                        batched       = False,
                        exclusion_set = {"num_digits"}
                    )
                
                    self._core.tok_ref_query     .append(torch.tensor(sample[  "question_tok"]))
                    self._core.tok_ref_answer    .append(torch.tensor(sample[    "answer_tok"]))
                    self._core.tok_ref_scratchpad.append(torch.tensor(sample["scratchpad_tok"]))

                    self._core.detok_ref_query     .append(sample[  "question_detok_skip"])
                    self._core.detok_ref_answer    .append(sample[    "answer_detok_skip"])
                    self._core.detok_ref_scratchpad.append(sample["scratchpad_detok_skip"])

            self._max_num_digits = max(
                self._core.extra_information, 
                key=lambda x: x["num_digits"]
            )["num_digits"]


    def _setup_sft_mode(self, target_files):
        self._core = lib_utils.DictDataset(
            keys=[
                "ref_qa_question", 
                "ref_qa_answer", 
                "ref_qa_scratchpad",
                "num_digits",
            ]
        )

        for file in tqdm.tqdm(
            target_files, 
            desc="Loading files", 
            disable=RANK != 0,
        ):
            assert not self._question_prefix
            with jsonl.open(file) as f:
                for sample in tqdm.tqdm(f, desc="Building DictDataset"):
                    scratchpad = sample["scratchpad"] + "\n" + sample["answer"]
                    self._core.append(dict(
                        ref_qa_question   = sample["input"],
                        ref_qa_answer     = sample["answer"],
                        ref_qa_scratchpad = scratchpad,
                        num_digits        = int(sample["num_digits"]),
                ))
        self._max_num_digits = max(self._core["num_digits"])

    @property
    def max_num_digits(self):
        return self._max_num_digits

    def _prepare_few_shots(self):
        text_list = []
        
        for entry in libs_data.arithmetic.arithmetic_10_shot.make_few_shots(
            self._dataset_root_folder_dir
        ):
            text_list.append(
                f"Q: {entry['input']}\n"
                f"{entry['scratchpad']}\n"
                f"A:\n" +
                str(entry['answer']) + "\n" +
                f"!\n"
            )
        final_text = "\n".join(text_list)        

        return final_text

    @property
    def use_few_shots(self):
        return self._use_few_shots

    def __len__(self):
        return len(self._core) # type: ignore

    def __getitem__(
        self, idx_or_slice: typing.Union[int, slice]
    ) -> lib_utils.DictDataset:
        return {
            key: (torch.tensor(value) if key.startswith("tok_") else value) 
            for key, value in self._core[idx_or_slice].items()
        }

    @property
    def question_prefix(self):
        return self._question_prefix

    @property
    def question_suffix(self):
        return self._question_suffix

    def get_extractor(self):
        return self._extractor

    def post_process_gen_fewshots(self, *, raw_gen_outputs, any_tokenizer) -> str:
        text_list = any_tokenizer.batch_decode(raw_gen_outputs)
        return [
            torch.tensor(x, dtype=torch.long, device=torch.device(LOCAL_RANK)) 
            for x in any_tokenizer(text_list)["input_ids"]
        ]


def main():
    forward_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    path = "/network/scratch/g/gagnonju/arithmetic"

    ds = Arithmetic(
            dataset_root_folder_dir=path,
            any_tokenizer=forward_tokenizer,
            question_prefix=None,
            question_suffix=None,
            split=lib_utils.CVSets.TRAIN,
            use_few_shots=False,
        )
    for i in range(len(ds)):
        print(ds[i])
        break
     
    ds = Arithmetic(
            dataset_root_folder_dir=path,
            any_tokenizer=forward_tokenizer,
            question_prefix=None,
            question_suffix=None,
            split=lib_utils.CVSets.VALID,
            use_few_shots=False,
        )
    for i in range(len(ds)):
        print(ds[i])
        break



class PerNumberOfDigitsAccuracy(lib_base_classes.Metric):
    """Exact match answer accuracy.
    """

    def __init__(self, extractor, num_digits: int, pad_token):
        self._accuracy = lib_metric.ScratchpadAnswerAccuracy(
            extractor=extractor,
            pad_token=pad_token,
        )
        self._num_digits = num_digits

    @property
    def num_digits(self):
        return self._num_digits

    def __call__(
        self,
        *,
        responses: list[lib_base_classes.BatchedUnrollReturn],
        batch: lib_base_classes.DataListContainer,
    ) -> lib_base_classes.MetricOutput:

        #######################################################################
        # For each entry, replace the entry by None when the number of digits
        # is not the same as the one we are looking for.
        #######################################################################
        filtered_batch = lib_base_classes.DataListContainer()
        for i, entry in enumerate(batch):
            num_digits = batch.extra_information[i]["num_digits"]
            
            if num_digits != self._num_digits:
                new_entry = {k: v for k, v in vars(entry).items()}
            else:
                new_entry = {k: None for k  in vars(entry)}

            filtered_batch.append(**new_entry)

        #######################################################################
        # Compute individual accuracies, then merge them.
        #######################################################################
        accuracies = []
        for r, b in mit.zip_equal(responses, filtered_batch):
            none_test     = [x is None for x in b.values()]
            any_none      = any(none_test)
            all_none      = all(none_test)
            assert any_none == all_none, (
                any_none, all_none, none_test)
            
            new_batch     = lib_base_classes.DataListContainer(
                **{k: [v] for k, v in b.items()})
            new_responses = [r]

            if all_none:
                accuracies.append(None)
            else:
                accuracies.append(
                    self._accuracy(
                        batch     = new_batch,
                        responses = new_responses,
                    ) 
                )

        # Merge
        output = lib_base_classes.MetricOutput(
            extracted_gen   = [],
            extracted_ref   = [],
            logging_columns = [],
            moving_averages = None,
            name            = f"accuracy_with_{self._num_digits}_digits",
            values          = [],
        )

        for metric in accuracies:
            if metric is None:
                output.values          .append(None)
                output.logging_columns .append(None)
                output.extracted_gen   .append(None)
                output.extracted_ref   .append(None)
            else:
                output.values          .extend(metric.         values)
                output.logging_columns .extend(metric.logging_columns)
                output.extracted_gen   .extend(metric.  extracted_gen)
                output.extracted_ref   .extend(metric.  extracted_ref)

        assert len(output.values) == len(batch), (len(output.values), len(batch))
        return output

    
if __name__ == "__main__":
    main()