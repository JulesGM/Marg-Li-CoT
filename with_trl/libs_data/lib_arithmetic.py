from __future__ import annotations
import collections
import logging
import os
import pathlib
import random
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
import numpy as np
import rich
import rich.logging
import rich.markup
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
            detok_skip = any_tokenizer.batch_decode(
                tok.input_ids, 
                skip_special_tokens=True
            )
            # detok_not_skip = any_tokenizer.batch_decode(tok.input_ids, skip_special_tokens=False)
        else:
            detok_skip = any_tokenizer.decode(
                tok.input_ids, 
                skip_special_tokens=True
            )
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

class DSIterator:
    def __init__(self, ds, use_curriculum: bool):
        self._ds = ds

        self._proportion_difficulties = None
        self._use_curriculum = use_curriculum

        if self._use_curriculum:
            assert isinstance(self._ds, dict), type(self._ds).mro()
            self._indices = {k: 0 for k in self._ds}
            self._idx = None
        else:
            self._indices = None
            self._idx = 0

    def set_proportion_difficulties(self, proportions):
        assert self._use_curriculum, "Trying to set proportions when use-curriculum is disabled"
        
        if RANK == 0:
            rich.print(rich.rule.Rule(style="red"))
            rich.print(rich.rule.Rule(style="red"))
            rich.print(rich.rule.Rule(style="red"))
            rich.print(f"SETTING DIFFICULTY PROPORTIONS TO {proportions}")
            rich.print(rich.rule.Rule(style="red"))
            rich.print(rich.rule.Rule(style="red"))
            rich.print(rich.rule.Rule(style="red"))

        self._proportion_difficulties = proportions

    def __next__(self):
        """
        Each dataset difficulty is sharded to len / num_gpus.

        This is the most naive way to do it.

        The main advantage is that it's the most straightforward to parallelize. 

        The main disadvantage is that a process can run out of data while another has too much.

        """

        assert self._use_curriculum

        if self._use_curriculum:
            """
            . Sample a difficulty
            . Get the dataset object linked to the difficulty
            . get the index of the sharded data we're at with that difficulty dataset
            . increment the index, possibly restarting the epoch

            """
            difficulty = np.random.choice(
                list(self._proportion_difficulties.keys()),
                p=list(self._proportion_difficulties.values()),
            )
            
            ds_obj = self._ds[difficulty]
            base_idx = self._indices[difficulty]
            self._indices[difficulty] += 1
        else:
            ds_obj = self._ds
            base_idx = self._idx
            self._idx += 1
            
        # Check that the sharded idx is within bounds
        # Adjust it if it't not
        
        sharded_idx = (base_idx * WORLD_SIZE + RANK) % len(ds_obj)
        value = ds_obj[sharded_idx]

        return value


class Arithmetic(
    libs_data.lib_base.FewShotMixin,
    libs_data.lib_base.IterableDataset,
):
    def __init__(
            self, 
            *, 
            answer_only: bool,
            any_tokenizer: Optional[transformers.PreTrainedTokenizerBase] = None,
            dataset_root_folder_dir: str | pathlib.Path,
            eos_token: str,
            extractor_ignore_one_line: bool,
            pad_token: str,
            question_prefix: str, 
            question_suffix: Optional[str],
            sft_mode: bool,
            shuffle_once: bool,
            split: lib_utils.CVSets,
            use_few_shots: bool,
            use_curriculum: bool,
            use_cached_dataset: bool = False,
        ):

        dataset_root_folder_dir = pathlib.Path(dataset_root_folder_dir)
        self._dataset_root_folder_dir = dataset_root_folder_dir
        assert dataset_root_folder_dir.exists(), dataset_root_folder_dir

        assert question_prefix is None, f"question_prefix: `{question_prefix}`"
        assert question_suffix is None, f"question_suffix: `{question_suffix}`"
        
        if answer_only:
            assert not sft_mode, "Not done yet"

        self._answer_only = answer_only
        self._question_prefix = question_prefix
        self._question_suffix = question_suffix
        self._eos_token = eos_token
        self._use_curriculum = use_curriculum

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

        self._inner_iterator = DSIterator(
            ds=self._core, 
            use_curriculum=self._use_curriculum,
        )
        self._already_started = False

    def _load_cached_rl_mode_dataset(self, hf_dataset_path):
        if RANK == 0:
            rich.print(f"[green bold]Loading hf_dataset from disk:[/]  {hf_dataset_path}")

        self._core = datasets.load_from_disk(hf_dataset_path)

    def _save_rl_mode_dataset(self, hf_dataset_path):
        if RANK == 0:
            rich.print(f"[green bold]Saving hf_dataset to disk:[/] {hf_dataset_path}")

        hf_dataset_obj = datasets.Dataset.from_dict(
            dict(self._core.items()))
        hf_dataset_obj.save_to_disk(hf_dataset_path)
        self._core = hf_dataset_obj

    def _prepare_rl_mode_data(self, target_files, any_tokenizer):
        self._core = lib_base_classes.DataListContainer()
        target_files.sort(key=lambda x: x.name)

        for file in tqdm.tqdm(target_files, desc="Loading files", disable=RANK != 0):
            num = _count_lines(file)

            with jsonl.open(file) as f:
                for sample in tqdm.tqdm(
                    f, 
                    desc=f"{file.name}: Building DataListContainer, including Tok-detok.", 
                    total=num,
                    disable=RANK != 0,
                ):
                    if self._answer_only:
                        sample["question"] = (
                            self._question_prefix + "\n" +
                            "Q: " + sample["input"] + "\n" +
                            sample["scratchpad"] + "\nA:\n"
                        )
                    else:
                        sample["question"] = (
                            self._question_prefix + "\n" +
                            "Q: " + sample["input"] + "\n" +
                            self._question_suffix + "\n"
                        )

                    difficulty_level = sample["num_digits"]

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
                    self._core.extra_information   .append({})
                    self._core.difficulty_level.append(difficulty_level)

            self._max_num_digits = max(
                self._core.difficulty_level
            )

        if self._use_curriculum:
            final = collections.defaultdict(list)
            for entry in self._core:
                final[entry.difficulty_level].append(entry)
            self._core = final


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

    def __iter__(self):
        assert self._already_started is False
        self._already_started = True

        return self

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

    def set_proportion_difficulties(self, proportions):
        self._inner_iterator.set_proportion_difficulties(proportions)

    def __next__(self):
        
        assert self._inner_iterator is not None
        return next(self._inner_iterator)


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


def main(n=1):
    forward_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Open-Orca/Mistral-7B-OpenOrca"
    )
    path = (
        "/home/mila/g/gagnonju/Marg-Li-CoT/with_trl/libs_data/arithmetic/"
    )

    ds = Arithmetic(
            any_tokenizer=forward_tokenizer,
            dataset_root_folder_dir=path,
            pad_token=forward_tokenizer.pad_token,
            eos_token=forward_tokenizer.eos_token,
            extractor_ignore_one_line=False,
            question_prefix=None,
            question_suffix=None,
            sft_mode=False,
            shuffle_once=False,
            split=lib_utils.CVSets.TRAIN,
            use_cached_dataset=False,
            use_few_shots=True,
        )
    
    # for i in range(n):
    #     sample = ds[i]
    #     sorted_keys = sorted(sample.keys())

    #     table = rich.table.Table(
    #         "Keys", "Values",
    #         highlight=True, 
    #         show_lines=True,
    #     )
    #     table.add_row("Keys", rich.markup.escape(str(sorted_keys)))

    #     table.add_row("Non-tensor vals", rich.markup.escape(str(
    #         {
    #             k: sample[k]
    #             for k in sorted_keys
    #             if not isinstance(sample[k], torch.Tensor)
    #         }
    #     )))
        
    #     table.add_row(
    #         "Few Shots", 
    #         rich.markup.escape(str(sample["detok_ref_query"])))
    #     rich.print(table)

    rich.print(ds[0]["detok_ref_query"])

if __name__ == "__main__":
    fire.Fire(main)