from __future__ import annotations
import collections
import enum
import itertools as it
import logging
import math
import os
import pathlib
import random
import subprocess
import sys
from typing import Optional

import datasets
import fire
import jsonlines as jsonl
import more_itertools as mit
import numpy as np
import rich
import rich.logging
import rich.markup
import rich.rule
import rich.traceback
import torch
import torch.utils.data
import tqdm
import transformers

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR.parent))
import lib_base_classes
import lib_metric
import lib_utils
from libs_extraction import lib_final_line
from libs_data import lib_base
from libs_data.arithmetic import arithmetic_10_shot

rich.traceback.install(show_locals=True)
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
    """

    Init:    
    Set the index of each curriculum level to 0.
    or
    Set the index to zero.

    

    """

    def __init__(self, ds, use_curriculum: bool, return_idx: bool, use_few_shots: bool):
        self._ds = ds
        self._return_idx = return_idx
        self._use_few_shots = use_few_shots

        self._proportion_difficulties = None
        self._use_curriculum = use_curriculum

        if self._use_curriculum:
            assert isinstance(self._ds, dict), type(self._ds).mro()
            self._indices = {k: 0 for k in self._ds}
            self._is_done = {k: False for k in self._ds}
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
        assert not all(
            math.isclose(v, 0) 
            for v in proportions.values()
        ), proportions

    def __next__(self):
        """
        Each dataset difficulty is sharded to len / num_gpus.

        This is the most naive way to do it.

        The main advantage is that it's the most straightforward to parallelize. 

        The main disadvantage is that a process can run out of data while another has too much.

        """

        if self._use_curriculum:
            """
            . Look up unfinished difficulties
            . Normalize the probabilities of unfinished difficulties
            . Sample a difficulty
            . Get the dataset object linked to the difficulty
            . get the index of the sharded data we're at with that difficulty dataset
            . increment the index, possibly restarting the epoch

            The issue with that is that the proportions break down as the epoch progresses.
            Anything that has a non zero proportion will eventually be fully trained on.

            Using uniform probabilities between a set of difficulties would be a more robust way to do it, if less flexible.
            
            """

            #######################################################################################
            # Look up unfinished difficulties
            #######################################################################################
            proportion_keys = self._proportion_difficulties.keys()
            not_done = {k for k in proportion_keys if not self._is_done[k]}
            not_dones = proportion_keys & not_done
            
            if not not_dones:
                raise StopIteration

            #######################################################################################
            # Normalize the probabilities of unfinished difficulties
            #######################################################################################
            normalization_factor = sum(self._proportion_difficulties[k] for k in not_dones)
            normalized_proportions = {
                k: self._proportion_difficulties[k] / normalization_factor
                for k in not_dones
            }
            for k in self._proportion_difficulties:
                if k not in not_dones:
                    normalized_proportions[k] = 0.

            #######################################################################################
            # Pick a difficulty
            #######################################################################################
            assert math.isclose(sum(normalized_proportions.values()), 1), normalized_proportions

            difficulty = np.random.choice(
                list(normalized_proportions.keys()),
                p=list(self._proportion_difficulties.values()),
            )

            #######################################################################################
            # Increment the index.
            #######################################################################################
            self._indices [difficulty]   +=  1
            ds_obj                        =  self._ds      [difficulty]
            base_idx                      =  self._indices [difficulty]
            
            if self._indices[difficulty] + 1 >= len(ds_obj):
                assert self._indices[difficulty] + 1 == len(ds_obj), (
                    self._indices[difficulty] + 1, 
                    len(ds_obj),
                )
                self._is_done[difficulty] = True

        else:
            ds_obj     = self._ds
            base_idx   = self._idx
            self._idx += 1
            if self._idx >= len(ds_obj):
                assert self._idx == len(ds_obj), (self._idx, len(ds_obj))
                raise StopIteration
            
        # Check that the sharded idx is within bounds
        # Adjust it if it't not

        value = ds_obj[base_idx]

        if self._use_few_shots:
            value.detok_ref_query = self.make_few_shot_toks() + "\n\n" + value.detok_ref_query

        if self._return_idx:
            return value, base_idx
        else:
            return value


    def __iter__(self):
        return self


class TrainModes(enum.Enum):
    RL = "RL"
    SFT = "SFT"


class Arithmetic(
    lib_base.FewShotMixin,
    lib_base.IterableDataset,
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
            return_idx: bool,
            sft_mode: bool,
            shuffle_once: bool,
            split: lib_utils.CVSets,
            use_few_shots: bool,
            use_curriculum: bool,
            use_cached_dataset: bool,
        ):

        self._return_idx              = return_idx
        self._dataset_root_folder_dir = pathlib.Path(dataset_root_folder_dir)
        del dataset_root_folder_dir
        assert self._dataset_root_folder_dir.exists(), self._dataset_root_folder_dir

        if sft_mode:
            self._mode = TrainModes.SFT
        else:
            self._mode = TrainModes.RL
        del sft_mode    

        if use_curriculum:
            assert self._mode != TrainModes.SFT, "sft mode is not compatible with curriculum."

        self._answer_only            = answer_only # This is handled in the collator.
        self._eos_token              = eos_token
        self._use_curriculum         = use_curriculum
        self._curriculum_proportions = None

        self._extractor = lib_final_line.FinalLineExtractor(
            pad_token=pad_token,
            ignore_one_line=extractor_ignore_one_line,
        )

        split = lib_utils.CVSets(split)
        folder = self._dataset_root_folder_dir / f"{split.value}_scratch"
        assert folder.exists(), folder
        assert folder.is_dir(), folder
        glob_pattern = "*.jsonl"
        target_files = list(folder.glob(glob_pattern))
        assert target_files, (glob_pattern, folder, list(folder.iterdir()))
        
        self._max_num_digits = max(int(file.stem) for file in target_files)
        
        self._use_few_shots = use_few_shots
        if use_few_shots:
            assert not self._mode == TrainModes.SFT, self._mode
            self._few_shot_text = self._prepare_few_shots()
        else:
            self._few_shot_text = None

        #######################################################################
        if self._mode == TrainModes.SFT:
            self._setup_sft_mode(target_files)
            
        elif self._mode == TrainModes.RL:
            tok_name = any_tokenizer.name_or_path.replace("/", "_")
            hf_dataset_path = folder / f"{split}_{tok_name}_{self._answer_only}.hf_dataset"
            already_exists = (
                hf_dataset_path.exists() or 
                list(hf_dataset_path.parent.glob(hf_dataset_path.name + "*"))
            )

            if not use_cached_dataset or not already_exists:
                self._prepare_rl_mode_data(target_files, any_tokenizer)
                if use_cached_dataset:
                    print("SAVE RL MODE")
                    self._save_rl_mode_dataset(hf_dataset_path)
            else:
                self._load_cached_rl_mode_dataset(hf_dataset_path)        
        else:
            raise ValueError(self._mode)
        #######################################################################

        if shuffle_once:
            self._core.shuffle()


    def make_few_shot_toks(self) -> torch.Tensor:
        assert False, "Move this to the iterator."
        if not self._use_curriculum:
            raise NotImplementedError("Few shot only works with curriculum")
    
        assert not self._train_mode == TrainModes.SFT, self._train_mode

        per_digit = {}
        for num_digits, proportion in self._curriculum_proportions.items():
            rounded_prop = round(self._few_shot_qty * proportion)
            indices = np.random.permutation(self._few_shot_qty)[:rounded_prop]
            per_digit[num_digits] = [self._few_shot_text[num_digits][i] for i in indices]
        
        flattened_few_shots = list(it.chain.from_iterable(per_digit.values()))
        random.shuffle(flattened_few_shots)
        return "\n".join(flattened_few_shots).strip()
    
    @property
    def training_mode(self):
        return self._mode
        
    @property
    def few_shot_text(self) -> str:
        assert False
        return self._few_shot_text

    def _load_cached_rl_mode_dataset(self, hf_dataset_path):
        if RANK == 0:
            rich.print(f"[green bold]Loading hf_dataset from disk:[/]  {hf_dataset_path}")
        
        if self._use_curriculum:
            self._core = {}
            files = hf_dataset_path.parent.glob(hf_dataset_path.name + "*")
            for file in files:
                difficulty = int(file.name.rsplit("_", 1)[-1])
                data = datasets.load_from_disk(file)
                self._core[difficulty] = lib_base_classes.DataListContainer.from_list_of_items([
                    lib_base_classes.DataItemContainer(**data) for data in data
                ])
        else:
            self._core = lib_base_classes.DataListContainer.from_list_of_items([
                lib_base_classes.DataItemContainer(**data) for data in 
                datasets.load_from_disk(hf_dataset_path)
            ])


    def _save_rl_mode_dataset(self, hf_dataset_path: pathlib.Path):
        if RANK == 0:
            rich.print(f"[green bold]Saving hf_dataset to disk:[/] {hf_dataset_path}")

        print("Converting to basic types")

        if self._use_curriculum:
            progress = tqdm.tqdm(self._core.items())
            for dificulty_level, difficulty_data  in progress:
                assert isinstance(dificulty_level, int), (type(dificulty_level), dificulty_level)

                progress.set_description(f"Converting to basic types: {dificulty_level}")
                dataset = datasets.Dataset.from_list([x.to_dict() for x in difficulty_data])
                path = hf_dataset_path.parent / f"{hf_dataset_path.name}_{dificulty_level}"
                dataset.save_to_disk(path)
        else:
            dataset = datasets.Dataset.from_list([x.to_dict() for x in self._core])
            path = hf_dataset_path.parent / f"{hf_dataset_path.name}"
            dataset.save_to_disk(path)


    def _prepare_rl_mode_data(self, target_files, any_tokenizer):
        rich.print("[bold blue]_prepare_rl_mode_data:[/]", target_files)

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
                            f"Q: {sample['input']}\n" +
                            f"{sample['scratchpad'].strip()}\n" +
                            "A:"
                        )
                    else:
                        sample["question"] = (
                            f"Q: {sample['input']}\n" +
                            "<scratch>"
                        )

                    difficulty_level = sample["num_digits"]

                    # sample = _tok_detok(
                    #     any_tokenizer = any_tokenizer, 
                    #     batch         = sample, 
                    #     batched       = False,
                    #     exclusion_set = {"num_digits"}
                    # )
                
                    # self._core.tok_ref_query       .append(None)
                    # self._core.tok_ref_answer      .append(None)
                    # self._core.tok_ref_scratchpad  .append(None)

                    self._core.detok_ref_query     .append(sample[  "question"])
                    self._core.detok_ref_answer    .append(sample[    "answer"])
                    self._core.detok_ref_scratchpad.append(sample["scratchpad"])
                    
                    self._core.extra_information   .append({})
                    self._core.difficulty_level    .append(difficulty_level)

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
            with jsonl.open(file) as f:
                for sample in tqdm.tqdm(f, desc="Building DictDataset"):
                    scratchpad = sample["scratchpad"] + "\n" + sample["answer"]
                    self._core.append(dict(
                        ref_qa_question   = sample["input"],
                        ref_qa_answer     = sample["answer"],
                        ref_qa_scratchpad = scratchpad, 
                        num_digits        = int(sample["num_digits"]),
                ))

    @property
    def max_num_digits(self):
        return self._max_num_digits

    def _prepare_few_shots(self):
        self._few_shot_qty = 10
        few_shots = arithmetic_10_shot.make_few_shots(
            max_digits=self.max_num_digits,
            num_per=self._few_shot_qty,
            root=self._dataset_root_folder_dir,
        )
        
        few_shot_text = collections.defaultdict(list)
        for l, entries in few_shots.items():
            for entry in entries:
                few_shot_text[l].append(
                    f"Q: {entry['input']}\n"
                    f"{entry['scratchpad'].strip()}\n"
                    f"A:\n" +
                    str(entry['answer']) + "\n" +
                    "!\n"
                )

        return few_shot_text

    @property
    def use_few_shots(self):
        return self._use_few_shots

    def __iter__(self):
        return DSIterator(
            ds=self._core, 
            use_curriculum=self._use_curriculum,
            return_idx=self._return_idx,
            use_few_shots=self._use_few_shots,
        )

    def get_extractor(self):
        return self._extractor

    def post_process_gen_fewshots(self, *, raw_gen_outputs, any_tokenizer) -> str:
        text_list = any_tokenizer.batch_decode(raw_gen_outputs)
        return [
            torch.tensor(x, dtype=torch.long, device=torch.device(LOCAL_RANK)) 
            for x in any_tokenizer(text_list)["input_ids"]
        ]

    def set_proportion_difficulties(self, proportions):
        raise DeprecationWarning("Call it in the iterator instead.")
        self._curriculum_proportions = proportions
        self._inner_iterator.set_proportion_difficulties(proportions)


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


def main(use_curriculum=True):
    forward_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Open-Orca/Mistral-7B-OpenOrca"
    )
    path = (
        "/home/mila/g/gagnonju/marglicot/with_trl/libs_data/arithmetic/"
    )

    ds = Arithmetic(
        answer_only               = False,
        dataset_root_folder_dir   = path,
        eos_token                 = forward_tokenizer.eos_token,
        extractor_ignore_one_line = False,
        pad_token                 = forward_tokenizer.pad_token,
        split                     = lib_utils.CVSets.TRAIN,
        shuffle_once              = False,
        
        sft_mode                  = True,
        use_few_shots             = False,
        use_curriculum            = use_curriculum,
        use_cached_dataset        = True,

        any_tokenizer             = forward_tokenizer,
        return_idx                = False,
    )

    iterator_ = iter(ds)

    if use_curriculum:
        iterator_.set_proportion_difficulties({
            1: 0.5,
            2: 0.5,
        })

    for i, _ in enumerate(iterator_):
        print(i)


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


    # rich.print(ds[0]["detok_ref_query"])

if __name__ == "__main__":
    fire.Fire(main)