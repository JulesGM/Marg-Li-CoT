import collections
import os
import pathlib
import sys

import datasets
import fire
import jsonlines as jsonl
import rich
import rich.box
import rich.traceback

rich.traceback.install()

import approach_sft.lib_sft_constants as lib_sft_constants
from with_trl import libs_extraction
from with_trl import lib_utils

datasets.disable_caching()
RANK = int(os.environ.get("RANK", 0))


def openai_commonsense_qa_output(root_path, filter_bads: bool):

    path = pathlib.Path(root_path)
    assert path.exists(), f"{path} does not exist"
    assert path.is_dir(), f"{path} is not a directory"

    paths = {
        lib_sft_constants.CVSet.TRAIN: path/"commonsenseqa.chatgpt.train.jsonl",
        lib_sft_constants.CVSet.VALIDATION: path/"commonsenseqa.chatgpt.validation.jsonl",
    }

    for split, path in paths.items():
        assert path.exists(), f"{split}: {path} does not exist"
        assert path.is_file(), f"{split}: {path} is not a file"

    # Read the data
    data = {}
    for split, path in paths.items():
        with jsonl.open(path) as f:
            data[split] = list(f)

    extractor = libs_extraction.lib_multiple_choice.MultipleChoiceRfindExtractor(
        choices=["(A)", "(B)", "(C)", "(D)", "(E)"])
    
    data_by_columns = collections.defaultdict(lambda: collections.defaultdict(list))
    for split_name, split_data in data.items():
        is_train = split_name == lib_sft_constants.CVSet.TRAIN
        
        for d in split_data:
            answer_good = extractor(d["output"]) == d["ref_qa_answer"]
            if filter_bads and is_train and not answer_good:
                continue

            for k, v in d.items():
                data_by_columns[split_name][k].append(v)

        # Make sure all the lists are the same length
        len_first = len(next(iter(data_by_columns[split_name].values())))
        assert all(len(x) == len_first for x in data_by_columns[split_name].values()), (
            split_name, [len(x) for x in data_by_columns[split_name].values()])

        final_qty = len_first
        init_qty = len(split_data)
        kept_ratio = final_qty / init_qty
        assert kept_ratio > 0.7, f"{kept_ratio:.1%} is too low for {split_name}, this is probably a bug"

        if RANK == 0:
            if filter_bads:
                rich.print(f"[bold blue]{split_name}:[/] Kept {final_qty} / {init_qty}, which is {kept_ratio:.1%}")
            else:
                rich.print(f"[bold blue]{split_name}:[/] Kept all {final_qty} / {init_qty} examples, since filter_bads=False")

    dict_datasets = {}
    for split, split_data in data_by_columns.items():
        dict_datasets[split] = lib_utils.DictDataset(split_data)

    return dict_datasets


def main(
    filter_bads=True,
):
    import approach_sft.lib_sft_collators as lib_sft_collators
    import lib_trl_utils
    import transformers
    
    data_directory = "/network/scratch/g/gagnonju/saved_scratchpad_gen_outputs/chatgpt-3.5-commonsenseqa-scratchpads/not-cond-on-answers"
    model_name_or_path = "EleutherAI/pythia-410m"
    output_type = lib_sft_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER

    tmp_tokenizers = lib_trl_utils.load_tokenizers(
        model_name=model_name_or_path, 
        config=transformers.AutoConfig.from_pretrained(model_name_or_path),
    )

    forward_tokenizer = tmp_tokenizers["forward_tokenizer"]
    prediction_tokenizer = tmp_tokenizers["prediction_tokenizer"]
    del tmp_tokenizers

    openai_commonsense_qa_output(
        data_directory, filter_bads=filter_bads,
    )
    lib_sft_collators.CausalFullCollator(
        output_type=output_type,
        forward_tokenizer=forward_tokenizer,
        prediction_tokenizer=prediction_tokenizer,
    )


if __name__ == "__main__":
    fire.Fire(main)