#!/usr/bin/env python
# coding: utf-8

import collections
import concurrent.futures
import itertools as it
import logging
import pathlib
import time
import sys

import fire
import jsonlines as jsonl
import more_itertools as mit
import openai
import rich
import rich.table
import rich.markup
import rich.status
from tqdm import tqdm

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR / "Marg-Li-CoT/with_trl/approach_answer/"))
sys.path.append(str(SCRIPT_DIR / "Marg-Li-CoT/with_trl/"))
print(" - " + "\n - ".join(sorted(sys.path)))
import lib_data_commonsense_qa

import threading
MODEL = "gpt-3.5-turbo"
DEFAULT_PATH_SECRET = SCRIPT_DIR / "openai.txt"
WAIT_TIME = 60
N_THREADS = 1

def work(sample):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": sample["ref_fs_scratchpad_gen_query"]}],
        temperature=0,
        max_tokens=1300,
        stop=["\n"],
    )

    output = mit.one(response.choices)["message"]["content"]
    copy = sample.copy()
    assert "output" not in copy
    copy["output"] = output
    return copy

def _query_one_error_handler(error, thread_name):
    error_name = type(error).__name__
    header = f"{error_name} - {thread_name}"
    rich.print(f"[bold red on white]{header}: Waiting {WAIT_TIME} seconds")
    time.sleep(WAIT_TIME)
    rich.print(f"[bold green on white]{header}: Retrying")


def query_one(sample):
    thread_name = threading.current_thread().getName()
    prompt = sample["ref_fs_scratchpad_gen_query"]
    
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1300,
                stop=["\n"],
            )
        except openai.error.ServiceUnavailableError as e:
            _query_one_error_handler(e, thread_name)
            continue
        except openai.error.RateLimitError as e:
            _query_one_error_handler(e, thread_name)
            continue
        break

    output = mit.one(response.choices)["message"]["content"]
    return dict(**sample, output=output)


def _check_matches_setup_resuming(dataset_sample, saved_sample):
    for k in dataset_sample:
        assert dataset_sample[k] == saved_sample[k], (k, dataset_sample[k], saved_sample[k])


def setup_resuming(*, data, output_path, split):
    with jsonl.open(output_path, "r") as f:
        existing = list(f)
    existing_dict = {x["ref_qa_id"]: x for x in existing}

    rich.print(f"[bold blue on white]({split}): Checking existing data.")
    for dataset_sample in it.islice(data, len(existing_dict)):
        saved_sample = existing_dict[dataset_sample["ref_qa_id"]]
        _check_matches_setup_resuming(
            dataset_sample=dataset_sample, 
            saved_sample=saved_sample,
        )
        
    rich.print(f"[bold green on white]({split}): Did {len(existing)} of {len(data)} already.")

    for dataset_sample, saved_sample in mit.zip_equal(it.islice(data, len(existing)), existing):
        _check_matches_setup_resuming(
            dataset_sample=dataset_sample, 
            saved_sample=saved_sample,
        )

    rich.print(f"[bold green on white]({split}): Did {len(existing)} of {len(data)} already.")

    return len(existing)


def inference_on_split(*, split, directory, test_run):
    directory = pathlib.Path(directory)
    path = directory / f"commonsenseqa.chatgpt.{split}.jsonl"
    
    print(f"{split}: {path}")

    data = lib_data_commonsense_qa.CommonSenseScratchpadGenMC(
        any_tokenizer=None, 
        split=split, 
        text_only=True,
    )

    if path.exists():
        initial_count = setup_resuming(output_path=path, data=data, split=split)
    else:
        initial_count = 0

    rich.print(f"[bold blue on white]({split}): Starting inference.")
    if not test_run:
        with jsonl.open(path, "a", flush=True) as f:
            for sample in tqdm(
                it.islice(data, initial_count, None, 1),
                total=len(data),    
                initial=initial_count,
            ):
                output = query_one(sample)
                f.write(output)
        
def main(
        output_dir=SCRIPT_DIR / "outputs",
        secret_path=DEFAULT_PATH_SECRET,
        test_run=False,
    ):
    
    args = locals().copy()
    table = rich.table.Table(
        "Key", 
        "Value", 
        title="[bold blue on white]Arguments:", 
        show_lines=True,
    )

    for k, v in args.items():
        table.add_row(
            "[bold]" + rich.markup.escape(str(k)), 
            rich.markup.escape(str(v)),
        )
    rich.print(table)

    output_dir = pathlib.Path(output_dir)
    assert output_dir.exists(), output_dir
    assert output_dir.is_dir(), output_dir

    with rich.status.Status("[bold blue on white]Reading secret key."):
        with open(secret_path, "r") as fin:
            openai.api_key = fin.read().strip()

    for split in ["train", "validation"]:
        rich.print(f"[bold blue on white]Running split [green]{split}")
        inference_on_split(split=split, directory=output_dir, test_run=test_run)


if __name__ == "__main__":
    fire.Fire(main)

