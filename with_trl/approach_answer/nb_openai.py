#!/usr/bin/env python
# coding: utf-8

import collections
import concurrent.futures
import itertools as it
import pathlib
import time
import sys
sys.path.append("..")

import fire
import jsonlines as jsonl
import more_itertools as mit
import openai
import torch
from tqdm import tqdm

import data_few_shot_commonsense_qa_scratchpads
import lib_data_commonsense_qa

MODEL = "gpt-3.5-turbo"
DEFAULT_PATH_SECRET = "/home/mila/g/gagnonju/openai/openai.txt"

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

def inference(*, split, directory, n):
    directory = pathlib.Path(directory)
    path = directory / f"commonsenseqa.chatgpt.{split}.jsonl"
    print(f"{split}: {path}")
    assert not path.exists(), path

    with jsonl.open(path, "w", flush=True) as f:
        data = lib_data_commonsense_qa.CommonSenseScratchpadGenMC(
            any_tokenizer=None, 
            split=split, 
            text_only=True,
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            arguments = list(it.islice(data, n))
            imap = executor.map(work, arguments)
            
            for output in tqdm(imap, total=n if n is not None else len(data)):
                f.write(output)
    


def main(
        output_dir="/network/scratch/g/gagnonju/scratchpad_gen_outputs/",
        n=4,
        secret_path=DEFAULT_PATH_SECRET,
        ):
    args = locals().copy()
    for k, v in args.items():
        print(f"{k}: {v}")

    output_dir = pathlib.Path(output_dir)
    assert output_dir.exists()

    with open(secret_path, "r") as fin:
        openai.api_key = fin.read().strip()

    for split in ["train", "validation"]:
        inference(split=split, directory=output_dir, n=n)


if __name__ == "__main__":
    fire.Fire(main)

