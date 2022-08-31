#!/usr/bin/env python3
# coding: utf-8

print("Doing imports")
import collections
import fire
import json
from pathlib import Path
import pickle
import rich
import time
from typing import *

import jsonlines as jsonl  # type: ignore
from tqdm import tqdm  # type: ignore

print("Done with imports")


SPLIT_CONVERSION_MAP = dict(
    train="training",
    eval="validation",
)

def main(
    name="349_6_6_200000",
    source_data_folder="../SelfLearnedExplanations/data",
    target_data_root="data/basic_arithmetic/",
):

    source_data_folder = Path(source_data_folder)
    target_data_root = Path(target_data_root)

    target_data_folder = target_data_root / name
    data_path = source_data_folder / f"{name}.json.pkl"

    rich.print(list(source_data_folder.iterdir()))

    # Load the pickled data
    rich.print("\n[bold]Loading the pickled data.")
    rich.print(f"From path: \"{data_path}\".")
    start = time.perf_counter()
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    rich.print(f"Done loading data: {time.perf_counter() - start:0.2f} seconds.")

    # Save the config file
    rich.print("\n[bold]Saving the config file.")
    rich.print(f"Output folder: \"{target_data_folder}\"")
    target_data_folder.mkdir(exist_ok=True)
    with open(target_data_folder / "config.json", "w") as f:
        json.dump({k: v for k, v in data["config"].items() if k != "output_name"}, f)

    # Prepare the entries
    rich.print("\n[bold]Preparing entries.")
    output = collections.defaultdict(list)
    for split_name, split_data in tqdm(data["data"].items()):
        for level_idx, level_data in tqdm(split_data.items()):
            for i, entry in enumerate(level_data):
                new_entry = {}
                new_entry["input"] = entry["input_str"]
                new_entry["value"] = entry["value"]
                new_entry["scratchpad"] = entry["oracle_without_top_val"]
                new_entry["scratchpad_with_value"] = entry["oracle_str"]
                new_entry["level"] = level_idx
                output[split_name].append(new_entry)

    # Write the data
    rich.print("\n[bold]Writing data.")
    for split in output:
        output_split_name = SPLIT_CONVERSION_MAP[split]
        with jsonl.open(target_data_folder / f"{output_split_name}.jsonl", mode="w") as writer:
            for entry in output[split]:
                writer.write(entry)

if __name__ == "__main__":
    fire.Fire(main)