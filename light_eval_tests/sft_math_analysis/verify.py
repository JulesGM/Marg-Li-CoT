import copy

import pathlib
import datasets
from ast import literal_eval
import more_itertools
import pandas as pd
import numpy as np
import gc
import tqdm
import polars as pl
import re
import rich
import rich.table
import functools
import rich.console
import enum
import sys
import itertools
import json
import hashlib
import collections
import time
import numpy as np
import contextlib
import os
os.environ["OPENINSTRUCT_PARSE_LATEX_BACKEND"] = "lark" 

sys.path.append("/home/mila/g/gagnonju/marglicot/with_open-instruct/open-instruct")
from open_instruct.math_utils import (
    last_boxed_only_string,
    remove_boxed,
    get_unnormalized_answer,
    normalize_final_answer,
    is_equiv,
    hendrycks_is_equiv
)

class Mode(enum.Enum):
    gsm8k = "gsm8k"
    math = "math"

class LearningType(enum.Enum):
    sft = "sft"
    rejection = "rejection"
    zero_shot = "zero_shot"
    few_shot = "few_shot"



def verify_math_sample(model_output, ground_truth_answer):
    ground_truth_answer = last_boxed_only_string(ground_truth_answer)
    if ground_truth_answer is not None:
        try:
            ground_truth_answer = remove_boxed(ground_truth_answer)
        except AssertionError:
            ground_truth_answer = None
    if ground_truth_answer is None:
        raise NotImplementedError(f"Bad ground truth: {ground_truth_answer}")

    raw_answer = model_output
    # for math, more complex. We will try a few different ways to extract the answer.
    # this roughly follows 'flex em' in oe-eval-internal
    all_answers = []
    # First, try find answer in \boxed{}.
    boxed_answer = last_boxed_only_string(raw_answer)
    if boxed_answer is not None:
        try:
            boxed_answer = remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None
    if boxed_answer is not None:
        all_answers.append(boxed_answer)
    # Second, try to extract via minerva format.
    minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)
    # If nothing still, try to find the last latex-formatted answer
    if len(all_answers) == 0:
        dollars = [m.start() for m in re.finditer("\\$", raw_answer)]
        if len(dollars) > 1:
            # Add the answer between the second to last and last dollar sign
            answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
            all_answers.append(answer)
    # otherwise, just take the full output. Probably wont work, bit of a yolo.
    if len(all_answers) == 0:
        all_answers.append(normalize_final_answer(model_output))
    # now, compare all answers to ground truth.
    matched = False

    for answer in all_answers:
        if is_equiv(answer, ground_truth_answer):
            matched = True
            break
        elif hendrycks_is_equiv(answer, ground_truth_answer):
            matched = True
            break
    # if we got any match, we are good.
    return matched, all_answers



def verify_gsm8k_sample(model_output, ground_truth_answer, verbose=False):
    # model_output = model_output.split("<|assistant|>\n")[-1].strip()
    # gsm is easy: extract numbers, and then just compare last number with answer.
    # matches how we do eval.
    predictions = None
    # replace numbers like `x,xxx` with `xxxx`
    response = re.sub(r"(\d),(\d)", r"\1\2", model_output)
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", response)
    if numbers:
        predictions = numbers[-1]
    else:
        predictions = response
    if verbose:
        print(f"predictions: {predictions}, ground_truth_answer: {ground_truth_answer}")
    return str(predictions).lower() == str(ground_truth_answer).lower(), predictions


pattern = re.compile(r"-?(\d{1,3}(,\d{3})*|\d+)")


def extract_answer_predicted(text):
    found = more_itertools.last(pattern.findall(text), None)
    return found


def extract_answer_gold(text):
    return text.rsplit("####", 1)[-1].strip()


def get_of_expected(container, idx, expected_size: int):
        assert len(container) == expected_size
        return container[idx]


functools.lru_cache(maxsize=None)
def load_parquet(path):
    return pl.read_parquet(path)




class FractionTimeSpent:
    def __init__(self):
        self._data = collections.defaultdict(list)
        self._start_time = collections.defaultdict(float)

    def start(self, key: str):
        self._start_time[key] = time.perf_counter()

    def stop(self, key: str):
        self._data[key].append(time.perf_counter() - self._start_time[key])
        self._start_time[key] = None

    @contextlib.contextmanager
    def time_block(self, key: str):
        self.start(key)
        yield
        self.stop(key)

    def get(self):
        sum_ = 0
        means = {}
        for key in self._data:
            means[key] = np.mean(self._data[key])
            sum_ += means[key]

        normalized_data = {}
        for key in self._data:
            normalized_data[key] = means[key] / sum_

        return normalized_data, means


def compute_score(path, mode: Mode, time_spent: FractionTimeSpent, compute_score: bool, subset_qty: int | None = None):

    if mode == Mode.gsm8k:
        verify = verify_gsm8k_sample
        extract = extract_answer_gold
    elif mode == Mode.math:
        verify = verify_math_sample
        extract = lambda x: x
    else:
        raise ValueError(f"Invalid mode: {MODE}")

    with time_spent.time_block(key="read_parquet"):
        ds = load_parquet(path)

    with time_spent.time_block("convert_to_series"):
        parsed_predictions = pl.Series([
            more_itertools.one(get_of_expected(more_itertools.one(literal_eval(pred)), 0, 2)) 
            for pred in ds["predictions"]
        ])
        gold = pl.Series([more_itertools.one(literal_eval(pred)) for pred in ds["gold"]])
        original_score = pl.Series([literal_eval(x)["qem"] for x in ds["metrics"]])


    results_root = path.parent.parent.parent.parent / "results" 
    assert results_root.exists(), results_root
    second_part = results_root / path.relative_to(results_root.parent / "details").parent.parent
    assert second_part.exists(), second_part
    assert (second_part / "meta_info.json").exists(), second_part / "meta_info.json"
    meta_info = json.loads((second_part / "meta_info.json").read_text())

    if compute_score:
        with time_spent.time_block("verify"):
            ongoing = []
            extracted_predictions = []  
            extracted_golds = []
            is_equal = []

            for i, (generated, gold_individual) in enumerate(more_itertools.zip_equal(parsed_predictions, gold)):
                if subset_qty is not None and i >= subset_qty:
                    break
                if i % 1000 == 0:
                    print(f"i: {i / len(parsed_predictions):0.1%}")
                extracted_gold_i = extract(gold_individual).replace(",", "")
                
                # Check if we get the same answer for the gold from the verify function 
                # and from the reference gold. If not, this is a bug.
                # is_equal_golds, test_extracted_gold_as_pred = verify(
                #     model_output=extracted_gold_i, 
                #     ground_truth_answer=extracted_gold_i
                # )
                
                # if extracted_gold_i != test_extracted_gold_as_pred:
                #     print(f"extracted_gold_i: {extracted_gold_i}, test_extracted_gold_as_pred: {test_extracted_gold_as_pred}, is_equal: {is_equal_golds}")

                # Actually verify the prediction.
                verify_output, extracted_prediction_i = verify(model_output=generated, ground_truth_answer=extracted_gold_i)
                ongoing.append(extracted_prediction_i is not None and extracted_gold_i is not None and verify_output)
                extracted_predictions.append(extracted_prediction_i)
                extracted_golds.append(extracted_gold_i)
                is_equal.append(verify_output)
            score = np.mean(ongoing)
    
        main_output = pl.DataFrame(
            {
                "epoch": meta_info["epoch"], 
                "learning_rate": meta_info["cfg"]["learning_rate"],
                "score": score, 
                "original_score": original_score.mean(),
                "path": str(path), 
            }
        )
        predictions_output = pl.DataFrame({"predictions": parsed_predictions[:subset_qty], "gold": gold[:subset_qty], "extracted_predictions": extracted_predictions[:subset_qty], "extracted_golds": extracted_golds[:subset_qty], "is_equal": is_equal[:subset_qty]})
    else:
        main_output = pl.DataFrame({"original_score": original_score.mean(), "learning_rate": meta_info["cfg"]["learning_rate"], "epoch": meta_info["epoch"], "path": str(path)})
        predictions_output = None

    return main_output, predictions_output
