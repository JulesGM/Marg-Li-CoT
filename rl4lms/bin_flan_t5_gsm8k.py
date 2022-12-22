#!/usr/bin/env python
# coding: utf-8


"""
This is basic code to do inference with Flan-T5-* on zero or few-shot chain of thought problems.
The goal is to use this as as benchmark, to see what to expect of each model sizes, for each dataset,
with and without scratchpads, with and without majority vote, with and without group beam search,
for different number of shots, etc.

A few decisions.

###########################################################
# Parsing the answer:
###########################################################
We take the answer by taking the last integer in the answer string.
If we don't find an integer, we call word2num on the answer string, to maybe get a word form number.
If we don't find a number, in the majority vote case, we ignore that generation in the vote.
If none of the generations have a number, we return the default value.
Similarily, in the case without majority vote, if we don't find a number, we return the default value.

It would be fair to say that we should maybe not use the default value, and just assign the answer as incorrect.
We may do that eventually. It would be pretty straightfoward, just change the default value to a stopper object,
the comparison with the reference answer would always be false.

Floating point numbers:
We currently round to the closest integer.


###########################################################
# Picking questions in the dataset:
###########################################################
More than 85% of the questions in the ASDiv dataset have a single, integer answer.
We only consider those questions, as they make the task of parsing the answer easier.


###########################################################
# 
###########################################################

"""


import collections
import enum

import importlib
import itertools
import logging
import math
import more_itertools
import os
from pathlib import Path
import random
import re
import time
import xml

import datasets
import deepspeed
import matplotlib.pyplot as plt
import numpy as np
import rich
import rich.table
from text2digits import text2digits
import torch
from tqdm import tqdm
import transformers
import wget
import rl4lms.data_pools.text_generation_pool as rl4lms_pool

import general_utils as utils
import dataset_asdiv
import dataset_gsm8k
from our_scratchpad import bin_deepspeed_experim


LOGGER = logging.getLogger(__name__)
MODEL_NAME = "google/flan-t5-xl"

dataset_to_use = "asdiv"
parallelism = "deepspeed"

model_precision = torch.bfloat16
do_autocast = False
use_dp = True

verbose = True
shuffle = True
n_shots = 0
num_beams = 1
batch_size = 8
max_new_tokens = 197
with_scratchpads = True
use_majority_vote = False
use_group_beam_search = False
few_shot_context_rng_seed = 42  # Makes sure the context is the same if we want it to stay the same
generation_extra_kwargs = dict(repetition_penalty=0.5)

if use_group_beam_search:
    generation_extra_kwargs["diversity_penalty"] = 1.


###############################################################################
###############################################################################

text2digits_ = text2digits.Text2Digits()
num_pat = re.compile(r"\d+(?:[\,\.]\d+)?")

def deal_with_words(text):
    converted = text2digits_.convert(text)


    all_found = num_pat.findall(converted)
    
    if not all_found:
        raise ValueError(f"Could not find any numbers in `{converted}`. Original text: `{text}`.")

    output = all_found[-1]
    rich.print(
        f"[bold blue]text2digits[/]:\n"
        f" \t - [green]source:[/]    {text}\n"
        f" \t - [green]converted:[/] {converted}\n"
        f" \t - [green]final:[/]     {output}"
    )
    return output 


def split_fn(x):
    results = num_pat.findall(x)
    if not results:
        try:
            output = deal_with_words(x)
        except ValueError:
            output = None

        if output is not None:
            rich.print(f"[red]split_fn: no numbers found. Received:[/] `{x}`. Text2Digit worked. Output: `{output}`")
            output = str(output)
        else:
            rich.print(f"[red]split_fn: no numbers found. Received:[/] `{x}`")
            output = None
    else:
        output = results[-1]
    return output


class ContextGeneration:
    """
    
    Namespace

    """

    chain_of_thought_intro = "Let's think about it step by step. Chain-of-thought:"
    answer_intro           = "Answer:"
    question_intro         = "Question:"

    @classmethod
    def compose_fewshot_context(cls, dataset, n: int, with_scratchpad: bool, seed: int):
        """ 
        Creates a random few-shot context. Works fine with n = 0.
        """
        if n == 0:
            return ""

        rng = random.Random(seed)
        indices = rng.sample(range(len(dataset)), n)

        output = []
        for i in indices:
            scratchpad = dataset[i]["scratchpad"]
            answer = dataset[i]["answer"]

            assert "#" not in scratchpad, scratchpad
            assert "#" not in answer, answer

            text = "Question: " + dataset[i]["question"]
            if with_scratchpad:
                text += f" {cls.chain_of_thought_intro} " + scratchpad
            
            text += f" {cls.answer_intro} " + answer
            output.append(text)

        return " ".join(output)

    @classmethod
    def collate(cls, inputs, tokenizer, few_shot_context, with_scratchpad):
        """ Collates the inputs and prompts into a single list of strings. """

        first_context_addition = few_shot_context + f" {cls.question_intro} "
        final_context_addition = f" {cls.chain_of_thought_intro} " if with_scratchpad else f" {cls.answer_intro} "

        reformatted_samples = dict(question=[], answer=[], scratchpad=[])
        for entry in inputs:
            assert isinstance(entry, tuple), type(entry)
            assert len(entry) == 2, len(entry)
            sample = entry[0]
            assert isinstance(sample, rl4lms_pool.Sample), type(sample)
            assert len(sample.references) == 1, len(sample.references)
            reformatted_samples["question"  ].append(sample.prompt_or_input_text)
            reformatted_samples["scratchpad"].append(sample.meta_data["ref_scratchpad"])
            reformatted_samples["answer"    ].append(sample.references[0])

        question_text = [
            first_context_addition + question + final_context_addition 
            for question in reformatted_samples["question"]
        ]

        output = tokenizer(
            question_text,
            padding=True,
            return_tensors="pt"
        ) 

        output["answer"] = reformatted_samples["answer"]
        output["scratchpad"] = reformatted_samples["scratchpad"]
        
        return {
            k: v.to(int(os.getenv("LOCAL_RANK", "0"))) if isinstance(v, torch.Tensor) else v 
            for k, v in output.items()
    }


###############################################################################
# Dataset stuff
###############################################################################
def format_output(output):
    try:
        float_conversion = float(output.replace(",", ""))
    except (ValueError, ArithmeticError) as err:
        rich.print(
            f"[bold red]Failed to convert to float. "
            f"value is:[/] `{output}`, "
            f"[bold red]error is:[/] {type(err)} {err}"
        )
        return "0"
    
    rounding = round(float_conversion)
    return str(rounding)


def majority_vote(generated, tokenizer, answer_extraction_fn, verbose):
    answers = []
    for entry in generated:
        decoded = tokenizer.decode(entry, skip_special_tokens=True)
        output = answer_extraction_fn(decoded)
        if output is not None:
            answers.append(format_output(output))
    
    counter = collections.Counter(answers)
    if verbose:
        print(counter)
    if counter:
        return counter.most_common(1)[0][0]
    else:
        return "0"


def majority_vote_batch(generated, tokenizer, answer_extraction_fn, verbose):
    for entry in generated:
        yield majority_vote(entry, tokenizer, answer_extraction_fn, verbose)


def compare(pred, answ):
    return format_output(pred.strip()) == format_output(answ.strip())


class DatasetChoices(str, enum.Enum):
    gsm8k = "gsm8k"
    asdiv = "asdiv"

class ParallelismChoices(str, enum.Enum):
    naive_mp = "naive_mp"
    deepspeed = "deepspeed"
    no_parallelism = "none"


def run(
    *,
    parallelism,
    do_autocast,
    model_precision,
    shuffle,
    verbose,
    num_beams,
    batch_size,
    max_new_tokens,
    with_scratchpads,
    use_majority_vote,
    use_group_beam_search,
    generation_extra_kwargs,
    which_dataset_to_use: DatasetChoices,
    max_sum_squares=None, # 41957
):
    args = locals().copy()
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "0"))

    logging.basicConfig(
        level=logging.INFO, 
        format=f"[{global_rank + 1} / {world_size}]:\t%(message)s", 
        datefmt="[%X]", 
        handlers=[rich.logging.RichHandler(markup=True, rich_tracebacks=True)]
    )

    # The tokenizer is required to do length filtering of the datasets.
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    if which_dataset_to_use == DatasetChoices.asdiv:    
        dataset_train = dataset_asdiv.ZeroShotASDivTextGenPool.prepare("train")
        dataset_test  = dataset_asdiv.ZeroShotASDivTextGenPool.prepare("test")

    elif which_dataset_to_use == DatasetChoices.gsm8k:
        dataset_train = dataset_gsm8k.ZeroShotGSM8KTextGenPool.prepare("train", tokenizer, max_sum_squares=max_sum_squares)
        dataset_test  = dataset_gsm8k.ZeroShotGSM8KTextGenPool.prepare("test", tokenizer, max_sum_squares=max_sum_squares)

        print(dataset_train[0].keys())
        print(dataset_test [0].keys())
    else:
        raise ValueError(f"Unknown dataset: {which_dataset_to_use}, should be one of {list(DatasetChoices)}")

    context = ContextGeneration.compose_fewshot_context(
            dataset_train, 
            n_shots, 
            with_scratchpads, 
            few_shot_context_rng_seed,
        )

    rich.print(
        f"[bold blue]Context[/]:\n" +
        context
    )

    ###############################################################################
    # Load the model
    ###############################################################################

    rich.print(f"[bold blue]Loading the model [bold white]{MODEL_NAME}")
    with utils.ctx_timeit(f"Loading model `{MODEL_NAME}`"):
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=model_precision)

    if parallelism == ParallelismChoices.naive_mp:
        LOGGER.info("[bold red]Initializing with parallelize")
        model.parallelize()
    elif parallelism == ParallelismChoices.deepspeed:
        LOGGER.info("[bold red]Initializing deepspeed")
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        _, config_inference = bin_deepspeed_experim.make_deepspeed_config(batch_size=batch_size, wandb_config=None)
        model = deepspeed.init_inference(model=model, config=config_inference)
    elif parallelism == ParallelismChoices.no_parallelism:
        with utils.ctx_timeit(f"Moving model to GPU"):
            model = model.cuda()
    else:
        raise ValueError(f"Unknown parallelism: {parallelism}, should be one of {list(ParallelismChoices)}")

    rich.print(f"[bold blue]Model dtype:[/]  {model}")

    with torch.inference_mode():
        dataloader = torch.utils.data.DataLoader(
            dataset_test,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=lambda inputs: ContextGeneration.collate(
                inputs, tokenizer, context, with_scratchpads
            )
        )

        outputs = []
        tqdm_obj = tqdm(dataloader)

        extra_kwargs = generation_extra_kwargs.copy()
        if use_group_beam_search:
            extra_kwargs["num_beam_groups"] = num_beams

        for batch in tqdm_obj:
            if do_autocast:
                ctx_obj = torch.amp.autocast(device_type="cuda", dtype=model_precision)
                ctx_obj.__enter__()

            # utils.check_equal(batch["input_ids"].shape[0], batch_size)
            utils.check_equal(batch["input_ids"].shape[0], batch["attention_mask"].shape[0])
            utils.check_equal(batch["input_ids"].shape[1], batch["attention_mask"].shape[1])

            output = model.generate(
                input_ids            = batch["input_ids"],
                attention_mask       = batch["attention_mask"],
                num_beams            = num_beams,
                num_return_sequences = num_beams if use_majority_vote else 1,
                max_new_tokens       = max_new_tokens,
                **extra_kwargs
            ).reshape(batch_size, num_beams if use_majority_vote else 1, -1)

            if do_autocast:
                ctx_obj.__exit__(None, None, None)

            predictions = list(majority_vote_batch(output, tokenizer, split_fn, verbose))
            raw_decoded = [
                [tokenizer.decode(x, skip_special_tokens=True) for x in batch_entry] 
                for batch_entry in output
            ]

            answer_decoded = [list(map(split_fn, x)) for x in raw_decoded]

            if verbose:
                for prediction, answer, raw_decoded_entry, answer_decoded, input_text in zip(
                    predictions, 
                    batch["answer"], 
                    raw_decoded, 
                    answer_decoded, 
                    [tokenizer.decode(x) for x in batch["input_ids"]]
                ):

                    rich.print(
                        f"[bold blue]input_text[/]:      {input_text}\n"
                        f"[bold blue]ref_answer[/]:      {answer}\n"
                        f"[bold blue]prediction[/]:      {prediction}\n"
                        # f"[bold blue]raw_decoded[/]:     {raw_decoded_entry}\n"
                        f"[bold blue]answer_decoded[/]:  {answer_decoded}"
                    )
                    rich.print("[bold blue]Raw Decoded:[/]")
                    for v in raw_decoded_entry:
                        rich.print(f" [bold]-[/] {v}")

            good_bad_preds = [
                compare(pred=pred, answ=answ) 
                for pred, answ in zip(predictions, batch["answer"])
            ]

            # print([(format_output(a), format_output(b), a, b) for a, b in zip(predictions, batch["answer"])])
            # print(good_bad_preds)

            outputs.extend(good_bad_preds)
            tqdm_obj.set_description(f"Accuracy: {np.mean(outputs):.1%}")
            
        accuracy = np.mean(outputs)

        rich.print(args)
        rich.print(f"[bold green]Accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    run(
        parallelism=parallelism,
        model_precision=model_precision,
        do_autocast=do_autocast,
        shuffle=shuffle,
        verbose=verbose,
        num_beams=num_beams, 
        batch_size=batch_size, 
        max_new_tokens=max_new_tokens, 
        with_scratchpads=with_scratchpads, 
        use_majority_vote=use_majority_vote, 
        use_group_beam_search=use_group_beam_search,
        generation_extra_kwargs=generation_extra_kwargs,
        which_dataset_to_use=dataset_to_use,
)


# ### ASDiv Dataset:
# 
# - **At a glance**: float32, no context (except let's do this step by step), 8 beams majority vote, fixed number parser.
#   - **Accuracy**: 37.9%
#   - **Precision**: float32
#   - **num_beams**: 8
#   - **batch_size**: 1
#   - **max_new_tokens**: 100
#   - **use_majority_vote**: True
#   - **use_group_beam_search**: False
#   - **generation_extra_kwargs**: 
#     - *repetition_penalty*: 50.0
#     - *context*: ''
#   - **Notes**: 
#     - For the word outputs, the word2num often doesn't pickup the correct (last) word.
#     - Uses "Let's do this step by step".
# 
# - **At a glance**: 8 shot (random) context, with scrachpad. The scratchpads are the Formula from the dataset, so they are pretty poor.
#   - **Accuracy**: 41.2%
#   - **Precision**: float32
#   - **num_beams**: 8
#   - **batch_size**: 1
#   - **max_new_tokens**: 100
#   - **use_majority_vote**: True
#   - **use_group_beam_search**: False
#   - **generation_extra_kwargs**: 
#     - *repetition_penalty*: 50.0
#     - *context*: 8 examples in the context, WITH SCRATCHPADS.
#     - *context seed*: 42
# 