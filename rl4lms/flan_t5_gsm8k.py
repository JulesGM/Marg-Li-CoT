#!/usr/bin/env python
# coding: utf-8


"""
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

"""


import collections

import importlib
import itertools
import math
import more_itertools
from pathlib import Path
import random
import re
import time
import xml

import datasets
import matplotlib.pyplot as plt
import numpy as np
import rich
import rich.table
from text2digits import text2digits
import torch
from tqdm import tqdm
import transformers
import wget

import general_utils as utils

import asdiv_dataset

# importlib.reload(text2digits)



dataset_train = datasets.load_dataset("gsm8k", "main", split="train")
dataset_test  = datasets.load_dataset("gsm8k", "main", split="test")

def clean_text(sample):
    return {k: v.replace("<<", "(").replace(">>", ")").strip() for k, v in sample.items()}

def split_answer_scratchpad(sample):
    scratchpad, answer = sample["answer"].split("####")
    return {
        "question": sample["question"].strip(), 
        "answer": answer.strip(), 
        "scratchpad": scratchpad.strip()
    }


dataset_train = dataset_train.map(clean_text).map(split_answer_scratchpad)
dataset_test  = dataset_test .map(clean_text).map(split_answer_scratchpad)

print(dataset_train[0].keys())
print(dataset_test[0].keys())


# dataset_train = asdiv_dataset.ASDivInteger(cache_path="ASDiv.xml", quiet=False)
# dataset_test  = dataset_train


###############################################################################
# Load the model
###############################################################################
model_name = "google/flan-t5-xxl"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

with utils.ctx_timeit(f"Loading model `{model_name}`"):
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)

with utils.ctx_timeit("Converting model's type"):
    # model_cpu
    pass

with utils.ctx_timeit(f"Moving model to GPU"):
    model = model.cuda()

rich.print(f"[bold blue]Model dtype:[/]  {model.dtype}")
devices = collections.Counter(x.device.type for x in model.parameters())
rich.print(f"\n[bold blue]Model device:[/] {devices}")

assert len(devices) == 1 and "cuda" in devices, devices




text2digits_ = text2digits.Text2Digits()
num_pat = re.compile(r"\d+(?:[\,\.]\d+)?")


def deal_with_words(text):
    converted = text2digits_.convert(text)
    output = num_pat.findall(converted)[-1]
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
    def compose_fewshot_context(cls, dataset, n, with_scratchpad, seed):
        """ 
        Creates a random few-shot context. Works fine with n = 0.
        """
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

        inputs = utils.dict_unzip(inputs)

        question_text = [
            first_context_addition + question + final_context_addition 
            for question in inputs["question"]
        ]

        # rich.print(
        #     f"[bold blue]Question example:[/]\n" +
        #     random.choice(question_text) 
        # )

        output = tokenizer(
            question_text,
            padding=True,
            return_tensors="pt"
        ) 

        output["answer"] = inputs["answer"]
        output["scratchpad"] = inputs["scratchpad"]
        
        return {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v 
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


def run(
    *,
    shuffle,
    verbose,
    context,
    num_beams,
    batch_size,
    max_new_tokens,
    with_scratchpads,
    use_majority_vote,
    use_group_beam_search,
    generation_extra_kwargs
):
    args = locals().copy()
    
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
            output = model.generate(
                input_ids            = batch["input_ids"],
                attention_mask       = batch["attention_mask"],
                num_beams            = num_beams,
                num_return_sequences = num_beams if use_majority_vote else 1,
                max_new_tokens       = max_new_tokens,
                **extra_kwargs
            ).reshape(batch_size, num_beams if use_majority_vote else 1, -1)

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
        rich.print(f"[bold green]Accuracy, {model.dtype}: {accuracy:.1%}")


verbose = True
shuffle = True
n_shots = 16
num_beams = 8
batch_size = 1
max_new_tokens = 200
with_scratchpads = True
use_majority_vote = True
use_group_beam_search = False
few_shot_context_rng_seed = 42  # Makes sure the context is the same if we want it to stay the same

generation_extra_kwargs = dict(
    repetition_penalty=50.,
)


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


run(
    shuffle=shuffle,
    verbose=verbose,
    context=context,
    num_beams=num_beams, 
    batch_size=batch_size, 
    max_new_tokens=max_new_tokens, 
    with_scratchpads=with_scratchpads, 
    use_majority_vote=use_majority_vote, 
    use_group_beam_search=use_group_beam_search,
    generation_extra_kwargs=generation_extra_kwargs
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