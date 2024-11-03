#!/usr/bin/env python
# coding: utf-8

import outlines
import outlines.models.transformers
import outlines.samplers
from outlines.generate.generator import sequence_generator
import transformers
import more_itertools as mit
import rich
import rich.panel
from typing import *

import multi_regexes



MODEL          = "susnato/phi-2"
hf_model       = transformers .AutoModelForCausalLM .from_pretrained (MODEL).cuda()
hf_tokenizer   = transformers .AutoTokenizer        .from_pretrained (
    MODEL, padding_side="left")
outlines_model = outlines     .models               .Transformers    (hf_model, hf_tokenizer)


def clean_output(text):
    return "\n" + text.replace("\n", " ").strip() + "\n"

def show(text, title):
    rich.print(rich.panel.Panel(text, title=title, title_align="left"))

prompts = [
    r"1 1 1 \+ 3 3 3 = \? Solution: 1 1 1 \+ 3 3 3 = ",
    r"2 2 2 2 \+ 4 4 4 4 = \? Solution: 2 2 2 2 \+ 4 4 4 4 = "
]

formats = [
    r"\d( \d){2}",
    r"\d( \d){3}"
]

generator = multi_regexes.MultiRegexGenerator(
    outlines_model, 
    sampler=outlines.samplers.MultinomialSampler(10),
    # sampler=outlines.samplers.GreedySampler(),
)

output = generator(prompts=prompts, regexes_str=formats)

for o in output:
    rich.print(o)



