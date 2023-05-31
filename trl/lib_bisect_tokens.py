import bisect
import collections
import copy
import logging
import re
import time

import fire
import numpy as np
import rich
import rich.logging
import torch
import transformers

LOGGER = logging.getLogger(__name__)


# Python bisect
bisect_left = bisect.bisect_left
bisect_right = bisect.bisect_right


def find_lt(a, x):
    'Find rightmost value less than x'
    
    i = bisect_left(a, x)
    if i:
        return i - 1

    raise ValueError


def find_le(a, x):
    'Find rightmost value less than or equal to x'
    
    i = bisect_right(a, x)
    if i:    
        return i - 1
    
    raise ValueError


def find_gt(a, x):
    'Find leftmost value greater than x'

    i = bisect_right(a, x)
    if i != len(a):
        return i

    raise ValueError


def find_ge(a, x):
    'Find leftmost item greater than or equal to x'

    i = bisect_left(a, x)
    if i != len(a):
        return i

    raise ValueError


def extract_match_tokens(*, regexes, strings, tokenizer, tokenizer_kwargs=None, verbose=False):
    ########################################################################
    # Preliminary checks and setup
    ########################################################################
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    assert (
        "return_offsets_mapping" not in tokenizer_kwargs or 
        tokenizer_kwargs["return_offsets_mapping"]), (
        "`return_offsets_mapping` is required."
    )


    # compile the regexes in place:
    for i, regex in enumerate(regexes):
        if isinstance(regex, str):
            regexes[i] = re.compile(regex)
        else:
            assert isinstance(regex, re.Pattern), type(regex).mro()

    ########################################################################
    # Main bout
    ########################################################################
    
    # Tokenize
    tok_output = tokenizer(strings, return_offsets_mapping=True, **tokenizer_kwargs)
    tokens = tok_output["input_ids"]
    offsets = tok_output["offset_mapping"]

    left_boundaries = []
    right_boundaries = []
    for offset, mask in zip(offsets, tok_output["attention_mask"]):
        left_local = []
        right_local = []
        largest = 0
        for offset_seq, mask_seq in zip(offset, mask):
            if mask_seq != 0:
                if offset_seq[0] > largest:
                    largest = offset_seq[0]
                

                left_local.append(largest)

                if offset_seq[1] > largest:
                    largest = offset_seq[1]

                right_local.append(largest)

        left_boundaries.append(left_local)
        right_boundaries.append(right_local)

    outputs_boundaries = []
    outputs_str = []
    if verbose:
        LOGGER.debug(
            f"[bold blue]offsets:[/]           {offsets}\n"
            f"[bold blue]left_boundaries:[/]   {left_boundaries}\n"
            f"[bold blue]right_boundaries:[/]  {right_boundaries}\n"
            f"[bold blue]pairs:[/]             {[list(zip(a, b)) for a, b in zip(left_boundaries, right_boundaries)]}\n"
        )

    # Extract the matches
    for i, (toks, l_b, r_b, str_, regex) in enumerate(zip(tokens, left_boundaries, right_boundaries, strings, regexes)):
        matches = list(regex.finditer(str_))
        per_str_output_boundaries = []
        per_str_output_str = []
        
        for j, match in enumerate(matches):
            start_char, end_char = match.span()

            start_idx = find_le(l_b, start_char)
            end_idx   = find_ge(r_b, end_char)

            if verbose:
                LOGGER.debug(
                    "\n" +
                    "-" * 40 + "\n" +
                    f"[bold green]String {i + 1}/{len(strings)} Match {j + 1}/{len(matches)}:[/]\n" +
                    "-" * 40 + "\n" +
                    f"[bold blue]string match:[/]     `{str_[start_char:end_char]}`" + "\n" +
                    f"[bold blue]token match:[/]      `{tokenizer.decode(toks[start_idx:end_idx + 1])}`" + "\n" +
                    f"[bold blue]start_char:[/]        {start_char}" + "\n" +
                    f"[bold blue]end_char:[/]          {end_char}"   + "\n" +
                    f"[bold blue]start_idx:[/]         {start_idx}"  + "\n" +
                    f"[bold blue]end_idx:[/]           {end_idx}"    + "\n"+
                    # f"[bold blue]l_b_right:[/]         {lb_right}"   + "\n"+
                    # f"[bold blue]l_b_left:[/]          {lb_left}"    + "\n"+
                    # f"[bold blue]r_b_right:[/]         {rb_right}"   + "\n"+
                    # f"[bold blue]r_b_left:[/]          {rb_left}"    + "\n" +
                    f"[bold blue]l_b:[/]               " + str([(i, int(b)) for i, b in enumerate(l_b)]) + "\n" +
                    f"[bold blue]r_b:[/]               " + str([(i, int(b)) for i, b in enumerate(r_b)]) + "\n" +
                    f"[bold blue]both boundaries:[/]   " + str([(i, (int(l), int(r))) for i, (l, r) in enumerate(zip(l_b, r_b))]) + "\n" +
                    f"[bold blue]tokens:[/]            " + str([(i, tokenizer.decode([t], skip_special_tokens=False)) for i, t in enumerate(toks)]) + "\n" +
                    f"[bold blue]token ids:[/]         " + str([(i, int(t)) for i, t in enumerate(toks)]) + "\n" +
                    "-" * 40 + "\n"
                )
                assert start_idx <= end_idx, f"{start_idx = } {end_idx = }"

            per_str_output_boundaries.append((start_idx, end_idx))
            per_str_output_str.append(tokenizer.decode(toks[start_idx:end_idx + 1]))
        outputs_boundaries.append(per_str_output_boundaries)
        outputs_str.append(per_str_output_str)

    return tok_output, outputs_boundaries, outputs_str


# tok_outputs_copy = copy.deepcopy(tok_outputs)
# for i, (tok_output, output) in enumerate(zip(tok_outputs, outputs)):
#     print(f"{output = }")
#     for output_ in output:
#         print(f"{output_ = }")
#         tok_outputs_copy["input_ids"][i][output_[0] : output_[1] + 1] = 0
#     print(t.decode(tok_outputs_copy["input_ids"][i]))



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, 
        format="%(message)s", 
        handlers=[rich.logging.RichHandler(markup=True)]
    )
    
    strings = [
        "This is a string with 32362513213 potatoes and 15 apples",
        "This is #$1231,12.12. 32.1. 3. $212 3.3222.",
        "222",
        "222 ",
    ]

    tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-t5-base")

    start = time.perf_counter()
    tok_outputs, outputs = extract_match_tokens(
        regexes=[r"\d+"] * len(strings),
        strings=strings, 
        tokenizer=tokenizer,
        tokenizer_kwargs=dict(
            return_tensors="pt", padding=True
        ),
        verbose=False,

    )
    print(f"{time.perf_counter() - start = }")

    print(f"{outputs = }")