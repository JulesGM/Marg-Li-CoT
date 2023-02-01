#!/usr/bin/env python
# coding: utf-8

import collections
import enum
import fire
import re

import datasets
import editdistance
import jsonlines as jsonl
import numpy as np
import rich

from text2digits import text2digits
t2d_inst = text2digits.Text2Digits()


IN_PATH = "inputs/old_generations.jsonl"
NEW_PATH_ALL  = "outputs/reformated_old_generations_all.jsonl"
NEW_PATH_GOOD = "outputs/reformated_old_generations_goods.jsonl"


class KeysIn(str, enum.Enum):
    QUESTION = "input"
    REF_ANSWER = "value"
    GENERATED = "scratchpad"


class KeysOut(str, enum.Enum):
    QUESTION = "question"
    REF_ANSWER = "ref_answer"
    ALL_GENERATED = "all_generated"
    GENERATED_ANSWERS = "generated_answer"
    MAJORITY_ANSWER = "majority_generated_answer"


def extract_answer(text):
    text = text.replace(",", "")
    for char in ["(", ")", "/", "*", "+", "-", "=", "$", "%", ";"]:
        text = text.replace(char, f" {char} ")
    text = re.sub(r"(\.(?=\D))|(\.$)", " . ", text)
    text = re.sub(r"(\.\d+)(\D+)", "\g<1> \g<2>", text)
    text = re.sub(r"(\D)(\.\d+)", "\g<1> \g<2>", text)
    text = re.sub(r"(\d+\.\d+)(\D+)", "\g<1> \g<2>", text)
    text = re.sub(r"(\D)(\d+\.\d+)", "\g<1> \g<2>", text)

    try:
        converted = t2d_inst.convert(text)
    except:
        rich.print(
            f"[bold red]\"{text}\"" 
        )
        rich.print(
            t2d_inst._lex(text)
        )
        raise

    if len(converted) == 0:
        assert len(text) == 0, text

    matches = re.compile(r"\d+").findall(converted)
    if matches:
        return matches[-1], converted
    else:
        return None, converted


def prep(jsonl_iterable):
    formatted = collections.defaultdict(lambda : collections.defaultdict(list))
    generations_with_no_answer = 0
    empty_generations = 0
    num_generations = 0

    for entry in jsonl_iterable:
        key = entry[KeysIn.QUESTION]
        formatted_entry = formatted[key]
        
        ##############################################################################
        # Things done a single time
        ##############################################################################
        question_is_present = KeysOut.QUESTION in formatted_entry
        ref_answer_is_present = KeysOut.REF_ANSWER in formatted_entry
        assert (question_is_present) == (ref_answer_is_present), (
            question_is_present, ref_answer_is_present)
        
        # Add the ref answer
        if ref_answer_is_present:
            assert formatted_entry[KeysOut.REF_ANSWER] == entry[KeysIn.REF_ANSWER], (
                formatted_entry[KeysOut.REF_ANSWER], entry[KeysIn.REF_ANSWER]
            )
        else:
            formatted_entry["ref_answer"] = entry["value"]

        # Add the question
        if question_is_present:
            assert formatted_entry[KeysOut.QUESTION] == key, (
                formatted_entry[KeysOut.QUESTION], key)
        else:
            formatted_entry[KeysOut.QUESTION] = key

        ##############################################################################
        # Add a generated answer, done multiple times.
        ##############################################################################
        generated_answ, converted_scratchpad = extract_answer(entry[KeysIn.GENERATED])
        num_generations += 1
        if generated_answ is None:
            generations_with_no_answer += 1
            if len(entry[KeysIn.GENERATED].strip()) == 0:
                empty_generations += 1

            print(f"{entry[KeysIn.GENERATED] = }")
            if converted_scratchpad != entry[KeysIn.GENERATED]:
                #print("-" * 80)
                print(f"{converted_scratchpad    = }")
            print("-" * 80)
            formatted_entry[KeysOut.GENERATED_ANSWERS].append(None)
        else:
            formatted_entry[KeysOut.GENERATED_ANSWERS].append(generated_answ)
        
        formatted_entry[KeysOut.ALL_GENERATED].append(entry[KeysIn.GENERATED])

    ##############################################################################
    # Compute the most popular answer
    ##############################################################################
    for key, formatted_entry in formatted.items():
        formatted_entry[KeysOut.MAJORITY_ANSWER] = collections.Counter(
            formatted_entry[KeysOut.GENERATED_ANSWERS]).most_common(1)[0][0]

    ##############################################################################
    # Print some stats
    ##############################################################################
    print("")
    print(f"{generations_with_no_answer / num_generations = :0.4%}")
    print(f"{empty_generations / num_generations          = :0.4%}")
    print("")
    print(f"{generations_with_no_answer = } {num_generations = }")
    print(f"{empty_generations          = } {num_generations = }")
    print("")
    return formatted


def main(
    in_path=IN_PATH,
    new_path_all=NEW_PATH_ALL,
    new_path_good=NEW_PATH_GOOD,
):
    gsm8k = datasets.load_dataset("gsm8k", "main", split="train")
    gsm8k = gsm8k.map(lambda x: {
        "answer": x["answer"].split("####")[1].strip(), 
        "scratchpad": x["answer"].split("####")[0].strip()
    })

    with jsonl.open(in_path) as fin:
        generated = list(fin)

    formatted = prep(generated)
    dataset = {x["question"]: x["answer"] for x in gsm8k}

    for i, k in enumerate(formatted):
        if k.strip() not in dataset:
            print(i, k)
            min_edit = min(dataset, key=lambda x: editdistance.distance(x, k))
            print(min_edit)
            assert False

    containeds    = [         formatted[k][KeysOut.REF_ANSWER] in formatted[k][KeysOut.GENERATED_ANSWERS]              for k in dataset]
    majority_vote = [         formatted[k][KeysOut.REF_ANSWER] == formatted[k][KeysOut.MAJORITY_ANSWER]                for k in dataset]
    average_em    = [np.mean([formatted[k][KeysOut.REF_ANSWER] == x for x in formatted[k][KeysOut.GENERATED_ANSWERS]]) for k in dataset]

    success = []
    for question in dataset:
        for generated in formatted[question][KeysOut.GENERATED_ANSWERS]:
            success.append(generated == dataset[question])
                
    print(f"{np.mean(containeds   ) = :0.1%}")
    print(f"{np.mean(majority_vote) = :0.1%}")
    print(f"{np.mean(average_em   ) = :0.1%}")
    print(f"{np.mean(success      ) = :0.1%}")

    with jsonl.open(new_path_all, "w") as fout:
        for k, v in formatted.items():
            fout.write(v)

    with jsonl.open(new_path_good, "w") as fout:
        for k, v in formatted.items():
            entry = v.copy()
            assert len(entry[KeysOut.GENERATED_ANSWERS]) == len(entry[KeysOut.ALL_GENERATED]), (
                len(entry[KeysOut.GENERATED_ANSWERS]), len(entry[KeysOut.ALL_GENERATED]))

            # Only keep the generated text that leads to good answers
            all_generated_goods = []
            answer_generated_goods = []
            for idx, gen in enumerate(entry[KeysOut.GENERATED_ANSWERS]):
                if gen == entry[KeysOut.REF_ANSWER]:
                    all_generated_goods.append(entry[KeysOut.ALL_GENERATED][idx])
                    answer_generated_goods.append(gen)

            # If we do have good answers, write the entry
            if all_generated_goods:
                entry[KeysOut.ALL_GENERATED    ] = all_generated_goods
                entry[KeysOut.GENERATED_ANSWERS] = answer_generated_goods
                fout.write(entry)


if __name__ == "__main__":
    fire.Fire(main)

