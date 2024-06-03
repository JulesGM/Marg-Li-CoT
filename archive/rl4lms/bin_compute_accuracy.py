#!/usr/bin/env python
# coding: utf-8


"""

- Has multi-gpu and multi-node support through accelerate.
- Computes the accuracy on the GSM8K or the ASDiv dataset.
- Uses seq2seq (flan-T5-*) models.
- Uses the answer parser of the reward function.

"""

import collections
import enum
import logging
import os
import random
import time
import typing

import accelerate
import fire
import general_utils as utils
import more_itertools
import numpy as np
import rich
import rich.table
import torch
import transformers
from tqdm import tqdm

import dataset_asdiv
import dataset_gsm8k
import rl4lms.data_pools.text_generation_pool as rl4lms_pool
from metrics_wordmath_datasets import split_fn

LOGGER = logging.getLogger(__name__)

MODEL_NAME_OR_PATH = "google/flan-t5-xl"
MODEL_PRECISION = torch.float32

DATASET_TO_USE = "gsm8k"
N_SHOTS = 0
NUM_BEAMS = 8
GENERATION_EXTRA_KWARGS = dict(
    repetition_penalty=5.0
)
DIVERSITY_PENALTY = 15.0
USE_GROUP_BEAM_SEARCH = False
NUM_RETURN_SEQUENCES = 1
LOG_LEVEL = logging.INFO

SPLITS = ["val", "train"]



###############################################################################
# SHOULD NOT CHANGE
###############################################################################
MAX_NEW_TOKENS = 200
SHUFFLE = False
BATCH_SIZE = 10
MAX_QUESTION_LENGTH = 120
WITH_SCRATCHPADS = True
TOKENIZER_NAME_OR_PATH = MODEL_NAME_OR_PATH
MAX_ANSWER_LENGTH = MAX_NEW_TOKENS


FEW_SHOT_CONTEXT_RNG_SEED = 42
if USE_GROUP_BEAM_SEARCH:
    GENERATION_EXTRA_KWARGS["diversity_penalty"] = DIVERSITY_PENALTY
    pass

assert NUM_RETURN_SEQUENCES <= NUM_BEAMS, (NUM_RETURN_SEQUENCES, NUM_BEAMS)
USE_MAJORITY_VOTE = NUM_RETURN_SEQUENCES > 1

###############################################################################
###############################################################################

import contextlib


@contextlib.contextmanager
def one_by_one(accelerator):
    for _ in range(accelerator.process_index):
        accelerator.wait_for_everyone()
    
    yield

    for _ in range(accelerator.process_index, accelerator.num_processes):
        accelerator.wait_for_everyone()


class ContextGeneration:
    """
    
    Namespace

    """

    chain_of_thought_intro = "" # "Let's think about it step by step. Chain-of-thought:"
    answer_intro = "" # "Answer:"
    question_intro = "" # "Question:"

    @classmethod
    def compose_fewshot_context(
        cls, dataset, n: int, with_scratchpad: bool, seed: int
    ):
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
    def collate(cls, *, inputs, tokenizer, few_shot_context: str, with_scratchpads: bool, device: int):
        """ 
        
        Collates the inputs and prompts into a single list of strings. 
        
        """

        first_context_addition = few_shot_context + f" {cls.question_intro} "
        final_context_addition = (
            f" {cls.chain_of_thought_intro} "
            if with_scratchpads
            else f" {cls.answer_intro} "
        )

        reformatted_samples = dict(question=[], answer=[], scratchpad=[])
        for entry in inputs:
            assert isinstance(entry, tuple), type(entry)
            assert len(entry) == 2, len(entry)
            sample = entry[0]
            assert isinstance(sample, rl4lms_pool.Sample), type(sample)
            assert len(sample.references) == 1, len(sample.references)
            reformatted_samples["question"].append(sample.prompt_or_input_text)
            reformatted_samples["scratchpad"].append(sample.meta_data["ref_scratchpad"])
            reformatted_samples["answer"].append(more_itertools.one(sample.references))

        question_text = [
            first_context_addition + question + final_context_addition
            for question in reformatted_samples["question"]
        ]

        output = tokenizer(question_text, padding=True, return_tensors="pt")

        output["answer"] = reformatted_samples["answer"]
        output["scratchpad"] = reformatted_samples["scratchpad"]

        return {
            k: v.to(device)
            if isinstance(v, torch.Tensor)
            else v
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


def majority_vote(generated, tokenizer, answer_extraction_fn, is_main_process):
    answers = []
    for entry in generated:
        decoded = tokenizer.decode(entry, skip_special_tokens=True)
        output = answer_extraction_fn(decoded)
        if output is not None:
            answers.append(format_output(output))

    
    counter = collections.Counter(answers)

    if is_main_process:
        LOGGER.debug(f"majority_vote: {counter = }")
        
    if counter:
        return counter.most_common(1)[0][0]
    else:
        return "0"


def majority_vote_batch(generated, tokenizer, answer_extraction_fn, is_main_process):
    for entry in generated:
        yield majority_vote(entry, tokenizer, answer_extraction_fn, is_main_process)


def compare(pred, answ):
    return format_output(pred.strip()) == format_output(answ.strip())


class DatasetChoices(str, enum.Enum):
    gsm8k = "gsm8k"
    asdiv = "asdiv"



def eval_on_dataloader(
    *, 
    
    args: dict[str, typing.Any],
    varia: str,
    model: transformers.T5ForConditionalGeneration, 
    dataloader,

    tokenizer: transformers.PreTrainedTokenizer, 
    split_name: str,
    accelerator: accelerate.Accelerator,
    final_gen_kwargs: dict[str, typing.Any],
    
    num_return_sequences: int,
):
    if accelerator.is_main_process:
        all_good_bad = []
    else:
        all_good_bad = None
    per_process_good_bad = []

    tqdm_obj = tqdm(
        dataloader, 
        disable=not accelerator.is_main_process,
    )

    for batch in tqdm_obj:
        utils.check_equal(
            batch["input_ids"].shape[0], batch["attention_mask"].shape[0]
        )
        utils.check_equal(
            batch["input_ids"].shape[1], batch["attention_mask"].shape[1]
        )

        with torch.inference_mode():
            model.eval()
            output = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **final_gen_kwargs,
            )

        output = output.reshape(
            batch["input_ids"].shape[0], 
            num_return_sequences, 
            output.shape[-1],
        )

        predictions = list(
            majority_vote_batch(
                output, 
                tokenizer, 
                split_fn, 
                accelerator.is_main_process)
        )
        raw_decoded = [
            [tokenizer.decode(x, skip_special_tokens=True) 
            for x in batch_entry]
            for batch_entry in output
        ]
        def split_lambda(x):
            return split_fn(x, accelerator.process_index)
        answer_decoded = [list(map(split_lambda, x)) for x in raw_decoded]

        for (
            prediction,
            answer,
            raw_decoded_entry,
            answer_decoded,
            input_text,
        ) in zip(
            predictions,
            batch["answer"],
            raw_decoded,
            answer_decoded,
            [tokenizer.decode(x) for x in batch["input_ids"]],
        ):

            
            if accelerator.is_main_process:
                LOGGER.debug(f"[bold blue]split_name[/]:      {split_name}\n")
                LOGGER.debug(f"[bold blue]input_text[/]:      {input_text.replace('<pad>', '')}\n")
                LOGGER.debug(f"[bold blue]ref_answer[/]:      {answer}\n")
                LOGGER.debug(f"[bold blue]prediction[/]:      {prediction}\n")
            
            lines_raw_decoded = ["[bold blue]raw decoded:[/]"] + [f" [bold]-[/] {v}" for v in raw_decoded_entry]
            lines_kwargs = {f"[bold cyan]{k}[/]: {v}" for k, v in final_gen_kwargs.items()}

            if accelerator.is_main_process:
                LOGGER.debug(
                    f"[bold blue]answer_decoded[/]:  {answer_decoded}\n" +
                    varia + 
                    "\n".join(lines_raw_decoded) + "\n" + 
                    "[bold blue]Final kwargs:[/]\n" + 
                    "\t" + "\n\t".join(lines_kwargs) + 
                    "\n" * 2
                )


        good_bad_preds = [
            compare(pred=pred, answ=answ)
            for pred, answ in zip(predictions, batch["answer"])
        ]

        per_process_good_bad.extend(good_bad_preds)

        if accelerator.is_main_process:
            acc = np.mean(per_process_good_bad)
            tqdm_obj.set_description(
                f"Proc-Zero Accuracy: {acc:.1%}"
            )
            LOGGER.info(args)


    assert per_process_good_bad, (
        per_process_good_bad is None, 
        len(per_process_good_bad) if per_process_good_bad is not None else None  # Don't take the len if it's None
    )

    all_good_bad = accelerator.gather(torch.tensor(per_process_good_bad).to(accelerator.device)).tolist()
    
    for k in all_good_bad:
        assert isinstance(k, bool), type(k)

    assert all_good_bad, (
        all_good_bad is None, 
        len(all_good_bad) if all_good_bad is not None else None
    )
    
    accuracy = np.mean(all_good_bad)
    
    if accelerator.is_main_process:
        LOGGER.info(args)
        LOGGER.info(
            "\n" * 2 +
            "#" * 80 + 
            "\n" +
            f"[bold green]Split:    {split_name}\n" +
            f"[bold green]Accuracy: {accuracy:.1%}\n" +
            "#" * 80 + 
            "\n" * 2
        )

    return accuracy



def run(
    *,
    splits=SPLITS,

    n_shots=N_SHOTS,
    log_level=LOG_LEVEL,
    num_beams=NUM_BEAMS,
    batch_size=BATCH_SIZE,
    max_new_tokens=MAX_NEW_TOKENS,
    model_precision=MODEL_PRECISION,
    with_scratchpads=WITH_SCRATCHPADS,
    
    use_majority_vote=USE_MAJORITY_VOTE,

    model_name_or_path=MODEL_NAME_OR_PATH,
    num_return_sequences=NUM_RETURN_SEQUENCES,
    which_dataset_to_use=DATASET_TO_USE,
    use_group_beam_search=USE_GROUP_BEAM_SEARCH,
    tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH,
    generation_extra_kwargs=GENERATION_EXTRA_KWARGS,
    few_shot_context_rng_seed=FEW_SHOT_CONTEXT_RNG_SEED,

    max_answer_len=None,
    max_sum_squares=None,  # 41957
    max_question_len=None,

    local_rank=None,
):


    args = locals().copy()
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "0"))

    assert isinstance(log_level, (int, str)), type(log_level).mro()
    if isinstance(log_level, str):
        log_level = logging.getLevelName(log_level)
    
    logging.basicConfig(
        level=log_level,
        format=f"[{global_rank + 1} / {world_size}] - %(name)s:\t%(message)s",
        datefmt="[%X]",
        handlers=[
            rich.logging.RichHandler(
                markup=True, 
                rich_tracebacks=True,
            )
        ],
    )

    logging.getLogger("transformers.generation_utils").setLevel(logging.DEBUG)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    
    accelerator = accelerate.Accelerator()
    # The tokenizer is required to do length filtering of the datasets.

    with one_by_one(accelerator):
        LOGGER.info(
            f"[bold blue]Loading tokenizer:[white] "
            f"{tokenizer_name_or_path}"
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
    )
    with one_by_one(accelerator):
        LOGGER.info("[bold blue]Done loading tokenizer.")

    if which_dataset_to_use == DatasetChoices.asdiv:
        dataset_train = dataset_asdiv.ZeroShotASDivTextGenPool.prepare("train")
        datasets = {
            split: dataset_asdiv.ZeroShotASDivTextGenPool.prepare(split)
            for split in splits
        }

    elif which_dataset_to_use == DatasetChoices.gsm8k:

        dataset_train = dataset_gsm8k.ZeroShotGSM8KTextGenPool.prepare(
            "train",
            model_name_or_path,
            max_sum_squares=max_sum_squares,
            max_question_len=max_question_len,
            max_answer_len=max_answer_len,
        )
        datasets = {
            split:  dataset_gsm8k.ZeroShotGSM8KTextGenPool.prepare(
                split,
                model_name_or_path,
                max_sum_squares=max_sum_squares,
                max_question_len=max_question_len,
                max_answer_len=max_answer_len,
            )
            for split in splits
        }

    else:
        raise ValueError(
            f"Unknown dataset: {which_dataset_to_use}, should be "
            f"one of {list(DatasetChoices)}"
        )

    context = ContextGeneration.compose_fewshot_context(
        dataset_train, n_shots, with_scratchpads, few_shot_context_rng_seed,
    )
    
    if accelerator.is_main_process:
        if context:
            LOGGER.info(f"[bold blue]Few-shot Context (N={n_shots})[/]:\n" + context)
        else:
            LOGGER.info("[bold blue]No few-shot context.[/]")

    ###############################################################################
    # Load the model
    ###############################################################################

    with one_by_one(accelerator):
        LOGGER.info(
            f"[bold blue]Loading the model:[bold white] "
            f"{model_name_or_path} {model_precision}"
        )
    start = time.perf_counter()
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path, torch_dtype=model_precision
    )
    model = model.to(accelerator.device)
    duration = time.perf_counter() - start

    with one_by_one(accelerator):
        LOGGER.info(f"[bold blue]Loading took:[/]  {duration:0.1f} seconds")
        LOGGER.info(f"[bold blue]Model dtype:[/]   {type(model) = }")

    extra_kwargs = generation_extra_kwargs.copy()
    if use_group_beam_search:
        extra_kwargs["num_beam_groups"] = num_beams

    final_gen_kwargs = dict(
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        **extra_kwargs,
    )

    del extra_kwargs
    del generation_extra_kwargs

    def collate(inputs):
        return ContextGeneration.collate(
                inputs=inputs, 
                tokenizer=tokenizer, 
                few_shot_context=context, 
                with_scratchpads=with_scratchpads, 
                device=accelerator.device
            )

    assert batch_size is not None, batch_size

    with torch.inference_mode():
        dataloaders = {
            split: accelerator.prepare_data_loader(
                torch.utils.data.DataLoader(
                    split_dataset,
                    shuffle=False,
                    batch_size=batch_size,
                    collate_fn=collate,
                )
            )
            for split, split_dataset in datasets.items()
        }

        model = accelerator.prepare_model(
            model
        )

        varia = (
            f"[bold blue]n_shots[/]:         {n_shots}\n"
            f"[bold blue]model[/]:           {model_name_or_path}\n"
            f"[bold blue]num beams[/]:       {num_beams}\n"
            f"[bold blue]max new tokens[/]:  {max_new_tokens}\n"
            f"[bold blue]dtype[/]:           {model_precision}\n"
        )

        # They are all the same over the N processes.
        final_results = {}
        for split, dataloader in dataloaders.items():
            final_results[split] = eval_on_dataloader(
                args=args,
                varia=varia,
                model=model,
                split_name=split,

                dataloader=dataloader,
                tokenizer=tokenizer, 
                accelerator=accelerator,
                final_gen_kwargs=final_gen_kwargs,
                
                num_return_sequences=num_return_sequences,
            )

    if accelerator.is_main_process:
        table = rich.table.Table(
            title="Results",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
        )

        table.add_column("Split", justify="center", style="dim", no_wrap=True)
        table.add_column("Accuracy", justify="center", style="dim", no_wrap=True)

        for split, split_acc in final_results.items():
            assert split is not None
            assert split_acc is not None
            table.add_row(split, f"{split_acc:0.1%}")

        rich.print(table)


if __name__ == "__main__":
    fire.Fire(run)


# ### ASDiv Dataset:
#
# - **At a glance**: 
#       float32, no context (except let's do this step by step), 
#       8 beams majority vote, fixed number parser.
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
