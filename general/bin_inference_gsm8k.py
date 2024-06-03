#!/usr/bin/env python
# coding: utf-8


"""

- Has multi-gpu and multi-node support through accelerate.
- Computes the accuracy on the GSM8K or the ASDiv dataset.
- Uses seq2seq (flan-T5-*) models.
- Uses the answer parser of the reward function.

"""
import rich.status
import os
import contextlib

import contextlib
import collections
import enum
import itertools
import logging
import random
import time
import typing

import accelerate
from dataclasses import dataclass
import fire
import general_utils as utils
import more_itertools
import numpy as np
import rich
import rich.logging
import rich.progress
import rich.table
import torch
import transformers
from tqdm.rich import tqdm

import libs_compute_accuracy.dataset_asdiv as dataset_asdiv
import libs_compute_accuracy.dataset_gsm8k as dataset_gsm8k
from   libs_compute_accuracy.metrics_wordmath_datasets import split_fn

import general_utils
LOGGER = logging.getLogger(__name__)


# Model and precision
MODEL_NAME_OR_PATH      = "TheBloke/vicuna-13B-1.1-HF" # "MetaIX/GPT4-X-Alpasta-30b"  # "VMware/flan-ul2-alpaca-lora" # "google/flan-ul2"
TOKENIZER_NAME_OR_PATH  = "CarperAI/stable-vicuna-13b-delta"
MODEL_PRECISION         = torch.bfloat16

# Varia
GENERATION_EXTRA_KWARGS = dict(repetition_penalty=2.0)
MAX_NEW_TOKENS          = 200
BATCH_SIZE              = 8
N_SHOTS                 = 0

# Beam Search
NUM_BEAMS               = 8
NUM_RETURN_SEQUENCES    = 1

# Group Beam Search
USE_GROUP_BEAM_SEARCH   = False
DIVERSITY_PENALTY       = 15.0

###############################################################################
# SHOULD NOT CHANGE
###############################################################################
LOG_LEVEL = logging.INFO
SPLITS = ["val"]
DATASET_TO_USE = "gsm8k"
SHUFFLE = False
MAX_QUESTION_LENGTH = 120
WITH_SCRATCHPADS = True
MAX_ANSWER_LENGTH = MAX_NEW_TOKENS


FEW_SHOT_CONTEXT_RNG_SEED = 42
if USE_GROUP_BEAM_SEARCH:
    GENERATION_EXTRA_KWARGS["diversity_penalty"] = DIVERSITY_PENALTY
    pass

assert NUM_RETURN_SEQUENCES <= NUM_BEAMS, (NUM_RETURN_SEQUENCES, NUM_BEAMS)
USE_MAJORITY_VOTE = NUM_RETURN_SEQUENCES > 1

###############################################################################
# End of config
###############################################################################


@contextlib.contextmanager
def noop_contextmanager():
    yield


@contextlib.contextmanager
def one_by_one(accelerator):
    for _ in range(accelerator.process_index):
        accelerator.wait_for_everyone()
    
    yield

    for _ in range(accelerator.process_index, accelerator.num_processes):
        accelerator.wait_for_everyone()


class Collator:
    def __init__(
        self, 
        *, 
        few_shot_context, 
        with_scratchpads, 
        accelerator,
        tokenizer, 
    ):
        self.few_shot_context = few_shot_context
        self.with_scratchpads = with_scratchpads
        self.accelerator      = accelerator
        self.tokenizer        = tokenizer

    def collate(self, inputs):
        return ContextGeneration.collate(
                few_shot_context = self.few_shot_context, 
                with_scratchpads = self.with_scratchpads, 
                tokenizer        = self.tokenizer, 
                device           = self.accelerator.device,
                inputs           = inputs, 
            )

    __call__ = collate


class ContextGeneration:
    """
    
    Namespace

    """

    chain_of_thought_intro = "" # "Let's think about it step by step. Chain-of-thought:"
    answer_intro = "" # "Answer:"
    question_intro = "" # "Question:"

    @classmethod
    def compose_fewshot_context(
        cls, 
        *, 
        dataset, 
        n: int, 
        with_scratchpad: bool, 
        seed: int,
    ):
        """ 
        Creates a random few-shot context. 
        Works fine with n = 0.
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
    def collate(
        cls, 
        *, 
        inputs, 
        tokenizer, 
        few_shot_context: str, 
        with_scratchpads: bool, 
        device: int,
    ):
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
        for sample in inputs:
            reformatted_samples["question"  ].append(sample["prompt_or_input_text"])
            reformatted_samples["scratchpad"].append(sample["meta_data"]["ref_scratchpad"])
            reformatted_samples["answer"    ].append(more_itertools.one(sample["references"]))

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


def majority_vote_batch(
        generated, 
        tokenizer, 
        answer_extraction_fn, 
        is_main_process,
    ):
    
    for entry in generated:
        yield majority_vote(
            generated=entry, 
            tokenizer=tokenizer, 
            answer_extraction_fn=answer_extraction_fn, 
            is_main_process=is_main_process,
        )


def compare(pred, answ):
    return format_output(pred.strip()) == format_output(answ.strip())


class DatasetChoices(str, enum.Enum):
    gsm8k = "gsm8k"
    asdiv = "asdiv"


def batched_unroll(
    *, 
    num_return_sequences: int,
    accelerator: accelerate.Accelerator, 
    gen_kwargs,
    tokenizer: transformers.PreTrainedTokenizerBase, 
    batch: dict,
    model: transformers.PreTrainedModel,
):
    is_encoder_decoder = model.config.is_encoder_decoder

    if is_encoder_decoder:
        assert tokenizer.padding_side == "right", (
            tokenizer.padding_side)
    else:
        assert tokenizer.padding_side == "left", (
            tokenizer.padding_side)


    output = accelerator.unwrap_model(model).generate(
        input_ids      = batch["input_ids"],
        attention_mask = batch["attention_mask"],
        **gen_kwargs,
    )

    output = output.reshape(
        batch["input_ids"].shape[0], 
        num_return_sequences, 
        output.shape[-1],
    )

    if is_encoder_decoder:
        prepared = output
    else:
        query_len = batch["input_ids"].shape[1]
        prepared = output[:, :, query_len:]
        for i in range(output.shape[1]):
            if not (output[:, i, :query_len] == batch["input_ids"]).all().item():
                import ipdb; ipdb.set_trace()

    return prepared


@dataclass
class ExtractOutput: 
    answer_decoded: typing.Any
    predictions: typing.Any
    raw_decoded: typing.Any


def extract(
        *, 
        generated_samples,
        accelerator: accelerate.Accelerator,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ):
    
    predictions = list(
            majority_vote_batch(
                answer_extraction_fn = split_fn, 
                is_main_process      = accelerator.is_main_process,
                generated            = generated_samples, 
                tokenizer            = tokenizer, 
            )
        )
    raw_decoded = [
        tokenizer.batch_decode(batch_entry, skip_special_tokens=True)
        for batch_entry in generated_samples
    ]
    answer_decoded = [
        [split_fn(
            per_return_sequence, 
            accelerator.process_index) 
        for per_return_sequence in per_batch]
        for per_batch in raw_decoded
    ]
    return ExtractOutput(
        answer_decoded=answer_decoded,
        predictions=predictions,
        raw_decoded=raw_decoded,
    )

def log_generations(
        *, 
        answer_decoded,
        raw_decoded, 
        predictions, 
        split_name, 
        gen_kwargs,
        tokenizer, 
        batch,
        qty,
    ):

    table = rich.table.Table(
        "split_name",
        "input_text",
        "ref_answer",
        "prediction", 
        "raw decoded",
        "answer_decoded",
        show_lines=True,
    )

    for (
        prediction,
        answer,
        raw_decoded_entry,
        answer_decoded,
        input_text,
    ) in itertools.islice(more_itertools.zip_equal(
        predictions,
        batch["answer"],
        raw_decoded,
        answer_decoded,
        tokenizer.batch_decode(batch["input_ids"]),
    ), qty):

        lines_raw_decoded = [f" [bold]-[/] [black on white]{v}[/]" for v in raw_decoded_entry]
        lines_decoded     = [f" [bold]-[/] [black on white]{v}[/]" for v in answer_decoded]

        table.add_row(
            split_name, # split_name
            "[black on white]" + input_text + "[/]", # input_text
            answer, # ref_answer
            prediction, # prediction
            "\n".join(lines_raw_decoded), # answer_decoded
            "\n".join(lines_decoded), # answer_decoded   
        )
        
    lines_kwargs = {
        f"[bold cyan]{k}[/]: {v}" 
        for k, v in gen_kwargs.items()
    }

    rich.print(table)
    rich.print(lines_kwargs)
    rich.print("#" * 80)


def eval(
    *, 
    num_return_sequences: int,
    varia_logging: str,
    accelerator: accelerate.Accelerator,
    split_name: str,
    gen_kwargs: dict[str, typing.Any],
    dataloader,
    tokenizer: transformers.PreTrainedTokenizer, 
    model: transformers.T5ForConditionalGeneration, 
    args: dict[str, typing.Any],
):
    all_good_bad = []
    per_process_good_bad = []

    # Iterate per batch
    tqdm_obj = tqdm(
        dataloader, 
        disable=True, #not accelerator.is_main_process,
    )

    for batch in tqdm_obj:
        utils.check_equal(
            batch["input_ids"     ].shape[0], 
            batch["attention_mask"].shape[0],
        )
        utils.check_equal(
            batch["input_ids"     ].shape[1], 
            batch["attention_mask"].shape[1],
        )

        generated_samples = batched_unroll(
            num_return_sequences = num_return_sequences, 
            accelerator          = accelerator,
            gen_kwargs           = gen_kwargs,
            tokenizer            = tokenizer, 
            batch                = batch,
            model                = model,
        )

        extract_outputs = extract(
            generated_samples = generated_samples, 
            accelerator       = accelerator, 
            tokenizer         = tokenizer,
        )

        # Log some returned sequences.
        if accelerator.is_main_process:
            log_generations(    
                answer_decoded = extract_outputs.answer_decoded,
                predictions    = extract_outputs.predictions,
                raw_decoded    = extract_outputs.raw_decoded, 
                split_name     = split_name,
                gen_kwargs     = gen_kwargs,
                tokenizer      = tokenizer, 
                batch          = batch,
                qty            = 5,
            )

        good_bad_preds = [
            compare(pred=pred, answ=answ)
            for pred, answ in more_itertools.zip_equal(
                extract_outputs.predictions, 
                batch["answer"],
            )
        ]

        per_process_good_bad.extend(good_bad_preds)

        if accelerator.is_main_process:
            acc = np.mean(per_process_good_bad)
            mem = accelerator.unwrap_model(model).get_memory_footprint() // 1024 ** 3
            tqdm_obj.set_description(
                f"({split_name})({mem}GB)Proc-Zero Accuracy: {acc:.1%}"
            )
            LOGGER.info(args)


    assert per_process_good_bad, (
        per_process_good_bad is None, 
        len(per_process_good_bad) if per_process_good_bad is not None else None  # Don't take the len if it's None
    )

    per_process_good_bad = torch.tensor(per_process_good_bad).to(accelerator.device)
    all_good_bad = accelerator.gather(
        per_process_good_bad
    ).tolist()
    
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


def init_model(
    *,
    is_encoder_decoder,
    model_name_or_path, 
    model_precision, 
):
    
    if is_encoder_decoder:
        model_cls = transformers.AutoModelForSeq2SeqLM
    else:
        model_cls = transformers.AutoModelForCausalLM

    if model_precision in (torch.float16, torch.bfloat16):
        assert model_precision in (
            torch.bfloat16, torch.float16
        ), (model_precision)
        model = model_cls.from_pretrained(
            model_name_or_path, torch_dtype=model_precision
        )
    elif model_precision == "8bit":
        # dmap_keys = ["encoder", "lm_head", "shared", "decoder"]
        # dmap = {k: accelerator_device for k in dmap_keys}
        # import ipdb; ipdb.set_trace()

        dmap = "auto"
        model = model_cls.from_pretrained(
            model_name_or_path, 
            device_map=dmap,
            load_in_8bit=True,
        )
    elif model_precision == torch.float32 or model_precision is None:
        model = model_cls.from_pretrained(
            model_name_or_path
        )
    else:
        raise ValueError(f"Unknown model precision: {model_precision}")
        
    return model

def init_tokenizer_and_datasets(
        *,
        few_shot_context_rng_seed,
        tokenizer_name_or_path, 
        which_dataset_to_use, 
        model_name_or_path, 
        is_encoder_decoder,
        max_question_len, 
        with_scratchpads, 
        max_sum_squares, 
        max_answer_len,
        accelerator, 
        n_shots, 
        splits,
    ):
    with one_by_one(accelerator):
        LOGGER.info(
            f"[bold blue]Loading tokenizer:[white] "
            f"{tokenizer_name_or_path}"
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,)

    if is_encoder_decoder:
        pass
    else:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
    with one_by_one(accelerator):
        LOGGER.info("[bold blue]Done loading tokenizer.")

    
    print(f"Doing data: {which_dataset_to_use}")
    if which_dataset_to_use == DatasetChoices.asdiv:
        # dataset_train = dataset_asdiv.ZeroShotASDivTextGenPool.prepare("train")
        
        datasets = {
            split: dataset_asdiv.ZeroShotASDivTextGenPool.prepare(split)
            for split in splits
        }

    elif which_dataset_to_use == DatasetChoices.gsm8k:

        datasets = {
            split:  dataset_gsm8k.ZeroShotGSM8KTextGenPool.prepare(
                split,
                tokenizer_or_name_or_path = tokenizer_name_or_path,
                max_sum_squares  = max_sum_squares,
                max_question_len = max_question_len,
                max_answer_len   = max_answer_len,
            )
            for split in splits
        }

        if n_shots:
            if "train" not in splits:
                dataset_train = dataset_gsm8k.ZeroShotGSM8KTextGenPool.prepare(
                    "train",
                    tokenizer_or_name_or_path = tokenizer_name_or_path,
                    max_question_len = max_question_len,
                    max_sum_squares  = max_sum_squares,
                    max_answer_len   = max_answer_len,
                )
            else:
                dataset_train = datasets["train"]

    else:
        raise ValueError(
            f"Unknown dataset: {which_dataset_to_use}, should be "
            f"one of {list(DatasetChoices)}"
        )

    if n_shots:
        context = ContextGeneration.compose_fewshot_context(
            n               = n_shots,
            seed            = few_shot_context_rng_seed,
            dataset         = dataset_train,
            with_scratchpad = with_scratchpads,
        )
    else:
        context = ""
    
    if accelerator.is_main_process:
        if context:
            LOGGER.info(f"[bold blue]Few-shot Context (N={n_shots})[/]:\n" + context)
        else:
            LOGGER.info("[bold blue]No few-shot context.[/]")
    
    collator = Collator(
        tokenizer=tokenizer, 
        few_shot_context=context, 
        with_scratchpads=with_scratchpads, 
        accelerator=accelerator,
    )
    return tokenizer, datasets, collator

def setup_logging(log_level, global_rank, world_size):
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


@contextlib.contextmanager
def status_and_time(msg, accelerator):
    if accelerator.is_main_process:
        with rich.status.Status(msg):
            start = time.time()
            yield
            end = time.time()
        LOGGER.info(f"[bold blue]`{msg}` done in {end - start:.2f} seconds[/]")
    else:
        yield


def print_table_final_results(final_results):
    table = rich.table.Table(
            header_style = "bold magenta",
            show_header  = True,
            show_lines   = True,
            title        = "Results",
        )

    table.add_column("Split",    justify="center", style="dim", no_wrap=True)
    table.add_column("Accuracy", justify="center", style="dim", no_wrap=True)

    for split, split_acc in final_results.items():
        assert split is not None
        assert split_acc is not None
        table.add_row(split, f"{split_acc:0.1%}")

    rich.print(table)

def prepare_dataloaders(
    *,
    accelerator: accelerate.Accelerator,
    batch_size: int,
    datasets,
    collate: typing.Callable,
):
    dataloaders = {
        split: accelerator.prepare_data_loader(
            torch.utils.data.DataLoader(
            split_dataset,
            batch_size = batch_size,
            collate_fn = collate,
            shuffle    = False,
    )) for split, split_dataset in datasets.items()}

    return dataloaders


def main(
    *,
    splits=SPLITS,
    n_shots=N_SHOTS,
    log_level=LOG_LEVEL,
    num_beams=NUM_BEAMS,
    batch_size=BATCH_SIZE,
    max_new_tokens=MAX_NEW_TOKENS,
    model_precision=MODEL_PRECISION,
    with_scratchpads=WITH_SCRATCHPADS,
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
):

    args = locals().copy()
    global_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "0"))
    accelerator = accelerate.Accelerator()
    assert model_precision in [
        torch.bfloat16, torch.float32, torch.float16, "8bit"
    ], model_precision
    is_encoder_decoder = transformers.AutoConfig.from_pretrained(
        model_name_or_path).is_encoder_decoder

    if int(os.getenv("RANK", 0)) == 0:
        sorted_environ = sorted(os.environ.items(), key=lambda kv: kv[0])
        accelerate_then_deepspeed_ones = {
            k: v for k, v in sorted_environ
            if "deepspeed" in k.lower()
        } | {
            k: v for k, v in sorted_environ
            if "accelerate" in k.lower() and 
            "deepspeed" not in k.lower()
        }

        if accelerate_then_deepspeed_ones:
            general_utils.print_dict(
                accelerate_then_deepspeed_ones)

    # -------------------------------------------------------------------------
    # Setup logging
    # -------------------------------------------------------------------------
    setup_logging(log_level, global_rank, world_size)
    
    # -------------------------------------------------------------------------
    # Build the tokenizer and the datasets
    # -------------------------------------------------------------------------
    with status_and_time(
        f"Loading the data ({which_dataset_to_use}) & the tokenizer.", 
        accelerator=accelerator
    ):
        tokenizer, datasets_, collate = init_tokenizer_and_datasets(
            few_shot_context_rng_seed = few_shot_context_rng_seed,
            tokenizer_name_or_path    = tokenizer_name_or_path,
            which_dataset_to_use      = which_dataset_to_use,
            model_name_or_path        = model_name_or_path,
            is_encoder_decoder        = is_encoder_decoder,
            with_scratchpads          = with_scratchpads,
            max_question_len          = max_question_len,
            max_sum_squares           = max_sum_squares,
            max_answer_len            = max_answer_len,
            accelerator               = accelerator,
            n_shots                   = n_shots,
            splits                    = splits,
        )

    # -------------------------------------------------------------------------
    # Load the model
    # -------------------------------------------------------------------------
    model = init_model(
        is_encoder_decoder = is_encoder_decoder,
        model_name_or_path = model_name_or_path,
        model_precision    = model_precision,
    )

    # -------------------------------------------------------------------------
    # Run the evaluation
    # -------------------------------------------------------------------------
    extra_kwargs = generation_extra_kwargs.copy()
    if use_group_beam_search:
        extra_kwargs["num_beam_groups"] = num_beams

    final_gen_kwargs = dict(
        num_return_sequences = num_return_sequences,
        max_new_tokens       = max_new_tokens,
        num_beams            = num_beams,
        **extra_kwargs,
    )

    del extra_kwargs
    del generation_extra_kwargs
    assert batch_size is not None, batch_size

    with torch.no_grad():
        model = accelerator.prepare(model)

        dataloaders = prepare_dataloaders(
            accelerator = accelerator,
            datasets    = datasets_,
            batch_size  = batch_size,
            collate     = collate,
        )

        varia_logging = (
            f"[bold blue]n_shots[/]:         {n_shots}\n" +
            f"[bold blue]model[/]:           {model_name_or_path}\n" +
            f"[bold blue]num beams[/]:       {num_beams}\n" +
            f"[bold blue]max new tokens[/]:  {max_new_tokens}\n" +
            f"[bold blue]dtype[/]:           {model_precision}\n"
        )

        # They are all the same over the N processes.
        final_results = {}
        for split, dataloader in dataloaders.items():
            final_results[split] = eval(
                num_return_sequences = num_return_sequences,
                varia_logging        = varia_logging,
                accelerator          = accelerator,
                dataloader           = dataloader,
                split_name           = split,
                gen_kwargs           = final_gen_kwargs,
                tokenizer            = tokenizer, 
                model                = model,
                args                 = args,
            )

    if accelerator.is_main_process:
        print_table_final_results(final_results)


if __name__ == "__main__":
    fire.Fire(main)


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
