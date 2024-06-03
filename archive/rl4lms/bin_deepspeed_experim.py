#!/usr/bin/env python
# coding: utf-8
import logging
import os
import time
from typing import Union

import datasets
import deepspeed
import general_utils as utils
import more_itertools
import rich
import rich.console
import rich.logging
import torch
import transformers
import transformers.deepspeed
from beartype import beartype
from tqdm import tqdm

import metrics_wordmath_datasets

# Required for deepspeed
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOGGER = logging.getLogger(__file__)
CONSOLE = rich.console.Console(force_terminal=True, width=80)
INFERENCE_MODE = True
NUM_BEAMS = 8

NO_DEEPSPEED_MODE = False # Don't use deepspeed
NO_DEEPSPEED_PARALLELIZE = False # When not using deepspeed, whether to use PreTrainedModel.parallelize()
DTYPE = torch.bfloat16

APPROX_BATCH_SIZE = 20 * int(os.getenv("WORLD_SIZE", "0"))
MODEL_NAME = "google/flan-t5-large"
ZERO_LEVEL = 3
ZERO_LEVEL_3_CPU_OFFLOAD = True

MAX_QUESTION_LENGTH = None

# 95th percentile. Answers are much longer 
# than questions. we care about l**2
MAX_ANSWER_LENGTH = None
MAX_SQUARED_SUM = 43205

WANDB_CONFIG = {
    "enabled": True,
    "project": "deepspeed-exploration",
    "team": "julesgm",  # Is really `entity``
}


def log_rank_0(level, message):
    if os.getenv("LOCAL_RANK", "0") == "0":
        LOGGER.log(
            level, "[white bold]\[log-zero]:[/] " + message)


def info_rank_0(message):
    log_rank_0(logging.INFO, message)


def debug_rank_0(message):
    log_rank_0(logging.DEBUG, message)


class OptimizerMerger:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def backward(self, loss):
        for optimizer in self.optimizers:
            optimizer.backward(loss)


@beartype
def make_deepspeed_config(
    batch_size: Union[int, str],
    wandb_config=None,
    zero_level: int = 3,
    zero_level_3_cpu_offload: bool = True,
):

    assert DTYPE == torch.bfloat16, (
        f"Only bfloat16 is supported for now. `{DTYPE = }`"
    )
    assert zero_level_3_cpu_offload, (
        f"Only cpu offload is supported for now. " 
        f"`{zero_level_3_cpu_offload = }`"
    )
    assert zero_level == 3, (
        f"Only zero level 3 is supported for now. "
        f"`{zero_level = }` is not supported."
    )

    if wandb_config is None:
        wandb_config = {"enabled": False}

    zero_3_optimization = {
        "stage": 3,
        "overlap_comm": True,
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "sub_group_size": 1e9,  # ... No info
        # Same as default. I wonder if we could switch to auto?
        "stage3_max_reuse_distance": 1e9,
        # Same as default. I wonder if we could switch to auto?
        "stage3_max_live_parameters": 1e9,
    }

    if zero_level_3_cpu_offload:
        zero_3_optimization["offload_param"] = {
            "device": "cpu", 
            "pin_memory": True,
        }

    zero_2_optimization = {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,  # Default is False
        "contiguous_gradients": True,  # Default is True
        "reduce_scatter": True,  # Default is True
        "allgather_partitions": True,  # Default is True
        # "allgather_bucket_size": 2e8, # Default is 5e8
        # "reduce_bucket_size": 2e8, # Default is 5e8
    }

    per_zero_level = {
        2: zero_2_optimization,
        3: zero_3_optimization,
    }

    ds_config_train = {
        "wandb": wandb_config,
        "wall_clock_breakdown": True,
        "micro_batch_size_per_gpu": "auto",
        "micro_batch_per_gpu": "auto",
        "train_batch_size": batch_size,
        "steps_per_print": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam", 
            "params": {
                "lr": 0.0001, 
                "betas": [0.9, 0.999,]
            }
        },
        "zero_optimization": per_zero_level[zero_level],
    }

    assert isinstance(per_zero_level[zero_level], dict), (
        f"per_zero_level[zero_level] should be a dict. "
        f"It is a {type(per_zero_level[zero_level])}"
    )

    ds_config_inference = {
        "micro_batch_size_per_gpu": "auto",
        "zero": per_zero_level[zero_level],
    }

    if DTYPE == torch.bfloat16:
        ds_config_inference["dtype"] = "bfloat16"
        ds_config_train["grad_accum_dtype"] = "bfloat16"
        ds_config_train["optimizer"]["grad_accum_dtype"] = "bfloat16"
        ds_config_train["bfloat16"] = {"enabled": True}
    elif DTYPE == torch.float16:
        ds_config_inference["dtype"] = "float16"
        ds_config_train["grad_accum_dtype"] = "fp16"
        ds_config_train["fp16"] = {"enabled": True}
    elif DTYPE == torch.float32:
        pass
    else:
        raise ValueError(f"Invalid DTYPE: {DTYPE}")

    return ds_config_train, ds_config_inference


def main():
    if NO_DEEPSPEED_MODE:
        local_rank = 0
        world_size = 1
        assert (
            "LOCAL_RANK" not in os.environ
        ), "LOCAL_RANK should not be set in NO_DEEPSPEED_MODE"
        assert (
            "WORLD_SIZE" not in os.environ
        ), "WORLD_SIZE should not be set in NO_DEEPSPEED_MODE"
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # Round the batch size to the closest multiple of the world size
    batch_size = world_size * (APPROX_BATCH_SIZE // world_size)

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{local_rank + 1} / {world_size}]:\t%(message)s",
        datefmt="[%X]",
        handlers=[rich.logging.RichHandler(markup=True, rich_tracebacks=True)],
    )

    info_dict = dict(
        DTYPE=DTYPE,
        ZERO_LEVEL=ZERO_LEVEL,
        BATCH_SIZE=batch_size,
        INFERENCE_MODE=INFERENCE_MODE,
        NO_DEEPSPEED_MODE=NO_DEEPSPEED_MODE,
        ZERO_LEVEL_3_CPU_OFFLOAD=ZERO_LEVEL_3_CPU_OFFLOAD,
        NO_DEEPSPEED_PARALLELIZE=NO_DEEPSPEED_PARALLELIZE,
    )

    info_rank_0("\n" + utils.print_dict(info_dict, return_str=True))
    info_rank_0(f"[green]Starting main. zero_level: {ZERO_LEVEL}[/green]")

    ####################################################################################
    # Instantiate dataset
    ####################################################################################
    info_rank_0("[green bold]LOADING DATA GSM8K")
    dataset = datasets.load_dataset("gsm8k", "main", split="train")

    ####################################################################################
    # Instantiate models
    ####################################################################################
    info_rank_0(f"[green bold]LOADING MODEL {MODEL_NAME}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    policy_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    info_rank_0(f"[green bold]DONE LOADING MODEL {MODEL_NAME} :)")

    # value_model  = transformers.AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # ref_model    = copy.deepcopy(policy_model).eval()
    # for parameter in ref_model.parameters():
    # parameter.requires_grad = False
    # value_head   = torch.nn.Linear(value_model.config.hidden_size, 1, bias=False)

    ####################################################################################
    # Instantiate engines
    ####################################################################################
    ds_config_train, ds_config_inference = make_deepspeed_config(
        batch_size=batch_size,
        wandb_config=WANDB_CONFIG,
        zero_level=ZERO_LEVEL,
        zero_level_3_cpu_offload=ZERO_LEVEL_3_CPU_OFFLOAD,
    )

    if not NO_DEEPSPEED_MODE:
        if INFERENCE_MODE:
            transformers.deepspeed.HfDeepSpeedConfig(ds_config_inference)
            # deepspeed.init_distributed()
        else:
            transformers.deepspeed.HfDeepSpeedConfig(ds_config_train)

    models = {
        "policy_model": policy_model,
        # "value_model":  value_model,
        # "ref_model":    ref_model,
        # "value_head":   value_head,
    }

    engines = {}
    optimizers = {}
    info_rank_0("[red bold]Starting LOOP")
    if NO_DEEPSPEED_MODE:
        engines = {k: v.cuda() for k, v in models.items()}
        optimizers = {
            k: torch.optim.Adam(v.parameters(), lr=0.0001) 
            for k, v in models.items()
        }
        if NO_DEEPSPEED_PARALLELIZE:
            engines = {k: v.parallelize() for k, v in engines.items()}
    else:
        for idx, (model_name, model) in enumerate(models.items()):
            info_rank_0(f"[red bold]LOOP {idx}")
            if INFERENCE_MODE:
                engines[model_name] = deepspeed.init_inference(
                    model=model,
                    mp_size=world_size,
                    replace_method="auto",
                    replace_with_kernel_inject=True,
                    config=ds_config_inference,
                )
            else:
                (
                    engines[model_name],
                    optimizers[model_name],
                    _,
                    _,
                ) = deepspeed.initialize(
                    model=model,
                    # mp_size=world_size,
                    # replace_method="auto",
                    # replace_with_kernel_inject=True,
                    config_params=ds_config_train,
                    dist_init_required=idx == 0,
                )

    LOGGER.info("[red bold]Exiting LOOP")

    # optimizer = OptimizerMerger(list(models.values()))

    def max_length_dataset(
        max_length_question, max_length_answer, dataset,
    ):
        assert False
        for entry in dataset:
            question = tokenizer(entry["question"])
            assert isinstance(
                question, (dict, transformers.tokenization_utils_base.BatchEncoding)
            ), f"question is a {type(question)}"
            if max_length_question and len(question["input_ids"]) > max_length_question:
                continue

            answer = tokenizer(entry["answer"])
            assert isinstance(
                answer, (dict, transformers.tokenization_utils_base.BatchEncoding)
            ), f"answer is a {type(answer)}"
            if max_length_answer and len(answer["input_ids"]) > max_length_answer:
                continue

            yield question, answer

    def max_squared_sum(
        max_squared_sum, dataset,
    ):
        for entry in dataset:
            question = tokenizer(entry["question"])
            assert isinstance(
                question, (dict, transformers.tokenization_utils_base.BatchEncoding)
            ), f"question is a {type(question)}"
            answer = tokenizer(entry["answer"])
            assert isinstance(
                answer, (dict, transformers.tokenization_utils_base.BatchEncoding)
            ), f"answer is a {type(answer)}"

            if (
                len(question["input_ids"]) ** 2 + len(answer["input_ids"]) ** 2
                > max_squared_sum
            ):
                continue

            yield question, answer

    def padder(batch):
        batch = utils.dict_unzip(batch)
        max_length_input_ids = max(len(x) for x in batch["input_ids"])
        max_length_attention_mask = max(len(x) for x in batch["attention_mask"])
        assert max_length_input_ids == max_length_attention_mask, (
            max_length_input_ids,
            max_length_attention_mask,
        )
        max_length = max_length_input_ids
        assert len(batch["input_ids"]) == len(
            batch["attention_mask"]
        ), f"{len(batch['input_ids'])} != {len(batch['attention_mask'])}"

        tokens = []
        for x in batch["input_ids"]:
            assert isinstance(x, list), (type(x), x)
            output = x + [tokenizer.pad_token_id] * (max_length - len(x))
            assert isinstance(output, list), (type(output), output)
            assert len(output) == max_length, (len(output), len(x), max_length)
            tokens.append(output)

        attention_mask = []
        for x in batch["attention_mask"]:
            assert isinstance(x, list), (type(x), x)
            output = x + [0] * (max_length - len(x))
            assert isinstance(output, list), (type(output), output)
            assert len(output) == max_length, (len(output), len(x), max_length)
            attention_mask.append(x + [0] * (max_length - len(x)))

        assert (
            len(tokens) == len(attention_mask)
        ), (len(tokens), len(attention_mask))

        return dict(
            input_ids=torch.tensor(tokens), 
            attention_mask=torch.tensor(attention_mask),
        )

    assert MAX_SQUARED_SUM is None or (
        MAX_QUESTION_LENGTH is None and MAX_ANSWER_LENGTH is None
    ), (
        MAX_SQUARED_SUM is None,
        (MAX_QUESTION_LENGTH is None, MAX_ANSWER_LENGTH is None),
    )

    batcher = more_itertools.chunked(
        max_squared_sum(MAX_SQUARED_SUM, dataset)
        if MAX_SQUARED_SUM
        else max_length_dataset(MAX_QUESTION_LENGTH, MAX_ANSWER_LENGTH, dataset),
        batch_size,
    )

    batches = []
    for batch in tqdm(batcher):
        questions, answers = zip(*batch)
        input_ids = padder(questions)
        label_ids = padder(answers)
        batches.append((input_ids, label_ids))
    rich.print(f"Fraction kept: {(len(batches) * batch_size) / len(dataset) :0.1%} ")

    question_text = []
    answer_text = []
    generated_text = []

    for batch_idx, (input_ids, label_ids) in enumerate(tqdm(
        batches, disable=os.environ["RANK"] != "0")):
        if INFERENCE_MODE:
            ################################################################
            # Inference Test
            ################################################################
            engines["policy_model"].eval()
            start = time.perf_counter()
            output_ids = (
                engines["policy_model"]
                .generate(input_ids["input_ids"].to(local_rank), max_new_tokens=MAX_ANSWER_LENGTH, num_beams=NUM_BEAMS)
                .cpu()
                .numpy()
            )

            for i in range(len(input_ids["input_ids"])):
                question_text.append(tokenizer.decode(
                    input_ids["input_ids"][i], skip_special_tokens=True))
                answer_text.append([tokenizer.decode(
                    label_ids["input_ids"][i], skip_special_tokens=True)])
                generated_text.append(tokenizer.decode(
                    output_ids[i], skip_special_tokens=True))

            delta = time.perf_counter() - start
            LOGGER.info(
                f"[green bold]\[batch {batch_idx}/{len(batches)}]"
                f"Took {delta:0.3} seconds.")
            LOGGER.info(
                f"[green bold]\[batch {batch_idx}/{len(batches)}]"
                f"This is {delta / batch_size:0.3} "
                f"seconds per sample."
            )

        else:
            assert False
            ################################################################
            # Training Test
            ################################################################
            LOGGER.info(f"[red bold]\[batch {idx}] Training.")
            start = time.perf_counter()

            engines["policy_model"].zero_grad()

            inputs = {k: v.to(local_rank) for k, v in input_ids.items()}
            outputs = engines["policy_model"](
                **inputs,
                decoder_input_ids=label_ids["input_ids"].to(local_rank),
                decoder_attention_mask=label_ids["attention_mask"].to(local_rank),
                labels=label_ids["input_ids"].to(local_rank),
            )
            loss = outputs.loss
            LOGGER.info(f"[red bold]\[batch {idx}] Loss: {loss}")

            if NO_DEEPSPEED_MODE:
                loss.backward()
                optimizers["policy_model"].step()
            else:
                engines["policy_model"].backward(loss)
                engines["policy_model"].step()

            delta = time.perf_counter() - start
            LOGGER.info(f"[green bold]\[batch {idx}] Took {delta} seconds.")
            LOGGER.info(
                f"[green bold]\[batch {idx}] This is {delta / len(inputs['input_ids'])} seconds per sample."
            )
            info_rank_0("\n" + utils.print_dict(info_dict, return_str=True))


    if os.environ["RANK"] == "0":
        em_acc = metrics_wordmath_datasets.WordMathIntScratchpadAnswerAccuracy()
        computed = em_acc.compute(
            prompt_texts=question_text,
            reference_texts=answer_text,
            generated_texts=generated_text,
        )
        output = computed["em_accuracy"][1]
        LOGGER.info(f"[green bold]Final EM Accuracy: {output}")

if __name__ == "__main__":
    main()
