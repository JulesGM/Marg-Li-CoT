#!/usr/bin/env python
import os
import wandb


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"]             = "WARN"
os.environ["DATASETS_VERBOSITY"]     = "warning"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"

import collections
from dataclasses import dataclass
import itertools
import logging
import random
import typing

import fire
import torch
import numpy as np
import datasets
import peft
import rich
import rich.table
import rich.status
from tqdm import tqdm
import transformers
import trl
import trl.core 

import lib_trl_utils
from accelerate.utils import DistributedDataParallelKwargs


datasets    .logging.set_verbosity_warning()
transformers.logging.set_verbosity_warning()
logging.getLogger("datasets"    ).setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("deepspeed"   ).setLevel(logging.WARNING)


np.random            .seed(0)
random               .seed(1)
torch         .manual_seed(2)
torch.cuda.manual_seed_all(3)
trl              .set_seed(4)


DEFAULT_DATASET_NAME      = "imdb"
DEFAULT_INPUT_MIN_LENGTH  = 2
DEFAULT_INPUT_MAX_LENGTH  = 5
DEFAULT_OUTPUT_MIN_LENGTH = 10
DEFAULT_OUTPUT_MAX_LENGTH = 20

DEFAULT_LOG_STATS_VERBOSE = True
DEFAULT_REWARD_VERBOSE    = False
PROMPT =  "" 
DEFAULT_LORA_CONFIG = dict(
    inference_mode = False,
    lora_dropout   = 0.05,
    lora_alpha     = 32,
    task_type      = peft.TaskType.CAUSAL_LM,
    bias           = "none",
    r              = 8,
)
DEFAULT_GENERATION_KWARGS = dict(
    min_length   = 3,
    do_sample    = True,
    top_k        = 0.0,
    top_p        = 1.0,
)
DEFAULT_PIPE_SENT_KWARGS = dict(
    function_to_apply = "none", 
    batch_size        = 256,
    truncation        = True,
    top_k             = None,)
DEFAULT_PRECISION = torch.bfloat16


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    gradient_accumulation_steps: typing.Optional[int]   = 1 
    generation_batch_size:                        int   = 256
    mini_batch_size                                     = 16
    learning_rate:               typing.Optional[float] = 1.41e-5
    model_name:                  typing.Optional[str]   = "edbeeching/gpt-neo-125M-imdb-lora-adapter-merged"
    batch_size:                  typing.Optional[int]   = 256
    num_epochs:                  typing.Optional[int]   = 10000
    log_with:                    typing.Optional[str]   = None


def build_dataset(
    config:                dict[str, typing.Any],
    dataset_name:          str, 
    input_min_length: int, 
    input_max_length: int,
) -> datasets.Dataset:
    """

    Build dataset for training. This builds the dataset 
    from `load_dataset`, one should customize this function 
    to train the model on its own dataset.

    output needs to have input_ids, & query

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.
    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.

    """

    tokenizer = lib_trl_utils.build_tokenizer(config.model_name)
    ds = datasets.load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = trl.core.LengthSampler(
        input_min_length,
        input_max_length,
    )

    
    def tokenize(sample):
        prompt = PROMPT

        sample["input_ids"] = tokenizer.encode(
            prompt + sample["review"],
            truncation=True,
        )[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds


def collator(
    data: list[dict[str, lib_trl_utils.IntSequence]],
) -> dict[str, list[lib_trl_utils.IntSequence]]:
    
    output_dict = {key: [d[key] for d in data] for key in data[0]}

    return output_dict


class RewardFn:
    def __init__(
            self, 
            *, 
            device: int, 
            tokenizer: transformers, 
            pipe_sent_kwargs: dict[str, typing.Any],
            verbose: bool = False,
        ):

        self._sentiment_pipe = transformers.pipeline(
            "sentiment-analysis", 
            device = device,
            model  = "lvwerra/distilbert-imdb", 
        )

        self._pipe_sent_kwargs = pipe_sent_kwargs
        self._tokenizer        = tokenizer
        self._verbose          = verbose
    
    def __call__(self, response_tensors, batch_query):
        
        batch_response  = [self._tokenizer.decode(r.squeeze()) for r in response_tensors]
        texts           = [q + r for q, r in zip(batch_query, batch_response)]
        pipe_outputs    = self._sentiment_pipe(texts, **self._pipe_sent_kwargs)
        prepped_outputs = [{
            output["label"]: torch.tensor(output["score"]) for output in outputs} 
            for outputs in pipe_outputs
        ]
        final_outputs    = [{
            output["label"]: torch.tensor(output["score"]) for output in outputs}["POSITIVE"]
            for outputs in pipe_outputs
        ]

        if self._verbose:
            table = rich.table.Table(
                "Pipeline Inputs", 
                "POSITIVE",
                "NEGATIVE",
                "FINAL OUTPUT",
                title       = "Sentiment Analysis Pipeline",
                show_lines  = True,
            )


            for text, output, final_output in itertools.islice(
                zip(texts, prepped_outputs, final_outputs), 5):

                table.add_row(
                    str(text), 
                    str(output["POSITIVE"].item()),
                    str(output["NEGATIVE"].item()),
                    str(final_output.item()),
                )
            
            rich.print(self._pipe_sent_kwargs)
            rich.print(table)
    
        return final_outputs

def main(
    *, 
    reward_fn_verbose: bool                           = DEFAULT_REWARD_VERBOSE,
    log_stats_verbose: bool                           = DEFAULT_LOG_STATS_VERBOSE,
    generation_kwargs: dict[str, typing.Any]          = DEFAULT_GENERATION_KWARGS,
    lora_config_dict:  dict[str, typing.Any]          = DEFAULT_LORA_CONFIG, 
    precision:         typing.Union[str, torch.dtype] = DEFAULT_PRECISION,
    dataset_name:      str                            = DEFAULT_DATASET_NAME, 
    input_min_length:  str                            = DEFAULT_INPUT_MIN_LENGTH, 
    input_max_length:  str                            = DEFAULT_INPUT_MAX_LENGTH,
    output_min_length: str                            = DEFAULT_OUTPUT_MIN_LENGTH, 
    output_max_length: str                            = DEFAULT_OUTPUT_MAX_LENGTH,
    pipe_sent_kwargs:  dict[str, typing.Any]          = DEFAULT_PIPE_SENT_KWARGS,
):
    parser = transformers.HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if "gpt" in script_args.model_name.lower():
        assert lora_config_dict["task_type"] == peft.TaskType.CAUSAL_LM
        model_type = peft.TaskType.CAUSAL_LM
    elif "t5" in script_args.model_name.lower():
        assert lora_config_dict["task_type"] == peft.TaskType.SEQ_2_SEQ_LM
        model_type = peft.TaskType.SEQ_2_SEQ_LM
    else:
        raise ValueError(f"Unknown model type: {script_args.model_name}")

    config = trl.PPOConfig(
        gradient_accumulation_steps = script_args.gradient_accumulation_steps,
        mini_batch_size             = script_args.mini_batch_size,
        learning_rate               = script_args.learning_rate,
        model_name                  = script_args.model_name,
        batch_size                  = script_args.batch_size,
        log_with                    = script_args.log_with,
        accelerator_kwargs = dict(
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )
    )
    if lib_trl_utils.get_rank() == 0:
        wandb.init(
            project="trl", 
            entity="julesgm", 
            save_code=True,
        )

    
    dataset = build_dataset(
        config,
        dataset_name=dataset_name, 
        input_min_length=input_min_length, 
        input_max_length=input_max_length,
    )

    model, tokenizer = lib_trl_utils.init_model(
        lora_config_dict = lora_config_dict,
        model_name       = config.model_name,
        precision        = precision,
        model_type       = model_type,
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.learning_rate
    )
    ppo_trainer = trl.PPOTrainer(
        config, 
        model, 
        data_collator = collator,
        ref_model     = None,
        optimizer     = optimizer,
        tokenizer     = tokenizer,
        dataset       = dataset,
    )

    if "gpt" in config.model_name:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
        generation_kwargs["eos_token_id"] = -1
        generation_kwargs["min_length"]   = -1

    ###########################################################################
    ###########################################################################
    reward_fn = RewardFn(
        pipe_sent_kwargs = pipe_sent_kwargs,
        tokenizer        = tokenizer,
        verbose          = reward_fn_verbose,
        device           = lib_trl_utils.get_local_rank(),
    )
    output_length_sampler = trl.core.LengthSampler(
        output_min_length, 
        output_max_length,
    )
    assert output_min_length <= output_max_length, (
        output_min_length,
        output_max_length,
    )
    assert generation_kwargs["min_length"] <= output_min_length, (
        generation_kwargs["min_length"],
        output_min_length,
    )

    ###########################################################################
    ###########################################################################
    
    assert lib_trl_utils.print_trainable_parameters(model, False) > 0

    for epoch in range(script_args.num_epochs):
        for batch_idx, batch in tqdm(
            enumerate(ppo_trainer.dataloader), 
            desc="Training",
            disable=lib_trl_utils.get_rank() != 0
        ):
            batch["response"] = lib_trl_utils.batched_unroll(
                output_length_sampler = output_length_sampler,
                generation_batch_size = script_args.generation_batch_size,
                generation_kwargs     = generation_kwargs, 
                ppo_trainer           = ppo_trainer, 
                tokenizer             = tokenizer,
                batch                 = batch,
                model                 = model,
            )

            # Compute rewards
            rewards = reward_fn(
                response_tensors = batch["response"],
                batch_query      = batch["query"], 
            )

            ###########################################################################
            # Print Rewards
            ###########################################################################
            all_rewards = ppo_trainer.accelerator.gather_for_metrics(
                torch.tensor(rewards).to(ppo_trainer.accelerator.device)
            )

            rich.print(
                f"[bold blue]"
                f"({lib_trl_utils.get_rank()}/{lib_trl_utils.get_world_size()}) " +
                f"({epoch = } {batch_idx = }) " +
                f"[/][white bold]" +
                f"Average rewards: " +
                f"{all_rewards.mean().item():0.4} " +
                f"+- {all_rewards.std().item():0.1}" 
            )

            if lib_trl_utils.get_rank() == 0:
                wandb.log({"avg_all_rewards": all_rewards.mean().item()})


            ###########################################################################
            # Checks & Step
            ###########################################################################
            # PPO Step
            if ppo_trainer.is_encoder_decoder:
                lib_trl_utils.check_all_start_with_token_id(
                    batch["response"], tokenizer.pad_token_id,
                )

            assert all((response != tokenizer.pad_token_id).all() for response in batch["response"])
            assert all((inputs   != tokenizer.pad_token_id).all() for inputs   in batch["input_ids"])

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v.requires_grad = True

            stats = ppo_trainer.step(
                responses = batch["response"],
                queries   = batch["input_ids"],
                scores    = rewards,
            )

            # Log stats
            assert isinstance(rewards, list), type(rewards)
            assert isinstance(stats,   dict), type(stats)
            assert isinstance(batch,   dict), type(batch)

            ppo_trainer.log_stats(
                rewards = rewards,
                verbose = log_stats_verbose,
                batch   = batch,
                stats   = stats,
            )


if __name__ == "__main__":
    fire.Fire(main)