#!/usr/bin/env python
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["DATASETS_VERBOSITY"]     = "warning"
os.environ["WANDB_SILENT"]           = "true"
os.environ["NCCL_DEBUG"]             = "WARN"


import logging
import random
import typing

import accelerate
import fire
import numpy as np
import datasets
import peft
import rich
import rich.status
import rich.table
import torch
from tqdm import tqdm
import transformers
import trl
import trl.core 
import wandb

import lib_trl_utils
from accelerate.utils import DistributedDataParallelKwargs

import lib_data
import lib_metric
import lib_reward

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


DEFAULT_LOG_STATS_VERBOSE = True
DEFAULT_REWARD_VERBOSE    = False
PROMPT                    =  "" 

DEFAULT_LORA_CONFIG = dict(
    inference_mode = False,
    lora_dropout   = 0.05,
    lora_alpha     = 32,
    task_type      = peft.TaskType.SEQ_2_SEQ_LM,
    bias           = "none",
    r              = 16,
)

DEFAULT_GENERATION_KWARGS = dict(
    min_length   = 3,
    do_sample    = True,
    top_k        = 0.0,
    top_p        = 1.0,
)

DEFAULT_GRADIENT_ACCUMULATION_STEPS: int                  = 1
DEFAULT_GENERATION_BATCH_SIZE:       int                  = 16
DEFAULT_MINI_BATCH_SIZE:             int                  = 16
DEFAULT_LEARNING_RATE:               float                = 1.41e-5
DEFAULT_MODEL_NAME:                  str                  = "google/flan-t5-base" 
DEFAULT_BATCH_SIZE:                  int                  = 16
DEFAULT_NUM_EPOCHS:                  int                  = 10
DEFAULT_PRECISION                                         = torch.bfloat16
DEFAULT_LOG_WITH:                    typing.Optional[str] = None
DEFAULT_USE_PEFT:                    bool                 = True


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def evaluate_or_test(
    *,
    generation_batch_size: int,
    generation_kwargs: dict[str, typing.Any],
    logging_header: str,
    ppo_trainer: trl.core.PPOTrainer,
    dataloader, 
    reward_fn: typing.Callable[[list[str], list[str]], torch.Tensor],
    tokenizer: transformers.PreTrainedTokenizerBase,
    set_name: str, 
    model: trl.models.modeling_base.PreTrainedModelWrapper,
):
    
    rewards = []
    for batch_idx, batch in tqdm(
        enumerate(dataloader), 
        desc=logging_header
    ):
        
        batch["response"] = lib_trl_utils.batched_unroll(
            generation_batch_size = generation_batch_size,
            generation_kwargs     = generation_kwargs, 
            ppo_trainer           = ppo_trainer, 
            tokenizer             = tokenizer,
            batch                 = batch,
            model                 = model,
        )

        local_batch_rewards = reward_fn(
            response_tensors = batch["response"],
            batch_query      = batch["query"], 
        )

        gathered_batch_rewards = ppo_trainer.accelerator.gather_for_metrics(
            local_batch_rewards.to(ppo_trainer.accelerator.device),
        )

        rewards.append(gathered_batch_rewards)

    reward = torch.cat(rewards, dim=0)
    
    wandb.log({
        f"{set_name}/reward_mean": reward.mean().item(),
        f"{set_name}/reward_str":  reward.std ().item(),
        
    })



def main(
    *, 
    gradient_accumulation_steps: int          = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    generation_batch_size: int                = DEFAULT_GENERATION_BATCH_SIZE,
    reward_fn_verbose: bool                   = DEFAULT_REWARD_VERBOSE,
    generation_kwargs: dict[str, typing.Any]  = DEFAULT_GENERATION_KWARGS,
    log_stats_verbose: bool                   = DEFAULT_LOG_STATS_VERBOSE,
    lora_config_dict:  dict[str, typing.Any]  = DEFAULT_LORA_CONFIG, 
    mini_batch_size: int                      = DEFAULT_MINI_BATCH_SIZE,
    learning_rate: float                      = DEFAULT_LEARNING_RATE,
    model_name: str                           = DEFAULT_MODEL_NAME,
    batch_size: int                           = DEFAULT_BATCH_SIZE,
    num_epochs: int                           = DEFAULT_NUM_EPOCHS,
    precision: typing.Union[str, torch.dtype] = DEFAULT_PRECISION,
    log_with: str                             = DEFAULT_LOG_WITH,

    input_max_length: int                     = 115,
    dataset_name: str                         = lib_data.GSM8K,
    use_peft: bool                            = DEFAULT_USE_PEFT,
):
    args = locals().copy()

    ###########################################################################
    # Find the type of model we are using
    ###########################################################################
    if "gpt" in model_name.lower():
        assert lora_config_dict["task_type"] == peft.TaskType.CAUSAL_LM
        model_type = peft.TaskType.CAUSAL_LM
    elif "t5" in model_name.lower():
        assert lora_config_dict["task_type"] == peft.TaskType.SEQ_2_SEQ_LM
        model_type = peft.TaskType.SEQ_2_SEQ_LM
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    config = trl.PPOConfig(
        gradient_accumulation_steps = gradient_accumulation_steps,
        accelerator_kwargs          = dict(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]),
        mini_batch_size             = mini_batch_size,
        learning_rate               = learning_rate,
        model_name                  = model_name,
        batch_size                  = batch_size,
        log_with                    = log_with,
    )

    if lib_trl_utils.get_rank() == 0:
        wandb.init(
            save_code = True,
            project   = "trl", 
            config    = args,
            entity    = "julesgm", 
            name      = None,
        )

    if dataset_name == lib_data.GSM8K:
        dataset = lib_data.GSM8K(
            input_max_length, 
            tokenizer, 
            datasets.load_dataset("gsm8k", "main", split="train"),
        )
        eval_dataset = lib_data.GSM8K(
            input_max_length, 
            tokenizer, 
            datasets.load_dataset("gsm8k", "main", split="test")
        )
        
    elif dataset_name == lib_data.ASDiv:
        dataset = lib_data.ASDiv(
            input_max_length, 
            tokenizer, 
            datasets.load_dataset("asdiv"),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model, tokenizer = lib_trl_utils.init_model(
        lora_config_dict = lora_config_dict,
        model_name       = config.model_name,
        precision        = precision,
        model_type       = model_type,
    )


    ###########################################################################
    # Set model name specific flags
    ###########################################################################
    if "gpt" in config.model_name:
        assert lora_config_dict["task_type"] == peft.TaskType.CAUSAL_LM
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
        generation_kwargs["eos_token_id"] = -1
        generation_kwargs["min_length"]   = -1
    

    ###########################################################################
    # Prep Training
    ###########################################################################
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.learning_rate,
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
    metric_accuracy = lib_metric.ScratchpadAnswerAccuracy()
    reward_fn = lib_reward.ScratchpadRewardFn(
        ref_model = model,
        uses_peft = use_peft,
        tokenizer = tokenizer,
        metric_fn = metric_accuracy,
    )
    

    ###########################################################################
    # Training Loop
    ###########################################################################
    for epoch in range(num_epochs):
        for batch_idx, batch in tqdm(
            enumerate(ppo_trainer.dataloader), 
            desc="Training",
            disable=lib_trl_utils.get_rank() != 0
        ):
            batch["response"] = lib_trl_utils.batched_unroll(
                generation_batch_size = generation_batch_size,
                generation_kwargs     = generation_kwargs, 
                ppo_trainer           = ppo_trainer, 
                tokenizer             = tokenizer,
                batch                 = batch,
                model                 = model,
            )

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