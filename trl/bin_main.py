#!/usr/bin/env python
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["DATASETS_VERBOSITY"]     = "warning"
os.environ["WANDB_SILENT"]           = "true"
os.environ["NCCL_DEBUG"]             = "WARN"

import enum
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

from scratchpad import gpt2_sentiment_control

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

class RewardForwardWrapper:
    """
    Meant to work with either a fixed trlAutoModelWithValueHead or a PeftModel
    """
    def __init__(self, ppo_trainer_model, ppo_trainer_ref_model):
        self._ppo_model = ppo_trainer_model
        self._ppo_ref   = ppo_trainer_ref_model

    def __call__(self, *args, **kwargs):
        peft_mode = (
            isinstance(self._ppo_model, peft.PeftModel) and 
            self._ppo_ref is None
        )
        ref_mode = (
            (not isinstance(self._ppo_model, peft.PeftModel)) and 
            self._ppo_ref is not None
        )

        assert peft_mode ^ ref_mode
        rich.print(f"[red bold]{peft_mode = } {ref_mode = }")

        if peft_mode:
            assert isinstance(self._ppo_model, peft.PeftModel)
            with self._ppo_model.disable_adapter():
                with self._ppo_model.no_grad():
                    return self._ppo_model(*args, **kwargs)
            
        elif ref_mode:
            self._ppo_ref.eval()
            with self._ppo_ref.no_grad():
                return self._ppo_ref(*args, **kwargs)

        raise ValueError("Should not be here")

def make_sentiment_pipeline(
    *,
    pipeline_model_name, 
    accelerator_num_process, 
    accelerator_device,
):
    if accelerator_num_process == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    else:
        device = accelerator_device

    assert device != "cpu", "CPU is not supported for sentiment analysis"

    sentiment_pipe = transformers.pipeline(
        "sentiment-analysis", 
        pipeline_model_name,
        device=device)
    
    return sentiment_pipe


class Task(str, enum.Enum):
    SENTIMENT = "sentiment"
    GSM8K = "gsk8k"


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
    task_name: Task                           = Task("sentiment"),
    use_peft: bool                            = DEFAULT_USE_PEFT,
    name: str                                 = None,
):
    args = locals().copy()

    ###########################################################################
    # Find the type of model we are using
    ###########################################################################
    config = transformers.AutoConfig.from_pretrained(model_name)
    assert "task_type" not in lora_config_dict
    if not config.is_encoder_decoder:
        lora_config_dict["task_type"] == peft.TaskType.CAUSAL_LM
    elif config.is_encoder_decoder:
        lora_config_dict["task_type"] == peft.TaskType.SEQ_2_SEQ_LM
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    ppo_config_dict = dict(
        gradient_accumulation_steps = gradient_accumulation_steps,
        accelerator_kwargs          = dict(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]),
        mini_batch_size             = mini_batch_size,
        learning_rate               = learning_rate,
        model_name                  = model_name,
        batch_size                  = batch_size,
        log_with                    = log_with,
    )

    config = trl.PPOConfig(
        **ppo_config_dict,
    )

    wandb.init(
        save_code = True,
        project   = "trl-main",
        entity    = "julesgm",
        name      = name,
        config    = dict(
            generation_kwargs = generation_kwargs,
            lora_config_dict  = lora_config_dict,
            ppo_config_args   = ppo_config_dict,
            script_args       = args,
        ),
    )

    if task_name == Task.GSM8K:
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
        
    elif task_name == "asdiv":
        dataset = lib_data.ASDiv(
            input_max_length, 
            tokenizer, 
            datasets.load_dataset("asdiv"),
        )

    elif task_name == Task.SENTIMENT:
        dataset, tokenizer = gpt2_sentiment_control.prep_dataset(
            config.model_name, 
            txt_in_len = 5,
        )

    else:
        raise ValueError(f"Unknown task: {task_name}")

    model, tokenizer = lib_trl_utils.init_model(
        lora_config_dict = lora_config_dict,
        model_name       = config.model_name,
        precision        = precision,
    )

    ###########################################################################
    # Set model name specific flags
    ###########################################################################
    if not model.config.is_encoder_decoder:
        assert lora_config_dict["task_type"] == peft.TaskType.CAUSAL_LM
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
        generation_kwargs["eos_token_id"] = -1
        generation_kwargs["min_length"]   = -1
    

    ###########################################################################
    # Prep Training
    ###########################################################################
    ppo_trainer = trl.PPOTrainer(
        config, 
        model,
        data_collator = collator,
        ref_model     = None,
        tokenizer     = tokenizer,
        dataset       = dataset,
    )

    if task_name == Task.GSM8K:
        reward_forward_fn = RewardForwardWrapper(
            ppo_trainer_model     = ppo_trainer.model,
            ppo_trainer_ref_model = ppo_trainer.ref_model,
        )
        reward_fn = lib_reward.ScratchpadRewardFn(
            ref_model = reward_forward_fn,
            tokenizer = tokenizer, 
            uses_peft = use_peft,
            metric_fn = metric_accuracy,
        )

    elif task_name == Task.SENTIMENT:
        pipeline = gpt2_sentiment_control.make_reward(
            pipeline_model_name="lvwerra/distilbert-imdb", 
            accelerator_num_process=ppo_trainer.accelerator.num_processes,
            accelerator_device=ppo_trainer.accelerator.device,
        )

        def reward_fn(prompts, outputs, ref_answers):
            assert ref_answers is None, "ref_answers should be None with task 'sentiment'"
            return gpt2_sentiment_control.compute_reward(
                query_no_ctrl = prompts, 
                reward_kwargs = pipeline, 
                generated     = outputs, 
                reward_fn     = reward_fn, 
                task_list     = ["positive"] * len(prompts),
            )
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    if task_name == Task.GSM8K:
        metric_accuracy = lib_metric.ScratchpadAnswerAccuracy()
    elif task_name == Task.SENTIMENT:
        metric_accuracy = reward_fn

    ###########################################################################
    # Training Loop
    ###########################################################################
    for epoch in range(num_epochs):
        for batch_idx, batch in tqdm(
            enumerate(ppo_trainer.dataloader), 
            desc=f"Training - Epoch {epoch}",
            disable=lib_trl_utils.get_rank() != 0
        ):
            batch["response"], _ = lib_trl_utils.batched_unroll(
                generation_kwargs = generation_kwargs, 
                query_tensors     = batch["input_ids"], 
                ppo_trainer       = ppo_trainer, 
                tokenizer         = tokenizer,
            )

            if task_name == "sentiment":
                ref_answers = None
            else:
                ref_answers = batch["ref_answer"]

            rewards = reward_fn(
                prompts     = batch["query"],
                responses   = batch["response"],
                ref_answers = ref_answers,
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