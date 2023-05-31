#!/usr/bin/env python
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["DATASETS_VERBOSITY"]     = "warning"
os.environ["WANDB_SILENT"]           = "true"
os.environ["NCCL_DEBUG"]             = "WARN"


import enum
import itertools
import logging
import random
import typing

import accelerate
import datasets
import fire
import numpy as np
import peft
import rich
import rich.console
import rich.status
import rich.table
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.sampler
import transformers
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm

import lib_base_classes
import lib_data
import lib_metric
import lib_reward_exact_match
import lib_reward_ppl
import lib_sentiment_specific
import lib_trl_utils
import trl
import trl.core
import wandb

LOGGER = logging.getLogger(__name__)


def progress(seq, description, total=None, disable=False):
    yield from tqdm(seq, desc=description, total=total, disable=True)


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

##############################################################################
##############################################################################

DEFAULT_GEN_KWARGS = dict(
    repetition_penalty = 5.0,
    min_length         = 4,
    top_k              = 0.0,
    top_p              = 1.0,
    early_stopping     = True,
)

DEFAULT_TASK_NAME: str = "gsm8k"
DEFAULT_EVAL_EVERY: int = 25

if DEFAULT_TASK_NAME == "gsm8k":
    DEFAULT_WANDB_PROJECT: str                 = "gsm8k"

    # -------------------------------------------------------
    DEFAULT_GEN_KWARGS["temperature"]          = 0.1
    DEFAULT_GEN_KWARGS["do_sample"]            = True
    # -------------------------------------------------------
    # DEFAULT_GEN_KWARGS["num_beams"]:       int = 32
    DEFAULT_GEN_KWARGS["num_return_sequences"] = 32
    # -------------------------------------------------------

    DEFAULT_GEN_KWARGS["max_new_tokens"]       = 192
    DEFAULT_GENERATION_BATCH_SIZE:         int = 1
    DEFAULT_MINI_BATCH_SIZE:               int = 1
    DEFAULT_BATCH_SIZE:                    int = 1
    DEFAULT_GRADIENT_ACCUMULATION_STEPS:   int = 32

    DEFAULT_PRECISION                          = torch.bfloat16

    DEFAULT_REWARD_TYPE:  typing.Optional[str] = "exact_match"
    DEFAULT_MODEL_NAME:                    str = "google/flan-t5-small"
    DEFAULT_TASK_NAME:                     str = "gsm8k"

    DEFAULT_INFERENCE_GEN_KWARGS = DEFAULT_GEN_KWARGS.copy()
    DEFAULT_INFERENCE_GEN_KWARGS["num_beams"] = 8
    DEFAULT_INFERENCE_GEN_KWARGS["do_sample"] = False
    DEFAULT_INFERENCE_GEN_KWARGS["num_return_sequences"] = 1
    # We could use a custom batch size too.


elif DEFAULT_TASK_NAME == "sentiment":
    DEFAULT_WANDB_PROJECT: str                = "sentiment"
    DEFAULT_GEN_KWARGS["max_new_tokens"]      = 20
    DEFAULT_GEN_KWARGS["do_sample"]           = True
    DEFAULT_GENERATION_BATCH_SIZE:        int = 16
    DEFAULT_MINI_BATCH_SIZE:              int = 16
    DEFAULT_BATCH_SIZE:                   int = 16
    DEFAULT_GRADIENT_ACCUMULATION_STEPS: int  = 1

    DEFAULT_REWARD_TYPE: typing.Optional[str] = None

    DEFAULT_PRECISION                         = "int8"
    DEFAULT_MODEL_NAME:                   str = "edbeeching/gpt-neo-125M-imdb-lora-adapter-merged"

    # DEFAULT_MODEL_NAME:            str = "edbeeching/gpt-neo-2.7B-imdb"
    # DEFAULT_MODEL_NAME:            str = "edbeeching/gpt-neox-20b-imdb-lora-lr5e-5-adapter-merged"
    DEFAULT_INFERENCE_GEN_KWARGS = DEFAULT_GEN_KWARGS.copy()

else:
    raise ValueError(f"Unknown task name: {DEFAULT_TASK_NAME}")


##############################################################################
##############################################################################
DEFAULT_EVAL_QTY:                    int   = 200
DEFAULT_NUM_EPOCHS:                  int   = 10
DEFAULT_USE_PEFT:                    bool  = True

DEFAULT_LEARNING_RATE:               float = 1.41e-5

DEFAULT_PEFT_CONFIG = dict(
    inference_mode = False,
    lora_dropout   = 0.05,
    lora_alpha     = 256,
    bias           = "none",
    r              = 256,
)

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "0"))
RANK       = int(os.environ.get("RANK",       "0"))


class Task(str, enum.Enum):
    SENTIMENT = "sentiment"
    GSM8K     = "gsm8k"


class GSM8KRewardChoices(str, enum.Enum):
    EXACT_MATCH = "exact_match"
    REF_PPL = "ref_ppl"


def evaluate_or_test(
    *,
    generation_kwargs:     dict[str, typing.Any],
    logging_header:        str,
    ppo_trainer:           trl.PPOTrainer,
    dataloader, 
    reward_fn:             typing.Callable[[list[str], list[str]], torch.Tensor],
    tokenizer:             transformers.PreTrainedTokenizerBase,
    task_type:             Task,
    set_name:              str,
    metric,
):
    
    assert isinstance(
        dataloader.sampler, 
        torch.utils.data.sampler.SequentialSampler,
    )

    rewards = []
    metrics = []

    for batch_idx, batch in enumerate(progress(
        description = logging_header,
        total       = len(dataloader),
        seq         = dataloader,
    )):
        ############################################################
        # Keys of batch:
        #   - "query"
        #   - "input_ids"
        #   - "ref_answer"
        #   - "ref_scratchpad"
        ############################################################

        output = lib_trl_utils.batched_unroll(
            generation_kwargs = generation_kwargs, 
            query_tensors     = batch["input_ids"],
            ppo_trainer       = ppo_trainer, 
            tokenizer         = tokenizer,
        )

        reward_kwargs = dict(
            responses = output.response_text,
            queries = batch["query"], 
        )

        if task_type == Task.GSM8K:
            reward_kwargs["ref_answers"] = batch["ref_answer"]
        

        local_batch_rewards: lib_base_classes.RewardOutput = reward_fn(**reward_kwargs)
        local_batch_metrics: lib_base_classes.MetricOutput = metric   (**reward_kwargs)

        gathered_batch_rewards = ppo_trainer.accelerator.gather_for_metrics(
            tensor=torch.tensor(local_batch_rewards.values
                ).to(ppo_trainer.accelerator.device),
        )

        gathered_batch_metrics = (
            ppo_trainer.accelerator.gather_for_metrics(
                tensor=torch.tensor(
                    local_batch_metrics.values
                ).to(ppo_trainer.accelerator.device),
            )
        )

        rewards.append(gathered_batch_rewards)
        metrics.append(gathered_batch_metrics)

    reward = torch.cat(rewards, dim=0)
    metric = torch.cat(metrics, dim=0)

    wandb.log({
        f"inference_loop_fn/set_{set_name}/reward_mean": reward.mean().item(),
        f"inference_loop_fn/set_{set_name}/reward_std" : reward.std ().item(),
        f"inference_loop_fn/set_{set_name}/metric_mean": metric.mean().item(),
        f"inference_loop_fn/set_{set_name}/metric_std" : metric.std ().item(),
    })


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def prep_dataset(
    *,
    input_max_length: int, 
    task_name: str, 
    tokenizer: transformers.PreTrainedTokenizerBase,
    split: str,
) -> torch.utils.data.Dataset:
    
    if task_name == Task.GSM8K:
        assert isinstance(LOCAL_RANK, int), type(LOCAL_RANK)
        dataset = lib_data.GSM8K(
            max_length = input_max_length, 
            tokenizer  = tokenizer,
            device     = torch.device(LOCAL_RANK),
            ds         = datasets.load_dataset(
                split = split,
                path  = "gsm8k", 
                name  = "main", 
            ),
        )
        
    elif task_name == "asdiv":
        assert split is None, "split must be None for ASDiv"
        dataset = lib_data.ASDiv(
            input_max_length, 
            tokenizer, 
            datasets.load_dataset("asdiv"),
        )

    elif task_name == Task.SENTIMENT:
        assert split == "train", "split must be None for sentiment"
        dataset = lib_sentiment_specific.prep_dataset(
            txt_in_len = 5,
            tokenizer  = tokenizer,
        )

    else:
        raise ValueError(f"Unknown task: {task_name}")

    return dataset


def make_eval_dataloader(
    *,
    subset_size: typing.Optional[int] = None,
    accelerator: accelerate.Accelerator,
    batch_size: int,
    collator: typing.Callable,
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.DataLoader:
    
    if subset_size is not None:
        dataset = torch.utils.data.Subset(
            dataset, range(subset_size))

    dataloader = torch.utils.data.DataLoader(
        num_workers = 0,
        batch_size  = batch_size,
        collate_fn  = collator,
        dataset     = dataset,
        shuffle     = False,
    )

    prepared = accelerator.prepare_data_loader(
        dataloader)

    return prepared


def make_metric_and_reward_fn(
    *,
    ppo_trainer: trl.PPOTrainer,
    reward_type,
    task_name: Task,
    tokenizer: transformers.PreTrainedTokenizerBase,
    use_peft: bool,
) -> typing.Tuple[typing.Callable, typing.Callable]:
    
    if task_name == Task.GSM8K:  
        metric_accuracy = lib_metric.ScratchpadAnswerAccuracy()

        if reward_type == GSM8KRewardChoices.REF_PPL:
            reward_forward_fn = lib_reward_ppl.RewardForwardWrapper(
                ppo_trainer_ref_model = ppo_trainer.ref_model,
                ppo_trainer_model     = ppo_trainer.model,
                use_peft              = use_peft,
            )

            reward_fn = lib_reward_ppl.ScratchpadRewardFn(
                ref_model_is_encoder_decoder = ppo_trainer.model.config.is_encoder_decoder,
                ref_inference_fn             = reward_forward_fn,
                inputs_device                = ppo_trainer.accelerator.device,
                metric_fn                    = metric_accuracy,
                tokenizer                    = tokenizer,
            )

        elif reward_type == GSM8KRewardChoices.EXACT_MATCH:
            reward_fn = lib_reward_exact_match.ExactMatchReward(
                metric_fn=metric_accuracy,
            )

        else:
            raise ValueError(
                f"Unknown reward type: {reward_type}. "
                f"Must be one of {GSM8KRewardChoices}"
            )

    elif task_name == Task.SENTIMENT:
        reward_fn = lib_sentiment_specific.SentimentRewardFn(ppo_trainer)
        metric_accuracy = reward_fn

    else:
        raise ValueError(f"Unknown task: {task_name}")

    return metric_accuracy, reward_fn

class EvalLoop:
    def __init__(
            self,
            inference_gen_kwargs: typing.Dict[str, typing.Any],
            eval_subset_size:     int,
            metric_accuracy:      typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            ppo_trainer:          trl.PPOTrainer,
            reward_fn,
            tokenizer:            transformers.PreTrainedTokenizerBase,
            task_name:            Task,
            batch_size:           int,
            dataset:              torch.utils.data.Dataset,
            split:                str,
        ):

        dataloader = make_eval_dataloader(
                accelerator = ppo_trainer.accelerator,
                batch_size  = batch_size,
                collator    = collator,
                dataset     = dataset,
                subset_size = eval_subset_size,
            )

        self._inference_gen_kwargs = inference_gen_kwargs
        self._metric_accuracy      = metric_accuracy 
        self._set_dataloader       = dataloader
        self._ppo_trainer          = ppo_trainer
        self._reward_fn            = reward_fn
        self._tokenizer            = tokenizer
        self._task_name            = task_name
        self._split                = split

    def __call__(self):
        evaluate_or_test(
            generation_kwargs     = self._inference_gen_kwargs,
            logging_header        = f"Doing Evaluation of set: {self._split}",
            ppo_trainer           = self._ppo_trainer,
            dataloader            = self._set_dataloader,
            reward_fn             = self._reward_fn,
            tokenizer             = self._tokenizer,
            task_type             = self._task_name,
            set_name              = self._split,
            metric                = self._metric_accuracy,
        )


def main(
    *, 
    gradient_accumulation_steps: int          = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    inference_gen_kwargs: dict[str, typing.Any] = DEFAULT_INFERENCE_GEN_KWARGS,
    generation_kwargs: dict[str, typing.Any]  = DEFAULT_GEN_KWARGS,
    peft_config_dict:  dict[str, typing.Any]  = DEFAULT_PEFT_CONFIG, 
    input_max_length: int                     = 115,
    eval_subset_size: int                     = DEFAULT_EVAL_QTY,
    mini_batch_size: int                      = DEFAULT_MINI_BATCH_SIZE,
    learning_rate: float                      = DEFAULT_LEARNING_RATE,
    wandb_project: str                        = DEFAULT_WANDB_PROJECT,
    just_metrics: bool                        = False,
    reward_type: typing.Union[None, str, GSM8KRewardChoices] = DEFAULT_REWARD_TYPE,
    model_name: str                           = DEFAULT_MODEL_NAME,
    batch_size: int                           = DEFAULT_BATCH_SIZE,
    eval_every: int                           = DEFAULT_EVAL_EVERY,
    precision: typing.Union[str, torch.dtype] = DEFAULT_PRECISION,
    task_name: Task                           = Task(DEFAULT_TASK_NAME),
    use_peft: bool                            = DEFAULT_USE_PEFT,
    name: typing.Optional[str]                = None,
):
    
    args = locals().copy()
    task_name = Task(task_name)

    ###########################################################################
    # Find the type of model we are using
    ###########################################################################
    config = transformers.AutoConfig.from_pretrained(model_name)
    assert "task_type" not in peft_config_dict
    if not config.is_encoder_decoder:
        peft_config_dict["task_type"] = peft.TaskType.CAUSAL_LM
    elif config.is_encoder_decoder:
        peft_config_dict["task_type"] = peft.TaskType.SEQ_2_SEQ_LM
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    ppo_config_dict = dict(
        gradient_accumulation_steps = gradient_accumulation_steps,
        accelerator_kwargs          = dict(kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)]),
        mini_batch_size             = mini_batch_size,
        learning_rate               = learning_rate,
        model_name                  = model_name,
        batch_size                  = batch_size,
        log_with                    = "wandb",
    )

    config = trl.PPOConfig(
        **ppo_config_dict,
    )

    if task_name == Task.GSM8K:
        reward_type = GSM8KRewardChoices(reward_type)

    wandb.init(
        save_code = True,
        project   = wandb_project,
        entity    = "julesgm",
        name      = name,
        config    = dict(
            generation_kwargs = generation_kwargs,
            peft_config_dict  = peft_config_dict,
            ppo_config_args   = ppo_config_dict,
            script_args       = args,
        ),
    )

    assert isinstance(config.model_name, str), (
        type(config.model_name))
    
    model, tokenizer = lib_trl_utils.init_model(
        peft_config_dict = peft_config_dict,
        model_name       = config.model_name,
        precision        = precision,
        use_peft         = use_peft,
    )

    dataset = prep_dataset(
        input_max_length = input_max_length, 
        task_name        = task_name, 
        tokenizer        = tokenizer,
        split            = "train",
    )

    eval_dataset =  prep_dataset(
        input_max_length = input_max_length, 
        task_name        = task_name, 
        tokenizer        = tokenizer,
        split            = "test",
    )

    ###########################################################################
    # Set model name specific flags
    ###########################################################################
    if not model.config.is_encoder_decoder:
        assert peft_config_dict["task_type"] == peft.TaskType.CAUSAL_LM
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
        generation_kwargs["eos_token_id"] = -1
        generation_kwargs["min_length"]   = -1
    
    ###########################################################################
    # Prep Training
    ###########################################################################
    ppo_trainer = trl.PPOTrainer(
        data_collator = collator,
        ref_model     = None,
        tokenizer     = typing.cast(
            typing.Union[
                transformers.PreTrainedTokenizer, 
                transformers.PreTrainedTokenizerFast,],
            tokenizer,
        ),
        dataset       = dataset,
        config        = config, 
        model         = model,
    )

    metric_accuracy, reward_fn = make_metric_and_reward_fn(
        ppo_trainer = ppo_trainer,
        reward_type = reward_type,
        task_name   = task_name,
        tokenizer   = tokenizer,
        use_peft    = use_peft,
    )

    train_eval = EvalLoop(
        inference_gen_kwargs = inference_gen_kwargs,
        eval_subset_size     = eval_subset_size,
        metric_accuracy      = metric_accuracy,
        ppo_trainer          = ppo_trainer,
        batch_size           = batch_size,
        reward_fn            = reward_fn,
        tokenizer            = tokenizer,
        task_name            = task_name,
        dataset              = dataset,
        split                = "train",
    )

    eval_eval = EvalLoop(
        inference_gen_kwargs = inference_gen_kwargs,
        eval_subset_size     = eval_subset_size,
        metric_accuracy      = metric_accuracy,
        ppo_trainer          = ppo_trainer,
        batch_size           = batch_size,
        reward_fn            = reward_fn,
        tokenizer            = tokenizer,
        task_name            = task_name,
        dataset              = eval_dataset,
        split                = "eval",
    )    

    if just_metrics:
        train_eval()
        eval_eval()
        return

    ###########################################################################
    # Training Loop
    ###########################################################################
    answer_extractor = lambda sample: (
        metric_accuracy._make_comparable(
            metric_accuracy._extract_answer(sample)
        ))

    for epoch in itertools.count():
        for batch_idx, batch in enumerate(progress(
            description = f"Training - Epoch {epoch}",
            disable     = lib_trl_utils.get_rank() != 0,
            seq         = ppo_trainer.dataloader, 
        )):
            ############################################################
            # Keys of batch:
            #   - "query"
            #   - "input_ids"
            #   - "ref_answer"
            #   - "ref_scratchpad"
            ############################################################
            
            if batch_idx % eval_every == 0: 
                rich.print("[red bold]DOING EVAL")
                train_eval()
                eval_eval()
                rich.print("[red bold]DONE WITH EVAL")

            raw_gen_outputs = lib_trl_utils.batched_unroll(
                generation_kwargs = generation_kwargs, 
                query_tensors     = batch["input_ids"], 
                ppo_trainer       = ppo_trainer, 
                tokenizer         = tokenizer,
            )

            if task_name == Task.GSM8K:
                outputs = lib_trl_utils.keep_good_one_generation(
                    num_return_seq = generation_kwargs["num_return_sequences"],
                    other_rewards  = None, 
                    generations    = raw_gen_outputs, 
                    ref_answers    = batch["ref_answer"], 
                    extract_fn     = answer_extractor,
                    batch_size     = batch_size,
                    tokenizer      = tokenizer,
                )
            else:
                assert generation_kwargs["num_return_sequences"] == 1, (
                    generation_kwargs["num_return_sequences"])
                outputs = raw_gen_outputs
                assert len(outputs.response_tensors) == batch_size, (
                    len(outputs.response_tensors), batch_size)

            if task_name == Task.SENTIMENT:
                ref_answers = None
            else:
                ref_answers = batch["ref_answer"]

            reward_output = reward_fn(
                queries     = batch["query"],
                responses   = outputs.response_text,
                ref_answers = ref_answers,
            )

            ###########################################################################
            # Print Rewards
            ###########################################################################
            all_rewards = typing.cast(
                torch.Tensor, 
                ppo_trainer.accelerator.gather_for_metrics(
                    torch.tensor(reward_output.values).to(
                        ppo_trainer.accelerator.device
                    )
                )
            )

            rich.print(
                f"[bold blue]" +
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
                assert isinstance(tokenizer.pad_token_id, int), type(tokenizer.pad_token_id)
                lib_trl_utils.check_all_start_with_token_id(
                    outputs.response_tensors, tokenizer.pad_token_id)
            
            stats = ppo_trainer.step(
                queries   = batch["input_ids"],
                responses = outputs.response_tensors,
                scores    = reward_output.values,
            )

            # Log stats
            assert isinstance(reward_output.values, list), type(reward_output.values)
            assert isinstance(stats,   dict), type(stats)
            assert isinstance(batch,   dict), type(batch)

            batch["response"] = outputs.response_tensors

            ppo_trainer.log_stats(
                rewards = [x.to(torch.float32) for x in reward_output.values],
                batch   = batch,
                stats   = stats,
            )

            lib_trl_utils.print_table(
                extra_columns = reward_output.logging_columns,
                log_header    = f"(b{batch_idx}e{epoch}) ",
                responses     = outputs.response_text,
                queries       = batch["query"],
                rewards       = reward_output.values,
                name          = str(name), 
                qty           = 5,
            )

    

if __name__ == "__main__":
    fire.Fire(main)