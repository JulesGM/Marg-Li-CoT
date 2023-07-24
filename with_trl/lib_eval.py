""" Code used in the eval loops and nowhere else """
import logging
import os
import typing

import accelerate
import numpy as np
import rich
import torch
import torch.distributed
import torch.utils.data
import torch.utils.data.sampler
import transformers
import trl
import wandb

import lib_base_classes
import lib_metric
import lib_reward_exact_match
import lib_reward_ppl
import lib_sentiment_specific
import lib_trl_utils
import lib_utils

LOGGER = logging.getLogger(__name__)
RANK = int(os.getenv("RANK", 0))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def evaluate_or_test(
    *,
    generation_kwargs: dict[str, typing.Any],
    logging_header: str,
    accelerator: accelerate.Accelerator,
    accelerated_model,
    dataloader,
    reward_fn: typing.Callable[[list[str], list[str]], torch.Tensor],
    prediction_tokenizer: transformers.PreTrainedTokenizerBase,
    task_type: lib_utils.Task,
    set_name: str,
    metric,
):
    assert isinstance(
        dataloader.sampler,
        torch.utils.data.sampler.SequentialSampler,
    )

    rewards = []
    metrics = []

    for batch_idx, batch in enumerate(
        lib_utils.progress(
            description=logging_header,
            total=len(dataloader),
            seq=dataloader,
        )
    ):
        if RANK == 0:
            rich.print(
                f"Rank:   {RANK}/{WORLD_SIZE} - "
                + f"Split:  [bold white on blue]{set_name}[/] - "
                + f"Batch:  {batch_idx}/{len(dataloader)} - "
                + (f"Metric: {np.mean([x.item() for x in metrics]):0.2%} - ")
                if metrics
                else ""
            )

        ############################################################
        # Keys of batch:
        #   - "query"
        #   - "input_ids"
        #   - "ref_answer"
        #   - "ref_scratchpad"
        ############################################################
        assert batch

        output = lib_trl_utils.batched_unroll(
            prediction_tokenizer=prediction_tokenizer,
            generation_kwargs=generation_kwargs,
            accelerated_model=accelerated_model,
            accelerator=accelerator,
            query_tensors=batch.tok_ref_query,
        )


        local_batch_rewards: lib_base_classes.RewardOutput = reward_fn(
            responses=output.response_text,
            batch=batch,
        )
        local_batch_metrics: lib_base_classes.MetricOutput = metric(
            responses=output.response_text,
            batch=batch,
        )

        gathered_batch_rewards = accelerator.gather_for_metrics(
            tensor=torch.tensor(local_batch_rewards.values).to(
                accelerator.device
            ),
        )

        gathered_batch_metrics = accelerator.gather_for_metrics(
            tensor=torch.tensor(local_batch_metrics.values).to(
                accelerator.device
            ),
        )

        rewards.extend(gathered_batch_rewards)
        metrics.extend(gathered_batch_metrics)

    reward = torch.stack(rewards, dim=0)
    metric = torch.stack(metrics, dim=0)

    LOGGER.info(
        f"[bold red on white]\[{set_name}]EM Accuracy:[/] - {metric.mean().item():0.1%}"
    )

    if RANK == 0:
        wandb.log(
            {
                f"inference_loop_fn/set_{set_name}/reward_mean": reward.mean().item(),
                f"inference_loop_fn/set_{set_name}/reward_std": reward.std().item(),
                f"inference_loop_fn/set_{set_name}/metric_mean": metric.mean().item(),
                f"inference_loop_fn/set_{set_name}/metric_std": metric.std().item(),
            }
        )


def make_eval_dataloader(
    *,
    subset_size: typing.Optional[int] = None,
    accelerator: accelerate.Accelerator,
    batch_size: int,
    collator: typing.Callable,
    dataset: torch.utils.data.Dataset,
) -> torch.utils.data.DataLoader:
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(subset_size))

    dataloader = torch.utils.data.DataLoader(
        num_workers=0,
        batch_size=batch_size,
        collate_fn=collator,
        dataset=dataset,
        shuffle=False,
    )

    prepared = accelerator.prepare_data_loader(dataloader)

    return prepared


def make_metric_and_reward_fn(
    *,
    ppo_trainer: trl.PPOTrainer,
    reward_type,
    task_name: lib_utils.Task,
    use_peft: bool,
    extractor,
) -> typing.Tuple[typing.Callable, typing.Callable]:
    if task_name == lib_utils.Task.MAIN:
        metric_accuracy = lib_metric.ScratchpadAnswerAccuracy(extractor=extractor)

        if reward_type == lib_utils.RewardChoices.REF_PPL:
            assert False, "Not implemented"
            reward_forward_fn = lib_reward_ppl.RewardForwardWrapper(
                ppo_trainer_ref_model=ppo_trainer.ref_model,
                ppo_trainer_model=ppo_trainer.model,
                use_peft=use_peft,
            )

            reward_fn = lib_reward_ppl.ScratchpadRewardFn(
                ref_model_is_encoder_decoder=ppo_trainer.model.config.is_encoder_decoder,
                ref_inference_fn=reward_forward_fn,
                inputs_device=ppo_trainer.accelerator.device,
                metric_fn=metric_accuracy,
                tokenizer=tokenizer,
            )

        elif reward_type == lib_utils.RewardChoices.EXACT_MATCH:
            reward_fn = lib_reward_exact_match.ExactMatchReward(
                metric_fn=metric_accuracy,
            )

        else:
            raise ValueError(
                f"Unknown reward type: {reward_type}. "
                f"Must be one of {lib_utils.RewardChoices}"
            )

    elif task_name == lib_utils.Task.SENTIMENT:
        reward_fn = lib_sentiment_specific.SentimentRewardFn(ppo_trainer)
        metric_accuracy = reward_fn

    else:
        raise ValueError(f"Unknown task: {task_name}")

    return metric_accuracy, reward_fn


class EvalLoop:
    def __init__(
        self,
        inference_gen_kwargs: typing.Dict[str, typing.Any],
        eval_subset_size: int,
        metric_accuracy: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        accelerator: accelerate.Accelerator,
        accelerated_model,
        reward_fn,
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
        task_name: lib_utils.Task,
        batch_size: int,
        dataset: torch.utils.data.Dataset,
        split: str,
    ):
        dataloader = make_eval_dataloader(
            accelerator=accelerator,
            batch_size=batch_size,
            collator=lib_base_classes.DataListContainer.collate,
            dataset=dataset,
            subset_size=eval_subset_size,
        )

        self._inference_gen_kwargs = inference_gen_kwargs
        self._metric_accuracy = metric_accuracy
        self._set_dataloader = dataloader
        self._accelerator = accelerator
        self._accelerated_model = accelerated_model
        self._reward_fn = reward_fn
        self._prediction_tokenizer = prediction_tokenizer
        self._task_name = task_name
        self._split = split

    def __call__(self):
        evaluate_or_test(
            prediction_tokenizer=self._prediction_tokenizer,
            accelerated_model=self._accelerated_model,
            generation_kwargs=self._inference_gen_kwargs,
            logging_header=f"Doing Evaluation of set: {self._split}",
            accelerator=self._accelerator,
            dataloader=self._set_dataloader,
            reward_fn=self._reward_fn,
            task_type=self._task_name,
            set_name=self._split,
            metric=self._metric_accuracy,
        )
