""" Code used in the eval loops and nowhere else """
import itertools
import logging
import os
import random
import typing
from typing import Any, Optional, Union

import accelerate
import numpy as np
import rich
import torch
import torch.distributed
import torch.utils.data
import torch.utils.data.sampler
import transformers
import trl
import trl_fork
import wandb

import lib_base_classes
import lib_constant
import lib_data
import libs_data
import lib_metric
import lib_reward_exact_match
import lib_reward_ppl
import lib_sentiment_specific
import lib_trl_utils
import lib_utils


LOGGER = logging.getLogger(__name__)
RANK = int(os.getenv("RANK", 0))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))



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
    accelerator_device,
    accelerator_num_processes,
    dataset_name,
    dataset,
    pad_token,
    reward_type,
    task_name: lib_utils.Task,
    use_peft: bool,
    extractor,
) -> typing.Tuple[typing.Callable, typing.Callable]:
    if task_name == lib_utils.Task.MAIN:
        accuracy = lib_metric.ScratchpadAnswerAccuracy(
            extractor=extractor,
            pad_token=pad_token,
        )
        metrics = {
            f"accuracy_{type(extractor).__name__}": accuracy   
        }

        # if dataset_name == lib_data.DatasetChoices.ARITHMETIC:
        #     for i in range(dataset.max_num_digits):
        #         metrics[f"accuracy_with_{i}_digits"] = (
        #             libs_data.lib_arithmetic.PerNumberOfDigitsAccuracy(
        #                 extractor=dataset.get_extractor(), 
        #                 num_digits=i,
        #                 pad_token=pad_token,
        #         ))


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
                metric_fn=accuracy,
            )

        else:
            raise ValueError(
                f"Unknown reward type: {reward_type}. "
                f"Must be one of {lib_utils.RewardChoices}"
            )

    elif task_name == lib_utils.Task.SENTIMENT:
        reward_fn = lib_sentiment_specific.SentimentRewardFn(
            accelerator_device=accelerator_device,
            accelerator_num_processes=accelerator_num_processes,
        )
        metrics = {
            f"accuracy_{extractor}": 
            lib_metric.ScratchpadAnswerAccuracy(
                extractor=extractor, 
                pad_token=pad_token
            )
        }

    else:
        raise ValueError(f"Unknown task: {task_name}")

    return metrics, reward_fn


def unwrap_dataset(dataloader):
    seen = set([id(dataloader)])
    maybe_dataset = dataloader.dataset
    while hasattr(maybe_dataset, "dataset"):
        seen.add(id(maybe_dataset))
        assert id(maybe_dataset.dataset) not in seen # Prevent infinite loops
        maybe_dataset = maybe_dataset.dataset
    return maybe_dataset


class EvalLoop:
    def __init__(
        self,
        *,
        accelerated_model,
        accelerator: accelerate.Accelerator,
        batch_size: int,
        dataset: torch.utils.data.Dataset,
        dataset_type: lib_data.DatasetChoices,
        eval_subset_size: int,
        forward_tokenizer: transformers.PreTrainedTokenizerBase,
        inference_gen_kwargs: typing.Dict[str, typing.Any],
        metrics,
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
        reward_fn,
        split: str,
        task_name: lib_utils.Task,
        use_few_shots: bool,
        metric_exclusion = {f"accuracy_with_{i}_digits" for i in range(5)}
    ):
        
        dataloader = make_eval_dataloader(
            accelerator  = accelerator,
            batch_size   = batch_size,
            collator     = lib_base_classes.DataListContainer.collate,
            dataset      = dataset,
            subset_size  = eval_subset_size,
        )
        wandb_table_keys = tuple(itertools.chain((
                "call_idx",  "query_end",  "raw_gen", "cleaned_gen", "ref", 
            ), (f"{metric_name}_extract_gen" for metric_name in metrics 
            ), (f"{metric_name}_extract_ref" for metric_name in metrics)
        ))
        wandb_table = lib_utils.WandbTableRepair(
            wandb_kwargs= dict(columns=list(wandb_table_keys)))

        self._accelerated_model    = accelerated_model
        self._accelerator          = accelerator
        self._call_idx             = 0
        self._dataset_type         = dataset_type
        self._forward_tokenizer    = forward_tokenizer
        self._inference_gen_kwargs = inference_gen_kwargs
        self._metric_exclusion     = metric_exclusion
        self._metrics              = metrics
        self._prediction_tokenizer = prediction_tokenizer
        self._raw_dataset          = unwrap_dataset(dataloader)
        self._reward_fn            = reward_fn
        self._set_dataloader       = dataloader
        self._split                = split
        self._task_name            = task_name
        self._use_few_shots        = use_few_shots
        self._wandb_table          = wandb_table
        self._wandb_table_keys     = wandb_table_keys

        assert use_few_shots is self._raw_dataset.use_few_shots, (
            use_few_shots, self._raw_dataset.use_few_shots)
        
    def __call__(self):
        """
        
        1. Unroll
        2. Compute & Gather Rewards
        3. Compute & Gather Metrics
        4. Extract a random example from the batch to log
        5. Stack Rewards, Stack and Filter Metrics
        6. Create the Results Table

        """
        assert isinstance(
            self._set_dataloader.sampler,
            torch.utils.data.sampler.SequentialSampler,
        )

        with torch.no_grad():
            rewards = []
            filtered_metrics_values = lib_utils.DictDataset(keys=self._metrics.keys())
            table_information       = lib_utils.DictDataset(keys=self._wandb_table_keys)

            for batch_idx, batch in enumerate(
                lib_utils.progress(
                    self._set_dataloader,
                    description = f"Doing Evaluation of set: {self._split}",
                    total       = len(self._set_dataloader),
                )
            ):
                assert batch

                ############################################################
                # Unroll
                ############################################################
                output = lib_trl_utils.batched_unroll(
                    accelerated_model    = self._accelerated_model,
                    accelerator          = self._accelerator,
                    dataset_name         = self._dataset_type,
                    generation_kwargs    = self._inference_gen_kwargs,
                    post_process_gen_fewshots_fn = 
                        self._raw_dataset.post_process_gen_fewshots,
                    prediction_tokenizer = self._prediction_tokenizer,
                    query_tensors        = batch.tok_ref_query,
                    task_name            = self._task_name,
                    use_few_shots        = self._use_few_shots,
                )

                ############################################################
                # Compute & Gather Rewards
                ############################################################
                local_batch_rewards: lib_base_classes.RewardOutput = self._reward_fn(
                    responses=output.response_text,
                    batch=batch,
                )

                gathered_batch_rewards = self._accelerator.gather_for_metrics(
                    tensor=torch.tensor(local_batch_rewards.values).to(
                        self._accelerator.device
                    ),
                )           
                rewards.extend(gathered_batch_rewards)

                ############################################################
                # Compute & Gather Metrics
                ############################################################
                gathered_metric_values, local_metrics = lib_utils.compute_and_gather_metrics(
                    accelerator   = self._accelerator, 
                    batch         = batch, 
                    metrics       = self._metrics, 
                    response_text = output.response_text, 
                )
                filtered_metrics_values.extend(gathered_metric_values)

                ############################################################
                # Extract a random example from the batch to log
                ############################################################
                assert lib_utils.all_equal(
                    len(x.values) for x in local_metrics.values()), sorted(
                        (k, len(v.values)) for k, v in local_metrics.items()
                    )

                idx_in_local_batch = random.randint(0, len(batch.detok_ref_query) - 1)
                rindex = batch.detok_ref_query[idx_in_local_batch].rindex("Q:")

                sub_metric_gen = {
                    f"{metric_name}_extract_gen": 
                    metric_value   .extracted_gen[idx_in_local_batch] 
                    if metric_value.extracted_gen[idx_in_local_batch] 
                    is not None else "N/A"
                    for metric_name, metric_value in local_metrics.items()
                }

                sub_metric_ref = {
                    f"{metric_name}_extract_ref": 
                    metric_value   .extracted_ref[idx_in_local_batch] 
                    if metric_value.extracted_ref[idx_in_local_batch] 
                    is not None else "N/A"
                    for metric_name, metric_value in local_metrics.items()
                }

                table_information.append(dict(
                    call_idx    = str(self._call_idx),
                    cleaned_gen = str(output.response_text    [idx_in_local_batch]),
                    raw_gen     = str(output.raw_response_text[idx_in_local_batch]),
                    query_end   = str(batch.detok_ref_query   [idx_in_local_batch][rindex:]),
                    ref         = str(batch.detok_ref_answer  [idx_in_local_batch]),
                ) | sub_metric_gen | sub_metric_ref)
            
            ############################################################
            # Stack Rewards, Stack and Filter Metrics
            ############################################################
            # Rewards
            reward = torch.stack(rewards, dim=0)

            # Metrics
            filtered_metrics_values = lib_utils.DictDataset(data={
                metric_name: 
                torch.stack(metric_values, dim=0) if metric_values else None
                for metric_name, metric_values in filtered_metrics_values.items()
                if not any(metric_value is None for metric_value in metric_values)
            })
            
            for metric_name, metric_values in filtered_metrics_values.items():
                LOGGER.info(
                    f"[bold red on white]\[{self._split}]{metric_name}:[/] - "
                    f"{metric_values.mean().item():0.1%}"
                )

            self._call_idx += 1

            ############################################################
            # Create the Results Table
            ############################################################
            if RANK == 0:
                self._wandb_table.add_data(*table_information.values())
                metrics_mean = {}
                metrics_std = {}

                for metric_name, metric_values in filtered_metrics_values.items():
                    if isinstance(metric_values, dict):
                        metrics_mean.update({
                            f"us/set_{self._split.value}/{metric_name}_{k}/mean":
                            v.mean().item()
                            for k, v in metric_values.items()
                        })
                        metrics_std.update({
                            f"us/set_{self._split.value}/{metric_name}_{k}/std":
                            v.std().item()
                            for k, v in metric_values.items()
                        })
                    else:
                        if metric_values:
                            metrics_mean.update({
                                f"us/set_{self._split.value}/{metric_name}/mean": 
                                metric_values.mean().item(),
                            })
                            metrics_std.update({
                                f"us/set_{self._split.value}/{metric_name}/std": 
                                metric_values.std().item(),
                            })

                assert all(key.startswith(f"{lib_constant.WANDB_NAMESPACE}/") for key in metrics_mean)
                assert all(key.startswith(f"{lib_constant.WANDB_NAMESPACE}/") for key in metrics_std)
                wandb.log(
                    {
                        f"{lib_constant.WANDB_NAMESPACE}/set_{self._split.value}/reward_mean": reward.mean().item(),
                        f"{lib_constant.WANDB_NAMESPACE}/set_{self._split.value}/reward_std" : reward.std().item(),
                        f"{lib_constant.WANDB_NAMESPACE}/set_{self._split.value}/table"      : self._wandb_table.get_loggable_object(),
                    } | metrics_mean | metrics_std
                )

