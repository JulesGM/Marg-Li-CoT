""" Code used in the eval loops and nowhere else """
import logging
import os
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
import lib_data
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
        reward_fn = lib_sentiment_specific.SentimentRewardFn(
            accelerator_device=accelerator_device,
            accelerator_num_processes=accelerator_num_processes,
        )
        metric_accuracy = reward_fn

    else:
        raise ValueError(f"Unknown task: {task_name}")

    return metric_accuracy, reward_fn


class EvalLoop:
    def __init__(
        self,
        *,
        inference_gen_kwargs: typing.Dict[str, typing.Any],
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
        forward_tokenizer: transformers.PreTrainedTokenizerBase,
        accelerated_model,
        eval_subset_size: int,
        metric_accuracy: typing.Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor],
        accelerator: accelerate.Accelerator,
        batch_size: int,
        reward_fn,
        task_name: lib_utils.Task,
        dataset: torch.utils.data.Dataset,
        split: str,
        use_few_shots: bool,
        dataset_type: lib_data.DatasetChoices,
    ):
        dataloader = make_eval_dataloader(
            accelerator=accelerator,
            batch_size=batch_size,
            collator=lib_base_classes.DataListContainer.collate,
            dataset=dataset,
            subset_size=eval_subset_size,
        )

        self._dataset_type = dataset_type
        self._use_few_shots = use_few_shots
        self._inference_gen_kwargs = inference_gen_kwargs
        self._prediction_tokenizer = prediction_tokenizer
        self._forward_tokenizer = forward_tokenizer
        self._accelerated_model = accelerated_model
        self._metric_accuracy = metric_accuracy
        self._set_dataloader = dataloader
        self._accelerator = accelerator
        self._wandb_table_keys = (
            "call_idx", 
            "query_end", 
            "raw_gen", 
            "cleaned_gen", 
            "ref", 
            "extracted_answer",
            "extracted_ref",
        )
        self._wandb_table = wandb.Table(columns=list(self._wandb_table_keys))
        self._reward_fn = reward_fn
        self._task_name = task_name
        self._split = split
        self._call_idx = 0

        # Unwrap dataset
        seen = set([id(dataloader)])
        maybe_dataset = dataloader.dataset
        while hasattr(maybe_dataset, "dataset"):
            seen.add(id(maybe_dataset))
            assert id(maybe_dataset.dataset) not in seen # Prevent infinite loops
            maybe_dataset = maybe_dataset.dataset
        self._raw_dataset = maybe_dataset

        assert use_few_shots is maybe_dataset.use_few_shots, (
            use_few_shots, maybe_dataset.use_few_shots)

    def __call__(self):
        assert isinstance(
            self._set_dataloader.sampler,
            torch.utils.data.sampler.SequentialSampler,
        )

        # Should not be needed but we take no chances.
        with torch.no_grad():
            rewards = []
            metrics = []
            table_information = lib_utils.DictDataset(keys=self._wandb_table_keys)

            for batch_idx, batch in enumerate(
                lib_utils.progress(
                    description=f"Doing Evaluation of set: {self._split}",
                    total=len(self._set_dataloader),
                    seq=self._set_dataloader,
                )
            ):
                if RANK == 0:
                    rich.print(
                        f"Rank:   {RANK}/{WORLD_SIZE} - "
                        + f"Split:  [bold white on blue]{self._split}[/] - "
                        + f"Batch:  {batch_idx}/{len(self._set_dataloader)} - "
                        + (f"Metric: {np.mean([x.item() for x in metrics]):0.2%} - {metrics = }")
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
                    prediction_tokenizer=self._prediction_tokenizer,
                    generation_kwargs=self._inference_gen_kwargs,
                    accelerated_model=self._accelerated_model,
                    accelerator=self._accelerator,
                    query_tensors=batch.tok_ref_query,
                    use_few_shots=self._use_few_shots,
                    dataset_name=self._dataset_type,
                    task_name=self._task_name,
                    dataset_obj=self._raw_dataset
                )

                local_batch_rewards: lib_base_classes.RewardOutput = self._reward_fn(
                    responses=output.response_text,
                    batch=batch,
                )
                local_batch_metrics: lib_base_classes.MetricOutput = self._metric_accuracy(
                    responses=output.response_text,
                    batch=batch,
                )

                assert local_batch_rewards.extracted_gen == local_batch_metrics.extracted_gen, (
                    local_batch_rewards.extracted_gen, local_batch_metrics.extracted_gen
                )
                assert local_batch_rewards.extracted_ref == local_batch_metrics.extracted_ref, (
                    local_batch_rewards.extracted_ref, local_batch_metrics.extracted_ref
                )

                gathered_batch_rewards = self._accelerator.gather_for_metrics(
                    tensor=torch.tensor(local_batch_rewards.values).to(
                        self._accelerator.device
                    ),
                )

                gathered_batch_metrics = self._accelerator.gather_for_metrics(
                    tensor=torch.tensor(local_batch_metrics.values).to(
                        self._accelerator.device
                    ),
                )            

                for idx_in_batch in range(len(batch.detok_ref_query)):
                    rindex = batch.detok_ref_query[idx_in_batch].rindex("Q:")
                    table_information.append(dict(
                        call_idx=str(self._call_idx),
                        query_end=str(batch.detok_ref_query[idx_in_batch][rindex:]),
                        raw_gen=str(output.raw_response_text[idx_in_batch]),
                        cleaned_gen=str(output.response_text[idx_in_batch]),
                        ref=str(batch.detok_ref_answer[idx_in_batch]),
                        extracted_answer=str(local_batch_metrics.extracted_gen[idx_in_batch]),
                        extracted_ref=str(local_batch_metrics.extracted_ref[idx_in_batch]),
                    ))

                rewards.extend(gathered_batch_rewards)
                metrics.extend(gathered_batch_metrics)

            reward = torch.stack(rewards, dim=0)
            metric = torch.stack(metrics, dim=0)

            LOGGER.info(
                f"[bold red on white]\[{self._split}]EM Accuracy:[/] - {metric.mean().item():0.1%}"
            )

            self._call_idx += 1
            if RANK == 0:
                self._wandb_table.add_data(*table_information.values())
                
                wandb.log(
                    {
                        f"inference_loop_fn/set_{self._split}/reward_mean": reward.mean().item(),
                        f"inference_loop_fn/set_{self._split}/reward_std": reward.std().item(),
                        f"inference_loop_fn/set_{self._split}/metric_mean": metric.mean().item(),
                        f"inference_loop_fn/set_{self._split}/metric_std": metric.std().item(),
                        f"set_{self._split}/table": self._wandb_table,
                    }
                )

