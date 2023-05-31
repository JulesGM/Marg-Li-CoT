import logging

import numpy as np
import rich
import torch

import lib_base_classes
import lib_data

LOGGER = logging.getLogger(__name__)

class ExactMatchReward(lib_base_classes.Reward):
    def __init__(
        self, *, 
        metric_fn,
    ):
        self._metric = metric_fn

    def __call__(
        self,
        *,
        queries:     list[str],
        responses:   list[str],
        ref_answers: list[str],
    ) -> lib_base_classes.RewardOutput:
        
        moving_average_accuracies = {
            n: lib_data.MovingAverage(n)
            for n in [8, 16, 32]}
        
        metric = self._metric(
            queries     = queries,
            responses   = responses,
            ref_answers = ref_answers,
        )

        avg_accuracy = np.mean(metric.values)
        mov_wind_acc = {
            str(k): v.update(avg_accuracy)
            for k, v in moving_average_accuracies.items()
        }

        return lib_base_classes.RewardOutput(
            moving_averages = {k: v[0] for k, v in mov_wind_acc.items()},
            logging_columns = metric.logging_columns,
            values          = [torch.tensor(x - 0.5) for x in metric.values],
            name            = "exact_match",
        )
