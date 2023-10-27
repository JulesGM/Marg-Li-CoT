
"""
TODO: make sure "exists_better" works

"""
import datetime
import json
import pathlib
import shutil
from typing import Any, Optional, Union

import torch
import trl
import transformers

class Saver:
    def __init__(
        self, 
        *, 
        args: dict[str, Any], 
        ckpt_root: pathlib.Path, 
        run_name: str, 
        wandb_run_id: str,
        metric_to_max_name: str,
        model: trl.AutoModelForCausalLMWithValueHead, 
        optimizer: torch.optim.Optimizer,
        forwad_tokenizer: transformers.PreTrainedTokenizerBase,
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
        top_n_ckpts: int = 3,
    ):
        self._ckpt_root    = ckpt_root
        self._metric_to_max_name = metric_to_max_name
        self._model        = model
        self._optimizer    = optimizer
        self._run_name     = run_name
        self._top_n_ckpts  = top_n_ckpts
        self._wandb_run_id = wandb_run_id

        assert self._ckpt_root and self._ckpt_root.exists() and self._ckpt_root.is_dir(), (
            self._ckpt_root and self._ckpt_root.exists() and self._ckpt_root.is_dir())
        
        _ckpt_root = pathlib.Path(ckpt_root)
        self._ckpt_dir = _ckpt_root / f"{self._run_name}_{self._wandb_run_id}"
        self._ckpt_dir.mkdir()

        with open(self._ckpt_dir / "args.json", "w") as f:
            json.dump(args, f, indent=4)
        
        forwad_tokenizer.save_pretrained(self._ckpt_dir / "forward_tokenizer")
        prediction_tokenizer.save_pretrained(self._ckpt_dir / "prediction_tokenizer")
        with open(self._ckpt_dir / "meta.json", "w") as f:
            json.dump({
                "metric_to_max_name": self._metric_to_max_name,
            }, f, indent=4


    def _exists_better(self, metric_to_max_value):
        metrics_worse = []
        metrics_beq = []
        
        # Check if we have enough better checkpoints
        for path in self._ckpt_dir.glob("ckpt_*"):
            with open(path / "meta.json", "r") as fin:
                meta = json.load(fin)
                their_metric_to_max_value = meta["metrics"][metric_to_max_name]

                if their_metric_to_max_value < metric_to_max_value:
                    metrics_worse.append((
                        str(path),
                        their_metric_to_max_value,
                    ))
                else:
                    metrics_beq.append((
                        str(path),
                        metric_to_max_value,
                    ))


        assert (
            len(set(x[0] for x in metrics_worse)) == len(metrics_worse)
        ), (metrics_worse)
        assert (
            len(set(x[0] for x in metrics_beq)) == len(metrics_beq)
        ), (metrics_beq)

        metrics_beq.sort(key=lambda x: x[1], reverse=True)
        metrics_worse.sort(key=lambda x: x[1], reverse=True)

        # If n = 3, & we have 3 beq, we are good
        we_have_enough_better = len(metrics_beq) >= self._top_n_ckpts
        # We have n = 3, & we have 2 beq, we need to delete 1 worse
        # We have n = 3, & we have 1 beq, we need to delete 1 worse
        qty_worse_to_keep = max(0, len(metrics_beq) - self._top_n_ckpts)
        paths_of_worse_to_delete = [x[0] for x in metrics_worse[:qty_worse_to_keep]]

        assert len(paths_of_worse_to_delete) <= 1, paths_of_worse_to_delete
        assert we_have_enough_better or (len(paths_of_worse_to_delete) == 0)

        return we_have_enough_better, paths_of_worse_to_delete


    def save_checkpoint(self, metric_to_max_value: float):
        if self._top_n_ckpts:
            we_have_enough_better, to_delete = self._exists_better(metric_to_max_value)
            if not we_have_enough_better:
                self._do_save(metric_to_max_value)

                for path in to_delete:
                    shutil.rmtree(path)
        else:
            self._do_save(metric_to_max_value)


    def _do_save(self, metric_to_max_value):
        # filename compatible timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_ = self._ckpt_dir / f"ckpt_{metric_to_max_value:0.2f}_{timestamp}"
        dir_.mkdir(exist_ok=False)
        self._model.save_pretrained(dir_ / "model")
        torch.save(self._optimizer.state_dict(), dir_ / "optimizer.pt")
