#!/usr/bin/env python3
# coding: utf-8

print("Importing modules.")

from beartype.typing import *

import collections
import itertools
import logging
import math
from pathlib import Path
import random
import re
import time

from beartype import beartype
import h5py  # type: ignore[import]
import fire  # type: ignore[import]
import json  # type: ignore[import]
import jsonlines as jsonl  # type: ignore
import more_itertools
import numpy as np
import os
import pytorch_lightning as pl
import rich
import torch
from tqdm import tqdm  # type: ignore
import transformers
import wandb

import pretty_traceback  # type: ignore
pretty_traceback.install()

import general_shared_constants
import general_utils
print("Done loading modules.\n")


SCRIPT_DIR = Path(__file__).absolute().parent

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
GRADIENT_CLIP_VAL = 0.1
WANDB_ENTITY = "julesgm"
WANDB_PROJECT = "self_learned_explanations"
PRECISION = 16
DATA_PATH = SCRIPT_DIR / "data"
DETERMINISTIC = True
NUM_GPUS = torch.cuda.device_count()
EVAL_EVERY_N_EPOCHS = 1

class _RefineLM(pl.LightningModule):
    def __init__(
        self,
        *,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        datasets: Dict[str, torch.utils.data.Dataset],
        batch_sizes: Dict[str, int],
        generation_kwargs: dict[str, Any],
        learning_rate: float,
        is_adamw: bool,
        weight_decay: Optional[float],
        # scheduler_type,
        # scheduler_kwargs,
        path_log_results: Path,
    ):
        super().__init__()
        self._model: transformers.PreTrainedModel = model
        self._tokenizer: Final = tokenizer
        self._batch_size: Final[int] = batch_sizes[general_shared_constants.CVSets.TRAINING]
        self._eval_batch_size: Final[int] = batch_sizes[general_shared_constants.CVSets.VALIDATION]
        self._generation_kwargs: Final[dict[str, Any]] = generation_kwargs
        self._logging_conf: Final[dict[str, bool]] = dict(
            prog_bar=True, on_step=True, on_epoch=True, logger=True
        )

        ################################################################################
        # Related to datasets
        ################################################################################
        self._shuffle_train: Final[bool] = True
        self._shuffle_val: Final[bool] = False
        
        train_ds = datasets[general_shared_constants.CVSets.TRAINING]
        eval_ds = datasets[general_shared_constants.CVSets.VALIDATION]
        assert train_ds is not eval_ds, "train_ds and eval_ds must be different objects"
        self._train_ds: Final[torch.utils.data.Dataset] = train_ds
        self._eval_ds: Final[torch.utils.data.Dataset] = eval_ds
        
        ################################################################################
        # Rel. to logging results for answer overlap estim.
        ################################################################################
        self._path_log_results: Final[Path] = path_log_results
        self._results_to_log: Optional[dict[str, dict[bool, dict[str, torch.Tensor]]]] = {}
        self._labels_to_log: dict[str, str] = {}

        ################################################################################
        # Specific to the optimizer, its scheduler
        ################################################################################
        self._learning_rate: Final[float] = learning_rate
        self._is_adamw: Final[bool] = is_adamw
        self._weight_decay: Final[Optional[float]] = weight_decay

        # Related to the scheduler:
        # self.scheduler_type =         scheduler_type
        # self.scheduler_kwargs =       scheduler_kwargs

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, **self._logging_conf)
        return outputs.loss

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, list[str]]], batch_idx):  # type: ignore[override]
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        """
        See ref
        https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        """
        if self._is_adamw:
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.Adam

        optimizer = optimizer_class(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
            capturable=True,
        )

        return dict(optimizer=optimizer)

    @beartype
    def _make_regular_dataloader(
        self,
        ds: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=transformers.DataCollatorWithPadding(tokenizer=self._tokenizer),
            batch_size=batch_size,
            num_workers=os.cpu_count() - 1,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._make_regular_dataloader(
            ds=self._train_ds,
            batch_size=self._batch_size,
            shuffle=self._shuffle_train,
        )

    def val_dataloader(self):
        return self._make_regular_dataloader(
            ds=self._eval_ds,
            batch_size=self._batch_size,
            shuffle=self._shuffle_val,
        )

def _get_last_checkpoint_path(checkpoints_folder, wandb_run_id):
    assert False
    rich.print(f"[red bold]{wandb_run_id = }")
    checkpoint_files = list(checkpoints_folder.glob("**/*.ckpt"))
    assert len(checkpoint_files) == 1, checkpoint_files
    checkpoints = list((checkpoints_folder / WANDB_PROJECT / wandb_run_id / "checkpoints").glob("*.ckpt"))
    assert len(checkpoints) == 1, checkpoints
    checkpoint_path = checkpoints[0]
    rich.print(f"[red bold]{checkpoint_path = }")
    return checkpoint_path


def _set_resumed_state(checkpoint_dir: Union[Path, str], arg_meta_info: dict[str, Any]) -> tuple[dict[str, Any], transformers.PreTrainedModel]:
    checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / "meta_info.json", "r") as f:
        meta_info = json.load(f)

    # Check that the values that need to match do match
    arg_meta_info = arg_meta_info.copy()
    none_or_equal = {"run_name", "seed", "wandb_run_id", "transformers_model_name", "run_name"}
    none_or_absent = {"torch_rng_state", "numpy_rng_state", "python_rng_state"}

    for k in none_or_equal:
        arg_val = arg_meta_info.pop(k)
        assert arg_val is None or arg_val == meta_info[k], (arg_val, meta_info[k])
    
    for k in none_or_absent:
        if k in arg_meta_info:
            arg_val = arg_meta_info.pop(k)
            assert arg_val is None, arg_val
    
    # We should have no remaining keys
    assert not arg_meta_info, arg_meta_info

    # Load the variables
    wandb_run_id = meta_info["wandb_run_id"]
    seed = meta_info["seed"]
    torch_rng_state = meta_info["torch_rng_state"]
    numpy_rng_state = meta_info["numpy_rng_state"]
    python_rng_state = meta_info["python_rng_state"]
    run_name = meta_info["run_name"]
    transformers_model_name = meta_info["transformers_model_name"]

    # Deal with random seeds and states
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.set_rng_state(torch.ByteTensor(torch_rng_state))
    np.random.set_state(numpy_rng_state)
    for i, v in enumerate(python_rng_state):
        if isinstance(v, list):
            python_rng_state[i] = tuple(
                python_rng_state[i])
    random.setstate(tuple(python_rng_state))

    # Resume the wandb run
    rich.print("\n[red bold]Resuming Wandb run:", wandb_run_id)
    wandb.init(project=WANDB_PROJECT, resume="must", id=wandb_run_id)
    assert wandb.run.resumed, wandb.run.resumed
    assert wandb.run.project == WANDB_PROJECT, (wandb.run.project, WANDB_PROJECT)
    assert wandb.run.id == wandb_run_id, (wandb.run.id, wandb_run_id)

    rich.print(f"\n[bold]Loading model from:[/bold] {checkpoint_dir / 'model'}, [bold]Of type[/bold] {transformers_model_name}")
    start = time.perf_counter()
    model = transformers.from_pretrained(checkpoint_dir / "model")
    rich.print(f"Loaded model in {time.perf_counter() - start:.2f}s")

    return meta_info, model


def _set_initial_state(checkpoint_dir: Union[Path, str], arg_meta_info: dict[str, Any]) -> tuple[dict[str, Any], transformers.PreTrainedModel]:
    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir()

    # Deal with random seeds and states
    seed = arg_meta_info["seed"]
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    rich.print("\n[bold]Loading model from hf, name:[/bold]", arg_meta_info["transformers_model_name"])
    start = time.perf_counter()
    model = transformers.from_pretrained(arg_meta_info["transformers_model_name"])
    rich.print(f"Loaded model in {time.perf_counter() - start:.2f}s")
    
    return arg_meta_info, model


def _build_meta_info(**kwargs):
    return kwargs


def _text_mode_load_data(
    dataset_path: Union[str, Path], 
    tokenizer, 
    label_scratchpad_joiner= lambda sp, l: sp + " => " + l
):
    dataset_path = Path(dataset_path)
    
    sets = [
        general_shared_constants.CVSets.TRAINING, 
        general_shared_constants.CVSets.VALIDATION
    ]
    tokenized_data = {}

    for set_ in sets:
        with jsonl.open(dataset_path / "train.jsonl") as f:
            raw_data = [json.loads(line) for line in f]

        # Not bad not great.
        # Should really be pre-tokenized and put into an hdf5 file.
        # TODO: Make it pre-tokenized, & think about the format.
        tokenized_data[set_] = {
            "inputs": tokenizer([x["input_text"]  for x in raw_data], add_special_tokens=True),
            "labels": tokenizer([x["label"] for x in raw_data], add_special_tokens=True),
            "scratchpads": tokenizer([x["scratchpad"] for x in raw_data], add_special_tokens=True),
            "scratchpad_with_labels": tokenizer([label_scratchpad_joiner(x["scratchpad"], x["label"]) for x in raw_data], add_special_tokens=True),
        }

    return tokenized_data

def _text_mode_build_dataset(dataset_path, tokenizer):
    tokenized_data = _text_mode_load_data(dataset_path, tokenizer)
    output_datasets = {}
    keys_in_ds = {"inputs": "input_ids", "scratchpads_with_labels": "labels"}

    for set_, set_columns in tokenized_data.items():
        output_datasets[set_] = torch.utils.data.Dataset.from_dict({
            dataset_key: set_columns[tokenized_key] for tokenized_key, dataset_key in keys_in_ds.items()}
        )

    return output_datasets


def main(
    dataset_path: Union[Path, str],
    checkpoints_folder: Union[Path, str],
    wandb_run_id: Optional[str],
    seed: int,
    
):
    all_arguments = locals().copy()
    general_utils.check_and_print_args(all_arguments, main)

    dataset_path = Path(dataset_path)
    checkpoints_folder = Path(checkpoints_folder)

    assert dataset_path.exists(), dataset_path
    assert checkpoints_folder.exists(), checkpoints_folder
    assert checkpoints_folder.is_dir(), checkpoints_folder
    torch.use_deterministic_algorithms(mode=DETERMINISTIC)

    arg_meta_info = _build_meta_info(
        seed=seed, 
        dataset_path=dataset_path,
        checkpoints_folder=checkpoints_folder,
        wandb_run_id=wandb_run_id,
        run_name=f"noob_gsm8k",
        transformers_model_name="distilgpt",
    )

    resuming = checkpoints_folder.exists()
    if resuming:
        meta_info, hf_model = _set_resumed_state(checkpoints_folder, arg_meta_info)
    else:
        checkpoints_folder.mkdir()
        meta_info, hf_model = _set_initial_state(checkpoints_folder, arg_meta_info)

    rich.print(f"[bold]Run name:[/bold] [green]\"{meta_info['run_name']}\"\n")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(meta_info["transformers_model_name"])
    datasets = _text_mode_build_dataset(dataset_path, tokenizer)
    

    ###############################################################
    # Build the pt-lightning dataloader
    ###############################################################
    pl_object = _RefineLM(
        model=hf_model,
        tokenizer=tokenizer,
        datasets=datasets,
        batch_sizes=meta_info["batch_sizes"],
        generation_kwargs=meta_info["batch_sizes"],
        learning_rate=meta_info["learning_rate"],
        is_adamw=meta_info["is_adamw"],
        weight_decay=meta_info["weight_decay"],
        # scheduler_type="WarmupLinear",
        # scheduler_kwargs=dict(),
        # do_allen_nlp_predictions=False,
        path_log_results=meta_info["path_log_results"],
    )
    logger = pl.loggers.WandbLogger(
        project=WANDB_PROJECT,
        name=meta_info["run_name"],
        entity=WANDB_ENTITY,
        log_model=False,
        config=dict(
            meta_info=meta_info,
            num_gpus=NUM_GPUS,
            precision=PRECISION,
            arguments=all_arguments,
        ),
    )
    wandb.run.log_code(SCRIPT_DIR)

    trainer = pl.Trainer(
        enable_checkpointing=pl.callbacks.ModelCheckpoint( # type: ignore[arg-type]
            dirpath=checkpoints_folder,
            every_n_epochs=1, 
            save_on_train_epoch_end=True, 
            save_last=True
        ),
        deterministic=DETERMINISTIC,
        default_root_dir=str(checkpoints_folder),
        logger=logger,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        precision=PRECISION,
        max_epochs=meta_info["max_epochs"],
        gpus=NUM_GPUS,
        check_val_every_n_epoch=EVAL_EVERY_N_EPOCHS,
        accumulate_grad_batches=meta_info.get("accumulate_grad_batches", 1),
    )
    
    if resuming:
        trainer.fit(pl_object, ckpt_path=path_last_checkpoint)
    else:
        trainer.fit(pl_object)


if __name__ == "__main__":
    fire.Fire(main)