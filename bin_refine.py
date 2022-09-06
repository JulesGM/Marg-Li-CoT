#!/usr/bin/env python3
# coding: utf-8

print("Importing modules.")
import collections
import dataclasses
import enum
import itertools
import json  # type: ignore[import]
import logging
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import *

from beartype import beartype
import fire  # type: ignore[import]
import h5py  # type: ignore[import]
import jsonlines as jsonl  # type: ignore
import numpy as np
import pretty_traceback  # type: ignore
import pytorch_lightning as pl
import rich
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm  # type: ignore
import transformers  # type: ignore[import]
import wandb

import general_shared_constants as constants
import general_utils as utils
import data_tokenizer

pretty_traceback.install()
print("Done loading modules.\n")


###############################################################################################
# Constants that should be changed from time to time
###############################################################################################
class TokenizerModes(str, enum.Enum):
    PRETRAINED = "pretrained"
    ARITHMETIC = "arithmetic"


ACCELERATOR = "cuda"
DEFAULT_WANDB_ID = None  # "336o97pe"
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DISTRIBUTE_STRATEGIES = "ddp_find_unused_parameters_false"  # "ddp"
DATA_MODE = constants.DataModes.HDF5_PRETOK # constants.DataModes.JSONL
TOKENIZER_MODE = TokenizerModes.PRETRAINED
CUSTOM_MODEL_CONFIG = dict(
    n_embd=64,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
)
DEFAULT_GENERATION_KWARGS = {
    constants.PipelineModes.VALIDATION: 
    dict(
        num_beams=1,
        sample=False,
        min_length=0,
        use_cache=True,
        do_sample=False,
        constraints=None,
        max_new_tokens=80,
        repetition_penalty=None,
    ),
    constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING: 
    dict(
        num_beams=5, 
        num_beam_groups=5,
        max_new_tokens=122,
        num_return_sequences=5, 
        diversity_penalty=0.25,
    ),
}


###############################################################################################
# Should not change
###############################################################################################
SCRIPT_DIR = Path(__file__).absolute().parent

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wandb stuff
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WANDB_ENTITY = "julesgm"
WANDB_PROJECT = "SAG"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training loop stuff
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
EVAL_EVERY_N_EPOCHS = 1
LIMIT_VAL_BATCHES = 20
LIMIT_TRAIN_BATCHES = None
DETERMINISTIC = False
DEFAULT_CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Gradients and optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GRADIENT_CLIP_VAL = 0.1
GRADIENT_CLIP_ALGORITHM = "value"
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_USE_ADAMW = False
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data / Dataloaders
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATALOADER_NUM_WORKERS = 0 # int(os.environ.get("SLURM_CPUS_PER_TASK", 6)) - 1
SHUFFLE_TRAINING_DATA = True
SHUFFLE_VALIDATION_DATA = True
DATA_PATH = SCRIPT_DIR / "data"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Varia
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PRECISION = 16
DEFAULT_HUGGING_FACE = "distilgpt2"

def _print_predictions(*, inputs, masks, generated_decoded, labels, all_generated, all_labels):
    for in_, mask, gen, ref, all_g, all_l in zip(inputs, masks, generated_decoded, labels, all_generated, all_labels):
        if gen == ref:
            color == "green"
        else:
            color = "yellow"
        rich.print(f"[bold {color}]\[gen-input][/] {in_}")
        rich.print(f"[bold {color}]\[gen-mask][/] {mask}")
        rich.print(f"[bold blue]\[gen-reference][/] {ref}")
        rich.print(f"[bold {color}]\[gen-generated][/] {gen}")
        rich.print(f"[bold]\[gen-all-labels] {all_l}")
        rich.print(f"[bold]\[gen-all-gen] {all_g}")
        rich.print(f"[bold]" + "=" * 80)


class _RefineLM(pl.LightningModule):
    def __init__(
        self,
        *,
        wandb_logger: Optional[pl.loggers.WandbLogger],
        model: transformers.PreTrainedModel,
        datasets: Dict[str, torch.utils.data.Dataset],
        tokenizer: transformers.PreTrainedTokenizer,
        batch_sizes: Dict[str, int],
        generation_kwargs: dict[str, Any],
        learning_rate: float,
        is_adamw: bool,
        weight_decay: Optional[float],
        path_log_results: Path,
        scheduler_type,
        scheduler_kwargs,
        meta_info: dict,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "datasets", "tokenizer"])
        assert scheduler_type is None, "scheduler support is not yet implemented"
        assert scheduler_kwargs is None, "scheduler support is not yet implemented"

        self._dataloader_num_workers: Final[int] = DATALOADER_NUM_WORKERS
        self._wandb_logger: Final[pl.loggers.WandbLogger] = wandb_logger
        self._model: Final[transformers.PreTrainedModel] = model
        self._datasets: Final[Dict[str, torch.utils.data.Dataset]] = datasets
        self._tokenizer: Final[transformers.PreTrainedTokenizer] = tokenizer
        self._batch_size: Final[dict[str, int]] = batch_sizes
        self._generation_kwargs: Final[dict[str, Any]] = generation_kwargs
        self._logging_conf: Final[dict[str, bool]] = dict(
            prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        self._meta_info = meta_info

        ################################################################################
        # Related to datasets
        ################################################################################
        self._shuffle_train: Final[bool] = SHUFFLE_TRAINING_DATA
        self._shuffle_val: Final[bool] = SHUFFLE_VALIDATION_DATA
        self._active_training_mode: Final[str] = constants.PipelineModes.MLE_TRAINING
        self._training_collators = {
            constants.PipelineModes.MLE_TRAINING: MLETrainingCollator(self._tokenizer),

        }

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
        self._scheduler_type =         scheduler_type
        self._scheduler_kwargs =       scheduler_kwargs


    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)


    def _training_step_mle(self, batch, batch_idx):
        utils.check_equal(
            self._active_training_mode, 
            constants.PipelineModes.MLE_TRAINING,
        )

        assert "labels" in batch, (
            "Labels must be in batch. We must mask the input section with -100"
        )

        batch = {
            k: v for k, v in batch.items() 
            if k in ["input_ids", "attention_mask", "labels"]
        }
        bsz = batch["attention_mask"].shape[0]
        
        # input_ids_eos = self._tokenizer.eos_token_id * torch.ones([bsz, 1], dtype=batch["input_ids"].dtype).to(batch["input_ids"].device)
        # attention_mask_eos = torch.ones([bsz, 1], dtype=batch["attention_mask"].dtype).to(batch["attention_mask"].device)
        # batch["input_ids"] = torch.concat([batch["input_ids"], input_ids_eos], axis=1)
        # batch["attention_mask"] = torch.concat([batch["attention_mask"], attention_mask_eos], axis=1)

        if batch_idx == 0:
            all_masks = []
            for input_entry, mask_entry in zip(batch["input_ids"], batch["attention_mask"]):
                sent_masks = []
                for input_, mask_ in zip(input_entry, mask_entry):
                    sent_masks.append((
                        self._tokenizer.decode([input_.item()], skip_special_tokens=False), 
                        mask_.item()
                    ))
                all_masks.append(sent_masks)

            all_labels = []            
            for input_entry, label_entry in zip(batch["input_ids"], batch["labels"]):
                sent_labels = []
                for input_, label in zip(input_entry, label_entry):
                    sent_labels.append((
                        self._tokenizer.decode([input_.item()], skip_special_tokens=False), 
                        self._tokenizer.decode([label.item()], skip_special_tokens=False) if label.item() != -100 else "-100", 
                    ))
                all_labels.append(sent_labels)
            print("asdasd")

        outputs = self(**batch)

        self.log(
            "train_loss", outputs.loss, 
            batch_size=self._batch_size[self._active_training_mode], 
            **self._logging_conf
        )

        return outputs.loss

    def _training_step_marginal_likelihood(self, batch, batch_idx):
        """
        
        p(z|x): <generation>
            input_ids: masked, question, chainer
            keep the logits. 

        p(z|x): <after generation>
            input_ids: masked, question, chainer
            Labels: whatever the model has generated, value.
            Notes: 
                - We could keep the logits in generation and extract from that.
                - We need to add the value at the end so that we don't need to recompute everything.
                - We will need scratchpad masks and value masks, to extract the logits.
        
        p(y|z, x):
            input_ids: masked, question, chainer, scratchpad, value
            labels: value
            Note: I think we need to recompute over everything. This is not optimal clearly.


        """
        mode: Final[str] = constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING
        utils.check_equal(self._active_training_mode, mode)

        # TODO: Not sure about the options in these
        outputs = self._model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"], 
            **self._generation_kwargs[mode],
            jules_verbose=False
        )

        utils.check_equal(outputs.shape, (
            self._batch_size[self._active_training_mode], 
            self._generation_kwargs[mode]["beam_size"], 
            self._generation_kwargs[mode]["max_length"]
        ))

        ## Concatenate final value
        # [input, generated_scratchpad, answer]
        z_knowing_x_val, z_knowing_x_mask = unpadded_concatenation(
            [batch["input_ids"], outputs], 
            self._tokenizer.pad_token_id
        )
        

        label_mask = torch.ones_like(z_knowing_x_mask) * -100
        assert label_mask.dtype == torch.long, label_mask.dtype
        y_knowing_x_z_val, y_knowing_x_z_mask = unpadded_concatenation(
            [label_mask, batch["value"]], -100
        )
        
        # Compute loss
        prob = self._model(input_ids=y_knowing_x_z_val)

        pass


    def training_step(self, batch, batch_idx):
        if self._active_training_mode == constants.PipelineModes.MLE_TRAINING:
            return self._training_step_mle(batch, batch_idx)
        elif (
            self._active_training_mode == 
            constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING
        ):
            return self._training_step_marginal_likelihood(batch, batch_idx)
        else:
            raise ValueError(f"Unknown training mode: {self._active_training_mode}")


    def _generate(self, batch, generation_kwargs):
        assert "labels" in batch, "Labels must be in batch. We must mask the input section with -100"

        generation_inputs = batch["generation_input_ids"]
        generation_attention_mask = batch["generation_attention_mask"]

        outputs = self._model.generate(
            input_ids=generation_inputs, 
            attention_mask=generation_attention_mask, 
            **generation_kwargs,
        )
        
        generated_decoded = [
            self._tokenizer.decode(x, skip_special_tokens=False)
            for x in outputs]
        output_label = [
            self._tokenizer.decode(x, skip_special_tokens=False)
            for x in batch["input_and_scratchpad_with_value"]]
        inputs = [
            self._tokenizer.decode(x, skip_special_tokens=False)
            for x in generation_inputs
        ]

        all_masks = []
        for mask_entry, input_entry in zip(batch["generation_attention_mask"], generation_inputs, ):
            partial = []
            for mask, input_ in zip(mask_entry, input_entry.tolist()):
                partial.append((mask.item(), self._tokenizer.decode([input_], skip_special_tokens=False)))
            all_masks.append(partial)

        all_labels = []
        for label_entry, input_entry in itertools.zip_longest(batch["labels"], generation_inputs, fillvalue="<FILL_VALUE>"):
            partial = []
            for label in label_entry.tolist():
                decoded_label = self._tokenizer.decode([label], skip_special_tokens=False) if label > 0 else "<-100>"
                partial.append(decoded_label)
            all_labels.append(partial)

        all_generated = []
        for label_entry, gen_entry in itertools.zip_longest(batch["labels"], outputs, fillvalue="<FILL_VALUE>"):
            partial = []
            for label, gen_ in zip(label_entry.tolist(), gen_entry.tolist()):
                decoded_label = self._tokenizer.decode([label], skip_special_tokens=False) if label > 0 else "<-100>"
                decoded_gen = self._tokenizer.decode([gen_], skip_special_tokens=False)
                partial.append((decoded_label, decoded_gen))
            all_generated.append(partial)

        return inputs, all_masks, generated_decoded, output_label, all_generated, all_labels


    def validation_step(self, batch: Dict[str, torch.LongTensor], batch_idx):  # type: ignore[override]
        assert "labels" in batch, (
            "Labels must be in batch. We must mask the input section with -100"
        )
        mode: Final[str] = constants.PipelineModes.VALIDATION

        inputs, masks, generated_decoded, labels, all_generated, all_labels = self._generate(batch, self._generation_kwargs[mode])
        # if batch_idx == 0:
        #     _print_predictions(
        #         inputs=inputs, masks=masks, generated_decoded=generated_decoded, 
        #         labels=labels, all_generated=all_generated, all_labels=all_labels,
        #     )
            
        #     ###################################################################
        #     # Log Generated Text in Wandb.
        #     # Logging side by side makes the table unreadable.
        #     ###################################################################
        #     table_entry = []
        #     for index, (gen, lab) in enumerate(zip(generated_decoded, labels)):
        #         table_entry.append([self.current_epoch, index, "generation", gen])
        #         table_entry.append([self.current_epoch, index, "label", lab])

        #     self._wandb_logger.log_text(
        #         key="samples",
        #         columns=["epoch", "idx_in_batch", "type", "text"], 
        #         data=table_entry,
        #     )
        #     ###################################################################

        for_comparison = [(_clean_for_accuracy_computation(gen, self._tokenizer), _clean_for_accuracy_computation(l, self._tokenizer)) for gen, l in zip(generated_decoded, labels)]
        if batch_idx == 0 and os.environ["SLURM_PROCID"] == "0":
            for gen, ref in for_comparison:
                rich.print(f"[bold yellow]\[ref] {ref}")
                rich.print(f"[bold blue]\[gen] {gen}")
                rich.print("=" * 80)

        accuracy = np.mean([x == y for x, y in for_comparison])

        ppl_outputs = self._model(**{k: batch[k]for k in 
            ["input_ids", "attention_mask", "labels"]})

        self.log("val_em", accuracy, batch_size=self._batch_size[mode], **self._logging_conf)
        self.log("val_loss", ppl_outputs.loss, batch_size=self._batch_size[mode], **self._logging_conf)
        

        return ppl_outputs


    def predict_step(self, batch, batch_idx):
        batch = cast(Dict[str, torch.LongTensor], batch)
        mode = constants.PipelineModes.VALIDATION
        generated_decoded, label = self._generate(batch, self._generation_kwargs[mode])
        _print_predictions(generated_decoded, label)


    def on_validation_epoch_end(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
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
            capturable=False, # type: ignore
        )

        return dict(optimizer=optimizer)


    def train_dataloader(self):        
        return torch.utils.data.DataLoader(
            self._datasets[self._active_training_mode],
            collate_fn=self._training_collators[self._active_training_mode],
            batch_size=self._batch_size[self._active_training_mode],
            num_workers=self._dataloader_num_workers,
            shuffle=self._shuffle_train,
        )


    def val_dataloader(self):
        mode: Final[str] = constants.PipelineModes.VALIDATION
        return torch.utils.data.DataLoader(
            self._datasets[mode],
            collate_fn=ValitationCollator(self._tokenizer),
            batch_size=self._batch_size[mode],
            num_workers=self._dataloader_num_workers,
            shuffle=self._shuffle_val,
        )

    
    def predict_dataloader(self):
        return self.val_dataloader()


    def on_save_checkpoint(self, ckpt):
        return 
        # TODO: save info that varies
        # utils.dump_json(
        #     self._meta_info,
        #     _make_config_path(
        #         checkpoints_root_dir=self._checkpoints_root_dir, 
        #         run_name=self._meta_info["run_name"], 
        #         wandb_run_id=self._meta_info["wandb_run_id"],
        #         epoch=self.current_epoch,
        #         step=self.current_step,
        #     ),
        # )


def unpadded_concatenation(tensors, pad_token_id):
    lists_of_lists = [semi_vectorized_masked_2d_to_lol(x, x==pad_token_id) for x in tensors]
    concatenated = [list(itertools.chain(*list_of_lists)) for list_of_lists in zip(*lists_of_lists)]
    mask = generate_mask(concatenated, pad_token_id)
    padded = pad(concatenated, pad_token_id,)
    return padded, mask


def _clean_for_accuracy_computation(text, tokenizer):
    return re.sub(r"\s+", " ", text.replace(tokenizer.eos_token, "").strip())


@dataclasses.dataclass
class LastCkptInfo:
    path: Path
    run_name: str
    epoch: int
    step: int


def _get_last_checkpoint_path(
    checkpoints_folder, 
    run_name: Optional[str] = None, 
    wandb_run_id: Optional[str] = None
) -> LastCkptInfo:

    if wandb_run_id is None:
        return None

    rich.print(f"\n[red bold]_get_last_checkpoint_path: {wandb_run_id = }")

    if run_name:
        dir_ = checkpoints_folder / run_name / wandb_run_id / "checkpoints"
        checkpoints = list((dir_).glob("*.ckpt"))
        
    else:
        checkpoints = []
        for path in checkpoints_folder.glob("**/*.ckpt"):
            assert path.parent.name == "checkpoints", (
                f"{path.parent.name} != checkpoints")
            if path.parent.parent.name == wandb_run_id:
                checkpoints.append(path)

    if not checkpoints:
        return LastCkptInfo(None, run_name, None, None)

    assert len(checkpoints) == 1, checkpoints
    checkpoint_path = checkpoints[0]
    
    if run_name is None:
        # We recover the run name from the wandb run id
        run_name = path.parent.parent.parent.name
        rich.print(f"\n[red bold]Inferring `run_name` value of: {run_name = !s}")

    match_obj = re.match(r"epoch=(\d+)-step=(\d+).ckpt", checkpoint_path.name)
    epoch = int(match_obj.group(1))
    step = int(match_obj.group(2))

    rich.print(f"\n[red bold]_get_last_checkpoint_path: {checkpoint_path = !s}")

    return LastCkptInfo(checkpoint_path, run_name, epoch, step)


def _json_default_paths(entry: Any):
    if isinstance(entry, Path):
        return str(entry)
    return entry


def _set_resumed_state(
    checkpoints_root_dir: Union[Path, str], 
    arg_meta_info: dict[str, Any], 
    last_ckpt_info: LastCkptInfo,
) -> dict[str, Any]:

    """Resumes things that are in the global state, 
    ie. the wandb run and the random seeds and states.
    """
    checkpoints_root_dir = Path(checkpoints_root_dir)
    meta_info_path = _make_config_path(
        checkpoints_root_dir=checkpoints_root_dir, 
        run_name=arg_meta_info["run_name"], 
        wandb_run_id=arg_meta_info["wandb_run_id"],
        step=last_ckpt_info.step,
        epoch=last_ckpt_info.epoch,
    )
    meta_info = utils.load_json(meta_info_path)

    # Check that the values that need to match do match
    arg_meta_info = arg_meta_info.copy()
    none_or_equal = {
        "run_name", "seed", "wandb_run_id", 
        "transformers_model_name", "run_name"
    }
    none_or_absent = {
        "torch_rng_state", "numpy_rng_state", 
        "python_rng_state"
    }

    for k in none_or_equal:
        arg_val = arg_meta_info.pop(k)
        assert arg_val is None or arg_val == meta_info[k], (
            arg_val, meta_info[k])
    
    for k in none_or_absent:
        if k in arg_meta_info:
            arg_val = arg_meta_info.pop(k)
            assert arg_val is None, arg_val
    
    # We should have no remaining keys
    # assert not arg_meta_info, arg_meta_info

    # Load the variables
    wandb_run_id = meta_info["wandb_run_id"]
    seed = meta_info["seed"]

    # TODO: save the random states
    # torch_rng_state = meta_info["torch_rng_state"]
    # numpy_rng_state = meta_info["numpy_rng_state"]
    # python_rng_state = meta_info["python_rng_state"]
    # # run_name = meta_info["run_name"]
    # # transformers_model_name = meta_info["transformers_model_name"]

    # # Deal with random seeds and states
    # torch.cuda.manual_seed_all(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.random.set_rng_state(torch.ByteTensor(torch_rng_state))
    # np.random.set_state(numpy_rng_state)
    # for i, v in enumerate(python_rng_state):
    #     if isinstance(v, list):
    #         python_rng_state[i] = tuple(
    #             python_rng_state[i])
    # random.setstate(tuple(python_rng_state))

    # Resume the wandb run
    rich.print("\n[red bold]Resuming Wandb run:", wandb_run_id)

    
    return meta_info


def _set_initial_state(
    checkpoints_root_dir: Union[Path, str], 
    arg_meta_info: dict[str, Any], 
    global_rank: int,

) -> tuple[dict[str, Any], pl.loggers.WandbLogger]:
    """
    Sets the initial state of the global state, ie. the wandb run and the random seeds and states.

    checkpoints_root_dir
    arg_meta_info: dict[str, Any], 
    global_rank: int,

    """
    checkpoints_root_dir = Path(checkpoints_root_dir)
    

    assert ("wandb_run_id" not in arg_meta_info or 
        arg_meta_info["wandb_run_id"] is None), arg_meta_info 

    wandb_logger = pl.loggers.WandbLogger(
        project=WANDB_PROJECT,
        name=arg_meta_info["run_name"],
        entity=WANDB_ENTITY,
        log_model=False,
        config=dict(
            meta_info=arg_meta_info,
            accelerator="gpu",
            precision=PRECISION,
            arguments=arg_meta_info,
        ),
    )

    if global_rank == 0:
        wandb.run.log_code(SCRIPT_DIR)
        arg_meta_info["wandb_run_id"] = wandb.run.id

        # Deal with random seeds and states
        seed = arg_meta_info["seed"]
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        config_path = _make_config_path(
            checkpoints_root_dir, 
            arg_meta_info["run_name"], 
            arg_meta_info["wandb_run_id"], 
            0, 0)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        utils.dump_json(
            arg_meta_info, 
            config_path, 
            default=_json_default_paths,
        )

    return arg_meta_info, wandb_logger


def _build_meta_info(**kwargs):
    return kwargs


def semi_vectorized_masked_2d_to_lol(array: np.ndarray, mask: np.ndarray) -> List[List[Any]]:
    if isinstance(mask, np.ndarray):
        assert mask.dtype == bool, mask.dtype
    elif isinstance(mask, torch.Tensor):
        assert mask.dtype == torch.bool, mask.dtype
    else:
        raise ValueError(type(mask))

    utils.check_equal(array.shape, mask.shape)
    output = []
    for i in range(mask.shape[0]):
        vectorized_version = array[i][mask[i]]
        output.append(vectorized_version)

    return output


def _load_data(
    dataset_path: Union[str, Path], 
    tokenizer: transformers.PreTrainedTokenizer,
    mode: constants.DataModes,
    cv_sets: Optional[Iterable[str]],
):
    """Loads the textual entries, tokenizes them and returns a dict with the columns.
    The parallelization is done by the fast tokenizers, which are truely parallel with real Rust-based threads.
    There is no need to add more parallism here.
    """
    dataset_path = Path(dataset_path)
    
    if cv_sets is None:
        cv_sets = [
            constants.CVSets.TRAINING, 
            constants.CVSets.VALIDATION,
        ]
    tokenized_data = {}

    for set_ in cv_sets:
        start = time.perf_counter()
        if mode == constants.DataModes.JSONL:
            cv_path = dataset_path / f"{set_}.jsonl"
            
            with jsonl.open(cv_path) as f:
                rich.print(f"\n[bold]Loading a dataset file: [/bold]", str(cv_path))
                raw_data = list(f)
                rich.print(f"\n[bold]Done loading a dataset file: [/bold] {cv_path}, took {time.perf_counter() - start:0.2f}s", )

            chainer = " => "
            tokenized_data[set_] = {
                "input": tokenizer([x["input"] + chainer for x in raw_data], add_special_tokens=False)["input_ids"],
                "input_and_scratchpad_with_value": tokenizer([x["input"] + chainer + x["scratchpad_with_value"] for x in raw_data], add_special_tokens=False)["input_ids"],
                # "value_text": [x["value"] for x in raw_data]["input_ids"],
                "value": tokenizer([x["value"] for x in raw_data], add_special_tokens=False)["input_ids"],
                "scratchpad_with_value": tokenizer([x["scratchpad_with_value"] for x in raw_data], add_special_tokens=False)["input_ids"],
                "scratchpad": tokenizer([x["scratchpad"] for x in raw_data], add_special_tokens=False)["input_ids"],
            }

        elif mode == constants.DataModes.HDF5_PRETOK:
            cv_path = dataset_path / f"{set_}.h5"
            rich.print(f"\n[bold]Loading a dataset file: [/bold]", str(cv_path))
            with h5py.File(cv_path, "r") as f:
                keys_to_do = [key for key in f if not key.endswith("_text")]
                rich.print(f"Keys to do:")
                utils.print_list(keys_to_do)
                rich.print(f"Associated dtypes:")
                utils.print_dict({k: f[k].dtype for k in keys_to_do})
                rich.print(f"Associated shape:")
                utils.print_dict({k: f[k].shape for k in keys_to_do})
                for key in keys_to_do:
                    assert f[key].dtype != object, key

                cached = {k: f[k][:] for k in tqdm(keys_to_do, desc="Reading from file.")}
                for k, v in cached.items():
                    assert isinstance(v, np.ndarray), f"Field `{k}` is of type {type(v)}, expected np.ndarray."

            rich.print("Done reading the file.")
            for k, v in cached.items():
                rich.print(f"\t- {k}: {v.shape}")

            tokenized_data[set_] = {}
            mask_keys = set()
            ids_keys = set()
            text_keys = set()

            for key in cached.keys():
                if "attention_mask" in key:
                    mask_keys.add(key)
                elif not key.endswith("_attention_mask") and not key.endswith("_text"):
                    ids_keys.add(key)
                else:
                    text_keys.add(key)

            for key in ids_keys:
                assert (key + "_attention_mask") in mask_keys, (key, mask_keys)
            
            # Remove the padding
            # Dynamic padding makes everything easier to deal with.
            for key in tqdm(ids_keys, desc="Removing padding"):
                mask_key = key + "_attention_mask"
                tokenized_data[set_][key] = []

                start_norm_vec = time.perf_counter()                
                tokenized_data[set_][key] = semi_vectorized_masked_2d_to_lol(cached[key], cached[mask_key] == 1)
                print(f"Normal {time.perf_counter() - start_norm_vec:0.2f}s")

            for key in tqdm(text_keys, desc="Tokenizing"):
                tokenized_data[set_][key] = cached[key]

            rich.print(f"\n[bold]Done loading a dataset file: [/bold] {cv_path}, took {time.perf_counter() - start:0.2f}s", )
        
        else:
            raise ValueError(mode)

        delta = time.perf_counter() - start
        rich.print(f"\n[bold]Done preparing \"{cv_path.name}\". It took {delta:0.2f}s overall. ")

    return tokenized_data


class DictDataset(torch.utils.data.Dataset):
    """
    A dataset built from a dictionary with colums that fit the typing.Sequence protocol (eg, lists, tuples, np.ndarrays, torch.Tensors).
    The first dimension of the sequences needs to be of the same size.
    """
    def __init__(self, data: dict[str, Sequence[Any]]):
        lens = {k: len(v) for k, v in data.items()}
        assert len(set(lens.values())) == 1, lens
        self._len = lens[list(lens.keys())[0]]
        self._data = data

    def __getitem__(self, index: int) -> dict[str, Any]:
        # TODO: why do we convert to tensor every step
        return{k: torch.tensor(v[index]) for k, v in self._data.items()}

    def __len__(self) -> int:
        return self._len


def pad(seq : Sequence[Sequence[int]], pad_token_id: int, direction: str) -> torch.LongTensor:
    utils.check_equal(direction, "left")
    max_len = max(len(x) for x in seq)
    output = []
    for i, x in enumerate(seq):
        if not isinstance(x, list):
            assert isinstance(x, (torch.Tensor, np.ndarray)), type(x)
            x = x.tolist()

        if direction == "left":
            output.append([pad_token_id] * (max_len - len(x)) + x)

        elif direction == "right":
            output.append(x + [pad_token_id] * (max_len - len(x)))

        else:
            raise ValueError(direction)

    return torch.LongTensor(output)


def generate_mask(list_of_list: list[list[int]], direction: str) -> torch.LongTensor:
    assert isinstance(list_of_list, list), type(list_of_list)

    mask: list[torch.Tensor] = []
    for x in list_of_list:
        mask.append(torch.ones(len(x), dtype=torch.long))
    attention_mask = pad(mask, 0, direction)
    return attention_mask


def prep_mle_train_and_valid(examples, bos_token_id: int, eos_token_id: int) -> None:
    
    for example in examples:
        # Transormations
        example["input_ids"] = example["input_and_scratchpad_with_value"].tolist() + [eos_token_id]
        len_question = (len(example["input_and_scratchpad_with_value"]) - len(example["scratchpad_with_value"]))
        example["labels"] = [-100] * len_question + example["scratchpad_with_value"].tolist() + [eos_token_id]
        
        # End checks
        utils.check_equal(len(example["input_ids"]), len(example["labels"]))


@dataclasses.dataclass
class MarginalLikelihoodTrainingCollator:
    _tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, examples):
        """
        - We have the questions, we have the answers. Nothing else.

        Input ids: [question, chainer]
        Labels: [answer]

        loss: likelihoodOf[question, chainer, Generate(question), answer]

        """

        examples = utils.dict_unzip(examples)
        examples["attention_mask"] = generate_mask(examples["input"], "left")
        examples["input_ids"] = pad(examples["input"], self._tokenizer.pad_token_id, "left")

        return examples


@dataclasses.dataclass
class MLETrainingCollator:
    _tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, examples):
        """
        - For perplexity evaluation:
            (The chainer should already be in place for input_ids and input_ids_and_scratchpad_with_value)
            - input_ids: question + chainer (e.g., " -> ") + scratchpad + value
            - attention_mask: the same as above, but with 0s everywhere there is padding
            - labels: -100 except scratchpad + value (so, for the question, the chainer and the padding.)

        """
        prep_mle_train_and_valid(examples, self._tokenizer.bos_token_id, self._tokenizer.eos_token_id)

        examples = utils.dict_unzip(examples)
        examples = cast(dict[str, Union[Sequence[Any], torch.Tensor]], examples)
        examples["attention_mask"] = generate_mask(examples["input_ids"], "left")  # NEEDS TO BE BEFORE PAD
        examples["input_ids"] = pad(examples["input_ids"], self._tokenizer.pad_token_id, "left")
        examples["labels"] = pad(examples["labels"], -100, "left")

        return examples


@dataclasses.dataclass
class ValitationCollator:
    _tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, raw_examples):
        """
        We need:
        
        - For perplexity evaluation:
            (The chainer should already be in place for input_ids and input_ids_and_scratchpad_with_value)
            - input_ids: question + chainer (e.g., " -> ") + scratchpad + value
            - attention_mask: the same as above, but with 0s everywhere there is padding
            - labels: -100 except scratchpad + value (so, for the question, the chainer and the padding.)

        - For generation evaluation:
            (The chainer should already be in place for input_ids and input_ids_and_scratchpad_with_value)
            - generation_input_ids: question + chainer
            - generation_attention_mask: the same as above, but with 0s everywhere there is padding
        
        - To verify the generation:
            - value text 

        """
        
        prep_mle_train_and_valid(raw_examples, self._tokenizer.bos_token_id, self._tokenizer.eos_token_id)

        examples = utils.dict_unzip(raw_examples)
        examples["attention_mask"] = generate_mask(examples["input_ids"], "left")  # NEEDS TO BE BEFORE PAD
        examples["input_ids"] = pad(examples["input_ids"], self._tokenizer.pad_token_id, "left")
        examples["labels"] = pad(examples["labels"], -100, "left")

        examples["generation_input_ids"] = pad(examples["input"], self._tokenizer.pad_token_id, "left")
        examples["generation_attention_mask"] = generate_mask(examples["input"], "left")
    
        return examples


def _text_mode_build_dataset(
    dataset_path: Path, tokenizer: transformers.PreTrainedTokenizer, cv_sets: Optional[Sequence[str]]
) -> dict[str, DictDataset]:
    """
    The following returns a dict with a subset of columns depending on the cv set and 
    the pipeline mode.

    We first make a list of the columns that we want to keep per pipeline mode.
    We then iterate on the cv sets of data that we have, find the associated pipeline modes,
    and subset the columns per pipeline mode.

    An important point is that a pipeline modes only have one cv set associated to them.
    The reverse is not true, cvsets can have multiple pipeline modes, and do, in the case 
    of training (MLE and Marginal Likelihood).
    """

    tokenized_data = _load_data(dataset_path, tokenizer, DATA_MODE, cv_sets=cv_sets)
    assert tokenized_data    
    output_datasets = {}

    ds_key_filter = {
        constants.PipelineModes.MLE_TRAINING: {
            "input_and_scratchpad_with_value",
            "scratchpad_with_value",
        },
        
        constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING: {
            "input",
            "value",
        },

        constants.PipelineModes.VALIDATION: {
            "scratchpad_with_value",
            "input_and_scratchpad_with_value",
            "input",
            "value",
        },
    }

    for cv_set in cv_sets:
        for pipeline_mode in constants.CV_SETS_TO_PILELINES_MODES[cv_set]:
            keys = ds_key_filter[pipeline_mode]
            columns = {k: tokenized_data[cv_set][k] for k in keys}
            assert pipeline_mode not in output_datasets
            output_datasets[pipeline_mode] = DictDataset(columns)

    return output_datasets


class DDPInfo:
    def __init__(self, distribute_strategy: str):
        if distribute_strategy is not None:
            assert (
                distribute_strategy == "ddp" or 
                distribute_strategy == "ddp_find_unused_parameters_false"), (
                "Only ddp is supported for now."
            )

            self.num_nodes = int(os.environ["SLURM_NNODES"])
            self.num_devices = int(len(os.environ["SLURM_JOB_GPUS"].split(",")))
            self.global_rank = int(os.environ["SLURM_PROCID"])
            self.local_rank = int(os.environ["SLURM_LOCALID"])
            self.node_rank = int(os.environ["SLURM_NODEID"])
        
            rich.print("[bold green]Distributed Data Parallel (DDP) enabled.")
            rich.print(f"[bold green]\t- NUM_NODES:   {self.num_nodes}")
            rich.print(f"[bold green]\t- NUM_DEVICES: {self.num_devices}")
        else:
            self.num_nodes = None
            self.num_devices = 1
            self.global_rank = 0
            self.local_rank = 0
            self.node_rank = None
            
    def __repr__(self):
        return f"{type(self).__name__}(" + ", ".join([f"{k}={v}" for k, v in self.__dict__.items()]) + ")"


def _setup_ddp(distribute_strategy: str) -> DDPInfo:
    ddp_info = DDPInfo(distribute_strategy)
    
    if ddp_info.global_rank > 0:
        assert distribute_strategy in ["ddp", "ddp_find_unused_parameters_false"]

    return ddp_info

def _setup_tokenizer(hf_name: str) -> transformers.PreTrainedTokenizer:
    if TOKENIZER_MODE == TokenizerModes.ARITHMETIC:
        return data_tokenizer.ArithmeticTokenizer()
    elif TOKENIZER_MODE == TokenizerModes.PRETRAINED:

        rich.print(f"\n[bold]Loading tokenizer.")
        tokenizer = transformers.AutoTokenizer.from_pretrained(hf_name)
        utils.setattr_must_exist(tokenizer, "padding_side", "left")

        # Ad hoc things like this aren't great, but we're a small project, not production.
        if "gpt2" in hf_name:
            utils.setattr_must_exist(tokenizer, "pad_token", tokenizer.eos_token)
        
        rich.print(f"Tokenizer loaded.")
        return tokenizer
    else:
        raise ValueError(f"Unsupported tokenizer mode: {TOKENIZER_MODE}")


def _setup_base_model(
    hf_name: str, 
    custom_model_config: Optional[dict[str, int]], 
    tokenizer: transformers.PreTrainedTokenizer,
    verbose: bool = True
) -> transformers.PreTrainedModel:
    
    config = transformers.AutoConfig.from_pretrained(hf_name)
    
    if custom_model_config is not None:
        for k, v in custom_model_config.items():
            utils.setattr_must_exist(config, k, v)
    
        utils.setattr_must_exist(config, "vocab_size", len(tokenizer.vocab))

    assert config.n_inner is None, config.n_inner

    base_model = transformers.AutoModelForCausalLM.from_config(config)
    utils.setattr_must_exist(base_model.config, "early_stopping", True)

    if base_model.config.model_type == "gpt2" and isinstance(
        tokenizer, (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast)):
        utils.check_equal(TOKENIZER_MODE, TokenizerModes.PRETRAINED)
        utils.setattr_must_exist(base_model.config, "pad_token_id", base_model.config.eos_token_id)
    else:
        utils.check_equal(TOKENIZER_MODE, TokenizerModes.ARITHMETIC)
        utils.check_isinstance(tokenizer, data_tokenizer.ArithmeticTokenizer)
        utils.setattr_must_exist(base_model.config, "pad_token_id", tokenizer.pad_token_id)
        utils.setattr_must_exist(base_model.config, "bos_token_id", tokenizer.bos_token_id)
        utils.setattr_must_exist(base_model.config, "eos_token_id", tokenizer.eos_token_id)

    utils.check_equal(base_model.config.pad_token_id, tokenizer.pad_token_id)
    utils.check_equal(base_model.config.bos_token_id, tokenizer.bos_token_id)
    utils.check_equal(base_model.config.eos_token_id, tokenizer.eos_token_id)
    utils.check_equal(base_model.config.vocab_size, len(tokenizer.vocab))

    return base_model


def _compute_batch_size_defaults(
    local_rank: int, hf_name: str, batch_sizes: Optional[dict[str, int]], accelerator
) -> dict[str, int]:
    """Ad-hoc function for default the batch sizes.
    """
    if accelerator == "cpu":
        base = 2
    else:
        base = 64 * 1
    return {
            constants.PipelineModes.MLE_TRAINING: base,
            constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING: base,
            constants.PipelineModes.VALIDATION: base * 2,
        }  

    
    assert isinstance(local_rank, int)
    gpu_mem_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1024 ** 3
    if batch_sizes is None and hf_name == "distilgpt2" and gpu_mem_gb > 44:
        base = 64 * 4
        batch_sizes = {
            constants.PipelineModes.MLE_TRAINING: base,
            constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING: base,
            constants.PipelineModes.VALIDATION: base * 2,
        }  #384

    elif batch_sizes is None and hf_name == "distilgpt2" and gpu_mem_gb > 14:
        base = 64 * 4
        batch_sizes = {
            constants.PipelineModes.MLE_TRAINING: base,
            constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING: base,
            constants.PipelineModes.VALIDATION: base * 2,
        }

    else:
        raise ValueError("We don't know what batch size to use for this GPU.")
    return batch_sizes


@beartype
def _make_config_path(checkpoints_root_dir: Path, run_name: str, wandb_run_id: str, step: int, epoch: int) -> Path:
    return checkpoints_root_dir / run_name / wandb_run_id / "checkpoints" / f"epoch={epoch}-step={step}.json"


DATA_DIR = SCRIPT_DIR / "data"

class EntryPoints:
    @classmethod
    @beartype
    def main(
        cls,
        *, 
        seed: int = 453345,
        dataset_path: Union[Path, str] = DATA_DIR / "basic_arithmetic/80_3_6_200000",  # DATA_DIR / "basic_arithmetic/349_6_6_200000"
        path_log_results=DEFAULT_CHECKPOINTS_DIR / "logs",
        scheduler_type=None,
        batch_sizes=None,
        scheduler_kwargs=None,
        switch_to_maginal_after=False,
        checkpoints_folder: Union[Path, str] = DEFAULT_CHECKPOINTS_DIR,
        wandb_run_id: Optional[str] = DEFAULT_WANDB_ID,
        is_adamw: bool = DEFAULT_USE_ADAMW,
        weight_decay: Optional[float] = DEFAULT_WEIGHT_DECAY,
        transformers_model_name: str = DEFAULT_HUGGING_FACE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        generation_kwargs=DEFAULT_GENERATION_KWARGS,
        distribute_strategy=DEFAULT_DISTRIBUTE_STRATEGIES,  # "ddp_find_unused_parameters_false",
        custom_model_config=CUSTOM_MODEL_CONFIG,
    ):
        all_arguments = locals().copy()
        utils.check_and_print_args(all_arguments, cls.main, True, SCRIPT_DIR)

        if TOKENIZER_MODE == TokenizerModes.ARITHMETIC:
            assert DATA_MODE == constants.DataModes.JSONL, (
                f"We only support JSONL for arithmetic tokenizer, as things "
                "are pre-tokenized in the h5 mode. {DATA_MODE}"
            )

        dataset_path = Path(dataset_path)
        assert dataset_path.exists(), dataset_path

        checkpoints_folder = Path(checkpoints_folder)
        assert checkpoints_folder.exists(), checkpoints_folder
        assert checkpoints_folder.is_dir(), checkpoints_folder

        torch.use_deterministic_algorithms(mode=DETERMINISTIC)
        run_name = dataset_path.name + "_0.01_wd"   
        last_ckpt_info = _get_last_checkpoint_path(
            checkpoints_folder, None, wandb_run_id)
        resuming = wandb_run_id is not None
        if resuming:
            assert last_ckpt_info is not None, last_ckpt_info

        if resuming:
            latest_checkpoint = last_ckpt_info.path
            rich.print(f"[bold red] Will resume from \"{latest_checkpoint}\"")
        else:
            latest_checkpoint = None
            rich.print(f"[bold green]Not resuming: Will start from scratch.")

        ddp_info = _setup_ddp(distribute_strategy)

        if batch_sizes is None:
            batch_sizes = _compute_batch_size_defaults(
                ddp_info.local_rank, transformers_model_name, batch_sizes, ACCELERATOR
            )

        arg_meta_info = _build_meta_info(
            batch_sizes=batch_sizes,
            checkpoints_folder=checkpoints_folder,
            dataset_path=dataset_path,
            generation_kwargs=generation_kwargs,
            is_adamw=is_adamw,
            learning_rate=learning_rate,
            num_devices=ddp_info.num_devices,
            num_nodes=ddp_info.num_nodes,
            path_log_results=path_log_results,
            run_name=run_name,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_type=scheduler_type,
            seed=seed, 
            transformers_model_name=transformers_model_name,
            wandb_run_id=wandb_run_id,
            weight_decay=weight_decay,
            custom_model_config=custom_model_config,
        )
        
        # Load the pretrained model. If a checkpoint is used, it will
        # be loaded with the trainer.fit call, further in the code.
        
        
        if resuming:
            rich.print("\n[bold red]Resuming from checkpoint:[/]", latest_checkpoint)
            meta_info = _set_resumed_state(checkpoints_folder, arg_meta_info, last_ckpt_info)
            
            del arg_meta_info
            logger = pl.loggers.WandbLogger(
                resume="must", 
                id=wandb_run_id,
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                log_model=False,
                name=meta_info["run_name"],
                config=dict(
                    num_nodes=ddp_info.num_nodes,
                    num_devices=ddp_info.num_devices,
                    meta_info=meta_info,
                    precision=PRECISION,
                    arguments=all_arguments,
                ),
            )

            if ddp_info.global_rank == 0:
                assert wandb.run
                wandb.run.log_code(SCRIPT_DIR)
        else:
            rich.print("\n[bold green]Not Resuming: Setting the initial state.")
            meta_info, logger = _set_initial_state(
                checkpoints_folder, arg_meta_info, ddp_info.global_rank)
            del arg_meta_info

        tokenizer = _setup_tokenizer(meta_info["transformers_model_name"])
        base_model = _setup_base_model(meta_info["transformers_model_name"], meta_info["custom_model_config"], tokenizer)
        rich.print(f"\n[bold]Run name:[/bold] [green]\"{meta_info['run_name']}\"\n")
        datasets = _text_mode_build_dataset(dataset_path, tokenizer, 
            [constants.CVSets.TRAINING, constants.CVSets.VALIDATION]
        )

        rich.print(f"[bold red]Strategy: {distribute_strategy}[/]")
        rich.print(f"[bold red]ddp_info: {vars(ddp_info)}[/]")

        ###############################################################
        # Build the pt-lightning dataloader
        ###############################################################
        pl_object = _RefineLM(
            batch_sizes=batch_sizes,
            wandb_logger=logger,
            model=base_model,
            datasets=datasets,
            tokenizer=tokenizer,
            generation_kwargs=meta_info["generation_kwargs"],
            learning_rate=meta_info["learning_rate"],
            path_log_results=meta_info["path_log_results"],
            is_adamw=meta_info["is_adamw"],
            weight_decay=meta_info["weight_decay"],
            scheduler_type=meta_info["scheduler_type"],
            scheduler_kwargs=meta_info["scheduler_kwargs"],
            meta_info=meta_info,
        )

        ###############################################################
        # All of the follwing arguments are very stable
        ###############################################################
        trainer = pl.Trainer(
            accumulate_grad_batches=meta_info.get("accumulate_grad_batches", 1),
            enable_checkpointing=pl.callbacks.ModelCheckpoint( # type: ignore[arg-type]
                dirpath=checkpoints_folder,
                every_n_epochs=1, 
                save_on_train_epoch_end=True, 
                save_last=True
            ),
            default_root_dir=str(checkpoints_folder),
            max_epochs=10000,
            logger=logger,
            num_nodes=ddp_info.num_nodes,
            devices=ddp_info.num_devices,
            strategy=distribute_strategy,
            deterministic=DETERMINISTIC,
            gradient_clip_val=GRADIENT_CLIP_VAL,
            precision=PRECISION,
            accelerator=ACCELERATOR,
            check_val_every_n_epoch=EVAL_EVERY_N_EPOCHS,
            limit_val_batches=LIMIT_VAL_BATCHES,
            limit_train_batches=LIMIT_TRAIN_BATCHES,
        )
        
        if resuming:
            assert latest_checkpoint
            trainer.fit(pl_object, ckpt_path=str(latest_checkpoint))
        else:
            trainer.fit(pl_object)


    train = main


    @classmethod
    def json(cls, name: str, path: Path) -> Dict[str, Any]:
        """
        Run by loading a json file.
        """

        all_arguments = locals().copy()
        utils.check_and_print_args(all_arguments, cls.main, True)

        entrypoint_names = {
            k for k in cls.__dict__.keys() - {'json'} if not k.startswith("_")}
        assert hasattr(cls, name), f"{cls.__name__}.{name} doesn't exist. Valid options are: {entrypoint_names}"

        with open(path, "r") as f:
            args = json.load(f)
        
        return getattr(cls, name)(**args)


    @classmethod
    @beartype
    def predict(
        cls, 
        *,
        wandb_run_id: str = DEFAULT_WANDB_ID,
        run_name: Optional[str] = None,
        qty: int = 1,
        dataset_path: Path = DATA_DIR / "basic_arithmetic/80_3_6_200000",
        checkpoints_root_dir: Path = DEFAULT_CHECKPOINTS_DIR, 
        distribute_strategy: Optional[str] = None,
        batch_sizes: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Run by loading a json file.
        """
        all_arguments = locals().copy()
        utils.check_and_print_args(all_arguments, cls.predict, True)
        
        mode = constants.CVSets.VALIDATION
        last_ckpt_info = _get_last_checkpoint_path(
            checkpoints_root_dir, run_name, wandb_run_id)
        run_name = last_ckpt_info.run_name
        meta_info = utils.load_json(_make_config_path(
            checkpoints_root_dir=checkpoints_root_dir, 
            run_name=run_name, 
            wandb_run_id=wandb_run_id,
            step=last_ckpt_info.step,
            epoch=last_ckpt_info.epoch,
        ))
        base_model = _setup_base_model(meta_info["transformers_model_name"])
        tokenizer = _setup_tokenizer(meta_info["transformers_model_name"])  
        datasets = _text_mode_build_dataset(
            dataset_path, tokenizer, cv_sets=[constants.CVSets.VALIDATION])
        ddp_info = _setup_ddp(distribute_strategy)
        if batch_sizes is None:
            batch_sizes = _compute_batch_size_defaults(
                ddp_info.local_rank, meta_info["transformers_model_name"], batch_sizes)

        pl_object = _RefineLM(
            model=base_model,
            datasets=datasets,
            tokenizer=tokenizer,
            batch_sizes=meta_info["batch_sizes"],
            generation_kwargs=meta_info["generation_kwargs"],
            learning_rate=meta_info["learning_rate"],
            path_log_results=meta_info["path_log_results"],
            is_adamw=meta_info["is_adamw"],
            weight_decay=meta_info["weight_decay"],
            scheduler_type=meta_info["scheduler_type"],
            scheduler_kwargs=meta_info["scheduler_kwargs"],
            meta_info=meta_info,
        )

        trainer = pl.Trainer(
            deterministic=DETERMINISTIC,
            default_root_dir=str(checkpoints_root_dir),
            precision=PRECISION,
            accelerator=ACCELERATOR,
            limit_val_batches=LIMIT_VAL_BATCHES,
            strategy=distribute_strategy,
            num_nodes=ddp_info.num_nodes,
            devices=ddp_info.num_devices,
            limit_predict_batches=math.ceil(qty / batch_sizes[mode]),
        )

        trainer.predict(
            pl_object,
            ckpt_path=str(last_ckpt_info.path),
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Defaulting to the `main` entrypoint.")
        EntryPoints.main()
    else:
        fire.Fire(EntryPoints)
