#!/usr/bin/env python3
# coding: utf-8

"""

Some notes:

- Without a wandb_run_id -> Will not resume.
- With a wandb_run_id -> Will resume.
- With a wandb_run_id and new_wandb_run_id -> Will resume and create a new wandb run.

"""

print("Importing modules.")
import collections
import dataclasses
import enum
import itertools
import json  # type: ignore[import]
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import *

from beartype import beartype  # type: ignore[import]
import fire  # type: ignore[import]
import h5py  # type: ignore[import]
import jsonlines as jsonl  # type: ignore
import more_itertools  # type: ignore[import]
import numpy as np
import pretty_traceback  # type: ignore
import pytorch_lightning as pl  # type: ignore[import]
import rich.table as table
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm  # type: ignore[import]
import transformers  # type: ignore[import]
import wandb

import general_utils as utils

import console
import constants
import fast_ckpt_reader
import marginal
import pretrain
import train_utils
import with_trlx.trlx_exp as trlx_exp
    
Sequence.register(torch.Tensor)

CONSOLE = console.Console(force_terminal=True, force_interactive=True, width=200)

pretty_traceback.install()
print("Done loading modules.\n")

torch.autograd.set_detect_anomaly(True)
torch.multiprocessing.set_sharing_strategy("file_system")

CHAINER = " => "

###############################################################################################
# Constants that should be changed from time to time
###############################################################################################
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_STEP_3_LR = 0.0001
DEFAULT_STEP_3_LOSS_MODE = constants.LossModes.MARGINAL_KL_W_FIXED 

DEFAULT_SWITCH_TO_MARGINAL_AFTER: Final[Optional[dict[str, int]]] = 1
LIMIT_VAL_BATCHES = 10
VAL_CHECK_INTERVAL = 1 / 3
VAL_CHECK_INTERVAL_STEP_3 = 30

DEFAULT_NUM_BEAMS = 50
MARGINAL_LIKELIHOOD_BS = (64 * 2) // DEFAULT_NUM_BEAMS

DEFAULT_LM_MASKING_MODE = constants.LMMaskingMode.MASK_INPUT
DEFAULT_SHARED_BATCH_SIZE = 64 * 2
DEFAULT_GRADIENT_ACCUM = 2
DEFAULT_SCHEDULER_TYPE = constants.SchedulerTypes.LINEAR_WARMUP_CONSTANT

WARMUP_EPOCHS = 1
MAX_EPOCHS = 53

DEFAULT_WANDB_ID: Optional[str] = "22luh7ae"  # 22luh7ae, 1 epoch
DEFAULT_FIXED_ANSWER_MODEL_WANDB_RUN_ID: Optional[str] = "14i2wrva"
DEFAULT_DISTRIBUTE_STRATEGIES = "ddp"  # "ddp"

DATA_MODE = constants.DataModes.HDF5_PRETOK
TOKENIZER_MODE = constants.TokenizerModes.PRETRAINED

DEFAULT_HUGGING_FACE = "distilgpt2"
# DEFAULT_MODEL_MODE = constants.ModelModes.PRETRAINED
# DEFAULT_CUSTOM_MODEL_CONFIG: Optional[dict[str, Any]] = None

DEFAULT_MODEL_MODE = constants.ModelModes.RANDOM
DEFAULT_CUSTOM_MODEL_CONFIG = dict(
    n_ctx=64,
    n_embd=64,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
)

DEFAULT_GENERATION_KWARGS = {
    constants.PipelineModes.VALIDATION: 
    dict(
        num_beams=          1,
        min_length=         0,
        use_cache=          True,
        do_sample=          False,
        constraints=        None,
        max_new_tokens=     80,
        length_penalty=     1.0,
        repetition_penalty= None,
    ),
    constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING: 
    dict(
        min_length=     0,
        do_sample=      False,
        max_new_tokens= 80, # This is a very important knob
        
        # Not changing
        use_cache=            True,
        constraints=          None,
        repetition_penalty=   None,
        num_beams=            DEFAULT_NUM_BEAMS, 
        num_return_sequences= DEFAULT_NUM_BEAMS, 
        # diversity_penalty=0.25, # This needs to be tuned
        # num_beam_groups=DEFAULT_NUM_BEAMS,
    ),
}


# TODO: this is just a test, delete it
GLOBAL_MODEL_CONTAINER = None


###############################################################################################
# Should not change
###############################################################################################
SCRIPT_DIR = Path(__file__).absolute().parent
ACCELERATOR = "cuda"
DEFAULT_WANDB_CONFIG_PATH = SCRIPT_DIR / "wandb_config.json"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training loop stuff
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
EVAL_EVERY_N_EPOCHS = 1
DETERMINISTIC = False
DEFAULT_CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Gradients and optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GRADIENT_CLIP_VAL = 0.1
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_USE_ADAMW = False
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data / Dataloaders
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATALOADER_NUM_WORKERS = 0 # int(os.environ.get("SLURM_CPUS_PER_TASK", 6)) - 1
SHUFFLE_TRAINING_DATA = True
SHUFFLE_VALIDATION_DATA = True
DATA_PATH = SCRIPT_DIR / "data"
LIMIT_TRAIN_BATCHES = None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Varia
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PRECISION = "bf16" if torch.cuda.is_bf16_supported() else 32 # "mixed"
CONSOLE.print_zero_rank(f"\n[bold {'green' if PRECISION == 'bf16' else 'red'}]Precision:[/] {PRECISION}\n")

SCHEDULER_FN = {
    constants.SchedulerTypes.CONSTANT:
        lambda optimizer, steps_per_epoch:
            transformers.get_constant_schedule(
                optimizer
            ),

    constants.SchedulerTypes.LINEAR_WARMUP_CONSTANT: 
        lambda optimizer, steps_per_epoch:
            transformers.get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=steps_per_epoch * WARMUP_EPOCHS
            ),

    constants.SchedulerTypes.LINEAR_WARMUP_LINEAR: 
        lambda optimizer, steps_per_epoch:
            transformers.get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=steps_per_epoch * WARMUP_EPOCHS,
                num_training_steps=steps_per_epoch * MAX_EPOCHS,
            ),
}


def show_scratchpad_padding_table(
    *,
    mask_x,
    mask_y,
    tokenizer,
    batch_size,
    shift_prob,
    shift_MITGSWRV,
    demo_input_sp,
    demo_input_idx, 
    num_scratchpads,
    shift_MITGSWRV_is_pad,
    padded_final_input_ids,
):
    table_ = table.Table("input", "label", "mask_x", "mask_y", "pad_label", "label_log_prob", "label_prob")
    for input_, label, mask_x_, mask_y_, pad, prob in more_itertools.zip_equal(
        padded_final_input_ids.view(batch_size, num_scratchpads, -1)[demo_input_idx, demo_input_sp, :-1],
        shift_MITGSWRV                                             [demo_input_idx, demo_input_sp],
        mask_x                                                      [demo_input_idx, demo_input_sp], 
        mask_y                                                      [demo_input_idx, demo_input_sp], 
        shift_MITGSWRV_is_pad                                      [demo_input_idx, demo_input_sp].logical_not(), 
        shift_prob                                                  [demo_input_idx, demo_input_sp]
    ):
        table_.add_row(
            str(tokenizer.decode(input_)),
            str(tokenizer.decode(label )),
            str(mask_x_.item()), 
            str(mask_y_.item()), 
            str(pad.item()), 
            str(prob.item()),
        )
    CONSOLE.print_zero_rank(table_)



def show_small_generation_table(
    *, 
    scores,
    tokenizer, 
    demo_input_sp, 
    inputs_outputs, 
    demo_input_idx, 
):
    table_ = table.Table("tok", "logit", "prob")

    MITGSWRV_index_start = - scores[demo_input_idx][demo_input_sp].shape[0]
    tokens = inputs_outputs[demo_input_idx][demo_input_sp][MITGSWRV_index_start:]
    scores_to_show = scores[demo_input_idx][demo_input_sp]
    assert tokens is not None, tokens
    assert len(tokens), tokens
    assert scores_to_show is not None, scores_to_show
    assert len(scores_to_show), scores_to_show

    for tok, score in more_itertools.zip_equal(tokens, scores_to_show):
        table_.add_row(
            tokenizer.decode(tok), 
            str(score[tok].item()), 
            str(score[tok].exp().item())
        )

    CONSOLE.print_zero_rank(table_)


def _get_last_checkpoint_path(
    checkpoints_folder, 
    run_name: Optional[str] = None, 
    wandb_run_id: Optional[str] = None
) -> str:

    if wandb_run_id is None:
        return None

    dir_ = checkpoints_folder / run_name / wandb_run_id 
    checkpoint_path = dir_ / "last.ckpt"

    if not checkpoint_path.exists():
        return None

    return checkpoint_path


def _json_default_paths(entry: Any):
    if isinstance(entry, Path):
        return str(entry)
    return entry


# def _set_resumed_state(
#     checkpoints_root_dir: Union[Path, str], 
#     arg_meta_info: dict[str, Any], 
#     last_ckpt_info: LastCkptInfo,
# ) -> dict[str, Any]:

#     """Resumes things that are in the global state, 
#     ie. the wandb run and the random seeds and states.
#     """
#     checkpoints_root_dir = Path(checkpoints_root_dir)

#     json_path = checkpoints_root_dir / arg_meta_info["run_name"] / arg_meta_info["wandb_run_id"] / "last.json"

#     meta_info = utils.load_json(json_path)

#     # Check that the values that need to match do match
#     arg_meta_info = arg_meta_info.copy()
#     none_or_equal = {
#         "run_name", "seed", "wandb_run_id", 
#         "transformers_model_name", "run_name"
#     }
#     none_or_absent = {
#         "torch_rng_state", "numpy_rng_state", 
#         "python_rng_state"
#     }

#     for k in none_or_equal:
#         arg_val = arg_meta_info.pop(k)
#         assert arg_val is None or arg_val == meta_info[k], (
#             arg_val, meta_info[k])
    
#     for k in none_or_absent:
#         if k in arg_meta_info:
#             arg_val = arg_meta_info.pop(k)
#             assert arg_val is None, arg_val
    
#     # We should have no remaining keys
#     # assert not arg_meta_info, arg_meta_info

#     # Load the variables
#     wandb_run_id = meta_info["wandb_run_id"]
#     seed = meta_info["seed"]

#     # TODO: save the random states
#     # torch_rng_state = meta_info["torch_rng_state"]
#     # numpy_rng_state = meta_info["numpy_rng_state"]
#     # python_rng_state = meta_info["python_rng_state"]
#     # # run_name = meta_info["run_name"]
#     # # transformers_model_name = meta_info["transformers_model_name"]

#     # # Deal with random seeds and states
#     # torch.cuda.manual_seed_all(seed)
#     # random.seed(seed)
#     # np.random.seed(seed)
#     # torch.random.set_rng_state(torch.ByteTensor(torch_rng_state))
#     # np.random.set_state(numpy_rng_state)
#     # for i, v in enumerate(python_rng_state):
#     #     if isinstance(v, list):
#     #         python_rng_state[i] = tuple(
#     #             python_rng_state[i])
#     # random.setstate(tuple(python_rng_state))

#     # Resume the wandb run
#     CONSOLE.print_zero_rank("\n[red bold]Resuming Wandb run:", wandb_run_id)

    
#     return meta_info


def _set_initial_state(
    *,
    checkpoints_root_dir: Union[Path, str], 
    arg_meta_info: dict[str, Any], 
    global_rank: int,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
) -> tuple[dict[str, Any], pl.loggers.WandbLogger, int]:
    """
    Sets the initial state of the global state, ie. 
    the wandb run and the random seeds and states.

    checkpoints_root_dir
    arg_meta_info: dict[str, Any], 
    global_rank: int,

    """
    checkpoints_root_dir = Path(checkpoints_root_dir)
    
    assert ("wandb_run_id" not in arg_meta_info or arg_meta_info["wandb_run_id"] is None), arg_meta_info 

    wandb_logger = pl.loggers.WandbLogger(
        project=wandb_project,
        name=arg_meta_info["run_name"],
        entity=wandb_entity,
        log_model=False,
        config=dict(
            meta_info=arg_meta_info,
            accelerator="gpu",
            precision=PRECISION,
            arguments=arg_meta_info,
        ),
    )

    wandb_run_id = None
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
            arg_meta_info["wandb_run_id"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        utils.dump_json(
            arg_meta_info, 
            config_path, 
            default=_json_default_paths,
        )
        wandb_run_id = wandb.run.id
    
    # Broadcast the wandb run id
    return arg_meta_info, wandb_logger, wandb_run_id


def _build_meta_info(**kwargs):
    return kwargs


def _load_data(
    dataset_path: Union[str, Path], 
    tokenizer: transformers.PreTrainedTokenizer,
    mode: constants.DataModes,
    cv_sets: Optional[Iterable[str]],
    verbose: bool = False,
):
    """Loads the textual entries, tokenizes them and returns a dict with the columns.
    The parallelization is done by the fast tokenizers, 
    which are truely parallel with real Rust-based threads.
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
                CONSOLE.print_zero_rank(f"\n[bold]Loading a dataset file: [/bold]", str(cv_path))
                raw_data = list(f)
                CONSOLE.print_zero_rank(f"\n[bold]Done loading a dataset file: [/bold] {cv_path}, took {time.perf_counter() - start:0.2f}s", )

            tokenized_data[set_] = {
                "input":      tokenizer([x["input"] + CHAINER for x in raw_data], add_special_tokens=False)["input_ids"],
                "value":      tokenizer([x["value"] for x in raw_data], add_special_tokens=False)["input_ids"],
                "scratchpad": tokenizer([x["scratchpad"]      for x in raw_data], add_special_tokens=False)["input_ids"],
            }

        elif mode == constants.DataModes.HDF5_PRETOK:
            cv_path = dataset_path / f"{set_}.h5"
            
            CONSOLE.print_zero_rank(f"\n[bold]Loading a dataset file: [/bold]", str(cv_path))
            with h5py.File(cv_path, "r") as f:
                keys_to_do = [key for key in f if not key.endswith("_text")]
                for key in keys_to_do:
                    assert f[key].dtype != object, key

                cached = {k: f[k][:] for k in tqdm(keys_to_do, desc="Reading from file.", disable=not verbose)}
                for k, v in cached.items():
                    assert isinstance(v, np.ndarray), f"Field `{k}` is of type {type(v)}, expected np.ndarray."

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
            for key in tqdm(ids_keys, desc="Removing padding", disable=not verbose):
                mask_key = key + "_attention_mask"
                tokenized_data[set_][key] = []

                tokenized_data[set_][key] = train_utils.remove_padding(cached[key], cached[mask_key] == 1)

            for key in tqdm(text_keys, desc="Tokenizing", disable=not verbose):
                tokenized_data[set_][key] = cached[key]

            CONSOLE.print_zero_rank(f"\n[bold]Done loading a dataset file: [/bold] {cv_path}, took {time.perf_counter() - start:0.2f}s", )
        
        else:
            raise ValueError(mode)

        delta = time.perf_counter() - start
        CONSOLE.print_zero_rank(f"\n[bold]Done preparing \"{cv_path.name}\". It took {delta:0.2f}s overall. ")

    return tokenized_data


class DictDataset(torch.utils.data.Dataset):
    """
    A dataset built from a dictionary with colums that fit the typing.Sequence 
    protocol (eg, lists, tuples, np.ndarrays, torch.Tensors).
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
    output_datasets: dict[str, DictDataset] = {}

    ds_key_filter = {
        constants.PipelineModes.MLE_TRAINING: {
            "input",
            "scratchpad",
            "value",
        },
        
        constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING: {
            "input",
            "scratchpad",
            "value",
        },

        constants.PipelineModes.VALIDATION: {
            "input",
            "scratchpad",
            "value",
        },
    }

    for cv_set in cv_sets:
        for pipeline_mode in constants.CV_SETS_TO_PILELINES_MODES[cv_set]:
            if cv_set == constants.CVSets.VALIDATION:
                assert pipeline_mode == constants.PipelineModes.VALIDATION
            if cv_set == constants.CVSets.TRAINING:
                assert pipeline_mode in {
                    constants.PipelineModes.MLE_TRAINING,
                    constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING,
                }

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

            assert "SLURM_NNODES" in os.environ, f"You probably didn't launch with `srun`, but still set the strategy to `{distribute_strategy}`."

            self.num_nodes = int(os.environ["SLURM_NNODES"])
            self.num_devices = "auto"
            self.global_rank = int(os.environ["SLURM_PROCID"])
            self.local_rank = int(os.environ["SLURM_LOCALID"])
            self.node_rank = int(os.environ["SLURM_NODEID"])
            header = f"[{self.node_rank}:{self.local_rank}] "
            CONSOLE.print(f"[bold green]{header}Distributed Data Parallel (DDP) enabled.")
            CONSOLE.print(f"[bold green]{header}\t- NUM_NODES:   {self.num_nodes}")
            CONSOLE.print(f"[bold green]{header}\t- NUM_DEVICES: {self.num_devices}")
            CONSOLE.print("")
        else:
            self.num_nodes = None
            self.num_devices = 1
            self.global_rank = 0
            self.local_rank = 0
            self.node_rank = None
            
    def __repr__(self):
        return f"{type(self).__name__}(" + ", (".join([f"{k}={v}" for k, v in self.__dict__.items()]) + ")"


def _setup_ddp(distribute_strategy: str) -> DDPInfo:
    ddp_info = DDPInfo(distribute_strategy)
    
    if ddp_info.global_rank > 0:
        assert distribute_strategy in {"ddp", "ddp_find_unused_parameters_false"}

    return ddp_info


def _setup_tokenizer(hf_name: str, is_gpt2_model) -> transformers.PreTrainedTokenizer:
    if TOKENIZER_MODE == constants.TokenizerModes.ARITHMETIC:
        assert False, "This is not supported anymore."
    elif TOKENIZER_MODE == constants.TokenizerModes.PRETRAINED:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_name, pad_token="<|pad|>", 
            cls_token="<|cls|>"
        )
        
        utils.setattr_must_exist(tokenizer, "padding_side", "left")        
        CONSOLE.print_zero_rank(f"Tokenizer loaded.")
        return tokenizer
    else:
        raise ValueError(f"Unsupported tokenizer mode: {TOKENIZER_MODE}")


def _initialize_base_model(
    *,
    custom_model_config: Optional[dict[str, int]], 
    hf_name: str, 
    is_gpt2_model: bool,
    model_mode: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> transformers.GPT2LMHeadModel:
    """
    ------------------------------------------
    Structure:
    ------------------------------------------
    - Random Model
    --- Random Model with custom config
    - Pretrained Model
    ------------------------------------------
    - Shared modifications
    - GPT2 model specific changes
    - Shared checks
    ------------------------------------------
    """

    ###########################################################################
    # Random Model
    ###########################################################################
    if model_mode == constants.ModelModes.RANDOM :
        CONSOLE.print_zero_rank(f"\n[bold]USING A NON PRETRAINED MODEL.")

        config = transformers.AutoConfig.from_pretrained(hf_name)
        utils.setattr_must_exist(config, "vocab_size", len(tokenizer.vocab))  # type: ignore[attr-defined]

        ###########################################################################
        # Random model with custom config
        ###########################################################################
        if custom_model_config is not None:
            CONSOLE.print_zero_rank(f"\n[bold]USING CUSTOM MODEL CONFIGURATION.")
            for k, v in custom_model_config.items():
                utils.setattr_must_exist(config, k, v)

        base_model = transformers.GPT2LMHeadModel(config)
        assert config.n_inner is None, config.n_inner

    ###########################################################################
    # Pretrained model
    ###########################################################################
    elif model_mode == constants.ModelModes.PRETRAINED:
        CONSOLE.print_zero_rank(f"\n[bold GREEN]USING A PRETRAINED MODEL.")
        base_model = transformers.GPT2LMHeadModel.from_pretrained(hf_name)

    else:
        raise ValueError(f"Unsupported model mode: {model_mode}")


    base_model.resize_token_embeddings(len(tokenizer))

    ###########################################################################
    # Shared modifications
    ###########################################################################
    utils.setattr_must_exist(base_model.config, "early_stopping", True)
    del base_model.config.task_specific_params

    assert is_gpt2_model == (base_model.config.model_type == "gpt2"), (
        is_gpt2_model, base_model.config.model_type)

    ###########################################################################
    # GPT2 model type specific modifications, invariant to the config of the model
    ###########################################################################
    utils.setattr_must_exist(base_model.config, "pad_token_id", tokenizer.pad_token_id)
    utils.setattr_must_exist(base_model.config, "eos_token_id", tokenizer.eos_token_id)

    ###########################################################################
    # Shared checks
    ###########################################################################
    utils.check_equal(base_model.config.vocab_size, len(tokenizer.vocab))  # type: ignore[attr-defined]

    return base_model


def _compute_batch_size_defaults(
    local_rank: int, hf_name: str, batch_sizes: Optional[dict[str, int]], accelerator,
) -> dict[str, int]:
    """Ad-hoc function for default the batch sizes.
    """
    if accelerator == "cpu":
        base = 2
    else:
        base = DEFAULT_SHARED_BATCH_SIZE
    return {
            constants.PipelineModes.MLE_TRAINING: base,
            constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING: MARGINAL_LIKELIHOOD_BS,
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
def _make_config_path(checkpoints_root_dir: Path, run_name: str, wandb_run_id: str) -> Path:
    return checkpoints_root_dir / run_name / wandb_run_id / f"last.json"



def _setup_resuming_wandb_logger(
    use_new_wandb_run_id, wandb_run_id, wandb_config, meta_info, ddp_info, all_arguments
):
    if use_new_wandb_run_id:
            wandb_run_id_to_use = None
    else:
        wandb_run_id_to_use = wandb_run_id

    logger = pl.loggers.WandbLogger(
        resume=False if use_new_wandb_run_id else "must", 
        id=wandb_run_id_to_use,
        project=wandb_config["project"],
        entity=wandb_config["entity"],
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

    if utils.global_rank() == 0:
        wandb.run.log_code(SCRIPT_DIR)

    if ddp_info.global_rank == 0:
        assert wandb.run
        wandb.run.log_code(SCRIPT_DIR)
    
    return logger


def _ensure_answer_model_is_more_trained(
    path_fixed_answer_model,
    path_fixed_scratchpad_model,
):
    assert path_fixed_answer_model != path_fixed_scratchpad_model, (
            path_fixed_answer_model, path_fixed_scratchpad_model
        )

    fixed_scratchpad_model = fast_ckpt_reader.load(path_fixed_scratchpad_model)
    fixed_answer_model = fast_ckpt_reader.load(path_fixed_answer_model)
    
    assert fixed_scratchpad_model["global_step"] < fixed_answer_model["global_step"], (
        fixed_scratchpad_model["global_step"], fixed_answer_model["global_step"]
    )


def _init_in_rl_mode(
    *,
    base_model,
    main_checkpoint_path,
    answer_model_checkpoint_path,
    datasets,
    tokenizer,
    meta_info,
    batch_sizes,
    logger,
):
    #######################################################################
    # Initialize in RL mode.
    #######################################################################
    scratchpad_pl_model = pretrain.PreTrain.load_from_checkpoint(
        main_checkpoint_path,
        datasets=datasets,
        model=base_model,
        scheduler_fn=SCHEDULER_FN,
        tokenizer=tokenizer,
    )

    scratchpad_model = train_utils.clone_hf_model(scratchpad_pl_model._model)
    # fixed_model = train_utils.clone_hf_model(scratchpad_pl_model._model)
    # train_utils.fix_model_params_in_place(fixed_model)
    
    fixed_answer_model = None
    if answer_model_checkpoint_path:
        fixed_answer_refine_lm = pretrain.PreTrain.load_from_checkpoint(
            answer_model_checkpoint_path,
            datasets=datasets,
            model=base_model,
            scheduler_fn=SCHEDULER_FN,
            tokenizer=tokenizer,
        )
        
        fixed_answer_model = train_utils.clone_hf_model(fixed_answer_refine_lm._model)
        train_utils.fix_model_params_in_place(fixed_answer_model)

        _ensure_answer_model_is_more_trained(
            path_fixed_scratchpad_model=main_checkpoint_path,
            path_fixed_answer_model=answer_model_checkpoint_path, 
        )        

    return scratchpad_model, fixed_answer_model

    if fixed_answer_model is None:
        assert False, "This approach does not currently work"
        fixed_answer_model = fixed_model.clone()
        train_utils.fix_model_params_in_place(fixed_answer_model) # likely useless


    assert scratchpad_model is not fixed_model
    assert fixed_model is not fixed_answer_model
    assert fixed_answer_model is not scratchpad_model

    assert scratchpad_model is not base_model
    assert fixed_model is not base_model
    assert fixed_answer_model is not base_model
    

    return marginal.RLTraining(
        model                   = scratchpad_model,
        fixed_scratchpad_model  = fixed_model,
        fixed_answer_model      = fixed_answer_model, 

        loss_mode               = meta_info["step_3_loss_mode"],
        chainer                 = CHAINER,
        batch_sizes             = batch_sizes,
        datasets                = datasets,
        generation_kwargs       = meta_info["generation_kwargs"],
        is_adamw                = meta_info["is_adamw"],
        learning_rate           = meta_info["learning_rate"],
        lm_masking_mode         = meta_info["lm_masking_mode"],
        meta_info               = meta_info,
        path_log_results        = meta_info["path_log_results"],
        scheduler_type          = constants.SchedulerTypes.CONSTANT,
        scheduler_fn            = SCHEDULER_FN,
        shuffle_training_data   = SHUFFLE_TRAINING_DATA,
        shuffle_validation_data = SHUFFLE_VALIDATION_DATA,
        tokenizer               = tokenizer,
        wandb_logger            = logger,
        weight_decay            = meta_info["weight_decay"],
    )


def _init_in_mle_mode(batch_sizes, datasets, meta_info, base_model, tokenizer, logger):
    return pretrain.PreTrain(
            batch_sizes             = batch_sizes,
            chainer                 = meta_info["chainer"],
            datasets                = datasets,
            generation_kwargs       = meta_info["generation_kwargs"],
            is_adamw                = meta_info["is_adamw"],
            learning_rate           = meta_info["learning_rate"],
            lm_masking_mode         = meta_info["lm_masking_mode"],
            meta_info               = meta_info,
            model                   = base_model,
            path_log_results        = meta_info["path_log_results"],
            scheduler_type          = meta_info["scheduler_type"],
            scheduler_fn            = SCHEDULER_FN,
            shuffle_training_data   = SHUFFLE_TRAINING_DATA,
            shuffle_validation_data = SHUFFLE_VALIDATION_DATA,
            tokenizer               = tokenizer,
            wandb_logger            = logger,
            weight_decay            = meta_info["weight_decay"],
        )


def _should_init_rl_mode(meta_info, main_checkpoint_path):
    #######################################################################
    # Figure out if we need to resume in MLE or RL mode.
    #######################################################################
    # Warning:
    # We only support starting RL mode from it's start, we don't support
    # resuming from a checkpoint in RL mode. That's an important distinction.
    #######################################################################
    ckpt = fast_ckpt_reader.load(main_checkpoint_path)
    epoch = ckpt["epoch"]
    global_step = ckpt["global_step"]
    CONSOLE.print_zero_rank(f"\n[bold]Epoch in checkpoint:[/] {epoch}, [bold]global step in checkpoint:[/] {global_step}")
    del ckpt
    assert epoch <= meta_info["switch_to_maginal_after"] - 1, epoch

    return epoch == meta_info["switch_to_maginal_after"] - 1 and global_step > 0


DATA_DIR = SCRIPT_DIR / "data"


@beartype
def main(
    *, 
    seed: int = 453345,
    generation_kwargs=DEFAULT_GENERATION_KWARGS,
    dataset_path: Union[Path, str] = DATA_DIR / "basic_arithmetic/80_3_6_200000",  # DATA_DIR / "basic_arithmetic/349_6_6_200000"
    path_log_results=DEFAULT_CHECKPOINTS_DIR / "logs",
    batch_sizes=None,
    strategy=DEFAULT_DISTRIBUTE_STRATEGIES,
    is_gpt2_model=True,
    lm_masking_mode=DEFAULT_LM_MASKING_MODE,
    use_new_wandb_run_id: bool = False,
    step_3_loss_mode: str = DEFAULT_STEP_3_LOSS_MODE,
    switch_to_maginal_after: Optional[dict[str, int]] = DEFAULT_SWITCH_TO_MARGINAL_AFTER,

    #######################################################################
    # Model config
    #######################################################################
    model_mode=DEFAULT_MODEL_MODE,
    transformers_model_name: str = DEFAULT_HUGGING_FACE,
    custom_model_config=DEFAULT_CUSTOM_MODEL_CONFIG,
    
    #######################################################################
    # Optimization and regularization
    #######################################################################
    is_adamw: bool = DEFAULT_USE_ADAMW,
    weight_decay: Optional[float] = DEFAULT_WEIGHT_DECAY,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    step_3_lr: float = DEFAULT_STEP_3_LR,
    scheduler_type=DEFAULT_SCHEDULER_TYPE,
    accumulate_grad_batches: int = DEFAULT_GRADIENT_ACCUM,

    #######################################################################
    # Related to resuming
    #######################################################################
    wandb_run_id: Optional[Union[str, Path]] = DEFAULT_WANDB_ID,
    # This is if we pretrain a separate model & want to use it for step 3 as the fixed model:
    fixed_answer_model_wandb_run_id: Optional[Union[str, Path]] = DEFAULT_FIXED_ANSWER_MODEL_WANDB_RUN_ID,  
    checkpoints_folder: Union[Path, str] = DEFAULT_CHECKPOINTS_DIR,
    wandb_config_path: Union[Path, str] = DEFAULT_WANDB_CONFIG_PATH,
):
    all_arguments = locals().copy()
    
    if utils.is_rank_zero():
        utils.check_and_print_args(all_arguments, main, False, SCRIPT_DIR)

    if TOKENIZER_MODE == constants.TokenizerModes.ARITHMETIC:
        assert DATA_MODE == constants.DataModes.JSONL, (
            f"We only support JSONL for arithmetic tokenizer, as things "
            "are pre-tokenized in the h5 mode. {DATA_MODE}"
        )

    wandb_config_path = Path(wandb_config_path)
    assert wandb_config_path.exists(), wandb_config_path
    wandb_config = utils.load_json(wandb_config_path)

    dataset_path = Path(dataset_path)
    assert dataset_path.exists(), dataset_path

    checkpoints_folder = Path(checkpoints_folder)
    assert checkpoints_folder.exists(), checkpoints_folder
    assert checkpoints_folder.is_dir(), checkpoints_folder

    assert not (model_mode == constants.ModelModes.PRETRAINED and custom_model_config), (
        "If you are not using a pretrained model, you can't use a custom model config."
    )

    torch.use_deterministic_algorithms(mode=DETERMINISTIC)

    run_name = dataset_path.name
    resuming = wandb_run_id is not None
    main_checkpoint_path = _get_last_checkpoint_path(checkpoints_folder, run_name, wandb_run_id)
    answer_model_checkpoint_path = _get_last_checkpoint_path(checkpoints_folder, run_name, fixed_answer_model_wandb_run_id)

    ddp_info = _setup_ddp(strategy)
    arg_meta_info = _build_meta_info(
        accumulate_grad_batches=accumulate_grad_batches,
        batch_sizes=batch_sizes,
        chainer=CHAINER,
        checkpoints_folder=checkpoints_folder,
        custom_model_config=custom_model_config,
        dataset_path=dataset_path,
        generation_kwargs=generation_kwargs,
        is_adamw=is_adamw,
        is_gpt2_model=is_gpt2_model,
        lm_masking_mode=lm_masking_mode,
        learning_rate=learning_rate,
        step_3_loss_mode=step_3_loss_mode, 
        step_3_lr=step_3_lr,
        model_mode=model_mode,
        num_devices=ddp_info.num_devices,
        num_nodes=ddp_info.num_nodes,
        path_log_results=path_log_results,
        run_name=run_name,
        scheduler_type=scheduler_type,
        seed=seed, 
        transformers_model_name=transformers_model_name,
        wandb_run_id=wandb_run_id,
        weight_decay=weight_decay,
        switch_to_maginal_after=switch_to_maginal_after,
    )
    
    # Load the pretrained model. If a checkpoint is used, it will
    # be loaded with the trainer.fit call, further in the code.
    
    base_model = None
    tokenizer = None
    datasets = None
    pl_object = None
        
    if batch_sizes is None:
        batch_sizes = _compute_batch_size_defaults(
            ddp_info.local_rank, 
            transformers_model_name, 
            batch_sizes, 
            ACCELERATOR, 
        )

    if resuming:
        #######################################################################
        # Deal with argument conf vs checkpoint conf
        #######################################################################
        # TODO: fix this logic wrt what actually gets loaded in the checkpoint.
        # The thing is that, we used to do this by hand, 
        # but lighthing does a lot of it already.
        # It's likely not necessary.
        # meta_info = _set_resumed_state(
        #   checkpoints_folder, 
        #   arg_meta_info, 
        #   last_ckpt_info,
        # )
        meta_info = arg_meta_info
        del arg_meta_info
        
        #######################################################################
        # Setup the Wandb logger. 
        # Depending on the value of the arg, we may
        # resume from the run of the checkpoint, or start a new run.
        #######################################################################
        logger = _setup_resuming_wandb_logger(
            use_new_wandb_run_id, 
            wandb_run_id, 
            wandb_config, 
            meta_info, 
            ddp_info, 
            all_arguments,
        )

        tokenizer = _setup_tokenizer(
            meta_info["transformers_model_name"], 
            is_gpt2_model=meta_info["is_gpt2_model"],
        )
        
        datasets = _text_mode_build_dataset(dataset_path, tokenizer, 
            [constants.CVSets.TRAINING, constants.CVSets.VALIDATION]
        )

        base_model = _initialize_base_model(
            custom_model_config=meta_info["custom_model_config"], 
            hf_name=meta_info["transformers_model_name"], 
            is_gpt2_model=meta_info["is_gpt2_model"],
            model_mode=meta_info["model_mode"], 
            tokenizer=tokenizer,
        )

        if _should_init_rl_mode(meta_info, main_checkpoint_path):
            scratchpad_model, fixed_answer_model = _init_in_rl_mode(
                base_model=base_model,
                datasets=datasets,
                tokenizer=tokenizer,
                answer_model_checkpoint_path=answer_model_checkpoint_path,
                main_checkpoint_path=main_checkpoint_path,
                meta_info=meta_info,
                batch_sizes=batch_sizes,
                logger=logger,
            )

            trlx_exp.train(
                model        = scratchpad_model,
                reward_model = fixed_answer_model,
                tokenizer    = tokenizer, 
                ds_train     = datasets[constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING], 
                ds_eval      = datasets[constants.PipelineModes.VALIDATION], 
                ensure_not_from_scratch = True,
                trlx_config_path = "/home/mila/g/gagnonju/Marg-Li-CoT/our_scratchpad/configs/ppo_config.yml",
            )

            sys.exit(0)

            pl_object = _init_in_rl_mode(
                base_model=base_model,
                datasets=datasets,
                tokenizer=tokenizer,
                answer_model_checkpoint_path=answer_model_checkpoint_path,
                main_checkpoint_path=main_checkpoint_path,
                meta_info=meta_info,
                batch_sizes=batch_sizes,
                logger=logger,
            )
        else:
            pl_object = _init_in_mle_mode(
                batch_sizes=batch_sizes,
                datasets=datasets,
                meta_info=meta_info,
                base_model=base_model,
                tokenizer=tokenizer,
                logger=logger,
            )
    else:
        CONSOLE.print_zero_rank("\n[bold green]Not Resuming: Setting the initial state.")
        pl_object = None

        meta_info, logger, wandb_run_id = _set_initial_state(
            checkpoints_root_dir=checkpoints_folder,
            arg_meta_info=arg_meta_info, 
            global_rank=ddp_info.global_rank,
            wandb_project=wandb_config["project"],
            wandb_entity=wandb_config["entity"],
        )
        del arg_meta_info

        tokenizer = _setup_tokenizer(
            meta_info["transformers_model_name"], 
            is_gpt2_model=meta_info["is_gpt2_model"],
        )

        base_model = _initialize_base_model(
            custom_model_config=meta_info["custom_model_config"], 
            hf_name=meta_info["transformers_model_name"], 
            is_gpt2_model=meta_info["is_gpt2_model"],
            model_mode=meta_info["model_mode"], 
            tokenizer=tokenizer,
        )

        CONSOLE.print_zero_rank(f"\n[bold]Run name:[/bold] [green]\"{meta_info['run_name']}\"")
        datasets = _text_mode_build_dataset(dataset_path, tokenizer, 
            [constants.CVSets.TRAINING, constants.CVSets.VALIDATION]
        )

        pl_object = _init_in_mle_mode(
            batch_sizes, 
            datasets, 
            meta_info, 
            base_model, 
            tokenizer, 
            logger,
        )


    assert pl_object
    CONSOLE.print_zero_rank(f"\n[bold]PL_OBJECT_TYPE:[/bold] {type(pl_object)}")
    CONSOLE.print_zero_rank(f"\n[bold]Strategy:[/] {strategy}")
    CONSOLE.print(f"\n[bold]ddp_info:[/] {vars(ddp_info)}\n")

    ###############################################################
    # All of the follwing arguments are very stable
    ###############################################################
    trainer = pl.Trainer(
        accumulate_grad_batches=meta_info["accumulate_grad_batches"],
        logger=logger,
        num_nodes=ddp_info.num_nodes,
        devices=ddp_info.num_devices,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        default_root_dir=str(checkpoints_folder),
        
        # Accelerators 
        deterministic=DETERMINISTIC,
        strategy=strategy,
        accelerator=ACCELERATOR,
        precision=PRECISION,

        # Looping stuff
        max_epochs=switch_to_maginal_after,
        val_check_interval=VAL_CHECK_INTERVAL if isinstance(pl_object, pretrain.PreTrain) else VAL_CHECK_INTERVAL_STEP_3,
        limit_train_batches=LIMIT_TRAIN_BATCHES,
        limit_val_batches=LIMIT_VAL_BATCHES,

        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint( # type: ignore[arg-type]
                dirpath=(checkpoints_folder / meta_info["run_name"] / wandb_run_id) if wandb_run_id else "this is nonsense",
                every_n_epochs=1, 
                save_on_train_epoch_end=True, 
                save_last=True
            ),
        ]
    )
    
    if resuming and isinstance(pl_object, pretrain.PreTrain): 
        trainer.fit(pl_object, ckpt_path=main_checkpoint_path)
    else:
        trainer.fit(pl_object,)


@classmethod
def json(cls, name: str, path: Path) -> Dict[str, Any]:
    """
    Run by loading a json file.
    """

    all_arguments = locals().copy()
    utils.check_and_print_args(all_arguments, cls.main, True)

    entrypoint_names = {k for k in cls.__dict__.keys() - {"json"} if not k.startswith("_")}
    assert hasattr(cls, name), (
        f"{cls.__name__}.{name} doesn't exist. "
        f"Valid options are: {entrypoint_names}"
    )

    with open(path, "r") as f:
        args = json.load(f)
    
    return getattr(cls, name)(**args)


@beartype
def predict(
    *,
    wandb_run_id: str = DEFAULT_WANDB_ID,
    run_name: Optional[str] = None,
    qty: int = 1,
    dataset_path: Path = DATA_DIR / "basic_arithmetic/80_3_6_200000",
    checkpoints_root_dir: Path = DEFAULT_CHECKPOINTS_DIR, 
    distribute_strategy: Optional[str] = None,
    batch_sizes: Optional[dict[str, int]] = None,
    lm_masking_mode = DEFAULT_LM_MASKING_MODE,
) -> None:
    """
    Run by loading a json file.
    """
    all_arguments = locals().copy()
    utils.check_and_print_args(all_arguments, None)
    
    mode = constants.CVSets.VALIDATION
    last_ckpt_info = _get_last_checkpoint_path(
        checkpoints_root_dir, 
        run_name, 
        wandb_run_id,
    )
    run_name = last_ckpt_info.run_name
    meta_info = utils.load_json(_make_config_path(
        checkpoints_root_dir=checkpoints_root_dir, 
        run_name=run_name, 
        wandb_run_id=wandb_run_id,
    ))
    tokenizer = _setup_tokenizer(
        meta_info["transformers_model_name"], 
        is_gpt2_model=meta_info["is_gpt2_model"]
    )  
    base_model = _initialize_base_model(
        custom_model_config=meta_info["custom_model_config"], 
        hf_name=meta_info["transformers_model_name"], 
        is_gpt2_model=meta_info["is_gpt2_model"],
        model_mode=meta_info["model_mode"], 
        tokenizer=tokenizer,
    )
    datasets = _text_mode_build_dataset(
        dataset_path, tokenizer, cv_sets=[constants.CVSets.VALIDATION])
    ddp_info = _setup_ddp(distribute_strategy)

    if batch_sizes is None:
        batch_sizes = _compute_batch_size_defaults(
            ddp_info.local_rank, 
            meta_info["transformers_model_name"], 
            batch_sizes,
            accelerator=ACCELERATOR,
        )

    pl_object = pretrain.PreTrain(
        model=base_model,
        datasets=datasets,
        tokenizer=tokenizer,
        batch_sizes=batch_sizes,
        generation_kwargs=meta_info["generation_kwargs"],
        learning_rate=meta_info["learning_rate"],
        path_log_results=meta_info["path_log_results"],
        is_adamw=meta_info["is_adamw"],
        weight_decay=meta_info["weight_decay"],
        scheduler_type=meta_info["scheduler_type"],
        meta_info=meta_info,
        lm_masking_mode=lm_masking_mode,
        switch_to_maginal_after=None,
        wandb_logger=None,
    )

    # THIS IS FOR PREDICT
    trainer = pl.Trainer(
        precision=PRECISION,
        accelerator=ACCELERATOR,
        deterministic=DETERMINISTIC,
        devices=ddp_info.num_devices,
        num_nodes=ddp_info.num_nodes,
        strategy=distribute_strategy,
        default_root_dir=str(checkpoints_root_dir),
        limit_predict_batches=math.ceil(qty / batch_sizes[mode]),
        val_check_interval=VAL_CHECK_INTERVAL,
    )

    trainer.validate(
        pl_object,
        ckpt_path=str(last_ckpt_info),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Defaulting to the `main` entrypoint.")
        main()
    else:
        fire.Fire(dict(
            main=main, 
            train=main,
            predict=predict,
            json=json, 
        ))
