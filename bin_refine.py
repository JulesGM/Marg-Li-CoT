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

import console
import general_shared_constants as constants
import general_utils as utils


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

DEFAULT_SWITCH_TO_MARGINAL_AFTER: Final[Optional[dict[str, int]]] = dict(epochs=1)
LIMIT_VAL_BATCHES = 10
VAL_CHECK_INTERVAL = 1 / 3
VAL_CHECK_INTERVAL_STEP_3 = 30

DEFAULT_NUM_BEAMS = 20
MARGINAL_LIKELIHOOD_BS = (64 * 2) // DEFAULT_NUM_BEAMS

DEFAULT_LM_MASKING_MODE = constants.LMMaskingMode.MASK_INPUT
DEFAULT_SHARED_BATCH_SIZE = 64 * 2
DEFAULT_GRADIENT_ACCUM = 2
DEFAULT_SCHEDULER_TYPE = constants.SchedulerTypes.LINEAR_WARMUP_CONSTANT

WARMUP_EPOCHS = 1
MAX_EPOCHS = 53

DEFAULT_WANDB_ID: Optional[str] = "23pxj5xe" 
DEFAULT_FIXED_ANSWER_MODEL_WANDB_RUN_ID: Optional[str] = "1iis607f"
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
PRECISION = "bf16"  # Also try mixed

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


def _show_multi_scratchpad_table_format_text(ids, tokenizer) -> str:
    return tokenizer.decode(ids
                ).replace("<|pad|>", "").replace("<|endoftext|>", "<eos>"
                ).replace("<|cls|>", "<cls>") 


def show_multi_scratchpad_table(
    *,
    labels,
    y_prob, 
    z_prob, 
    tokenizer, 
    shift_MITGSWRV,
    demo_input_idx,
    num_scratchpads, 
):
    
    y_prob = y_prob.clone().detach()
    z_prob = z_prob.clone().detach()

    table_ = table.Table("Text", "y score", "z score", "y rank", "z rank")

    label_text = _show_multi_scratchpad_table_format_text([x for x in labels[demo_input_idx] if x > 0], tokenizer)
    table_.add_row(f"[bold magenta]{label_text}[/]", "", "", "", "", end_section=True)

    sort_y = torch.tensor(sorted(range(num_scratchpads), key=lambda i: y_prob[demo_input_idx, i], reverse=True))
    y_prob_entry = y_prob[demo_input_idx, sort_y]
    z_prob_entry = z_prob[demo_input_idx, sort_y]

    argsort = sorted(range(num_scratchpads), key=lambda i: z_prob_entry[i], reverse=True)
    ranks_z = {}
    for i, pos in enumerate(argsort):
        ranks_z[pos] = i

    for rank_y in range(num_scratchpads):
        maybe_color = ""
        if rank_y == 0 and ranks_z[rank_y] == 0:
            maybe_color = "[green bold]"
        elif rank_y == 0:
            maybe_color = "[blue bold]"
        elif ranks_z[rank_y] == 0:
            maybe_color = "[yellow bold]"

        maybe_close = ""        
        if maybe_color:
            maybe_close = "[/]"

        y_prob_color_coeff = int(y_prob_entry[rank_y].item() * 255)
        y_prob_color = f"white on #{y_prob_color_coeff:02x}{y_prob_color_coeff:02x}{y_prob_color_coeff:02x}"

        z_prob_color_coeff = int(z_prob_entry[rank_y].item() * 255)
        z_prob_color = f"white on #{z_prob_color_coeff:02x}{z_prob_color_coeff:02x}{z_prob_color_coeff:02x}"

        generated_text = _show_multi_scratchpad_table_format_text(shift_MITGSWRV[demo_input_idx, rank_y], tokenizer) 

        output_str = []
        for lab, gen in itertools.zip_longest(label_text, generated_text, fillvalue=" "):
            if lab == gen:
                output_str.append(gen)
            else:
                output_str.append(f"[red]{gen}[/red]")
        diff_colored = "".join(output_str)
        
        if generated_text == label_text:
            diff_colored = f"[white on green]{generated_text}[/white on green]"


        table_.add_row(
            maybe_color + diff_colored + maybe_close,
            f"[{y_prob_color}]{y_prob_entry[rank_y].item():.3}", 
            f"[{z_prob_color}]{z_prob_entry[rank_y].item():.3}",
            str(rank_y),
            str(ranks_z[rank_y])
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

    if run_name:
        dir_ = checkpoints_folder / run_name / wandb_run_id 
        checkpoints = list((dir_).glob("*.ckpt"))
        
    else:
        checkpoints = []
        for path in checkpoints_folder.glob("**/*.ckpt"):
            if path.parent.name == wandb_run_id and path.name == "last.ckpt":
                checkpoints.append(path)

    if not checkpoints:
        return LastCkptInfo(None, run_name, None, None)

    assert len(checkpoints) == 1, checkpoints
    checkpoint_path = checkpoints[0]
    
    if run_name is None:
        # We recover the run name from the wandb run id
        run_name = path.parent.parent.name
        CONSOLE.print_zero_rank(f"\n[bold]Inferring `run_name` value of:[/] {run_name = !s}")

    return LastCkptInfo(checkpoint_path, run_name, None, None)


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

                tokenized_data[set_][key] = remove_padding(cached[key], cached[mask_key] == 1)

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

@beartype
def pad(seq : Sequence, pad_token_id: int, direction: str) -> torch.LongTensor:
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


@beartype
def generate_mask(list_of_list: list, direction: str) -> torch.LongTensor:
    assert isinstance(list_of_list, list), type(list_of_list)

    mask: list[torch.Tensor] = []
    for x in list_of_list:
        mask.append(torch.ones(len(x), dtype=torch.long))
    attention_mask = pad(mask, 0, direction)
    return attention_mask


def prep_mle_train_and_valid(*, examples, eos_token_id: int, scratchpad_eos_token_id: int, lm_masking_mode: str, pad_token_id: int) -> None:
    # We take a mini bit of slowness vs bug potential any day of the week right now
    examples = examples.copy()

    for example in examples:
        # Transormations
        example["input_ids"] = example["input"].tolist()       + example["scratchpad"].tolist() + [scratchpad_eos_token_id] + example["value"].tolist() + [eos_token_id]
        if lm_masking_mode == constants.LMMaskingMode.MASK_INPUT:
            example["labels"] = [-100] * len(example["input"]) + example["scratchpad"].tolist() + [scratchpad_eos_token_id] + example["value"].tolist() + [eos_token_id]
        elif lm_masking_mode == constants.LMMaskingMode.PLAIN_AUTOREGRESSIVE:
            example["labels"] = example["input_ids"].copy()
        else:
            raise ValueError(lm_masking_mode)

    examples = utils.dict_unzip(examples)
    examples = cast(dict[str, Union[Sequence[Any], torch.Tensor]], examples)

    examples["attention_mask"] = generate_mask(examples["input_ids"], "right")  # NEEDS TO BE BEFORE PAD
    examples["input_ids"] = pad(examples["input_ids"], pad_token_id, "right")
    examples["labels"] = pad(examples["labels"], -100, "right")

    return examples


@dataclasses.dataclass
class MarginalLikelihoodTrainingCollator:
    _tokenizer: transformers.PreTrainedTokenizer
    _lm_masking_mode: str

    def __call__(self, raw_examples):
        """
        - We have the questions, we have the answers. Nothing else.

        Input ids: [question, chainer]
        Labels: [answer]

        loss: likelihoodOf[question, chainer, Generate(question), answer]

        """

        # We can only prepare the inputs for generation. 
        # These need to be padded to the left.
        examples = prep_mle_train_and_valid(
            examples=raw_examples, 
            eos_token_id=self._tokenizer.eos_token_id,
            scratchpad_eos_token_id=self._tokenizer.cls_token_id, 
            pad_token_id=self._tokenizer.pad_token_id,
            lm_masking_mode=self._lm_masking_mode,
        )
        
        examples["generation_attention_mask"] = generate_mask(examples["input"], "left")
        examples["generation_input_ids"] = pad(examples["input"], self._tokenizer.pad_token_id, "left")

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

            assert "SLURM_NNODES" in os.environ, f"You probably didn't launch with `srun`, but still setp the strategy to `{distribute_strategy}`."

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


def _setup_base_model(
    *,
    custom_model_config: Optional[dict[str, int]], 
    hf_name: str, 
    is_gpt2_model: bool,
    model_mode: str,
    tokenizer: transformers.PreTrainedTokenizer,
    verbose: bool = True
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


DATA_DIR = SCRIPT_DIR / "data"


@beartype
def main(
    cls,
    *, 
    seed: int = 453345,
    generation_kwargs=DEFAULT_GENERATION_KWARGS,
    dataset_path: Union[Path, str] = DATA_DIR / "basic_arithmetic/80_3_6_200000",  # DATA_DIR / "basic_arithmetic/349_6_6_200000"
    path_log_results=DEFAULT_CHECKPOINTS_DIR / "logs",
    batch_sizes=None,
    strategy=DEFAULT_DISTRIBUTE_STRATEGIES,
    is_gpt2_model=True,
    lm_masking_mode=DEFAULT_LM_MASKING_MODE,
    new_wandb_run_id: bool = False,
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
        utils.check_and_print_args(all_arguments, cls.main, True, SCRIPT_DIR)

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
    last_ckpt_info = _get_last_checkpoint_path(
        checkpoints_folder, None, wandb_run_id)
    resuming = wandb_run_id is not None
    
    if resuming:
        assert last_ckpt_info is not None, last_ckpt_info

    if resuming:
        latest_checkpoint = last_ckpt_info.path
        CONSOLE.print_zero_rank(f"\n[bold]Will resume from:[/] \"{latest_checkpoint}\"")
    else:
        latest_checkpoint = None
        CONSOLE.print_zero_rank(f"\n[bold]Not resuming: Will start from scratch.")

    ddp_info = _setup_ddp(strategy)
    arg_meta_info = _build_meta_info(
        accumulate_grad_batches=accumulate_grad_batches,
        batch_sizes=batch_sizes,
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

    if resuming:
        CONSOLE.print_zero_rank("\n[bold]Resuming from checkpoint:[/]", latest_checkpoint)
        # meta_info = _set_resumed_state(checkpoints_folder, arg_meta_info, last_ckpt_info)
        CONSOLE.print_zero_rank("\n[red bold]WATCH OUT:[/] Not loading meta info from checkpoint.")
        meta_info = arg_meta_info
        
        if new_wandb_run_id:
            wandb_run_id_to_use = None
        else:
            wandb_run_id_to_use = wandb_run_id
            

        del arg_meta_info
        logger = pl.loggers.WandbLogger(
            resume=False if new_wandb_run_id else "must", 
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

        tokenizer = _setup_tokenizer(
                meta_info["transformers_model_name"], 
                is_gpt2_model=meta_info["is_gpt2_model"],
            )
        
        CONSOLE.print_zero_rank(f"\n[bold]Run name:[/bold] [green]\"{meta_info['run_name']}\"")
        datasets = _text_mode_build_dataset(dataset_path, tokenizer, 
            [constants.CVSets.TRAINING, constants.CVSets.VALIDATION]
        )

        base_model = _setup_base_model(
            custom_model_config=meta_info["custom_model_config"], 
            hf_name=meta_info["transformers_model_name"], 
            is_gpt2_model=meta_info["is_gpt2_model"],
            model_mode=meta_info["model_mode"], 
            tokenizer=tokenizer,
        )

        pl_object = _RefineLM.load_from_checkpoint(
            latest_checkpoint,
            datasets=datasets,
            model=base_model,
            tokenizer=tokenizer,
        )

        if fixed_answer_model_wandb_run_id:
            latest_checkpoint_fixed = _get_last_checkpoint_path(
                checkpoints_folder, None, fixed_answer_model_wandb_run_id)
            
            fixed_answer_refine_lm = _RefineLM.load_from_checkpoint(
                latest_checkpoint_fixed.path,
                datasets=datasets,
                model=base_model,
                tokenizer=tokenizer,
            )
            
            fixed_answer_model_lm = fixed_answer_refine_lm._model
            for p in fixed_answer_model_lm.parameters():
                p.requires_grad = False
            pl_object._fixed_answer_model = fixed_answer_model_lm
            
    else:
        pl_object = None

        CONSOLE.print_zero_rank("\n[bold green]Not Resuming: Setting the initial state.")
        meta_info, logger, wandb_run_id = _set_initial_state(
            checkpoints_root_dir=checkpoints_folder,
            arg_meta_info=arg_meta_info, 
            global_rank=ddp_info.global_rank,
            wandb_project=wandb_config["project"],
            wandb_entity=wandb_config["entity"],
        )
        del arg_meta_info
    
    if batch_sizes is None:
        batch_sizes = _compute_batch_size_defaults(
            ddp_info.local_rank, transformers_model_name, batch_sizes, ACCELERATOR, 
        )

    if tokenizer is None:
        tokenizer = _setup_tokenizer(
            meta_info["transformers_model_name"], 
            is_gpt2_model=meta_info["is_gpt2_model"],
        )

    if base_model is None:
        base_model = _setup_base_model(
            custom_model_config=meta_info["custom_model_config"], 
            hf_name=meta_info["transformers_model_name"], 
            is_gpt2_model=meta_info["is_gpt2_model"],
            model_mode=meta_info["model_mode"], 
            tokenizer=tokenizer,
        )

    if datasets is None:
        CONSOLE.print_zero_rank(f"\n[bold]Run name:[/bold] [green]\"{meta_info['run_name']}\"")
        datasets = _text_mode_build_dataset(dataset_path, tokenizer, 
            [constants.CVSets.TRAINING, constants.CVSets.VALIDATION]
        )

    CONSOLE.print(f"\n[bold]Strategy:[/] {strategy}")
    CONSOLE.print(f"\n[bold]ddp_info:[/] {vars(ddp_info)}\n")

    ###############################################################
    # Build the pt-lightning dataloader
    ###############################################################
    if pl_object is None:
        pl_object = _RefineLM(
            batch_sizes=batch_sizes,
            datasets=datasets,
            generation_kwargs=meta_info["generation_kwargs"],
            is_adamw=meta_info["is_adamw"],
            learning_rate=meta_info["learning_rate"],
            step_3_loss_mode=meta_info["step_3_loss_mode"],
            step_3_lr=meta_info["step_3_lr"],
            lm_masking_mode=meta_info["lm_masking_mode"],
            meta_info=meta_info,
            model=base_model,
            path_log_results=meta_info["path_log_results"],
            scheduler_type=meta_info["scheduler_type"],
            switch_to_maginal_after=meta_info["switch_to_maginal_after"],
            tokenizer=tokenizer,
            wandb_logger=logger,
            weight_decay=meta_info["weight_decay"],
            
        )

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
        max_epochs=MAX_EPOCHS,
        val_check_interval=VAL_CHECK_INTERVAL,
        limit_train_batches=LIMIT_TRAIN_BATCHES,
        limit_val_batches=LIMIT_VAL_BATCHES,
        reload_dataloaders_every_n_epochs=1,

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
    
    if resuming: 
        trainer.fit(pl_object, ckpt_path=latest_checkpoint)
    else:
        trainer.fit(pl_object,)


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
    lm_masking_mode = DEFAULT_LM_MASKING_MODE,
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
    ))
    tokenizer = _setup_tokenizer(
        meta_info["transformers_model_name"], 
        is_gpt2_model=meta_info["is_gpt2_model"]
    )  
    base_model = _setup_base_model(
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

    pl_object = _RefineLM(
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
        ckpt_path=str(last_ckpt_info.path),
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
