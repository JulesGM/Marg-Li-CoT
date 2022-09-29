#!/usr/bin/env python3
# coding: utf-8

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

import general_shared_constants as constants
import general_utils as utils
import console

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

DEFAULT_SWITCH_TO_MARGINAL_AFTER: Final[Optional[dict[str, int]]] = dict(epochs=3)
LIMIT_VAL_BATCHES = 10
VAL_CHECK_INTERVAL = 0.01

DEFAULT_NUM_BEAMS = 20
MARGINAL_LIKELIHOOD_BS = (64 * 2) // DEFAULT_NUM_BEAMS

DEFAULT_LM_MASKING_MODE = constants.LMMaskingMode.MASK_INPUT
DEFAULT_SHARED_BATCH_SIZE = 64 * 2
DEFAULT_GRADIENT_ACCUM = 2
DEFAULT_SCHEDULER_TYPE = constants.SchedulerTypes.LINEAR_WARMUP_LINEAR

WARMUP_EPOCHS = 1
MAX_EPOCHS = 53

DEFAULT_WANDB_ID: Optional[str] = None # "4sr5c621" 
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
        num_beams=1,
        sample=False,
        min_length=0,
        use_cache=True,
        do_sample=False,
        constraints=None,
        max_new_tokens=80,
        length_penalty=1.0,
        repetition_penalty=None,
    ),
    constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING: 
    dict(
        min_length=0,
        do_sample=False,
        max_new_tokens=80, # This is a very important knob
        
        # Not changing
        use_cache=True,
        constraints=None,
        repetition_penalty=None,
        num_beams=DEFAULT_NUM_BEAMS, 
        num_return_sequences=DEFAULT_NUM_BEAMS, 
        # diversity_penalty=0.25, # This needs to be tuned
        # num_beam_groups=DEFAULT_NUM_BEAMS,
    ),
}


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
                optimizer),

    constants.SchedulerTypes.LINEAR_WARMUP_CONSTANT: 
        lambda optimizer, steps_per_epoch:
            transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=steps_per_epoch * WARMUP_EPOCHS),

    constants.SchedulerTypes.LINEAR_WARMUP_LINEAR: 
        lambda optimizer, steps_per_epoch:
            transformers.get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=steps_per_epoch * WARMUP_EPOCHS,
                num_training_steps=steps_per_epoch * MAX_EPOCHS,),
}


def clone_hf_model(model):
    new_model = model.__class__(model.config)
    new_model.load_state_dict(model.state_dict())
    return new_model


def _compute_length_stats(*, target, pad_token_id):
    good_mask = target != pad_token_id
    lengths = collections.Counter(torch.sum(good_mask, dim=1).tolist())
    sorted_lengths = {k: v for k, v in sorted(lengths.items(), key=lambda x: -x[0])}
    return sorted_lengths


def _compute_steps_per_epoch(trainer: pl.Trainer):
    if trainer.train_dataloader is None:
        trainer.reset_train_dataloader()

    total_batches = trainer.num_training_batches
    accumulate_grad_batches = trainer.accumulation_scheduler.get_accumulate_grad_batches(
        trainer.current_epoch)
    effective_batch_size = accumulate_grad_batches
    return math.ceil(total_batches / effective_batch_size)


def _get_final_number(s: str) -> str:
    maybe_minus = r"(?:\-\s?)?"
    maybe_decimal = r"(?:\.\d+)?"
    core = r"\d+"
    finds = re.findall(maybe_minus + core + maybe_decimal, s)
    if finds: 
        return finds[-1]
    return None


def _print_predictions(*, inputs, masks, generated_decoded, labels, all_generated, all_labels):
    for in_, mask, gen, ref, all_g, all_l in zip(
        inputs, masks, generated_decoded, labels, all_generated, all_labels):
        if gen == ref:
            color == "green"
        else:
            color = "yellow"
        CONSOLE.print(f"[bold {color}]\[gen-input][/] {in_}")
        CONSOLE.print(f"[bold {color}]\[gen-mask][/] {mask}")
        CONSOLE.print(f"[bold blue]\[gen-reference][/] {ref}")
        CONSOLE.print(f"[bold {color}]\[gen-generated][/] {gen}")
        CONSOLE.print(f"[bold]\[gen-all-labels] {all_l}")
        CONSOLE.print(f"[bold]\[gen-all-gen] {all_g}")
        CONSOLE.print(f"[bold]" + "=" * 80)


def _prefix_match(s1: str, s2: str) -> bool:
    s1 = s1.strip()
    s2 = s2.strip()
    return s1.startswith(s2) or s2.startswith(s1)


@beartype
def _last_non_masked(target: torch.Tensor, mask_token_id: int):
    utils.check_equal(target.ndim, 2)
    bsz = target.shape[0]
    label_range = torch.arange(target.shape[1]).repeat(bsz, 1).to(target.device)
    utils.check_equal(label_range.shape, target.shape)
    label_range[target == mask_token_id] = 0
    last_unmasked_token_pos = label_range.max(dim=1).values
    return torch.gather(target, 1, last_unmasked_token_pos.reshape(-1, 1))


@dataclasses.dataclass
class SamplesMarginal:
    y_mask_is_not_pad      : torch.Tensor
    z_mask_is_not_pad      : torch.Tensor
    MITGSWRV_ids           : torch.Tensor  # MITGSWRV: Masked Inputs Then Generated Samples With Reference Values
    ITGSWRV_ids            : torch.Tensor  # ITGSWRV: Inputs Then Generated Samples With Reference Values
    ITGSWRV_attention_mask : torch.Tensor  # ITGSWRV: Inputs Then Generated Samples With Reference Values


def prep_samples_marginal(
    *, 
    inputs_outputs:      torch.Tensor, 
    batch:               dict[str, torch.Tensor], 
    generation_kwargs:   dict[str, Any], 
    disable_timing:      bool, 
    eos_token_id:        int, 
    cls_token_id:        int, 
    label_pad_token_id:  int, 
    inputs_pad_token_id: int,
    batch_size:          int, 
    num_scratchpads:     int, 
) -> SamplesMarginal:
    
    utils.check_equal(label_pad_token_id, inputs_pad_token_id)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Unpad and interleave.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # utils.repeat_interleave is just a generator, doesn't not unnecessarily build a tensor.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with utils.cuda_timeit("[bold]bin_refine.py::PSI-A:[/] Unpad, Repeat-interleave", disable=disable_timing):
        with utils.cuda_timeit("[bold]bin_refine.py::PSI-A:[/] unpad then repeat-interleave", disable=disable_timing):
            unpadded_inputs_outputs = remove_padding(
                inputs_outputs, 
                inputs_outputs != inputs_pad_token_id,
            )
            unpadded_values = batch["value"]
            unpadded_inputs = remove_padding(
                batch["generation_input_ids"], 
                batch["generation_attention_mask"] == 1,
            )

            unpadded_repeated_inputs = utils.repeat_interleave(
                unpadded_inputs, 
                generation_kwargs["num_return_sequences"],
            )
            # Reproduce the structure of the multiple beams per input tensor
            unpadded_repeated_values = utils.repeat_interleave(
                unpadded_values, 
                generation_kwargs["num_return_sequences"],
            ) 

    final_ITGSWRV: list[list[int]] = []
    final_MITGSWRV: list[list[int]] = []
    y_mask: list[list[int]] = []
    z_mask: list[list[int]] = []

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MITGSWRV: Masked Input Then Generated Scratchpad With Rerefence Values
    # ITGSWRV:  Input Then Generated Scratchpad With Rerefence Values
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with utils.cuda_timeit("[bold]bin_refine.py::PSI-B:[/] Loop", disable=disable_timing):
        for inputs, io, value  in zip(
            unpadded_repeated_inputs,
            unpadded_inputs_outputs ,
            unpadded_repeated_values,
        ):
            # TODO: Future optimization: don't convert to a list. Probably faster.
            value = value.tolist()
            io_list = io.tolist()
            del io

            if not io_list[-1] == cls_token_id:
                io_list.append(cls_token_id)
            
            scratchpad = io_list[len(inputs):]

            final_ITGSWRV_entry   = io_list                                                    + value + [eos_token_id]
            final_MITGSWRV_entry  = len(inputs) * [label_pad_token_id] + scratchpad            + value + [eos_token_id]
            y_mask_entry          = len(inputs) * [0]                  + len(scratchpad) * [0] + (len(value) + 1) * [1]
            z_mask_entry          = len(inputs) * [0]                  + len(scratchpad) * [1] + (len(value) + 1) * [0]

            final_ITGSWRV.append(final_ITGSWRV_entry)
            final_MITGSWRV.append(final_MITGSWRV_entry)
            y_mask.append         (y_mask_entry)
            z_mask.append         (z_mask_entry)

            utils.check_equal(len(final_MITGSWRV_entry), len(final_ITGSWRV_entry))
            utils.check_equal(len(y_mask_entry),         len(final_ITGSWRV_entry))
            utils.check_equal(len(z_mask_entry),         len(final_ITGSWRV_entry))
            
    with utils.cuda_timeit("[bold]bin_refine.py::PSI-C:[/] Pad", disable=disable_timing):
        utils.check_equal(len(final_ITGSWRV), len(final_MITGSWRV))
        padded_final_ITGSWRV = pad(final_ITGSWRV, inputs_pad_token_id, "right").to(inputs_outputs.device)
        padded_final_MITGSWRV = pad(final_MITGSWRV, label_pad_token_id , "right").to(inputs_outputs.device)
        padded_final_attention_mask = generate_mask(final_ITGSWRV, "right").to(inputs_outputs.device)
        padded_y_mask_is_not_pad = pad(y_mask, 0, "right").to(inputs_outputs.device, dtype=padded_final_attention_mask.dtype)
        padded_z_mask_is_not_pad = pad(z_mask, 0, "right").to(inputs_outputs.device, dtype=padded_final_attention_mask.dtype)
        padded_y_mask_is_not_pad = padded_y_mask_is_not_pad.reshape(batch_size, num_scratchpads, -1)
        padded_z_mask_is_not_pad = padded_z_mask_is_not_pad.reshape(batch_size, num_scratchpads, -1)

        utils.check_equal(padded_final_MITGSWRV       .shape[-1], padded_final_ITGSWRV.shape[-1])
        utils.check_equal(padded_y_mask_is_not_pad    .shape[-1], padded_final_ITGSWRV.shape[-1])
        utils.check_equal(padded_z_mask_is_not_pad    .shape[-1], padded_final_ITGSWRV.shape[-1])
        utils.check_equal(padded_final_attention_mask .shape[-1], padded_final_ITGSWRV.shape[-1])

    with utils.cuda_timeit("bin_refine.py::Score", disable=disable_timing):
        utils.check_equal(padded_final_attention_mask  .shape,       padded_final_MITGSWRV.shape)
        utils.check_equal(padded_final_ITGSWRV         .shape,       padded_final_attention_mask.shape)
        utils.check_equal(padded_y_mask_is_not_pad     .shape[:-1], (batch_size, num_scratchpads,))

    return SamplesMarginal(
        ITGSWRV_attention_mask = padded_final_attention_mask, 
        ITGSWRV_ids            = padded_final_ITGSWRV, 
        MITGSWRV_ids           = padded_final_MITGSWRV, 
        y_mask_is_not_pad      = padded_y_mask_is_not_pad, 
        z_mask_is_not_pad      = padded_z_mask_is_not_pad,
    )


def print_table_marginal_likelihood(
    padded_final_input_ids, padded_final_attention_mask, 
    padded_final_labels, label_logprobs, batch_size, 
    num_scratchpads, seq_len, flat_labels, label_pad_token_id,
    tokenizer
):
    table_ = table.Table("Input", "Label", "Attn-Mask", "prob")
    sample_index = random.randint(0, len(padded_final_input_ids) - 1)
    for input_id, attention_mask, label, logprobs, masked in zip(
        padded_final_input_ids[sample_index], 
        padded_final_attention_mask[sample_index], 
        padded_final_labels[sample_index],
        label_logprobs.reshape(batch_size * num_scratchpads, seq_len)[sample_index],
        (flat_labels == label_pad_token_id).reshape(batch_size * num_scratchpads, seq_len)[sample_index]
    ):
        if label == label_pad_token_id:
            label = tokenizer.pad_token_id
    
        table_.add_row(
            f"'{tokenizer.decode(input_id)}'", 
            f"'{tokenizer.decode(label)   }'", 
            f"'{attention_mask.item()           }'",
            f"'{torch.exp(logprobs)             }'",
            f"'{masked                          }'",
        )

    CONSOLE.print(table_)


def prep_logits_and_MITGSWRV(
    *,
    vocab_size:         int,
    batch_size:         int, 
    num_scratchpads:    int, 
    label_pad_token_id: int,
    ITGSWRV_logits:     torch.Tensor, 
    MITGSWRV_ids:       torch.Tensor, 
    tokenizer:          transformers.GPT2Tokenizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    assert -100 not in MITGSWRV_ids
    assert num_scratchpads != -100, num_scratchpads
    assert label_pad_token_id != -100, label_pad_token_id

    shift_MITGSWRV = MITGSWRV_ids[..., 1:].contiguous()
    flat_MITGSWRV = shift_MITGSWRV.view(-1).unsqueeze(-1).contiguous()
    bsz_times_num_beams = ITGSWRV_logits.shape[0]

    seq_len = shift_MITGSWRV.shape[1]
    shift_MITGSWRV = shift_MITGSWRV.reshape(batch_size, num_scratchpads, seq_len)
    shift_MITGSWRV_is_pad = (shift_MITGSWRV == label_pad_token_id).to(MITGSWRV_ids.dtype)

    # Shift the logits, prep them for the gather, do the gather
    shift_ITGSWRV_logits = ITGSWRV_logits[..., :-1, :].contiguous()
    flat_shift_ITGSWRV_logits = shift_ITGSWRV_logits.view(-1, shift_ITGSWRV_logits.shape[-1]).contiguous()
    
    flat_shift_ITGSWRV_log_softmax   = flat_shift_ITGSWRV_logits.log_softmax(dim=-1)
    flat_shift_MITGSWRV_log_softmax  = flat_shift_ITGSWRV_log_softmax.gather(dim=1, index=flat_MITGSWRV)
    MITGSWRV_log_softmax = flat_shift_MITGSWRV_log_softmax.reshape(batch_size, num_scratchpads, seq_len)
        
    utils.check_equal(MITGSWRV_log_softmax.shape[-1],         seq_len)
    utils.check_equal(shift_MITGSWRV.shape,                  (batch_size,  num_scratchpads,  seq_len))
    utils.check_equal(shift_MITGSWRV_is_pad.shape,           (batch_size,  num_scratchpads,  seq_len))
    utils.check_equal(flat_shift_MITGSWRV_log_softmax.shape, (batch_size * num_scratchpads * seq_len,     1))
    utils.check_equal(flat_shift_ITGSWRV_logits.shape,       (batch_size * num_scratchpads * seq_len,     vocab_size))
    utils.check_equal(bsz_times_num_beams,                    batch_size * num_scratchpads)
    utils.check_equal(ITGSWRV_logits.shape,                  (batch_size * num_scratchpads,  seq_len + 1, vocab_size))

    return MITGSWRV_log_softmax, shift_MITGSWRV_is_pad, shift_MITGSWRV


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


def _show_multi_scratchpad_table_format_text(ids, tokenizer):
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
    
    y_prob = y_prob.clone()
    z_prob = z_prob.clone()

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

        table_.add_row(
            maybe_color + _show_multi_scratchpad_table_format_text(shift_MITGSWRV[demo_input_idx, rank_y], tokenizer) 
            + maybe_close,
            f"{y_prob_entry[rank_y].item():.3}", 
            f"{z_prob_entry[rank_y].item():.3}",
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


class Losses:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "This class is a static namespace, it is not meant to be instantiated."
        )

    @classmethod
    def most_basic_loss(cls, y_part_log_prob, z_part_log_prob):

        return (y_part_log_prob + z_part_log_prob).logsumexp(dim=-1)
    
    @classmethod        
    def js_divergence_loss(cls, *, y_log_probs, z_log_probs):
        y_part_log_prob = y_log_probs.sum(dim=-1).log_softmax(dim=-1)
        z_part_log_prob = z_log_probs.sum(dim=-1).log_softmax(dim=-1)

        jsd = (y_part_log_prob.exp().detach() - z_part_log_prob.exp()
            ) * (y_part_log_prob.detach() - z_part_log_prob)

        return torch.mean(jsd)                

    @classmethod
    def squared_loss(cls, y_log_probs, z_log_probs):
        y_part_log_prob = y_log_probs.sum(dim=-1)
        z_part_log_prob = z_log_probs.sum(dim=-1)

        return (y_part_log_prob.detach() - z_part_log_prob) ** 2


def ref_special_gather(
    *, 
    tensor:           torch.Tensor, 
    most_helpful_idx: torch.Tensor, 
    batch_size:       int,
) -> torch.Tensor:
    """
    Copy over the tokens of the scratchpad with the highest p( y | z, x ).

    Reference, non-vectorized implementation.
    """
    ref_final_final = torch.zeros(
        batch_size, 
        tensor.shape[-1], 
        dtype=torch.long, 
        device=tensor.device
    )
    for i in range(batch_size):
        for k in range(tensor.shape[-1]):
            ref_final_final[i, k] = tensor[i, most_helpful_idx[i], k]

    return ref_final_final

def special_gather(
    *, 
    tensor:           torch.Tensor, 
    most_helpful_idx: torch.Tensor, 
    batch_size:       int, 
    num_scratchpads:  int,
) -> torch.Tensor:
    """
    Copy over the tokens of the scratchpad with the highest p( y | z, x )

    Vectorized implementation.

    tensor is of shape (batch_size, num_scratchpads, seq_len)

    """
    index = most_helpful_idx.unsqueeze(-1).unsqueeze(-1).expand(
        batch_size, num_scratchpads, tensor.shape[-1]
    )
    final_final_MITGSWRV = tensor.gather(
        dim=1, index=index,
    )[:, 0, :]    

    return final_final_MITGSWRV


class _RefineLM(pl.LightningModule):
    def __init__(
        self,
        *,
        batch_sizes: Dict[str, int],
        datasets: Dict[str, torch.utils.data.Dataset],
        generation_kwargs: dict[str, Any],
        learning_rate: float,
        meta_info: dict,
        model: transformers.GPT2LMHeadModel,
        is_adamw: bool,
        lm_masking_mode: str,
        path_log_results: Path,
        scheduler_type: str,
        switch_to_maginal_after: dict[str, int],
        tokenizer: transformers.PreTrainedTokenizer,
        wandb_logger: Optional[pl.loggers.WandbLogger],
        weight_decay: Optional[float],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "datasets", "tokenizer"])

        self._dataloader_num_workers: Final[int] = DATALOADER_NUM_WORKERS
        self._wandb_logger: Final[pl.loggers.WandbLogger] = wandb_logger
        self._model: Final[transformers.GPT2LMHeadModel] = model
        self._fixed_model: Optional[transformers.GPT2LMHeadModel] = None
        self._datasets: Final[Dict[str, torch.utils.data.Dataset]] = datasets
        self._tokenizer: Final[transformers.PreTrainedTokenizer] = tokenizer
        self._batch_size: Final[dict[str, int]] = batch_sizes
        self._generation_kwargs: Final[dict[str, Any]] = generation_kwargs
        self._logging_conf: Final[dict[str, bool]] = dict(
            prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        self._meta_info = meta_info
        self._lm_masking_mode: Final[str] = lm_masking_mode
        self._switch_to_maginal_after: dict[str, int] = switch_to_maginal_after
        if switch_to_maginal_after:
            assert utils.safe_xor(
                "epochs" in switch_to_maginal_after, 
                "steps" in switch_to_maginal_after,
            )

        ################################################################################
        # Related to datasets
        ################################################################################
        self._shuffle_train: Final[bool] = SHUFFLE_TRAINING_DATA
        self._shuffle_val: Final[bool] = SHUFFLE_VALIDATION_DATA
        self._training_collators = {
            constants.PipelineModes.MLE_TRAINING: 
                MLETrainingCollator(self._tokenizer, self._lm_masking_mode),
            constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING:
                MarginalLikelihoodTrainingCollator(self._tokenizer, self._lm_masking_mode),
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
        self._scheduler_type: Final[str] = scheduler_type
        self._scheduler = None


    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)


    def _training_step_mle(self, batch, batch_idx):

        assert "labels" in batch, (
            "Labels must be in batch. We must mask the input section with -100"
        )

        batch = {
            k: v for k, v in batch.items() 
            if k in ["input_ids", "attention_mask", "labels"]
        }
        
        outputs = self(**batch)

        self.log(
            "train_loss", 
            outputs.loss, 
            batch_size=self._batch_size[constants.PipelineModes.MLE_TRAINING], 
            **self._logging_conf
        )

        # TODO: this is costly
        assert not torch.any(torch.isnan(outputs.loss)), "Loss is NaN"

        return outputs.loss


    def _training_step_marginal_likelihood(self, batch, batch_idx):
        """
        
        p(z|x): <generation>
            generation_input_ids: masked, question, chainer. *Padded left*.
            generation_attention_mask

        p(y, z| x): 
            input_ids: input, 
            labels:
        ---

        p(y, z | x) = p(y | z, x) * p(z | x)

        """

        #######################################################################
        # Useful constants
        #######################################################################
        mode: Final[str] = constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING
        batch_size = self._batch_size[mode]
        vocab_size = len(self._tokenizer)
        num_scratchpads = self._generation_kwargs[mode]["num_return_sequences"]
        disable_timing = True 
        SHOW_SCRATCHPAD_PADDING_TABLE = False
        SHOW_MULTI_SCRATCHPAD_TABLE = True
        SHOW_SMALL_GENERATION_TABLE = False
        torch.autograd.set_detect_anomaly(True)                


        if not disable_timing:
            CONSOLE.print_zero_rank(f"batch size:           {self._batch_size[mode]}")
            CONSOLE.print_zero_rank(f"num return sequences: {self._generation_kwargs[mode]['num_return_sequences']}")

        #######################################################################
        # Generate the scratchpads
        #######################################################################
        with utils.cuda_timeit("bin_refine.py::Generation", disable=disable_timing):
            self._model.eval()
            # Generating until CLS makes us only generate the scratchpad. 
            # Generating until EOS makes us generate the whole output.
            # That's how the model is pre-trained with the mle objective.
            generate_output_dict = self._model.generate(
                input_ids               = batch["generation_input_ids"],
                attention_mask          = batch["generation_attention_mask"], 
                # Config stuff
                eos_token_id            = self._tokenizer.cls_token_id,  
                output_scores           = True,
                return_dict_in_generate = True,
                **self._generation_kwargs[mode],
            )
            self._model.train()


        #######################################################################
        # Preparations common to the different losses
        #######################################################################
        batch_size = batch["generation_input_ids"].shape[0]
        num_scratchpads = self._generation_kwargs[mode]["num_return_sequences"]
        vocab_size = len(self._tokenizer)
        utils.check_equal(self._batch_size[mode], batch_size)

        input_ids_then_scratchpads = generate_output_dict["sequences"].reshape(
            batch_size * num_scratchpads,
            -1,
        )
        
        scores = torch.stack(generate_output_dict["scores"]).reshape(
            batch_size,
            num_scratchpads,
            len(generate_output_dict["scores"]),
            vocab_size
        )

        class LossModes(str, enum.Enum):
            PPO = "ppo"
            STRONGEST_MLE = "strongest_mle"

        loss_mode = LossModes.PPO
        label_pad_token_id = self._tokenizer.pad_token_id
        utils.check_equal(label_pad_token_id, self._tokenizer.pad_token_id)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sample preparation for per scratchpad stuff
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        samples_marginal = prep_samples_marginal(
            batch               = batch, 
            batch_size          = batch_size, 
            eos_token_id        = self._tokenizer.eos_token_id, 
            cls_token_id        = self._tokenizer.cls_token_id, 
            disable_timing      = disable_timing, 
            inputs_outputs      = input_ids_then_scratchpads, 
            num_scratchpads     = num_scratchpads, 
            generation_kwargs   = self._generation_kwargs[mode], 
            label_pad_token_id  = label_pad_token_id,
            inputs_pad_token_id = self._tokenizer.pad_token_id,
        ) 

        ITGSWRV_attention_mask      = samples_marginal.ITGSWRV_attention_mask
        ITGSWRV_ids                 = samples_marginal.ITGSWRV_ids
        
        # MITGSWRV: Masked Input Then Generated Scratchpad With Reference Value
        MITGSWRV_ids                = samples_marginal.MITGSWRV_ids
        y_mask_is_not_pad           = samples_marginal.y_mask_is_not_pad
        z_mask_is_not_pad           = samples_marginal.z_mask_is_not_pad
        shift_y_mask_is_not_pad     = y_mask_is_not_pad[:, :, 1:]
        shift_z_mask_is_not_pad     = z_mask_is_not_pad[:, :, 1:]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Learning rate
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert torch.all(
            ITGSWRV_ids[:, 0] != self._tokenizer.pad_token_id
        )
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]["lr"]                
        utils.check_equal(len(self.trainer.optimizers), 1)
        utils.check_equal(len(optimizer.param_groups), 1)
        CONSOLE.print_zero_rank(f"[bold]Learning rate:[/] {lr:.3}")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute the logits
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert (ITGSWRV_ids[:, 0] != self._tokenizer.pad_token_id).all()
        
        ITGSWRV_logits = self._model(
            input_ids      = ITGSWRV_ids,
            attention_mask = ITGSWRV_attention_mask,
        ).logits

        with torch.no_grad():
            ITGSWRV_logits_fixed_model = self._fixed_model(
                input_ids      = ITGSWRV_ids,
                attention_mask = ITGSWRV_attention_mask,
            ).logits

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Shift the logits and MITGSWRV
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        shift_MITGSWRV_log_softmax, shift_MITGSWRV_is_pad, shift_MITGSWRV = prep_logits_and_MITGSWRV(
            vocab_size         = vocab_size,
            ITGSWRV_logits     = ITGSWRV_logits, 
            batch_size         = batch_size, 
            MITGSWRV_ids       = MITGSWRV_ids, 
            num_scratchpads    = num_scratchpads, 
            label_pad_token_id = label_pad_token_id, 
            tokenizer          = self._tokenizer,
        ) 
        shift_seq_len = shift_MITGSWRV_log_softmax.shape[2]

        shift_MITGSWRV_log_softmax_fixed_model, _, _ = prep_logits_and_MITGSWRV(
            vocab_size         = vocab_size,
            ITGSWRV_logits     = ITGSWRV_logits_fixed_model,
            batch_size         = batch_size,
            MITGSWRV_ids       = MITGSWRV_ids,
            num_scratchpads    = num_scratchpads,
            label_pad_token_id = label_pad_token_id,
            tokenizer          = self._tokenizer,
        )


        ###############################################################
        # Extract the log-probs for the labels for y and z
        ###############################################################

        utils.check_equal(shift_MITGSWRV_log_softmax .shape, (batch_size, num_scratchpads, shift_seq_len))
        utils.check_equal(shift_MITGSWRV_is_pad      .shape, (batch_size, num_scratchpads, shift_seq_len))
        utils.check_equal(shift_y_mask_is_not_pad    .shape, (batch_size, num_scratchpads, shift_seq_len))
        utils.check_equal(shift_z_mask_is_not_pad    .shape, (batch_size, num_scratchpads, shift_seq_len))

        mask_y = shift_y_mask_is_not_pad * shift_MITGSWRV_is_pad.bool().logical_not().long()
        mask_z = shift_z_mask_is_not_pad * shift_MITGSWRV_is_pad.bool().logical_not().long()

        z_log_probs = shift_MITGSWRV_log_softmax * mask_z
        
        with torch.no_grad():
            y_log_probs_fixed = shift_MITGSWRV_log_softmax_fixed_model * mask_y
            z_log_probs_fixed = shift_MITGSWRV_log_softmax_fixed_model * mask_z
        

        ###############################################################
        # -> Log-likelihoods for y and for z
        # -> Importance Sampling Ratio
        ###############################################################                

        utils.check_equal(z_log_probs       .shape, (batch_size, num_scratchpads, shift_seq_len))
        utils.check_equal(z_log_probs_fixed .shape, (batch_size, num_scratchpads, shift_seq_len))

        y_log_probs_fixed_per_seq = y_log_probs_fixed.detach().sum(dim=-1)
        z_log_probs_per_seq       = z_log_probs               .sum(dim=-1)

        most_helpful_idx = y_log_probs_fixed_per_seq.argmax(dim=-1).detach()

        most_helpful_log_probs = z_log_probs_per_seq.gather(
            dim=-1, 
            index=most_helpful_idx.unsqueeze(-1),
        ).squeeze(-1)
        
        ###############################################################
        # COMPUTE THE CROSS-ENTROPY LOSS WITH THE MOST HELPFUL SEQUENCE
        ###############################################################
        MITGSWRV_ids           = MITGSWRV_ids               .reshape(batch_size, num_scratchpads, -1).clone()
        ITGSWRV_ids            = ITGSWRV_ids                .reshape(batch_size, num_scratchpads, -1).clone()
        ITGSWRV_attention_mask = ITGSWRV_attention_mask.reshape(batch_size, num_scratchpads, -1).clone()

        #######################################################################
        # Do MLE on the strongest scratchpad
        #######################################################################
        if loss_mode == LossModes.STRONGEST_MLE:
            ref_final_ITGSWRV_input_ids      = ref_special_gather(tensor=ITGSWRV_ids,            most_helpful_idx=most_helpful_idx, batch_size=batch_size)
            ref_final_ITGSWRV_attention_mask = ref_special_gather(tensor=ITGSWRV_attention_mask, most_helpful_idx=most_helpful_idx, batch_size=batch_size)
            ref_final_MITGSWRV_ids           = ref_special_gather(tensor=MITGSWRV_ids,           most_helpful_idx=most_helpful_idx, batch_size=batch_size)

            final_ITGSWRV_input_ids      = special_gather(tensor=ITGSWRV_ids,            most_helpful_idx=most_helpful_idx, batch_size=batch_size, num_scratchpads=num_scratchpads)
            final_ITGSWRV_attention_mask = special_gather(tensor=ITGSWRV_attention_mask, most_helpful_idx=most_helpful_idx, batch_size=batch_size, num_scratchpads=num_scratchpads)
            final_MITGSWRV_ids           = special_gather(tensor=MITGSWRV_ids,           most_helpful_idx=most_helpful_idx, batch_size=batch_size, num_scratchpads=num_scratchpads)

            assert -100 not in MITGSWRV_ids
            assert (ref_final_ITGSWRV_input_ids      == final_ITGSWRV_input_ids).all()
            assert (ref_final_ITGSWRV_attention_mask == final_ITGSWRV_attention_mask).all()
            assert (ref_final_MITGSWRV_ids           == final_MITGSWRV_ids).all()

            del ref_final_ITGSWRV_input_ids
            del ref_final_ITGSWRV_attention_mask
            del ref_final_MITGSWRV_ids
            
            final_MITGSWRV_ids[final_MITGSWRV_ids == self._tokenizer.pad_token_id] = -100

            assert self._tokenizer.pad_token_id not in final_MITGSWRV_ids
            assert (final_ITGSWRV_input_ids[:, 0] != self._tokenizer.pad_token_id).all()

            loss = self._model(
                input_ids      = final_ITGSWRV_input_ids,
                attention_mask = final_ITGSWRV_attention_mask,
                labels         = final_MITGSWRV_ids,
            ).loss

        elif loss_mode == "not ready":
            assert False
            import_ratio_w_fixed_z = z_log_probs.sum(dim=-1) - z_log_probs_fixed.sum(dim=-1)
        
            most_helpful_log_probs_ratio = import_ratio_w_fixed_z.gather(
                dim=-1, index=most_helpful_idx.unsqueeze(-1)
            ).squeeze(-1)
            
            average_log_probs_ratio = torch.mean(import_ratio_w_fixed_z, dim=-1)

            utils.check_equal(most_helpful_log_probs_ratio.shape, (batch_size,))
            utils.check_equal(average_log_probs_ratio.shape,      (batch_size,))
            utils.check_equal(most_helpful_log_probs.shape,       (batch_size,))

            beta = 1.5
            nll_best = - most_helpful_log_probs.mean(dim=-1)
            baseline_respect_loss = beta * average_log_probs_ratio.mean(dim=-1)
            nll = nll_best + beta * average_log_probs_ratio.mean(dim=-1)

            # E(x, y) ∼ D_{π^{RL}_φ} [r_θ(x, y) − β log (π^{RL}_φ (y | x) / π^{SFT}(y | x))] + γ E_{x}∼D_{pretrain} [log(π^{RL}_φ (x))]
            # E[p(y | x, y) - beta * (log p(y | x, y) - log p_{fixed}(y | x))] # + γ E[log(π^{RL}_φ (x))]
            loss = nll

            CONSOLE.print_zero_rank(f"{nll_best = }")
            CONSOLE.print_zero_rank(f"{baseline_respect_loss = }")
            self.log("nll_best", nll_best.item(), prog_bar=True)
            self.log("baseline_respect_loss", baseline_respect_loss.item(), prog_bar=True)
            utils.check_equal(loss.ndim, 0)

        elif loss_mode == LossModes.PPO:

            seq_z_log_probs = z_log_probs.sum(dim=-1)
            seq_z_log_probs_fixed = z_log_probs_fixed.sum(dim=-1)

            import_ratio_w_fixed_z = seq_z_log_probs - seq_z_log_probs_fixed
            utils.check_equal(most_helpful_log_probs.shape,       (batch_size,))

            beta = 5.
            rl_reward = seq_z_log_probs * y_log_probs_fixed_per_seq
            ppo_importance_sampling_penalty = seq_z_log_probs * import_ratio_w_fixed_z

            # E(x, y) ∼ D_{π^{RL}_φ} [r_θ(x, y) − β log (π^{RL}_φ (y | x) / π^{SFT}(y | x))] + γ E_{x}∼D_{pretrain} [log(π^{RL}_φ (x))]
            # E[p(y | x, y) - beta * (log p(y | x, y) - log p_{fixed}(y | x))] # + γ E[log(π^{RL}_φ (x))]

            loss = (rl_reward - beta * ppo_importance_sampling_penalty).mean()

            self.log("rl_reward",                       rl_reward.mean().item(),                       **self._logging_conf)
            self.log("ppo_importance_sampling_penalty", ppo_importance_sampling_penalty.mean().item(), **self._logging_conf)
            self.log("loss",                            loss.item(),                                   **self._logging_conf)
            utils.check_equal(loss.ndim, 0)
        else:
            raise ValueError(f"Unknown loss mode {loss_mode}")


        with torch.no_grad():
            y_part_prob = y_log_probs_fixed_per_seq.exp()
            z_part_prob = z_log_probs_per_seq.exp()

        demo_input_idx = random.randint(0, batch_size - 1)
        demo_input_sp = random.randint(0, num_scratchpads - 1)

        if SHOW_SMALL_GENERATION_TABLE:
            show_small_generation_table(
                scores=scores,
                tokenizer=self._tokenizer,
                inputs_outputs=input_ids_then_scratchpads.view(batch_size, num_scratchpads, -1),
                demo_input_sp=demo_input_sp,
                demo_input_idx=demo_input_idx,
            )

        if SHOW_MULTI_SCRATCHPAD_TABLE:                    
            show_multi_scratchpad_table(
                y_prob=y_part_prob,
                z_prob=z_part_prob,
                labels=batch["labels"], 
                tokenizer=self._tokenizer,
                shift_MITGSWRV=shift_MITGSWRV,
                demo_input_idx=demo_input_idx,
                num_scratchpads=num_scratchpads,
            )

        if SHOW_SCRATCHPAD_PADDING_TABLE :                    
            show_scratchpad_padding_table(
                mask_x=mask_x,
                mask_y=mask_y,
                batch_size=batch_size,
                shift_MITGSWRV=shift_MITGSWRV,
                tokenizer=self._tokenizer,
                demo_input_sp=demo_input_sp,
                demo_input_idx=demo_input_idx, 
                num_scratchpads=num_scratchpads,
                shift_MITGSWRV_is_pad=shift_MITGSWRV_is_pad,
                padded_final_input_ids=final_ITGSWRV_input_ids,
                shift_prob=shift_log_probs.sum(dim=-1).exp() if 
                    "shift_log_probs" in locals() else shift_probs.prod(dim=-1),
            )

        print(f"[{self.trainer.global_rank}] [bold]Loss:[/] {loss}")
        
        return loss


    def _decide_training_mode(self):
        mode = constants.PipelineModes.MLE_TRAINING
        if self._switch_to_maginal_after and "epochs" in self._switch_to_maginal_after:
            assert "steps" not in self._switch_to_maginal_after, self._switch_to_maginal_after
            utils.check_equal(len(self._switch_to_maginal_after), 1)

            if self.current_epoch >= self._switch_to_maginal_after["epochs"]:
                mode = constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING

        elif self._switch_to_maginal_after and "steps" in self._switch_to_maginal_after:
            assert "epochs" not in self._switch_to_maginal_after, self._switch_to_maginal_after
            utils.check_equal(len(self._switch_to_maginal_after), 1)

            if self.global_step >= self._switch_to_maginal_after["steps"]:
                mode = constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING

        else:
            assert not self._switch_to_marginal_after, self._switch_to_marginal_after

        return mode


    def training_step(self, batch, batch_idx):
        mode = self._decide_training_mode()
        

        if mode == constants.PipelineModes.MLE_TRAINING:
            return self._training_step_mle(batch, batch_idx)
        elif (
            mode == 
            constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING
        ):
            return self._training_step_marginal_likelihood(batch, batch_idx)
        else:
            raise ValueError(f"Unknown training mode: {mode}")


    def _decode_per_token(self, batch):
        dps = []
        for entry in batch:
            per_entry = []
            for token in entry:
                per_entry.append(
                    self._tokenizer.decode([token if token >= 0 else self._tokenizer.pad_token_id], 
                    skip_special_tokens=False))
            dps.append(per_entry)
        return dps


    def _decode_per_sample(self, batch):
        dps = []
        for entry in batch:
            dps.append(self._tokenizer.decode(
                [x if x >= 0 else self._tokenizer.pad_token_id for x in entry], skip_special_tokens=False))
        return dps


    def _generate(self, *, batch, generation_kwargs, batch_mode, model):
        assert "labels" in batch, "Labels must be in batch. We must mask the input section with -100"
        
        generation_inputs = batch["generation_input_ids"]
        
        if batch_mode:
            generation_attention_mask = batch["generation_attention_mask"]
            
            raw_generation_outputs = model.generate(
                input_ids=generation_inputs, 
                attention_mask=generation_attention_mask, 
                **generation_kwargs,
            )
            raw_generation_outputs = raw_generation_outputs[:, generation_inputs.shape[1]:]
        else:
            raw_generation_outputs_list =  []
            for input_ids in generation_inputs:
                input_ids = input_ids[input_ids != self._tokenizer.pad_token_id]
                output = model.generate(
                    input_ids=input_ids.reshape(1, -1),
                    **generation_kwargs,
                )
                utils.check_equal(output.shape[0], 1)
                raw_generation_outputs_list.append(
                    output[0][input_ids.shape[0]:]
                )
            raw_generation_outputs = pad(
                seq=raw_generation_outputs_list, 
                pad_token_id=self._tokenizer.pad_token_id, 
                direction="left"
            )

        return raw_generation_outputs


    def on_train_epoch_start(self) -> None:
        assert (
            not "steps" in self._switch_to_maginal_after and 
            "epochs" in self._switch_to_maginal_after
        ), f"Can't be both. {self._switch_to_maginal_after}"
        
        mode = self._decide_training_mode()
        if mode == constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING:
            self.trainer.val_check_interval = VAL_CHECK_INTERVAL
        self.trainer.reset_train_dataloader()

        # If we're starting epoch n (like 1), and we have 
        # self._switch_to_maginal_after["epochs"] == n (like 1), 
        # then we need to make a copy of the model 


        is_changing_epoch = ("epochs" in self._switch_to_maginal_after 
            and self._switch_to_maginal_after["epochs"] == self.current_epoch
        )
        is_changing_step = ("steps" in self._switch_to_maginal_after 
            and self._switch_to_maginal_after["steps"] == self.global_step)

        if is_changing_epoch or is_changing_step:
            assert self._fixed_model is None, "We should only assign this once"
            CONSOLE.print_zero_rank(f"[red bold]MAKING A COPY OF THE MODEL")
            self._fixed_model = clone_hf_model(self._model).eval().to(self._model.device)
            for param in self._fixed_model.parameters():
                param.requires_grad = False

    def validation_step(self, batch: Dict[str, torch.LongTensor], batch_idx):  # type: ignore[override]
        assert "labels" in batch, (
            "Labels must be in batch. We must mask the input section with -100"
        )
        mode: Final[str] = constants.PipelineModes.VALIDATION
        
        # if self._fixed_model is None:
        gen_outputs = self._generate(
            model=self._model,
            batch=batch, 
            batch_mode=True,
            generation_kwargs=self._generation_kwargs[mode], 
        )
        # else:
        #     config_scratch_pad = self._generation_kwargs[mode].copy()
        #     config_scratch_pad["eos_token_id"] = self._tokenizer.cls_token_id
        #     gen_outputs = self._generate(
        #         model=self._model,
        #         batch=batch, 
        #         batch_mode=True,
        #         generation_kwargs=config_scratch_pad, 
        #     )
            
        #     config_answer = self._generation_kwargs[mode].copy()
        #     unpadded = remove_padding(gen_outputs)
        #     padded = pad(unpadded, "left", self._tokenizer.pad_token_id)
        #     attention_mask = mask(padded, self._tokenizer.pad_token_id)
        #     new_batch = dict(
        #         input_ids=padded,
        #         attention_mask=attention_mask,
        #     )

        #     gen_outputs = self._generate(
        #         model=self._model,
        #         batch=new_batch, 
        #         batch_mode=True,
        #         generation_kwargs=config_scratch_pad, 
        #     )


            

        dps_generation = self._decode_per_sample(gen_outputs)
        dps_labels = self._decode_per_sample(batch["input_ids"])

        for_comparison = [(
            _clean_for_accuracy_computation(gen, self._tokenizer).strip(), 
            _clean_for_accuracy_computation(l,   self._tokenizer).split(CHAINER, 1)[1].strip(),
            ) for gen, l in zip(dps_generation, dps_labels)
        ]

        if batch_idx == 0 and utils.is_rank_zero():
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Log Generated Text in Wandb.  
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            table_entry = []
            for index, (gen, lab) in enumerate(zip(
                dps_generation, dps_labels
            )):
                table_entry.append([self.current_epoch, index, "generation", gen])
                table_entry.append([self.current_epoch, index, "label", lab])

            self._wandb_logger.log_text(
                key="samples",
                columns=["epoch", "idx_in_batch", "type", "text"], 
                data=table_entry,
            )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Print the generated text
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for gen, ref in for_comparison:
                if gen == ref:
                    color = "green"
                    CONSOLE.print_zero_rank(f"[bold {color}]>>> Match")
                else:
                    color = "red"
                    CONSOLE.print_zero_rank("Mismatch")

                CONSOLE.print_zero_rank(f"[bold {color}]\[ref] {ref}")
                CONSOLE.print_zero_rank(f"[bold blue]\[gen] {gen}")
                CONSOLE.print_zero_rank("=" * 80)
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute different metrics
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        em_accuracy      = np.mean([x == y                                       for x, y in for_comparison])
        final_answer_acc = np.mean([_get_final_number(x) == _get_final_number(y) for x, y in for_comparison])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ppl_outputs = self._model(**{k: batch[k]for k in ["input_ids", "attention_mask", "labels"]})

        self.log("val_em"  , em_accuracy,      batch_size=self._batch_size[mode], **self._logging_conf)
        self.log("val_answ", final_answer_acc, batch_size=self._batch_size[mode], **self._logging_conf)
        self.log("val_loss", ppl_outputs.loss, batch_size=self._batch_size[mode], **self._logging_conf)

        assert not torch.any(torch.isnan(ppl_outputs.loss)), "Loss is NaN"        
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
        )

        self._scheduler = SCHEDULER_FN[self._scheduler_type](
            optimizer, _compute_steps_per_epoch(self.trainer)
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=self._scheduler,
                interval="step",
                frequency=1,
                name=type(self._scheduler).__name__,
            )
        )


    def train_dataloader(self):        
        mode = self._decide_training_mode()
        
        assert mode in {
            constants.PipelineModes.MLE_TRAINING, 
            constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING
        }, mode

        return torch.utils.data.DataLoader(
            self._datasets[mode],
            collate_fn=self._training_collators[mode],
            batch_size=self._batch_size[mode],
            num_workers=self._dataloader_num_workers,
            shuffle=self._shuffle_train,
        )


    def val_dataloader(self):
        mode: Final[str] = constants.PipelineModes.VALIDATION
        return torch.utils.data.DataLoader(
            self._datasets[mode],
            collate_fn=ValitationCollator(self._tokenizer, self._lm_masking_mode),
            batch_size=self._batch_size[mode],
            num_workers=self._dataloader_num_workers,
            shuffle=self._shuffle_val,
        )

    
    def predict_dataloader(self):
        return self.val_dataloader()


    def on_save_checkpoint(self, ckpt):
        return 


def unpad_concatenate_repad(*, 
    tensors: list[torch.LongTensor], 
    attention_masks: list[torch.LongTensor],
    new_padding_direction: str, 
    pad_token_id: int
):
    for tensor in tensors:
        utils.check_equal(tensor.ndims, 2), tensor.shape
    lists_of_lists = [remove_padding(x, mask) for x, mask in more_itertools.zip_equal(tensors, attention_masks)]
    concatenated = [list(itertools.chain.from_iterable(list_of_lists)) for list_of_lists in zip(*lists_of_lists)]
    mask = generate_mask(concatenated, pad_token_id, new_padding_direction)
    padded = pad(concatenated, pad_token_id, new_padding_direction)
    return padded, mask


def _clean_for_accuracy_computation(text, tokenizer):
    without_pad = text.replace(tokenizer.pad_token, "")
    without_extra_whitespace = re.sub(r"\s+", " ", without_pad)
    return without_extra_whitespace.strip()


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


def _set_resumed_state(
    checkpoints_root_dir: Union[Path, str], 
    arg_meta_info: dict[str, Any], 
    last_ckpt_info: LastCkptInfo,
) -> dict[str, Any]:

    """Resumes things that are in the global state, 
    ie. the wandb run and the random seeds and states.
    """
    checkpoints_root_dir = Path(checkpoints_root_dir)

    json_path = checkpoints_root_dir / arg_meta_info["run_name"] / arg_meta_info["wandb_run_id"] / "last.json"

    meta_info = utils.load_json(json_path)

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
    CONSOLE.print_zero_rank("\n[red bold]Resuming Wandb run:", wandb_run_id)

    
    return meta_info


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
            arg_meta_info["wandb_run_id"], 
            0, 0)
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


def remove_padding(
    array: Union[np.ndarray, torch.Tensor], 
    mask: Union[np.ndarray, torch.Tensor]
) -> List[Union[torch.Tensor, np.ndarray]]:
    """
    Removes padding.
    """

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


def pad(seq : Sequence[Sequence[int]], pad_token_id: int, direction: str) -> torch.LongTensor:
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


@dataclasses.dataclass
class MLETrainingCollator:
    _tokenizer: transformers.PreTrainedTokenizer
    _lm_masking_mode: str

    def __call__(self, raw_examples):
        """
        - For perplexity evaluation:
            (The chainer should already be in place for input_ids and input_ids_and_scratchpad_with_value)
            - input_ids: question + chainer (e.g., " -> ") + scratchpad + value
            - attention_mask: the same as above, but with 0s everywhere there is padding
            - labels: -100 except scratchpad + value (so, for the question, the chainer and the padding.)

        """

        examples = prep_mle_train_and_valid(
            examples=raw_examples, 
            eos_token_id=self._tokenizer.eos_token_id,
            scratchpad_eos_token_id=self._tokenizer.cls_token_id, 
            pad_token_id=self._tokenizer.pad_token_id,
            lm_masking_mode=self._lm_masking_mode,
        )

        return examples


@dataclasses.dataclass
class ValitationCollator:
    _tokenizer: transformers.PreTrainedTokenizer
    _lm_masking_mode: str

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
        
        examples = prep_mle_train_and_valid(
            examples=raw_examples, 
            eos_token_id=self._tokenizer.eos_token_id, 
            pad_token_id=self._tokenizer.pad_token_id,
            scratchpad_eos_token_id=self._tokenizer.cls_token_id,
            lm_masking_mode=self._lm_masking_mode,
        )

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
) -> transformers.PreTrainedModel:
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
def _make_config_path(checkpoints_root_dir: Path, run_name: str, wandb_run_id: str, step: int, epoch: int) -> Path:
    return checkpoints_root_dir / run_name / wandb_run_id / f"last.json"


DATA_DIR = SCRIPT_DIR / "data"


class EntryPoints:
    @classmethod
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
        switch_to_maginal_after: Optional[dict[str, int]] = DEFAULT_SWITCH_TO_MARGINAL_AFTER,
        new_wandb_run_id: bool = False,

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
        scheduler_type=DEFAULT_SCHEDULER_TYPE,
        accumulate_grad_batches: int = DEFAULT_GRADIENT_ACCUM,

        #######################################################################
        # Related to resuming
        #######################################################################
        wandb_run_id: Optional[str] = DEFAULT_WANDB_ID,
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

            if ddp_info.global_rank == 0:
                assert wandb.run
                wandb.run.log_code(SCRIPT_DIR)
        else:
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
        
        tokenizer = _setup_tokenizer(
            meta_info["transformers_model_name"], 
            is_gpt2_model=meta_info["is_gpt2_model"],
        )

        base_model = _setup_base_model(
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

        CONSOLE.print(f"\n[bold]Strategy:[/] {strategy}")
        CONSOLE.print(f"\n[bold]ddp_info:[/] {vars(ddp_info)}\n")

        ###############################################################
        # Build the pt-lightning dataloader
        ###############################################################
        pl_object = _RefineLM(
            batch_sizes=batch_sizes,
            datasets=datasets,
            generation_kwargs=meta_info["generation_kwargs"],
            is_adamw=meta_info["is_adamw"],
            learning_rate=meta_info["learning_rate"],
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
        base_model = _setup_base_model(
            custom_model_config=meta_info["custom_model_config"], 
            hf_name=meta_info["transformers_model_name"], 
            is_gpt2_model=meta_info["is_gpt2_model"],
            model_mode=meta_info["model_mode"], 
            tokenizer=tokenizer,
        )
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
            meta_info=meta_info,
        )

        trainer = pl.Trainer(
            precision=PRECISION,
            accelerator=ACCELERATOR,
            deterministic=DETERMINISTIC,
            devices=ddp_info.num_devices,
            num_nodes=ddp_info.num_nodes,
            strategy=distribute_strategy,
            default_root_dir=str(checkpoints_root_dir),
            limit_predict_batches=math.ceil(qty / batch_sizes[mode]),
        )

        trainer.validate(
            pl_object,
            ckpt_path=str(last_ckpt_info.path),
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Defaulting to the `main` entrypoint.")
        EntryPoints.main()
    else:
        fire.Fire(EntryPoints)
