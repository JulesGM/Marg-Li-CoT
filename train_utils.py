import dataclasses
import math
import more_itertools
import re
import rich.table as table
from typing import *

from beartype import beartype
import numpy as np
import pytorch_lightning as pl
import torch
import transformers

import console
import constants

import general_utils as utils


CONSOLE = console.Console(force_terminal=True, force_interactive=True, width=200)

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

def build_match_stat_matrixes(scratchpad_matches, value_matches):

    p_good_sp_baad_va = np.mean(np.logical_and(               scratchpad_matches ,  np.logical_not(value_matches)))  # p(good_sp, baad_v | x)
    p_baad_sp_good_va = np.mean(np.logical_and(np.logical_not(scratchpad_matches),                 value_matches) )  # p(baad_sp, good_v | x)
    p_good_sp_good_va = np.mean(np.logical_and(               scratchpad_matches ,                 value_matches) )  # p(good_sp, good_v | x)
    p_baad_sp_baad_va = np.mean(np.logical_and(np.logical_not(scratchpad_matches),  np.logical_not(value_matches)))  # p(baad_sp, baad_v | x)

    p_good_va = p_good_sp_good_va + p_baad_sp_good_va
    p_baad_va = p_good_sp_baad_va + p_baad_sp_baad_va
    p_good_sp = p_good_sp_good_va + p_good_sp_baad_va
    p_baad_sp = p_baad_sp_good_va + p_baad_sp_baad_va

    p_good_sp_knowing_good_va = p_good_sp_good_va / p_good_va  # p(good_sp | good_va, x)
    p_good_sp_knowing_baad_va = p_good_sp_baad_va / p_baad_va  # p(good_sp | baad_va, x)
    p_baad_sp_knowing_good_va = p_baad_sp_good_va / p_good_va  # p(baad_sp | good_va, x)
    p_baad_sp_knowing_baad_va = p_baad_sp_baad_va / p_baad_va  # p(baad_sp | baad_va, x)


    p_baad_va_knowing_good_sp = p_good_sp_baad_va / p_good_sp  # p(baad_va | good_sp, x)
    p_baad_va_knowing_baad_sp = p_baad_sp_baad_va / p_baad_sp  # p(baad_va | baad_sp, x)

    # Most important stat
    p_good_va_knowing_good_sp = p_good_sp_good_va / p_good_sp  # p(good_va | good_sp, x)
    # Second most important stat
    p_good_va_knowing_baad_sp = p_baad_sp_good_va / p_baad_sp  # p(good_va | baad_sp, x)

    return dict(
        # p_good_sp_baad_va=p_good_sp_baad_va,
        # p_good_sp_good_va=p_good_sp_good_va,
        # p_baad_sp_good_va=p_baad_sp_good_va,
        # p_baad_sp_baad_va=p_baad_sp_baad_va,

        # p_good_sp_knowing_good_va=p_good_sp_knowing_good_va,
        # p_good_sp_knowing_baad_va=p_good_sp_knowing_baad_va,
        # p_baad_sp_knowing_good_va=p_baad_sp_knowing_good_va,
        # p_baad_sp_knowing_baad_va=p_baad_sp_knowing_baad_va,

        p_good_va_knowing_good_sp=p_good_va_knowing_good_sp,
        p_good_va_knowing_baad_sp=p_good_va_knowing_baad_sp,
        # p_baad_va_knowing_good_sp=p_baad_va_knowing_good_sp,
        # p_baad_va_knowing_baad_sp=p_baad_va_knowing_baad_sp,

        learnability_ratio=p_good_va_knowing_good_sp / p_good_va_knowing_baad_sp,
    )


def fix_model_params_in_place(model):
    for param in model.parameters():
        param.requires_grad = False

def shared_validation_step(lightning_module, batch, batch_idx, chainer):
    utils.rich_print_zero_rank(f"\n\n[red bold]VALIDATION STEP!")
    utils.check_equal("cuda", lightning_module._model.device.type)

    assert "labels" in batch, (
        "Labels must be in batch. We must mask the input section with -100"
    )
    mode: Final[str] = constants.PipelineModes.VALIDATION


    if not hasattr(lightning_module, "_fixed_model") or lightning_module._fixed_model is None:
        gen_outputs = lightning_module._generate(
            model             = lightning_module._model,
            batch             = batch, 
            generation_kwargs = lightning_module._generation_kwargs[mode], 
        )
        generated_tokens = gen_outputs[:, batch["generation_input_ids"].shape[1]:]

        ###################################################################
        # Compute Scratchpad Accuracy
        ###################################################################
        pos_clss = (generated_tokens == lightning_module._tokenizer.cls_token_id).long() * torch.arange(generated_tokens.shape[1]).unsqueeze(0).to(generated_tokens.device)
        last_cls_pos = (pos_clss).max(dim=1).values.unsqueeze(-1)
        del pos_clss
        is_scratchpad = torch.arange(generated_tokens.shape[1]).repeat(
            (generated_tokens.shape[0], 1)).to(generated_tokens.device) < last_cls_pos
        gen_scratchpads = remove_padding(generated_tokens, is_scratchpad)
        scratchpad_texts = get_scratchpad_texts(gen_scratchpads, batch["scratchpad"], lightning_module._tokenizer)
        scratchpad_matches = np.fromiter((gen == ref for gen, ref in scratchpad_texts), dtype=bool)
        scratchpads_acc = np.mean(scratchpad_matches)


        ###################################################################
        # Compute Accuracy p(y | x, z) only
        ###################################################################
        gen_values     = remove_padding(generated_tokens, is_scratchpad.logical_not())
        values_texts   = get_values_texts(gen_values, batch["value"], tokenizer=lightning_module._tokenizer)
        values_matches = np.fromiter((gen == ref for gen, ref in values_texts), dtype=bool)
        values_acc     = np.mean(values_matches)

        gen_outputs = gen_outputs[:, batch["generation_input_ids"].shape[1]:]
    else:
        utils.check_equal("cuda", lightning_module._fixed_model.device.type)
        ###################################################################
        # Compute the scratchpad with the learnable model
        ###################################################################
        config_scratchpad                 = lightning_module._generation_kwargs[mode].copy()
        config_scratchpad["eos_token_id"] = lightning_module._tokenizer.cls_token_id
        gen_outputs = lightning_module._generate(
            model             = lightning_module._model,
            batch             = batch, 
            generation_kwargs = config_scratchpad, 
        )

        ###################################################################
        # Compute The accuracy of Scratchpads
        ###################################################################
        gen_scratchpads = gen_outputs[:, batch["generation_input_ids"].shape[1]:]
        scratchpad_texts = get_scratchpad_texts(gen_scratchpads, batch["scratchpad"], tokenizer=lightning_module._tokenizer)
        scratchpad_matches = np.fromiter((gen == ref for gen, ref in scratchpad_texts), dtype=bool)
        scratchpads_acc = np.mean(scratchpad_matches)

        ###################################################################
        # Compute the answer after the scratchpad
        ###################################################################
        config_answer  = lightning_module._generation_kwargs[mode].copy()
        unpadded       = remove_padding(gen_outputs, gen_outputs != lightning_module._tokenizer.pad_token_id)
        padded         = pad           (unpadded, pad_token_id=lightning_module._tokenizer.pad_token_id, direction="left")
        attention_mask = generate_mask (unpadded, "left")

        new_batch = dict(
            generation_input_ids      = padded        .to(lightning_module._fixed_model.device),
            generation_attention_mask = attention_mask.to(lightning_module._fixed_model.device),
        )
        gen_values = lightning_module._generate(
            model             = lightning_module._fixed_answer_model,
            generation_kwargs = config_answer, 
            batch             = new_batch, 
        )[:, new_batch["generation_input_ids"].shape[1]:]

        ###################################################################
        # Compute Accuracy p(y | x, z) only
        ###################################################################
        values_texts   = get_values_texts(gen_values, batch["value"], tokenizer=lightning_module._tokenizer)
        values_matches = np.fromiter((gen == ref for gen, ref in values_texts), dtype=bool)
        values_acc     = np.mean(values_matches)

        gen_outputs = torch.cat([gen_scratchpads, gen_values], dim=1)

    stats = build_match_stat_matrixes(
        scratchpad_matches=scratchpad_matches, 
        value_matches=values_matches,
        
    )

    if utils.is_rank_zero():
        stats_table = table.Table("Key", "Value")
        for k, v in stats.items():
            stats_table.add_row(f"[bold]{k}", f"{v:.0%}")
            lightning_module.log(f"val/{k}", v, batch_size=lightning_module._batch_size[mode], **lightning_module._logging_conf)
        utils.rich_print_zero_rank(stats_table)

    dps_generation = decode_per_sample(lightning_module._tokenizer, gen_outputs)
    dps_labels     = decode_per_sample(lightning_module._tokenizer, batch["input_ids"])

    for_comparison = [(
        _clean_for_accuracy_computation(gen, lightning_module._tokenizer).strip(), 
        _clean_for_accuracy_computation(l,   lightning_module._tokenizer).split(chainer, 1)[1].strip(),
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
            table_entry.append([lightning_module.current_epoch, index, "generation", gen])
            table_entry.append([lightning_module.current_epoch, index, "label",      lab])

        if lightning_module._wandb_logger:
            lightning_module._wandb_logger.log_text(
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
    final_answer_acc = np.mean([get_final_number(x) == get_final_number(y) for x, y in for_comparison])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ppl_outputs = lightning_module._model(**{k: batch[k]for k in ["input_ids", "attention_mask", "labels"]})

    lightning_module.log("val/em"  ,          em_accuracy,             batch_size=lightning_module._batch_size[mode], **lightning_module._logging_conf)
    lightning_module.log("val/answ",          final_answer_acc,        batch_size=lightning_module._batch_size[mode], **lightning_module._logging_conf)
    lightning_module.log("val/loss",          ppl_outputs.loss.item(), batch_size=lightning_module._batch_size[mode], **lightning_module._logging_conf)
    lightning_module.log("val/values_acc",    values_acc,              batch_size=lightning_module._batch_size[mode], **lightning_module._logging_conf)
    lightning_module.log("val/scratchpad_em", scratchpads_acc,         batch_size=lightning_module._batch_size[mode], **lightning_module._logging_conf)

    utils.rich_print_zero_rank(
        f"val_scratchpad_em: {scratchpads_acc:0.2%}, "
        f"values_acc: {values_acc}, "
        f"val_em: {em_accuracy:.2%}, "
        f"val_answ: {final_answer_acc:.2%}"
    )

    assert not torch.any(torch.isnan(ppl_outputs.loss)), "Loss is NaN"        
    return ppl_outputs


def _clean_for_accuracy_computation(text, tokenizer):
    without_pad = text.replace(tokenizer.pad_token, "")
    without_extra_whitespace = re.sub(r"\s+", " ", without_pad)
    return without_extra_whitespace.strip()


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



def clone_hf_model(model):
    new_model = model.__class__(model.config)
    new_model.load_state_dict(model.state_dict())
    return new_model


def compute_steps_per_epoch(trainer: pl.Trainer):
    if trainer.train_dataloader is None:
        trainer.reset_train_dataloader()

    total_batches = trainer.num_training_batches
    accumulate_grad_batches = trainer.accumulation_scheduler.get_accumulate_grad_batches(
        trainer.current_epoch)
    effective_batch_size = accumulate_grad_batches
    return math.ceil(total_batches / effective_batch_size)


def get_final_number(s: str) -> str:
    maybe_minus = r"(?:\-\s?)?"
    maybe_decimal = r"(?:\.\d+)?"
    core = r"\d+"
    finds = re.findall(maybe_minus + core + maybe_decimal, s)
    if finds: 
        return finds[-1]
    return None


def print_predictions(*, inputs, masks, generated_decoded, labels, all_generated, all_labels):
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


def get_scratchpad_texts(gen_scratchpads, ref_scratchpads, tokenizer):
    return [
        (
            _clean_for_accuracy_computation(tokenizer.decode(gen_scratch), tokenizer).removesuffix("<|cls|>").strip(),
            _clean_for_accuracy_computation(tokenizer.decode(ref_scratch), tokenizer)
        )
        for  gen_scratch, ref_scratch in more_itertools.zip_equal(gen_scratchpads, ref_scratchpads)
    ]

def get_values_texts(gen_values, ref_values, tokenizer):
    return [
        (
            _clean_for_accuracy_computation(tokenizer.decode(gen_value), tokenizer
                ).removesuffix("<|endoftext|>").strip().removeprefix("<|cls|>").strip(),
            _clean_for_accuracy_computation(tokenizer.decode(ref_value), tokenizer)
        )
        for  gen_value, ref_value in more_itertools.zip_equal(gen_values, ref_values)
    ]


def decode_per_token(tokenizer, batch):
    dps = []
    for entry in batch:
        per_entry = []
        for token in entry:
            per_entry.append(
                tokenizer.decode([token if token >= 0 else tokenizer.pad_token_id], 
                skip_special_tokens=False))
        dps.append(per_entry)
    return dps


def decode_per_sample(tokenizer, batch):
    dps = []
    for entry in batch:
        dps.append(tokenizer.decode(
            [x if x >= 0 else tokenizer.pad_token_id for x in entry], skip_special_tokens=False))
    return dps


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


def shared_predict_step(pytorch_lightning_module):
    batch = cast(Dict[str, torch.LongTensor], batch)
    mode = constants.PipelineModes.VALIDATION
    generated_decoded, label = pytorch_lightning_module._generate(
        pytorch_lightning_module._model, batch, pytorch_lightning_module._generation_kwargs[mode])
    print_predictions(generated_decoded, label)
