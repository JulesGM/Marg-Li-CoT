import bisect
import collections
import contextlib
import dataclasses
import itertools
import math
import more_itertools as mit
import os
import pathlib
import random
import sys
import typing
from typing import Any, Iterator, Optional, Tuple, Union

import accelerate
import datasets
import more_itertools
import numpy as np
import peft
# import peft_qlora
import rich
import rich.layout
import rich.markup
import rich.panel
import rich.table
import torch
from torch.nn.parameter import Parameter
import transformers
import transformers.tokenization_utils
import trl
import trl.core
import trl.models
import trl_fork
import trl_fork.core
import trl_fork.models
import wandb
from beartype import beartype

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent

sys.path.append(str(SCRIPT_DIR.parent))

import lib_base_classes
import lib_constant
import lib_data
import lib_utils
import lib_peft_utils

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

def generate(
        *, 
        accelerator,
        answer_extractor,
        batch,
        batch_size,
        dataset_name,
        generation_kwargs,
        policy_model,
        post_process_gen_fewshots_fn,
        prediction_tokenizer,
        task_name,
        use_few_shots,
        step_information,
    ):
    
    """
    
    step_information = dict(
        trainer_step = ppo_trainer.current_step,
        batch_idx    = batch_idx,
        epoch_idx    = epoch, 
    )

    """
    
    print(f"[RANK: {RANK}] lib_trl_utils.batched_unroll: ({step_information = }) >>>")
    raw_gen_outputs, scores = batched_unroll(
        accelerated_model = policy_model,
        accelerator       = accelerator,
        dataset_name      = dataset_name, 
        difficulty_levels = batch.difficulty_level,
        generation_kwargs = generation_kwargs,
        post_process_gen_fewshots_fn = post_process_gen_fewshots_fn,
        prediction_tokenizer         = prediction_tokenizer,
        query_tensors     = batch.tok_ref_query,
        task_name         = task_name, 
        use_few_shots     = use_few_shots,
    )
    print(f"{RANK} lib_trl_utils.batched_unroll <<<")

    if task_name == lib_utils.Task.MAIN:
        outputs = keep_good_one_generation(
            num_return_seq=generation_kwargs["num_return_sequences"],
            other_rewards=dict(scores=scores),
            generations=raw_gen_outputs,
            ref_answers=batch.detok_ref_answer,
            batch_size=batch_size,
            prediction_tokenizer=prediction_tokenizer,
            answer_extractor=answer_extractor,
            difficulty_levels=batch.difficulty_level,
        )
    else:
        assert False
        assert task_name == lib_utils.Task.SENTIMENT, task_name
        assert (
            generation_kwargs["num_return_sequences"] == 1
        ), generation_kwargs["num_return_sequences"]

        outputs = raw_gen_outputs
        assert len(outputs.response_tensors) == batch_size, (
            len(outputs.response_tensors),
            batch_size,
        )

    outputs = lib_base_classes.BatchedUnrollReturn(
        response_tensors=unpad(
            responses=outputs.response_tensors,
            eos_token_id=prediction_tokenizer.eos_token_id,
            pad_token_id=prediction_tokenizer.pad_token_id,
        ),
        raw_response_tensors=unpad(
            outputs.raw_response_tensors,
            eos_token_id=prediction_tokenizer.eos_token_id,
            pad_token_id=prediction_tokenizer.pad_token_id,
        ),
        any_tokenizer=prediction_tokenizer,
    )

    return outputs


def rich_escape(value):
    return rich.markup.escape(str(value))

IntSequence = typing.TypeVar(
    "IntSequence",
    list[int],
    torch.LongTensor,
)


IntSequenceContainer = typing.TypeVar(
    "IntSequenceContainer",
    torch.LongTensor,
    list[torch.LongTensor],
    list[list[int]],
)


def check_qty_of_token_id(
    list_of_sequences: list[torch.Tensor], qty: int, token_id: int
):
    for sequence in list_of_sequences:
        qty_found = (sequence == token_id).long().sum().item()
        assert qty_found == qty, (qty_found, qty)


def check_max_qty_of_token_id(
    list_of_sequences: list[torch.Tensor], max_qty: int, token_id: int
):
    for sequence in list_of_sequences:
        qty_found = (sequence == token_id).sum().item()
        assert qty_found <= max_qty, (qty_found, max_qty)


def keep_good_one_generation(
    *,
    num_return_seq: int,
    ref_answers: Union[list[list[str]], torch.Tensor],
    batch_size: int,
    prediction_tokenizer: Optional[transformers.PreTrainedTokenizerBase],  # type: ignore
    answer_extractor,
    other_rewards: Optional[torch.Tensor],
    generations: lib_base_classes.BatchedUnrollReturn,
    difficulty_levels: list[int],
) -> lib_base_classes.BatchedUnrollReturn:
    """

    Return the index of the generation with the:
        - Max reward of the generations with good answers if there is one
        - Max reward of the generations with bad  answers otherwise

    """
    assert isinstance(generations, lib_base_classes.BatchedUnrollReturn
        ), type(generations)

    array_response_text = np.array(
        generations.response_text,
        dtype=object,
    ).reshape((batch_size, num_return_seq))

    device = generations.response_tensors[0].device
    response_tensors = prediction_tokenizer.pad(
            dict(input_ids=generations.response_tensors), 
            return_tensors="pt", 
            padding=True,
    )["input_ids"].reshape(batch_size, num_return_seq, -1)

    raw_response_tensors = torch.cat(
        generations.raw_response_tensors,
        dim=0,
    ).reshape(batch_size, num_return_seq, -1)

    del generations
    assert isinstance(ref_answers[0][0], str), type(ref_answers[0][0])

    selections = []

    table = rich.table.Table(
        "Difficulty Level",
        "Ref Comparable", 
        "Generated answers", 
        "Lengths", 
        "Ratio Good",
        title=f"(RANK={RANK}) keep_good_one_generation", 
        show_lines=True,
        expand=True,
    )
    ratios_good = []

    for b_idx in range(batch_size):
        
        # Extract the reference answer
        ref_comparable = answer_extractor(ref_answers[b_idx])
        assert ref_comparable is not None, ref_answers[b_idx]

        # Extract the generated answers
        gen_answ_beams = [answer_extractor(gen) for gen in array_response_text[b_idx]]
        is_good_beams = [gen == ref_comparable for gen in gen_answ_beams]
        ratio_good_beams = np.mean(np.array(is_good_beams, dtype=np.float32))
        values_to_counts = collections.Counter(x for x in gen_answ_beams)

        # Count the length of the generated tokens
        lengths = collections.Counter(len(x) for x in response_tensors[b_idx])
        length = more_itertools.one(lengths.keys())

        inner_table = rich.table.Table(
            expand        = True,
            show_edge     = False,
            show_footer   = False,
            title_justify = "center",
        )
        inner_table.add_column(  "Gen", justify="left")
        inner_table.add_column("Count", justify="left")

        for val, count in values_to_counts.most_common():
            inner_table.add_row(val, str(count))
        
        table.add_row(
            rich_escape(difficulty_levels[b_idx]),
            rich_escape(ref_comparable),
            inner_table,
            rich_escape(length),
            f"[green]{ratio_good_beams:0.2%}[/] or " + 
            f"[green]{np.sum(is_good_beams)}[/]/[green]{len(is_good_beams)}[/]",
        )

        if any(is_good_beams):
            # Return the good with the max other reward
            good_idx = [i for i, g in enumerate(is_good_beams) if g]
            if other_rewards:
                good_rewards = other_rewards["scores"][b_idx][torch.tensor(good_idx)]
                selection = torch.argmax(good_rewards)
            else:
                assert False
                good_idx_id = torch.randint(0, len(good_idx), (1,))
                selection = good_idx[good_idx_id]
        else:
            # Return the one with the max other reward
            if other_rewards:
                selection = torch.argmax(other_rewards["scores"][b_idx])
            else:
                selection = torch.randint(0, num_return_seq, (1,))[0]
                
        selections.append(selection)
        ratios_good.append(ratio_good_beams)

    ratio_avg = np.mean(ratios_good)
    at_least_one_good = np.mean([ratio > 0 for ratio in ratios_good])
    table.caption = (
        f"Ratio Average: {ratio_avg:0.2%}, " +
        f"At least one good ratio: {at_least_one_good:0.2%}"
    )

    rich.print(table)

    selections = torch.tensor(selections)

    assert selections.shape == (batch_size,), f"{selections.shape = } {batch_size = }"

    output_generations_tensors = []
    output_raw_generation_tensors = []
    for idx in range(batch_size):
        output_generations_tensors.append(
            response_tensors[idx][selections[idx]].detach().clone().to(device)
        )
        output_raw_generation_tensors.append(
            raw_response_tensors[idx][selections[idx]].detach().clone().to(device)
        )

    return lib_base_classes.BatchedUnrollReturn(
        any_tokenizer=prediction_tokenizer,
        response_tensors=output_generations_tensors,
        raw_response_tensors=output_raw_generation_tensors,
    )

def print_table(
    *,
    call_source: str,
    extra_columns: Optional[dict[str, list]] = None,
    difficulty_levels: list[int],
    generation_kwargs,
    log_header: str,
    name: str,
    qty: int,
    queries: list[str],
    responses: list[str],
    rewards: list[float],
):
    
    if extra_columns is None:
        extra_columns = {}

    assert (
        len(rewards) == len(responses) == len(queries)
    ), f"{len(rewards) = } {len(responses) = } {len(queries) = }"

    for k, v in extra_columns.items():
        assert len(v) == len(rewards), f"{k = } {len(v) = } {len(rewards) = }"

    table = rich.table.Table(
        "Difficulty Level",
        "Training Query",
        "Training Response",
        "Reward",
        *extra_columns.keys(),
        show_lines=True,
        title=(
            f"{rich_escape(name)} - " +
            f"{rich_escape(log_header)} - " +
            f"<{call_source}>.lib_trl_utils.print_table(...): Samples:"
        ),
    )

    for difficulty_level, query, response, reward, *extra in itertools.islice(
        more_itertools.zip_equal(
            difficulty_levels,
            queries,
            responses,
            rewards,
            *extra_columns.values(),
        ),
        qty,
    ):
        # Escape brackets for rich
        rindex = query.rfind("Q:")
        if rindex == -1:
            rindex = 0
        
        table.add_row(
            rich_escape(difficulty_level),
            f"[black on white]{rich_escape(query[rindex:])}",
            f"[black on white]{rich_escape(response)}",
            f"{reward:0.3}",
            *map(rich_escape, extra),
        )

    if RANK == 0:
        kwargs_table = rich.table.Table("Key", "Value", title="Gen Kwargs")
        for k, v in generation_kwargs.items():
            kwargs_table.add_row(rich.markup.escape(str(k)), rich.markup.escape(str(v)))
        rich.print(kwargs_table)
        
        rich.print(table)


def check_all_start_with_token_id(tensors, token_id: int):
    """
    Description: Makes sure all the tensors in the list start with the same token id.

    Intent: All the decoder inputs of an encoder-decoder model should start with the same,
            fixed token id. This is a sanity check to make surethat they do.

    """
    starts = [r[0].item() for r in tensors]
    if not all(s == token_id for s in starts):
        raise ValueError(f"{token_id = }\n" f"{collections.Counter(starts) = }\n")


def check_all_end_with_token_id(tensors, token_id: int):
    """
    Description: Makes sure all the tensors in the list end with the same token id.

    Intent: All the decoder inputs of an encoder-decoder model should end with the same,
            fixed token id. This is a sanity check to make surethat they do.

    """
    ends = [r[-1].item() for r in tensors]
    if not all(s == token_id for s in ends):
        raise ValueError(f"{token_id = }\n" f"{collections.Counter(ends) = }\n")


def print_trainable_parameters(
    model: torch.nn.Module,
    do_print: bool = True,
) -> int:
    """

    Description: Prints the number of trainable parameters in the model, returns the number.

    Intent: Mostly of use with peft & LoRA, to see how many parameters are actually trainable.

    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if do_print:
        rich.print(
            f"[bold blue]({RANK}/{WORLD_SIZE}):[/] "
            f"trainable params: {rich_escape(trainable_params)} || "
            f"all params: {rich_escape(all_param)} || "
            f"trainable%: {100 * trainable_params / all_param}"
        )

    return trainable_params


def load_then_peft_ize_model(
    *,
    peft_config_dict,
    peft_do_all_lin_layers: bool,
    precision: lib_utils.ValidPrecisions,
    model_name: str,
    use_peft: bool,
    forward_tokenizer: transformers.PreTrainedTokenizer,
    prediction_tokenizer: transformers.PreTrainedTokenizer,
    just_device_map: bool,
    adapter_name: str,
):
    if use_peft:
        lora_config = peft.LoraConfig(**peft_config_dict)

    config = transformers.AutoConfig.from_pretrained(model_name)  # type: ignore
    if just_device_map:
        assert not use_peft, "not written with that in mind"
        assert precision == lib_utils.ValidPrecisions.bfloat16, precision


    ###########################################################################
    # Model Class Specific Options
    ###########################################################################
    if not config.is_encoder_decoder:
        if use_peft:
            assert lora_config.task_type == peft.TaskType.CAUSAL_LM, lora_config.task_type
        transformers_cls = transformers.AutoModelForCausalLM  # type: ignore

    else:
        assert not (
            (config.model_type == "t5")
            and (precision == lib_utils.ValidPrecisions.float16)
        ), "fp16 doesn't work with t5"

        if use_peft:
            assert lora_config.task_type == peft.TaskType.SEQ_2_SEQ_LM
        transformers_cls = transformers.AutoModelForSeq2SeqLM  # type: ignore

    ###########################################################################
    # Init the Pre-Trained Model
    # -> Precision specific
    ###########################################################################
    if precision in (
        lib_utils.ValidPrecisions._4bit,
        lib_utils.ValidPrecisions._8bit,
    ):
        
        quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=precision == lib_utils.ValidPrecisions._4bit,
                load_in_8bit=precision == lib_utils.ValidPrecisions._8bit,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        pretrained_model = transformers_cls.from_pretrained(
            model_name,
            device_map   = {"": torch.device(int(LOCAL_RANK))},
            load_in_4bit = precision == lib_utils.ValidPrecisions._4bit,
            load_in_8bit = precision == lib_utils.ValidPrecisions._8bit,
            quantization_config = quantization_config,
            low_cpu_mem_usage   = os.environ.get("SLURM_JOB_PARTITION", "") == "main",
        )
        
        # Make sure that there is only one device
        # & Make sure that it is a cuda device
        devices = set()
        for name, parameter in pretrained_model.named_parameters():
            assert parameter.device.type == "cuda", (
                f"{name = }\n" f"{parameter.device = }"
            )
            devices.add(parameter.device.index)
        assert len(devices) == 1, devices

    else:
        assert isinstance(precision.value, torch.dtype), (
            f"{type(precision.value) = }")
        
        pretrained_model = transformers_cls.from_pretrained(
            model_name,
            device_map   = "auto" if just_device_map else None,
            torch_dtype  = precision.value,
        )
        assert not just_device_map or all(
            x.device.type == "cuda" 
            for x in pretrained_model.parameters()
        )

    ###########################################################################
    # Fix tokenizers for causal models
    ###########################################################################
    if not config.is_encoder_decoder:
        config.pad_token_id = prediction_tokenizer.pad_token_id

        # Fix pretrained model to handle the new pad token
        assert len(forward_tokenizer) == len(prediction_tokenizer), (
            f"{len(forward_tokenizer) = }\n" 
            f"{len(prediction_tokenizer) = }"
        )
        assert forward_tokenizer.pad_token_id == prediction_tokenizer.pad_token_id, (
            f"{forward_tokenizer.pad_token_id = }\n"
            f"{prediction_tokenizer.pad_token_id = }"
        )
        assert forward_tokenizer.pad_token == prediction_tokenizer.pad_token, (
            f"{forward_tokenizer.pad_token = }\n"
            f"{prediction_tokenizer.pad_token = }"
        )
        pretrained_model.resize_token_embeddings(len(forward_tokenizer))

    ###########################################################################
    # Peft-ize the model
    ###########################################################################
    if use_peft:
        if (precision == lib_utils.ValidPrecisions._4bit or 
            precision == lib_utils.ValidPrecisions._8bit
        ):
            pretrained_model = peft.prepare_model_for_kbit_training(
                pretrained_model,
            )
        
        if peft_do_all_lin_layers:
            lora_config.target_modules = lib_peft_utils.find_all_linear_names(
                precision.value, 
                pretrained_model,
            )

        pretrained_model = peft.get_peft_model(
            pretrained_model,
            lora_config,
            adapter_name=adapter_name,
        )

    return pretrained_model

def load_tokenizers(model_name, config):
    forward_tokenizer = transformers.AutoTokenizer.from_pretrained(  # type: ignore
        model_name)
    prediction_tokenizer = transformers.AutoTokenizer.from_pretrained(  # type: ignore
        model_name)
    
    if not config.is_encoder_decoder:
        for tokenizer in (forward_tokenizer, prediction_tokenizer):
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        
        prediction_tokenizer.padding_side = "left"
        forward_tokenizer.padding_side = "right"

    return dict(
        forward_tokenizer=forward_tokenizer, 
        prediction_tokenizer=prediction_tokenizer,
    )


class MultiAdapterWrapper(torch.nn.Module):
    """
    The idea is to switch adapters whenever we use the model.
    """
    def __init__(
            self, 
            *, 
            trl_model_with_peft: trl.AutoModelForCausalLMWithValueHead,
            adapter_name: str, 
            peft_config: peft.PeftConfig,
            add_adapter: bool,
        ):
        super().__init__()
        
        self._adapter_name = adapter_name
        self._is_peft_model = True

        self._peft_config: peft.PeftConfig = peft_config
        self._peft_model: peft.PeftModel = self._trl_model.pretrained_model
        self._trl_model = trl_model_with_peft

        if add_adapter:
            self._peft_model.add_adapter(adapter_name, peft_config)


    def _activate(self):
        self._peft_model.set_adapter(self._adapter_name)
        return self

    @contextlib.contextmanager
    def disable_adapter(self):
        with self._peft_model.disable_adapter() as ctx:
            yield ctx

    def forward(self, *args, **kwargs):
        self._activate()
        return self._trl_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self._activate()
        return self._trl_model.generate(*args, **kwargs)
    
    @property
    def adapter_name(self):
        return self._name

    @property
    def active_adapter(self):
        return self._peft_model.active_adapter
    
    @property
    def config(self):
        return self._peft_model.config
    
    @property
    def is_peft_model(self):
        return self._is_peft_model

    @property
    def peft_config(self):
        return self._peft_config

    @property
    def pretrained_model(self):
        return self._trl_model.pretrained_model

    @property
    def v_head(self):
        return self._trl_model.v_head


def init_model(
    *,
    model_name: str,
    peft_do_all_lin_layers: bool,
    peft_config_dict: Optional[dict[str, typing.Any]],
    precision=None,
    trl_library_mode,
    use_peft: bool,
) -> tuple[
    typing.Union[
        trl.models.AutoModelForCausalLMWithValueHead,
        trl.models.AutoModelForSeq2SeqLMWithValueHead,
    ],
    transformers.PreTrainedTokenizerBase,  # type: ignore
    transformers.PreTrainedTokenizerBase,  # type: ignore
]:
    """

    Description: Initializes the model, tokenizer, and LoRA config, & does the precision stuff.

    Intent: The LoRA stuff & the precision stuff is repetitive.

    Currently:
        1. Init the pretrained model
            -> If int8, prepare for int8 training w/ peft.
        2. Peft-ize the pretrained model
        3. Initialize the trl ValueHead model from the peft-model

    .. That's what they use in the example:
    https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py#L194

    The question is, why does peft + value-model work with gpt and not t5?

    It further doesn't work with "prepare_for_int8" training, but that's not the question right now.

    """


    if precision is None:
        precision = lib_utils.ValidPrecisions.float32

    precision = lib_utils.ValidPrecisions(precision)
    config = transformers.AutoConfig.from_pretrained(  # type: ignore
        model_name,
        trust_remote_code=True,
    )

    ###########################################################################
    # Tokenizer stuff
    ###########################################################################
    tmp_tokenizers = load_tokenizers(model_name, config)
    forward_tokenizer = tmp_tokenizers["forward_tokenizer"]
    prediction_tokenizer = tmp_tokenizers["prediction_tokenizer"]
    del tmp_tokenizers

    ###########################################################################
    # HF Raw Model Stuff
    ###########################################################################


    pretrained_model = load_then_peft_ize_model(
        adapter_name="policy" if lib_utils.TrlLibraryMode.TRL_FORK else "default",
        forward_tokenizer      = forward_tokenizer,
        just_device_map        = False,
        model_name             = model_name,
        peft_do_all_lin_layers = peft_do_all_lin_layers,
        peft_config_dict       = peft_config_dict,
        precision              = precision,
        prediction_tokenizer   = prediction_tokenizer,
        use_peft               = use_peft,
    )

    ###########################################################################
    # TRL Model
    ###########################################################################
    # if not config.is_encoder_decoder:
    #     if trl_library_mode == lib_utils.TrlLibraryMode.TRL:
    #         trl_cls = trl.models.AutoModelForCausalLMWithValueHead
    #     elif trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
    #         policy
    # else:
    #     assert False
    #     assert not (
    #         (config.model_type == "t5")
    #         and (precision == lib_utils.ValidPrecisions.float16)
    #     ), "fp16 doesn't work with t5"
    #     trl_cls = trl.models.AutoModelForSeq2SeqLMWithValueHead

    if trl_library_mode == lib_utils.TrlLibraryMode.TRL:
        model = trl.AutoModelForCausalLMWithValueHead.from_pretrained(
            pretrained_model,)
        model.gradient_checkpointing_disable = (
            model.pretrained_model.gradient_checkpointing_disable)
        model.gradient_checkpointing_enable = (
            model.pretrained_model.gradient_checkpointing_enable)

    elif trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
        import ipdb; ipdb.set_trace() # TODO JULESGM: FIX
        assert False
        trl_fork.trainer.ppo_trainer.SUPPORTED_ARCHITECTURES += (MultiAdapterWrapper,)
        print(trl_fork.trainer.ppo_trainer.SUPPORTED_ARCHITECTURES)

        peft_model: peft.PeftModel = pretrained_model
        peft_config = peft.tuners.lora.LoraConfig(**peft_config_dict)

        # trl_policy_model = trl_fork.AutoModelForCausalLMWithoutValueHead.from_pretrained(
        #     peft_model)

        trl_value_model = trl_fork.AutoModelForCausalLMWithValueHead.from_pretrained(
            peft_model,
        )
        
        # JULESGM: FIX trl_policy_model
        for model in [
        #    trl_policy_model, 
            trl_value_model,
        ]:
            model.gradient_checkpointing_disable = (
                model.pretrained_model.gradient_checkpointing_disable)

            model.gradient_checkpointing_enable = (
                model.pretrained_model.gradient_checkpointing_enable)

        # policy_model = MultiAdapterWrapper(
        #     trl_model_with_peft=trl_policy_model,
        #     peft_config=peft_config,
        #     adapter_name="policy",
        #     add_adapter=False,
        # )

        # JULGM: FIX add_adapter=True, adapter_name="value"
        # value_model = MultiAdapterWrapper(
        #     trl_model_with_peft=trl_value_model,
        #     peft_config=peft_config,
        #     adapter_name="policy", # "value",
        #     add_adapter=False,
        # )
        
        value_model = trl_value_model
        policy_model = trl_value_model
        

    if precision in (
        lib_utils.ValidPrecisions.float16,
        lib_utils.ValidPrecisions.bfloat16,
    ):
        dtype = (
            torch.float16
            if precision == lib_utils.ValidPrecisions.float16
            else torch.bfloat16)
        
        if trl_library_mode == lib_utils.TrlLibraryMode.TRL:
            model.v_head.to(dtype=dtype)
        elif trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
            value_model.v_head.to(dtype=dtype)
    
    if trl_library_mode == lib_utils.TrlLibraryMode.TRL:
        output = print_trainable_parameters(model, True)
        assert output > 0
        return dict(
            model=model, 
            forward_tokenizer=forward_tokenizer, 
            prediction_tokenizer=prediction_tokenizer,
        )
    
    elif trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
        output = print_trainable_parameters(policy_model, True)
        assert output > 0
        return dict(
            policy_model=policy_model, 
            value_model=value_model, 
            forward_tokenizer=forward_tokenizer, 
            prediction_tokenizer=prediction_tokenizer,
        )
    
    raise ValueError(f"{trl_library_mode = }")



class CurriculumSchedule:
    @dataclasses.dataclass(frozen=True)
    class CurriculumEntry:
        """
        Only steps are comparable
        """

        step: int
        proportions: dict[int, float]

        def __post_init__(self):
            # Make sure the proportions sum to approx 1
            self.check()

        def check(self) -> None:
            sum_ = sum(self.proportions.values()) 
            assert math.isclose(sum_, 1), sum_
            # Check types
            assert isinstance(self.step, int), type(self.step)
            assert isinstance(self.proportions, dict), type(self.proportions)
            assert all(
                isinstance(x, int) 
                for x in self.proportions
            ), type(self.proportions)
            assert all(
                isinstance(x, float) 
                for x in self.proportions.values()
            ), type(self.proportions)

    CE = CurriculumEntry

    def __init__(self, entries: list[CE]=None, literals=None):
        # Make sure the indices are strictly increasing
        # and start at 0
        assert entries or literals, (entries, literals)
        assert not (entries and literals), (entries, literals)
        if not entries:
            assert isinstance(literals, list), literals
            assert all(isinstance(x, tuple) for x in literals), literals
            entries = [self.CE(*x) for x in literals]
            del literals

        self._entries = entries
        self.check()
    
    def __call__(self, step: int) -> CE:
        assert isinstance(step, int), type(step)

        # Index of first <= step.
        index = bisect.bisect_right(
            [x.step for x in self._entries],
            step
        ) - 1
        entry = self[index]

        assert entry.step <= step, (entry.step, step)
        not_last = index < len(self._entries) - 1

        if not_last:
            assert step < self[index + 1].step

        return entry
 
    def __len__(self) -> int:
        return len(self._entries)
    
    def __getitem__(self, index) -> CurriculumEntry:
        return self._entries[index]

    def __bool__(self) -> bool:
        return bool(self._entries)
    
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._entries})>"

    def __str__(self) -> str:
        return self.__repr__()

    def check(self) -> None:
        assert all(
            isinstance(x, self.CE) 
            for x in self._entries), self._entries
        assert all(
            x.step < y.step 
            for x, y in zip(self._entries, self._entries[1:])
        ), self._entries
        assert self._entries[0].step == 0, self._entries[0].step
        
    def check(self):
        for i, entry in enumerate(self._entries):
            assert entry.step == i, (entry.step, i)
            assert isinstance(entry.proportions, dict), type(entry.proportions)
            assert all(
                isinstance(x, int) 
                for x in entry.proportions
            ), type(entry.proportions)
            assert all(
                isinstance(x, float) 
                for x in entry.proportions.values()
            ), type(entry.proportions)


def batched_unroll(
    *,
    accelerated_model,
    accelerator: accelerate.Accelerator,
    dataset_name,
    difficulty_levels,
    generation_kwargs: dict[str, typing.Any],
    post_process_gen_fewshots_fn,
    prediction_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    query_tensors: list[torch.Tensor],
    task_name,
    use_few_shots,
) -> lib_base_classes.BatchedUnrollReturn:
    
    model: transformers.PreTrainedModel = typing.cast(  # type: ignore
        transformers.PreTrainedModel,  # type: ignore
        accelerator.unwrap_model(accelerated_model),
    )

    if not model.config.is_encoder_decoder:
        assert prediction_tokenizer.padding_side == "left", (
            prediction_tokenizer.padding_side)

    tokenized = prediction_tokenizer.pad(
        dict(
            input_ids=typing.cast(
                transformers.tokenization_utils.EncodedInput,
                query_tensors,
            )
        ),
        return_tensors="pt",
        padding=True,
    ).to(accelerator.device)

    model_outputs = model.generate(
        **tokenized,
        **generation_kwargs,
        pad_token_id = prediction_tokenizer.pad_token_id,
        output_scores=True,
        return_dict_in_generate=True, 
    )
    responses = model_outputs.sequences

    assert not model.config.is_encoder_decoder

    in_seq_len = tokenized["input_ids"].shape[1]
    responses = responses[:, in_seq_len:]

    raw_responses = responses
    if task_name == lib_utils.Task.MAIN:
        if (
            dataset_name == lib_data.DatasetChoices.COMMONSENSEQA_MC or 
            dataset_name == lib_data.DatasetChoices.ARITHMETIC
        ) and use_few_shots:
            responses = post_process_gen_fewshots_fn(
                raw_gen_outputs = responses, 
                any_tokenizer   = prediction_tokenizer,
            )
        elif use_few_shots:
            raise NotImplemented
            assert not hasattr(dataset_obj, "post_process_gen_fewshots"), (
                type(dataset_obj).mro())

    outputs = lib_base_classes.BatchedUnrollReturn(
        response_tensors=list(responses),
        raw_response_tensors=list(raw_responses),
        any_tokenizer=prediction_tokenizer,
    )

    ###########################################################################
    batch_size       = tokenized.input_ids.shape[0]
    output_length    = model_outputs.sequences.shape[-1]
    responses        = model_outputs.sequences       .reshape(batch_size, -1, output_length)
    sequences_scores = model_outputs.sequences_scores.reshape(batch_size, -1)
    ###########################################################################

    table = rich.table.Table(
        title="Batch Unroll", 
        show_lines=True,
        show_header=False,
        show_edge=True,
        highlight=True
    )

    output_text = np.array(outputs.response_text, dtype=object).reshape(
        (batch_size, -1)
    )

    for batch_idx in range(batch_size):
        
        sorted_idx = sorted(
            range(len(sequences_scores[batch_idx])),
            key     = lambda i: sequences_scores[batch_idx][i],
            reverse = True,
        )
        sorted_texts  = [output_text     [batch_idx][idx] for idx in sorted_idx]
        sorted_scores = [sequences_scores[batch_idx][idx] for idx in sorted_idx]

        for (seq_idx, val), score in more_itertools.zip_equal(
            enumerate(sorted_texts), sorted_scores
        ):
            
            score = str(score.item())
            table.add_row(
                (f"[bold green]Winner[/]\n" if seq_idx == 0 else "") +
                f"{difficulty_levels[batch_idx] = } \n"
                f"{batch_idx = } \n"
                f"{seq_idx   = } \n"
                f"{score     = }", 
                str(val).strip(),
            )

    rich.print(table)

    return outputs, sequences_scores


def unpad(responses, pad_token_id, eos_token_id):
    # Remove the padding
    final_responses = []
    for response in responses:
        init_len = len(response)
        # Remove the padding:
        response = response[response != pad_token_id]
        modified = init_len != len(response)

        # If we have modified the len, then we are guaranteed that the
        # sentence ends, and so there needs to be an eos token.
        # We might have cut it off if pad_token_id == eos_token_Id,
        # so we need to add it if it's not there.
        if modified and (
            not response.shape[0] or response[-1] != eos_token_id
        ):
            response = torch.cat(
                [response, torch.tensor([eos_token_id]).to(response.device)]
            )
        final_responses.append(response)
    return final_responses