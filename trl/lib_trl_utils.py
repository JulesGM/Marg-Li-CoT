import collections
import enum
import itertools
import os
import typing

import peft
import rich
import torch
import transformers
from tqdm import tqdm

import trl
import trl.core
import trl.models


def get_rank():
    return int(os.getenv("RANK", "0"))


def get_local_rank():
    return int(os.getenv("LOCAL_RANK", "0"))


def get_world_size():
    return int(os.getenv("WORLD_SIZE", "1"))


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


def print_table(
    *,
    name:       str, 
    log_header: bool, 
    queries:    list[str],
    response:   list[str],
    qty:        int,
    rewards:    list[float], 
    tasks:      typing.Optional[list[str]] = None,
):

    table = rich.table.Table(
        "Query", 
        "Response", 
        "Task", 
        title=f"{name} - {log_header} - Samples:",
        show_lines=True,
    )
    if tasks:
        assert len(tasks) == len(queries) == len(response), (
            f"{len(tasks) = } != {len(queries) = } != {len(response) = }")
        
        for t, q, resp, rew in itertools.islice(zip(tasks, queries, response, rewards), qty):
            table.add_row(t, q, f"[black on white]{resp}", f"{rew:0.3}")

    else:
        assert len(queries) == len(response), (
            f"{len(queries) = } != {len(response) = }")
        
        for q, resp, rew in itertools.islice(zip(queries, response, rewards), qty):
            table.add_row(f"[black on white]{q}", f"[black on white]{resp}", f"{rew:0.3}")
    
    rich.print(table)


def build_tokenizer(model_name):
    tok = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left")
    
    if "gpt" in model_name.lower():
        tok.pad_token = tok.eos_token
    
    return tok


def check_all_start_with_token_id(tensors: IntSequence, token_id: int):
    """
    Description: Makes sure all the tensors in the list start with the same token id.
    
    Intent: All the decoder inputs of an encoder-decoder model should start with the same, 
            fixed token id. This is a sanity check to make surethat they do.

    """
    starts = [r[0].item() for r in tensors]
    if not all(s == token_id for s in starts):
        raise ValueError(
            f"{token_id = }\n"
            f"{collections.Counter(starts) = }\n"
        )


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
            f"[bold blue]({get_rank()}/{get_world_size()}):[/] "
            f"trainable params: {trainable_params} || "
            f"all params: {all_param} || "
            f"trainable%: {100 * trainable_params / all_param}"
        )

    return trainable_params


def init_model(
    *, 
    model_name: str, 
    use_peft: bool,
    peft_config_dict: dict[str, typing.Any],
    precision = None, 
) -> transformers.PreTrainedModel:
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
    
    assert precision in [None, torch.bfloat16, torch.float32], precision

    lora_config = peft.LoraConfig(**peft_config_dict)
    tokenizer   = build_tokenizer(model_name)
    config      = transformers.AutoConfig.from_pretrained(model_name)

    ###########################################################################
    # Model Class Specific Options
    ###########################################################################
    if not config.is_encoder_decoder:
        assert lora_config.task_type == peft.TaskType.CAUSAL_LM
        transformers_cls      = transformers.AutoModelForCausalLM
        trl_cls               = trl.models.  AutoModelForCausalLMWithValueHead
        tokenizer.pad_token   = tokenizer.eos_token
        dmap_keys = ["transformer", "lm_head"]

    elif config.is_encoder_decoder:
        assert lora_config.task_type == peft.TaskType.SEQ_2_SEQ_LM
        transformers_cls      = transformers.AutoModelForSeq2SeqLM
        trl_cls               = trl.models.  AutoModelForSeq2SeqLMWithValueHead
    else:
        raise ValueError(model_name)

    ###########################################################################
    # Init the Pre-Trained Model
    # -> Precision specific
    ###########################################################################
    if precision == "int8":
        dmap = {k: int(get_local_rank()) for k in dmap_keys}
        pretrained_model = transformers_cls.from_pretrained(
            model_name, 
            load_in_8bit = True, 
            device_map   = dmap
        )
        # https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L35
        # Casts the layer norm to fp32 for stability purposes
        # Upcasts lm_head to fp32 for stability purposes
        # Make the output embedding layer require grads 

        pretrained_model = peft.prepare_model_for_int8_training(
            pretrained_model, 
            output_embedding_layer_name = "lm_head"
        )

    else:
        pretrained_model = transformers_cls.from_pretrained(
            model_name,
            torch_dtype=precision,
        )
    
    ###########################################################################
    # 
    ###########################################################################
    if use_peft:
        assert False
        pretrained_model = peft.get_peft_model(pretrained_model, lora_config,)

    if precision == "int8":
        assert False
        model = trl_cls.from_pretrained(pretrained_model)
    else:
        model = trl_cls.from_pretrained(pretrained_model, torch_dtype=precision)

    model.gradient_checkpointing_disable = model.pretrained_model.gradient_checkpointing_disable
    model.gradient_checkpointing_enable  = model.pretrained_model.gradient_checkpointing_enable
    output = print_trainable_parameters(model, True)
    assert output > 0

    return model, tokenizer


def batched_unroll(
    *,
    generation_kwargs:     dict[str, typing.Any],
    query_tensors:         list[torch.Tensor],
    ppo_trainer:           trl.PPOTrainer,
    tokenizer:             transformers.PreTrainedTokenizer,
) -> tuple[str, torch.Tensor]:
    
    """
    Requires 
    """

    model = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)

    # Assignment in python can create new attributes
    # so we make sure that it existed before
    assert hasattr(tokenizer, "padding_side"), tokenizer
    tokenizer.padding_side = "left"

    assert tokenizer.padding_side == "left", tokenizer.padding_side

    tokenized = tokenizer.pad(
        dict(input_ids=query_tensors),
        return_tensors = "pt", 
        padding        = True, 
    ).to(get_local_rank())
    
    responses = model.generate(
        **tokenized,
        **generation_kwargs
    )

    if not model.config.is_encoder_decoder:
        responses = responses[:, tokenized["input_ids"].shape[1]:]
    
    return [x for x in responses], tokenizer.batch_decode(responses)


