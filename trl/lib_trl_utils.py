import collections
import enum
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
    precision, 
    model_name: str, 
    lora_config_dict: dict[str, typing.Any],
    model_type,
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
    
    lora_config = peft.LoraConfig(**lora_config_dict)
    tokenizer = build_tokenizer(model_name)

    ###########################################################################
    # Model Class Specific Options
    ###########################################################################
    if model_type == peft.TaskType.CAUSAL_LM:
        lora_config.task_type = peft.TaskType.CAUSAL_LM
        transformers_cls      = transformers.AutoModelForCausalLM
        trl_cls               = trl.models.  AutoModelForCausalLMWithValueHead
        transformers.GPTNeoForCausalLM

        tokenizer.pad_token   = tokenizer.eos_token
        dmap_keys = ["transformer", "lm_head"]

    elif model_type == peft.TaskType.SEQ_2_SEQ_LM:
        lora_config.task_type = peft.TaskType.SEQ_2_SEQ_LM
        transformers_cls      = transformers.AutoModelForSeq2SeqLM
        trl_cls               = trl.models.  AutoModelForSeq2SeqLMWithValueHead
        transformers.T5ForConditionalGeneration

        assert "t5" in model_name.lower(), model_name
        dmap_keys = ["encoder", "decoder", "lm_head", "shared"]

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
            output_embedding_layer_name="lm_head"
        )

    else:
        assert precision in [torch.float16, torch.bfloat16], precision 
        pretrained_model = transformers_cls.from_pretrained(
            model_name,
            torch_dtype=precision,
        )

    ###########################################################################
    # Model Instance Specific Fixes
    ###########################################################################
    if "gpt-neox" in model_name.lower():
        assert False
        # workaround to use 8bit training on this model
        # hacky workaround due to issues with "EleutherAI/gpt-neox-20b"
        lora_config.target_modules = ["query_key_value", "xxx"]  

        for name, param in pretrained_model.named_parameters():
            # freeze base model's layers
            param.requires_grad = False

            if getattr(pretrained_model, "is_loaded_in_8bit", False):
                # cast layer norm in fp32 for stability for 8bit models
                if param.ndim == 1 and "layer_norm" in name:
                    param.data = param.data.to(torch.float16)
    
    ###########################################################################
    # 
    ###########################################################################
    peft_model = peft.get_peft_model(pretrained_model, lora_config,)

    if precision == "int8":
        model = trl_cls.from_pretrained(peft_model)
    else:
        peft_model.to(precision)
        model = trl_cls.from_pretrained(peft_model, torch_dtype=precision)
        model.to(precision)

    model.gradient_checkpointing_disable = model.pretrained_model.gradient_checkpointing_disable
    model.gradient_checkpointing_enable  = model.pretrained_model.gradient_checkpointing_enable
    print_trainable_parameters(model)

    assert print_trainable_parameters(model, False) > 0

    return model, tokenizer


def unroll(
    *,
    output_length_sampler: typing.Optional[trl.core.LengthSampler],
    generation_kwargs:     dict[str, typing.Any],
    ppo_trainer:           trl.PPOTrainer,
    batch:                 dict[str, IntSequence],
    model:                 transformers.PreTrainedModel,
) -> IntSequenceContainer:
    """
    Requires 
    """

    # Unroll the policy
    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True
    model.eval()

    response_tensors = []  
    for query in tqdm(
        batch["input_ids"], 
        disable = get_rank() != 0,
        desc    = "Unrolling policy", 
    ):
        if output_length_sampler:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len

        with torch.no_grad():
            response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze())
    
    model.train()
    model.gradient_checkpointing_enable()
    model.pretrained_model.config.use_cache = False
    
    exit()
    return response_tensors


def batched_unroll(
    *,
    output_length_sampler: typing.Optional[trl.core.LengthSampler],
    generation_batch_size: int,
    generation_kwargs:     dict[str, typing.Any],
    ppo_trainer:           trl.PPOTrainer,
    tokenizer:             transformers.PreTrainedTokenizer,
    batch:                 dict[str, IntSequence],
    model:                 transformers.PreTrainedModel,
) -> IntSequenceContainer:
    """
    Requires 
    """

    # Unroll the policy
    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True
    model.eval()
    response_tensors = []


    if output_length_sampler:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len

    model = ppo_trainer.accelerator.unwrap_model(model)

    for i in tqdm(
        range(0, len(batch["input_ids"]), generation_batch_size), 
        disable=get_rank() != 0,
        desc="Unrolling",
    ):
        responses = model.generate(
            **tokenizer(
                batch["query"][i:i + generation_batch_size],
                return_tensors="pt", 
                padding=True, 
                truncation=True,
            ).to(get_local_rank()),
            **generation_kwargs
        )
    
        for response in responses:
            response_tensors.append(
                response[response != tokenizer.pad_token_id]
            )

    model.train()
    model.gradient_checkpointing_enable()
    model.pretrained_model.config.use_cache = False
    
    return response_tensors


