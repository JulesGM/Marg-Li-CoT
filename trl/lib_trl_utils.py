import collections
import enum
import itertools
import os
import typing
from dataclasses import dataclass

import more_itertools
import numpy as np
import peft
import rich
import rich.table
import torch
import transformers
from beartype import beartype
from tqdm import tqdm

import lib_sentiment_specific
import trl
import trl.core
import trl.models


@beartype
@dataclass
class BatchedUnrollReturn:
    response_tensors: list[torch.Tensor]
    response_text:    list[str]

    @beartype
    @dataclass
    class IndivualReturn:
        response_tensor: torch.Tensor
        response_text:   str

    def __len__(self):
        assert len(self.response_tensors) == len(self.response_text), (
            f"{len(self.response_tensors) = } "
            f"{len(self.response_text) = } "
        )
        return len(self.response_tensors)

    def __iter__(self):
        for i in range(len(self)):
            yield self.IndivualReturn(
                response_tensor = self.response_tensors[i],
                response_text   = self.response_text[i], 
            )



def get_rank() -> int:
    return int(os.getenv("RANK", "0"))


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def get_world_size() -> int:
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


def keep_good_one_generation(
        *, 
        num_return_seq: int,
        other_rewards:  typing.Optional[torch.Tensor], 
        generations:    BatchedUnrollReturn, 
        ref_answers:    typing.Union[list[list[str]], torch.Tensor], 
        extract_fn:     typing.Callable,
        batch_size:     int,
        tokenizer:      typing.Optional[transformers.PreTrainedTokenizerBase],
    ) -> BatchedUnrollReturn:

    """

    Return the index of the generation with the:
        - Max reward of the generations with good answers if there is one
        - Max reward of the generations with bad  answers otherwise
        
    """
    assert isinstance(generations, BatchedUnrollReturn), type(
        generations)
    
    array_response_text = np.array(
        generations.response_text, 
        dtype=object,
    ).reshape((batch_size, num_return_seq))
    
    array_response_tensors = torch.stack(
        typing.cast(list[torch.Tensor], generations.response_tensors), dim=0
    ).reshape((
        batch_size, 
        num_return_seq, 
        generations.response_tensors[0].shape[-1],
    ))

    del generations

    assert isinstance(ref_answers[0][0], str), type(
        ref_answers[0][0])

    selections = []
    for b_idx in range(batch_size):
        ref_comparable = extract_fn(ref_answers[b_idx])
        generated_answers = [extract_fn(gen) for gen in array_response_text[b_idx]]
        goods = [ref_comparable == gen for gen in generated_answers]
        goods_str = " ".join([f"[green]{g}[/]" if g else str(g) for g in goods])
        ratio = np.mean(np.array(goods, dtype=np.float32))

        table = rich.table.Table("Name", "Value", show_lines=True)
        # table.add_row("[bold white on red]Ref Answer",        f"{ref_answers[b_idx]}")
        table.add_row("[bold white on red]Ref Comparable",    f"{ref_comparable}")
        table.add_row("[bold white on red]Generated answers", f"{generated_answers}")
        table.add_row("[bold white on red]Comparaisons",      f"{goods_str}")
        table.add_row("[bold white on red]Ratio",             f"[green]{ratio:0.2%}[/] or [green]{np.sum(goods)}[/]/[green]{len(goods)}[/]")
        rich.print(table)

        if any(goods):
            # Return the good with the max other reward
            good_idx = [i for i, g in enumerate(goods) if g]
            if other_rewards:
                good_rewards = other_rewards[b_idx][torch.tensor(good_idx)]
                selection = torch.argmax(good_rewards)
            else:
                good_idx_id = torch.randint(0, len(good_idx), (1,))
                selection = good_idx[good_idx_id]
        else:
            # Return the one with the max other reward
            if other_rewards:
                selection = torch.argmax(other_rewards[b_idx])
            else:
                selection = torch.randint(0, num_return_seq, (1,))[0]        
        selections.append(selection)

    selections = torch.tensor(selections)

    assert selections.shape == (batch_size,), (
        f"{selections.shape = } {batch_size = }"
    )

    output_generations_text = []
    output_generations_tensors = []
    for idx in range(batch_size):
        output_generations_text.append(
            array_response_text[idx][selections[idx]])
        output_generations_tensors.append(
            array_response_tensors[idx][selections[idx]])

    return BatchedUnrollReturn(
        response_text    = output_generations_text,
        response_tensors = output_generations_tensors,
    )


def print_table(
    *,
    extra_columns: typing.Optional[dict[str, list]] = None,
    log_header:    str, 
    responses:     list[str],
    queries:       list[str],
    rewards:       list[float],
    name:          str, 
    qty:           int,
    
    # queries_ids:   list[list[int]],
    # responses_ids: list[list[int]],
    # model: trl.AutoModelForSeq2SeqLMWithValueHead = None,
):
    
    # queries_ids_qty = queries_ids[:qty]
    # responses_ids_qty = responses_ids[:qty]
    # values = model.v_head(responses_ids_qty, queries_ids_qty).tolist()

    if extra_columns is None:
        extra_columns = {}

    assert len(rewards) == len(responses) == len(queries), (
        f"{len(rewards) = } {len(responses) = } {len(queries) = }")
    
    for k, v in extra_columns.items():
        assert len(v) == len(rewards), (
            f"{k = } {len(v) = } {len(rewards) = }")

    table = rich.table.Table(
        "Query", 
        "Response", 
        "Reward",
        *extra_columns.keys(), 
        show_lines = True,
        title      = f"{name} - {log_header} - Samples:",
    )

    for query, response, reward, *extra in itertools.islice(
        more_itertools.zip_equal(
            queries, 
            responses, 
            rewards, 
            *extra_columns.values(),
        ), qty,
    ):
        table.add_row(
            f"[black on white]{query}", 
            f"[black on white]{response}", 
            f"{reward:0.3}",
            *map(str, extra),
        )

    rich.print(table)


def build_tokenizer(model_name: str) -> transformers.PreTrainedTokenizerBase:
    tok = transformers.AutoTokenizer.from_pretrained(
        model_name, 
        padding_side="left",
    )
    
    if "gpt" in model_name.lower():
        tok.pad_token = tok.eos_token
    
    return tok


def check_all_start_with_token_id(tensors, token_id: int):
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
) -> tuple[
    typing.Union[
        trl.models.AutoModelForCausalLMWithValueHead, 
        trl.models.AutoModelForSeq2SeqLMWithValueHead
    ],
    transformers.PreTrainedTokenizerBase,
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
    
    assert precision in [
        None, "int8", torch.float16, torch.bfloat16, torch.float32
    ], precision

    if precision is None:
        precision = torch.float32

    lora_config = peft.LoraConfig(**peft_config_dict)
    tokenizer   = build_tokenizer(model_name)
    config      = transformers.AutoConfig.from_pretrained(model_name)

    ###########################################################################
    # Model Class Specific Options
    ###########################################################################
    if not config.is_encoder_decoder:
        assert lora_config.task_type == peft.TaskType.CAUSAL_LM
        transformers_cls    = transformers.AutoModelForCausalLM
        trl_cls             = trl.models.  AutoModelForCausalLMWithValueHead
        tokenizer.pad_token = tokenizer.eos_token
        dmap_keys = ["transformer", "lm_head"]
        output_layer_name     = "lm_head"

    elif config.is_encoder_decoder:
        assert not ((config.model_type == "t5") and (precision == torch.float16)), (
            "fp16 doesn't work with t5")
        assert lora_config.task_type == peft.TaskType.SEQ_2_SEQ_LM
        transformers_cls      = transformers.AutoModelForSeq2SeqLM
        trl_cls               = trl.models.  AutoModelForSeq2SeqLMWithValueHead
        dmap_keys             = ["decoder", "encoder", "lm_head", "shared"]
        output_layer_name     = "lm_head"

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
            device_map   = dmap,
        )
        # https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L35
        # Casts the layer norm to fp32 for stability purposes
        # Upcasts lm_head to fp32 for stability purposes
        # Make the output embedding layer require grads 

    else:
        pretrained_model = transformers_cls.from_pretrained(
            model_name,
            torch_dtype=precision,
        )
    
    # Peft-ize the model
    if use_peft:
        if precision == "int8":
            pretrained_model = peft.prepare_model_for_int8_training(
                pretrained_model,
                output_embedding_layer_name = output_layer_name,
            )
        pretrained_model = peft.get_peft_model(
            pretrained_model, lora_config,
        )

    # Initialize the trl model
    if precision == "int8":
        model = trl_cls.from_pretrained(
            pretrained_model,
        )
    else:
        model = trl_cls.from_pretrained(
            pretrained_model, 
            torch_dtype=precision,
        )

    # Set the precision of the value head
    if precision != torch.float32 and precision != "int8":
        model = model.to(precision)

    model.gradient_checkpointing_disable = typing.cast(
        transformers.PreTrainedModel, 
        model.pretrained_model,
    ).gradient_checkpointing_disable

    model.gradient_checkpointing_enable  = typing.cast(
        transformers.PreTrainedModel, 
        model.pretrained_model,
    ).gradient_checkpointing_enable

    output = print_trainable_parameters(model, True)

    assert output > 0

    return model, tokenizer


def batched_unroll(
    *,
    generation_kwargs:     dict[str, typing.Any],
    query_tensors:         list[torch.Tensor],
    ppo_trainer:           trl.PPOTrainer,
    tokenizer:             transformers.PreTrainedTokenizerBase,
) -> BatchedUnrollReturn:
    
    """
    Requires 
    """
    model: transformers.PreTrainedModel = typing.cast(
        transformers.PreTrainedModel,
        ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
    )
    # Assignment in python can create new attributes
    # so we make sure that it existed before
    assert hasattr(tokenizer, "padding_side"), tokenizer
    tokenizer.padding_side = "left"
    assert tokenizer.padding_side == "left", tokenizer.padding_side

    tokenized = tokenizer.pad(
        dict(input_ids=typing.cast(
            transformers.tokenization_utils.EncodedInput, 
            query_tensors,
        )),
        return_tensors = "pt", 
        padding        = True, 
    ).to(torch.device(get_local_rank()))

    responses = model.generate(
        **tokenized,
        **generation_kwargs,
    )

    if not model.config.is_encoder_decoder:
        seq_len = typing.cast(
            torch.Tensor, 
            tokenized["input_ids"],
        ).shape[1]
        responses = responses[:, seq_len:]
    
    return BatchedUnrollReturn(
        response_tensors = list(responses),
        response_text    = tokenizer.batch_decode(responses),
    )


