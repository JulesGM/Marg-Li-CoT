import collections
import itertools
import os
import typing
from dataclasses import dataclass
from typing import Any, Optional, Union

import more_itertools
import numpy as np
import peft
import peft_qlora
import rich
import rich.markup
import rich.table
import torch
import transformers
import transformers.tokenization_utils
import trl
import trl.core
import trl.models
import wandb
from beartype import beartype

import lib_utils

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))


def rich_escape(value):
    return rich.markup.escape(str(value))


@beartype
class BatchedUnrollReturn:
    def __init__(self, response_tensors, tokenizer):
        self._response_tensors = response_tensors
        self._tokenizer: transformers.PreTrainedTokenizerBase = tokenizer  # type: ignore
        self._response_text = self.tokenizer.batch_decode(self.response_tensors)

    @property
    def response_tensors(self):
        return self._response_tensors

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def response_text(self):
        return self._response_text

    @response_text.setter
    def response_text(self, value):
        raise RuntimeError("Cannot set response_text")

    @response_tensors.setter
    def response_tensors(self, value):
        raise RuntimeError("Cannot set response_tensors")

    @tokenizer.setter
    def tokenizer(self, value):
        raise RuntimeError("Cannot set tokenizer")

    @beartype
    @dataclass
    class IndivualReturn:
        response_tensor: torch.Tensor
        response_text: str

    def __len__(self):
        assert len(self.response_tensors) == len(self.response_text), (
            f"{len(self.response_tensors) = } " f"{len(self.response_text)    = } "
        )
        return len(self.response_tensors)

    def __iter__(self):
        response_text = self.response_text

        for i in range(len(self)):
            yield self.IndivualReturn(
                response_tensor=self.response_tensors[i],
                response_text=response_text[i],
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
        qty_found = (sequence == token_id).long().sum().item()
        assert qty_found <= max_qty, (qty_found, max_qty)


def keep_good_one_generation(
    *,
    num_return_seq: int,
    other_rewards: Optional[torch.Tensor],
    generations: BatchedUnrollReturn,
    ref_answers: Union[list[list[str]], torch.Tensor],
    extract_fn: typing.Callable,
    batch_size: int,
    tokenizer: Optional[transformers.PreTrainedTokenizerBase],  # type: ignore
) -> BatchedUnrollReturn:
    """

    Return the index of the generation with the:
        - Max reward of the generations with good answers if there is one
        - Max reward of the generations with bad  answers otherwise

    """
    assert isinstance(generations, BatchedUnrollReturn), type(generations)

    array_response_text = np.array(
        generations.response_text,
        dtype=object,
    ).reshape((batch_size, num_return_seq))

    device = generations.response_tensors[0].device
    response_tensors = torch.cat(
        generations.response_tensors,
        dim=0,
    ).reshape(batch_size, num_return_seq, -1)

    del generations

    assert isinstance(ref_answers[0][0], str), type(ref_answers[0][0])

    selections = []
    for b_idx in range(batch_size):
        ref_comparable = extract_fn(ref_answers[b_idx])
        generated_answers = [extract_fn(gen) for gen in array_response_text[b_idx]]
        goods = [ref_comparable == gen for gen in generated_answers]
        # goods_str = " ".join([f"[green]{g}[/]" if g else str(g) for g in goods])
        ratio = np.mean(np.array(goods, dtype=np.float32))

        table = rich.table.Table("Name", "Value", show_lines=True)
        # table.add_row("[bold white on red]Ref Answer",        f"{ref_answers[b_idx]}")
        table.add_row("[bold white on red]Ref Comparable", rich_escape(ref_comparable))
        table.add_row(
            "[bold white on red]Generated answers",
            rich_escape(collections.Counter(x for x in generated_answers)),
        )
        table.add_row(
            "[bold white on red]Lengths",
            rich_escape(collections.Counter(len(x) for x in response_tensors[b_idx])),
        )
        table.add_row(
            "[bold white on red]</s> present",
            rich_escape(
                collections.Counter("</s>" in x for x in array_response_text[b_idx])
            ),
        )
        table.add_row(
            "[bold white on red]Ratio",
            f"[green]{ratio:0.2%}[/] or [green]{np.sum(goods)}[/]/[green]{len(goods)}[/]",
        )
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

    assert selections.shape == (batch_size,), f"{selections.shape = } {batch_size = }"

    output_generations_text = []
    output_generations_tensors = []
    for idx in range(batch_size):
        output_generations_text.append(array_response_text[idx][selections[idx]])
        output_generations_tensors.append(
            torch.tensor(response_tensors[idx][selections[idx]]).to(device)
        )

    return BatchedUnrollReturn(
        tokenizer=tokenizer,
        response_tensors=output_generations_tensors,
    )


def print_table(
    *,
    extra_columns: Optional[dict[str, list]] = None,
    log_header: str,
    responses: list[str],
    queries: list[str],
    rewards: list[float],
    name: str,
    qty: int,
):
    if extra_columns is None:
        extra_columns = {}

    assert (
        len(rewards) == len(responses) == len(queries)
    ), f"{len(rewards) = } {len(responses) = } {len(queries) = }"

    for k, v in extra_columns.items():
        assert len(v) == len(rewards), f"{k = } {len(v) = } {len(rewards) = }"

    table = rich.table.Table(
        "Query",
        "Response",
        "Reward",
        *extra_columns.keys(),
        show_lines=True,
        title=f"{rich_escape(name)} - {rich_escape(log_header)} - Samples:",
    )

    for query, response, reward, *extra in itertools.islice(
        more_itertools.zip_equal(
            queries,
            responses,
            rewards,
            *extra_columns.values(),
        ),
        qty,
    ):
        # Escape brackets for rich
        table.add_row(
            f"[black on white]{rich_escape(query)}",
            f"[black on white]{rich_escape(response)}",
            f"{reward:0.3}",
            *map(rich_escape, extra),
        )

    rich.print(table)


def _build_tokenizer(
    model_config, model_name: str
) -> transformers.PreTrainedTokenizerBase:  # type: ignore
    optional_kwargs = {}
    if not model_config.is_encoder_decoder:
        optional_kwargs["padding_side"] = "left"

    tok = transformers.AutoTokenizer.from_pretrained(  # type: ignore
        model_name,
        **optional_kwargs,
    )

    if tok.pad_token is None:
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
            f"[bold blue]({get_rank()}/{get_world_size()}):[/] "
            f"trainable params: {rich_escape(trainable_params)} || "
            f"all params: {rich_escape(all_param)} || "
            f"trainable%: {100 * trainable_params / all_param}"
        )

    return trainable_params


def _load_then_peft_ize_model(
    *,
    precision,
    peft_config_dict,
    model_name,
    use_peft,
):
    lora_config = peft.LoraConfig(**peft_config_dict)
    config = transformers.AutoConfig.from_pretrained(model_name)  # type: ignore

    ###########################################################################
    # Model Class Specific Options
    ###########################################################################
    if not config.is_encoder_decoder:
        assert lora_config.task_type == peft.TaskType.CAUSAL_LM
        transformers_cls = transformers.AutoModelForCausalLM  # type: ignore

    else:
        assert not (
            (config.model_type == "t5")
            and (precision == lib_utils.ValidPrecisions.float16)
        ), "fp16 doesn't work with t5"

        assert lora_config.task_type == peft.TaskType.SEQ_2_SEQ_LM
        transformers_cls = transformers.AutoModelForSeq2SeqLM  # type: ignore

    ###########################################################################
    # Init the Pre-Trained Model
    # -> Precision specific
    ###########################################################################
    if precision in (
        lib_utils.ValidPrecisions.int4,
        lib_utils.ValidPrecisions.int8,
    ):
        pretrained_model = transformers_cls.from_pretrained(
            model_name,
            device_map={"": torch.device(int(get_local_rank()))},
            load_in_4bit=precision == lib_utils.ValidPrecisions.int4,
            load_in_8bit=precision == lib_utils.ValidPrecisions.int8,
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
        assert isinstance(precision.value, torch.dtype), f"{type(precision.value) = }"
        pretrained_model = transformers_cls.from_pretrained(
            model_name,
            torch_dtype=precision.value,
        )

    # Peft-ize the model
    if use_peft:
        if precision == lib_utils.ValidPrecisions.int8:
            pretrained_model = peft.prepare_model_for_int8_training(
                pretrained_model,
            )
        pretrained_model = peft.get_peft_model(
            pretrained_model,
            lora_config,
        )

    return pretrained_model


def init_model(
    *,
    model_name: str,
    use_peft: bool,
    peft_config_dict: Optional[dict[str, typing.Any]],
    peft_qlora_mode: bool,
    precision=None,
) -> tuple[
    typing.Union[
        trl.models.AutoModelForCausalLMWithValueHead,
        trl.models.AutoModelForSeq2SeqLMWithValueHead,
    ],
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

    if peft_qlora_mode:
        assert precision in (
            lib_utils.ValidPrecisions.bfloat16,
            lib_utils.ValidPrecisions.float16,
            lib_utils.ValidPrecisions.float32,
        ), precision

        pretrained_model = peft_qlora.from_pretrained(
            model_name,
            bf16=precision == lib_utils.ValidPrecisions.bfloat16,
            fp16=precision == lib_utils.ValidPrecisions.float16,
            trust_remote_code=True,
        )
    else:
        assert False, "This is not supported anymore."
        pretrained_model = _load_then_peft_ize_model(
            peft_config_dict=peft_config_dict,
            model_name=model_name,
            precision=precision,
            use_peft=use_peft,
        )

    config = transformers.AutoConfig.from_pretrained(  # type: ignore
        model_name,
        trust_remote_code=True,
    )

    tokenizer = _build_tokenizer(
        model_config=config,
        model_name=model_name,
    )
    if not config.is_encoder_decoder:
        trl_cls = trl.models.AutoModelForCausalLMWithValueHead
        tokenizer.pad_token = tokenizer.eos_token

    else:
        assert not (
            (config.model_type == "t5")
            and (precision == lib_utils.ValidPrecisions.float16)
        ), "fp16 doesn't work with t5"
        trl_cls = trl.models.AutoModelForSeq2SeqLMWithValueHead

    # Initialize the trl model
    if precision in (
        lib_utils.ValidPrecisions.float16,
        lib_utils.ValidPrecisions.bfloat16,
    ):
        dtype = (
            torch.float16
            if precision == lib_utils.ValidPrecisions.float16
            else torch.bfloat16
        )
        model = trl_cls.from_pretrained(
            pretrained_model,
        )
        model.v_head.to(dtype=dtype)
    else:
        model = trl_cls.from_pretrained(
            pretrained_model,
        )

    model.gradient_checkpointing_disable = typing.cast(
        transformers.PreTrainedModel,  # type: ignore
        model.pretrained_model,
    ).gradient_checkpointing_disable

    model.gradient_checkpointing_enable = typing.cast(
        transformers.PreTrainedModel,  # type: ignore
        model.pretrained_model,
    ).gradient_checkpointing_enable

    output = print_trainable_parameters(model, True)

    assert output > 0

    return model, tokenizer


def batched_unroll(
    *,
    generation_kwargs: dict[str, typing.Any],
    query_tensors: list[torch.Tensor],
    ppo_trainer: trl.PPOTrainer,
    tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
) -> BatchedUnrollReturn:
    model: transformers.PreTrainedModel = typing.cast(  # type: ignore
        transformers.PreTrainedModel,  # type: ignore
        ppo_trainer.accelerator.unwrap_model(ppo_trainer.model),
    )

    if not model.config.is_encoder_decoder:
        assert tokenizer.padding_side == "left", tokenizer.padding_side

    tokenized = tokenizer.pad(
        dict(
            input_ids=typing.cast(
                transformers.tokenization_utils.EncodedInput,
                query_tensors,
            )
        ),
        return_tensors="pt",
        padding=True,
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
        response_tensors=list(responses),
        tokenizer=tokenizer,
    )


def unpad(responses, tokenizer):
    # Remove the padding
    final_responses = []
    for response in responses:
        init_len = len(response)
        # Remove the padding:
        response = response[response != tokenizer.pad_token_id]
        modified = init_len != len(response)

        # If we have modified the len, then we are guaranteed that the
        # sentence ends, and so there needs to be an eos token.
        # We might have cut it off if pad_token_id == eos_token_Id,
        # so we need to add it if it's not there.
        if modified and (
            not response.shape[0] or response[-1] != tokenizer.eos_token_id
        ):
            response = torch.cat(
                [response, torch.tensor([tokenizer.eos_token_id]).to(response.device)]
            )
        final_responses.append(response)
    return final_responses


def log_reward(
    *,
    ppo_trainer,
    reward_output,
    metric_output,
    epoch,
    batch_idx,
):
    all_rewards = ppo_trainer.accelerator.gather_for_metrics(
        torch.tensor(reward_output.values).to(ppo_trainer.accelerator.device)
    )
    all_metrics = ppo_trainer.accelerator.gather_for_metrics(
        torch.tensor(metric_output.values).to(ppo_trainer.accelerator.device)
    )

    rich.print(
        f"[bold blue]"
        + f"({RANK}/{WORLD_SIZE}) "
        + f"({epoch = } {batch_idx = }) "
        + f"[/][white bold]"
        + f"Average rewards: "
        + f"{all_rewards.mean().item():0.4} "
        + f"+- {all_rewards.std().item():0.1}"
    )

    if RANK == 0:
        wandb.log({"avg_all_rewards": all_rewards.mean().item()})
        wandb.log({"avg_all_metrics": all_metrics.mean().item()})