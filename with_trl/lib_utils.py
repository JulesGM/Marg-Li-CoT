import contextlib
import enum
import os

import numpy as np
import rich
import rich.table
import torch
import transformers
from tqdm import tqdm

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

class MovingAverage:
    def __init__(self, window_size: int):
        self._window_size = window_size
        self._window = np.zeros(window_size)
        self._pointer = 0
        self._size = 0

    @property
    def window_size(self):
        return self._window_size

    @property
    def size(self):
        return self._size

    def update(self, value: float):
        self._window[self._pointer] = value
        self._pointer = (self._pointer + 1) % self._window_size
        self._size = min(self._size + 1, self._window_size)
        return self.get()

    def get(self) -> tuple[float, tuple[float, int]]:
        if self._size == 0:
            raise ValueError("No data in the moving average window. " "self._size == 0")

        window_sum = self._window.sum()
        return window_sum / self._size, (window_sum, self._size)


class RewardChoices(str, enum.Enum):
    EXACT_MATCH = "exact_match"
    REF_PPL = "ref_ppl"


class Task(str, enum.Enum):
    SENTIMENT = "sentiment"
    MAIN = "main"


class ValidPrecisions(enum.Enum):
    int4 = "int4"
    int8 = "int8"
    bfloat16 = torch.bfloat16
    float16 = torch.float16
    float32 = torch.float32

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ValidPrecisions):
            raise TypeError(f"Cannot compare {type(self)} with {type(value)}")

        return super().__eq__(value)


def make_tokenizers_sft(model_name_or_path, model, is_encoder_decoder):
    """
    
    In supervised conditional generation with a causal model, we need the pad token
    to be different from the eos token, so that the model can learn to stop generating.

    """
    forward_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)  # type: ignore
    prediction_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)  # type: ignore

    ###########################################################################
    # 🔍 Make there is no pad token, or that pad token is the same as the eos token.
    ###########################################################################
    tokenizers_dont_have_pad_token = (
        forward_tokenizer.pad_token is None and 
        prediction_tokenizer.pad_token is None)
    model_doesnt_have_pad_token = model.config.pad_token_id is None

    if not model_doesnt_have_pad_token:
        # If the model is using the eos token as the pad token, it doesn't count. 
        # We need a separate pad token.
        if model.config.pad_token_id == model.config.eos_token_id:
            model.config.pad_token_id = None
            model_doesnt_have_pad_token = True

    assert (
        tokenizers_dont_have_pad_token 
        ), (
        forward_tokenizer.pad_token,
        prediction_tokenizer.pad_token,
        model.config.pad_token_id,
    )

    assert model.config.eos_token_id is not None, model.config.eos_token_id
    for tokenizer in [forward_tokenizer, prediction_tokenizer]:
        assert tokenizer.eos_token is not None
        assert tokenizer.eos_token_id is not None

    ###########################################################################
    # 🔨 Create a new pad token & assign it to the tokenizers & to the model. 
    ###########################################################################
    if tokenizers_dont_have_pad_token:
        for i, tokenizer in enumerate([forward_tokenizer, prediction_tokenizer]):
            tokenizer.add_special_tokens(dict(pad_token="<|pad|>"))
            if i == 0:
                model.resize_token_embeddings(len(tokenizer))  # type: ignore
                model.config.pad_token_id = tokenizer.pad_token_id  # type: ignore

    ###########################################################################
    # 🔍 Verify things are setup correctly
    ###########################################################################
    assert (
        forward_tokenizer.pad_token is not None and
        prediction_tokenizer.pad_token is not None and
        model.config.pad_token_id is not None
        ), (
        forward_tokenizer.pad_token,
        prediction_tokenizer.pad_token,
        model.config.pad_token_id,
    ) 

    ###########################################################################
    # 🔨 Fix the padding side for causal models.
    ###########################################################################
    if not is_encoder_decoder:
        assert hasattr(prediction_tokenizer, "padding_side")
        assert hasattr(forward_tokenizer, "padding_side")
        prediction_tokenizer.padding_side = "left"
        forward_tokenizer.padding_side = "right"
    
    return dict(
        forward_tokenizer=forward_tokenizer,
        prediction_tokenizer=prediction_tokenizer,
    )


def not_first_token(*, tensor, forward_tokenizer):
    assert len(tensor.shape) == 2
    assert not (tensor[:, 0] == forward_tokenizer.pad_token_id).any()
    assert forward_tokenizer.padding_side == "right"


def not_last_token(*, tensor, predict_tokenizer):
    assert len(tensor.shape) == 2
    assert not (tensor[:, -1] == predict_tokenizer.pad_token_id).any()
    assert predict_tokenizer.padding_side == "left"


def progress(seq, description, total=None, disable=False):
    yield from tqdm(seq, desc=description, total=total, disable=disable)


def child_names(pt_module):
    return set(name for name, _ in pt_module.named_children())


def print_accelerate_envs():
    if RANK == 0:
        keys = [k for k in sorted(os.environ) if "accelerate" in k.lower()]

        table = rich.table.Table(
            "Key", "Value", title="Accelerate Environment Variables"
        )
        for k in keys:
            if "accelerate" in k.lower():
                form_k = k.replace(
                    "DEEPSPEED",
                    "[green]DEEPSPEED[/]",
                )
                table.add_row(form_k, os.environ[k])
        table.caption = str(len(table.rows))
        rich.print(table)


@contextlib.contextmanager
def maybe_context_manager(caller, disable):
    if disable:
        yield
    else:
        with caller():
            yield