#!/usr/bin/env python
# coding: utf-8


"""

Training script for the RL part of the project.

There are wrappers for GSM8K and for the ASDiv datasets.

By default, we support the GPT2 model.

"""

print("Doing imports.")
import os
import collections
import contextlib
import enum
import logging
import random
from pathlib import Path
from typing import *

import accelerate
import datasets
import fire
import general_utils
import numpy as np
import rich
import rich.logging
import torch
import transformers
import trlx
import yaml
from trlx.data.configs import TRLConfig

import lib_data
import lib_metric
import lib_reward
print("Done with imports")


DEFAULT_DETERMINISTIC   = False
DEFAULT_DO_SINGLE_PROC  = False
DEFAULT_DATASET_TO_USE  = "gsm8k"
DEFAULT_REWARD_MODEL    = "google/flan-t5-xl"
DEFAULT_MAIN_MODEL      = DEFAULT_REWARD_MODEL
DEFAULT_TOKENIZER_MODEL = DEFAULT_REWARD_MODEL


# -----------------------------------------------------------------------------
# Shouldn't ever change
# -----------------------------------------------------------------------------

os.environ["NCCL_DEBUG"] = "WARN"

RANK       = int(os.environ["RANK"      ])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

random               .seed(0)
np.random            .seed(1)
torch         .manual_seed(2)
torch.cuda.manual_seed_all(3)

LOGGER = logging.getLogger(__name__)
DEFAULT_PPO_CONFIG_PATH = str(Path(__file__).parent / "config_ppo.yml")
assert Path(DEFAULT_PPO_CONFIG_PATH).exists(), (
    f"{DEFAULT_PPO_CONFIG_PATH = }")
torch.cuda.set_device(LOCAL_RANK)



class MergedExtraInfo:
    def __init__(self, *, ds_train_obj, ds_eval_obj):
        
        self.ds_train_obj = ds_train_obj
        self.ds_eval_obj = ds_eval_obj
        
    def merged_get_extra_info(
            self,
            *,
            sample_str,
            miss_ok=False,
        ):
        if isinstance(sample_str, list):
            return [
                self.merged_get_extra_info(sample_str=x) 
                for x in sample_str
            ]

        assert isinstance(sample_str, str), type(sample_str).mro()
        output = self.ds_train_obj.get_extra_info(
            sample_str=sample_str, miss_ok=True)
        
        if output is None:
            output = self.ds_eval_obj.get_extra_info(
                sample_str=sample_str, miss_ok=False)
            assert output is not None, sample_str

        return output


def check_tokenizer(tokenizer):
    assert (
        tokenizer.pad_token != tokenizer.eos_token
    ), f"{tokenizer.pad_token = }, {tokenizer.eos_token = }"
    assert (
        tokenizer.pad_token != tokenizer.cls_token
    ), f"{tokenizer.pad_token = }, {tokenizer.cls_token = }"
    assert (
        tokenizer.eos_token != tokenizer.cls_token
    ), f"{tokenizer.eos_token = }, {tokenizer.cls_token = }"

    assert (
        tokenizer.pad_token_id != tokenizer.eos_token_id
    ), f"{tokenizer.pad_token_id = }, {tokenizer.eos_token_id = }"
    assert (
        tokenizer.pad_token_id != tokenizer.cls_token_id
    ), f"{tokenizer.pad_token_id = }, {tokenizer.cls_token_id = }"
    assert (
        tokenizer.eos_token_id != tokenizer.cls_token_id
    ), f"{tokenizer.eos_token_id = }, {tokenizer.cls_token_id = }"


@contextlib.contextmanager
def setup(
    *,
    model: Optional[transformers.PreTrainedModel],
    reward_model: Optional[transformers.PreTrainedModel],
    tokenizer: Optional[transformers.PreTrainedTokenizer],
    main_model_hf_name_or_path: Optional[Union[str, Path]],
    reward_model_hf_name_or_path: Optional[Union[str, Path]],
    tokenizer_hf_name_or_path: Optional[Union[str, Path]],
    model_class: Optional[Type[transformers.PreTrainedModel]],
):

    assert reward_model is None, f"{reward_model = }"
    assert model is None, f"{type(model    ) = }"
    assert tokenizer is None, f"{type(tokenizer) = }"

    LOGGER.info("[bold red]Loading from HF name or path")
    LOGGER.info(f"[bold red]{main_model_hf_name_or_path   = }")
    LOGGER.info(f"[bold red]{reward_model_hf_name_or_path = }")
    LOGGER.info(f"[bold red]{tokenizer_hf_name_or_path    = }")

    reward_model = model_class.from_pretrained(reward_model_hf_name_or_path)
    assert reward_model.config.model_type == "t5"
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_hf_name_or_path
    )

    return (
        reward_model_hf_name_or_path,
        tokenizer_hf_name_or_path,
        reward_model,
        reward_tokenizer,
    )


def stats_for_key(
    ds: lib_data.GSM8KLMDataset,
    field: str,
    reward_tokenizer: transformers.PreTrainedTokenizer,
):
    """
    Evaluate stats on the number of tokens per sample
    """
    stuff = collections.Counter()
    shortest = []
    field_options = ("inputs", "labels")

    for entry in ds:
        # 1. Extract the text of the inputs or of the labels

        # assert field in field_options, (
        #     f"inputs_or_outputs should be in {field_options}, "
        #     f"got `{field}`"
        # )
        target = entry[field]

        # 2. Tokenize the text
        # assert (
        #   target.endswith(reward_tokenizer.cls_token) or
        #   target.endswith(reward_tokenizer.eos_token)
        # ), f"{target = }"
        target = target.removesuffix(reward_tokenizer.cls_token).removesuffix(
            reward_tokenizer.eos_token
        )

        input_ids = reward_tokenizer(target)["input_ids"]
        if len(input_ids) <= 7:
            shortest.append((target, input_ids))

        stuff.update([len(input_ids)])

    keys = np.fromiter(stuff.keys(), dtype=float)
    values = np.fromiter(stuff.values(), dtype=float)

    mean = np.average(keys, weights=values)
    std = np.sqrt(np.average((keys - mean) ** 2, weights=values))
    max_ = np.max(keys)
    min_ = np.min(keys)

    LOGGER.info(f"\n[bold blue]{field}:")
    LOGGER.info(f"input max  = {int(max_)}")
    LOGGER.info(f"input min  = {int(min_)}")
    LOGGER.info(f"input mean = {mean:0.3}")
    LOGGER.info(f"input std  = {std :0.3}")

    # plt.title(field)
    # plt.hist(keys, bins=10, weights=values)
    # plt.gca().xaxis.set_major_locator(
    #     ticker.MaxNLocator(integer=True))
    # plt.show()


class ModelClassChoices(str, enum.Enum):
    SEQ2SEQ = "seq2seq"
    CAUSAL_LM = "causal_lm"


MODEL_CLASS_CHOICES = {
    ModelClassChoices.CAUSAL_LM: transformers.AutoModelForCausalLM,
    ModelClassChoices.SEQ2SEQ: transformers.AutoModelForSeq2SeqLM,
}

MODEL_TYPE_CHECKS = {
    ModelClassChoices.CAUSAL_LM: {"gpt2"},
    ModelClassChoices.SEQ2SEQ: {"bart", "t5"},
}

def _build_dataset(
    *,
    tokenizer_hf_name_or_path,
    val_subset_size,
    dataset_to_use,
    config_dict,
):
    assert dataset_to_use in list(lib_data.DatasetChoices), (
        f"{dataset_to_use = } not in {list(lib_data.DatasetChoices)}")
    assert dataset_to_use == lib_data.DatasetChoices.GSM8K, (
        f"{dataset_to_use = } not supported yet")
    assert torch.cuda.current_device() == LOCAL_RANK, (
        torch.cuda.current_device(), LOCAL_RANK)
    
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_hf_name_or_path)
    max_input_length = (
        config_dict["train" ]["seq_length"] -
        config_dict["method"]["gen_kwargs"]["max_new_tokens"])
    assert max_input_length >= 1, max_input_length

    #############################################################################
    # Train Dataset
    #############################################################################
    ds_train_obj = lib_data.GSM8KLMDataset(
        max_length = max_input_length,
        tokenizer  = reward_tokenizer,
        ds         = datasets.load_dataset("gsm8k", "main", split="train"),)

    #############################################################################
    # Eval Subset
    #############################################################################
    eval_dataset = datasets.load_dataset("gsm8k", "main", split="test")
    if val_subset_size:
        np_indices = np.random.permutation(len(eval_dataset))[:val_subset_size]
        np_indices.sort()
        indices = [int(x) for x in np_indices]
        eval_dataset = torch.utils.data.Subset(
            eval_dataset, 
            indices,)

    ds_eval_obj = lib_data.GSM8KLMDataset(
        max_length = max_input_length,
        tokenizer  = reward_tokenizer,
        ds         = eval_dataset,
    )
    return ds_train_obj, ds_eval_obj, reward_tokenizer


def _sanity_check_model_type(model_class_name: str, hf_name_or_path: str):
    """
    Check that the model type is compatible with the model class.
    Basically checks that we're not trying to instantiate a seq2seq gpt2
    model or something like that.
    """
    config = transformers.AutoConfig.from_pretrained(hf_name_or_path)
    assert (
        config.model_type in MODEL_TYPE_CHECKS[model_class_name]
    ), (
        f"Model type {config.model_type} is not "
        f"compatible with model class {model_class_name}. "
    )


def _logging(*, args, log_level: str):

    if RANK == 0:
        sorted_environ = sorted(os.environ.items(), key=lambda kv: kv[0])
        accelerate_then_deepspeed_ones = {
            k: v for k, v in sorted_environ
            if "deepspeed" in k.lower()
        } | {
            k: v for k, v in sorted_environ
            if "accelerate" in k.lower() and 
            "deepspeed" not in k.lower()
        }
        general_utils.print_dict(accelerate_then_deepspeed_ones)

    if RANK == 0:
        LOGGER.info("[bold blue]Arguments:")
        general_utils.print_dict(args, logger=LOGGER, log_level="INFO")
        print("")

    logging.basicConfig(
        level=log_level,
        format=f"[{RANK}/{WORLD_SIZE}] %(name)s - %(message)s ",
        handlers=[rich.logging.RichHandler(markup=True)],
    )
    logging.getLogger("reward").setLevel(logging.DEBUG)


def _setup_config(
    *,
    main_model_hf_name_or_path,
    tokenizer_hf_name_or_path,
    reward_tokenizer,
    config_dict,
):
    assert "model_path"     not in config_dict["model"    ]
    assert "tokenizer_path" not in config_dict["tokenizer"]
    hf_path = main_model_hf_name_or_path
    config_dict["tokenizer"]["tokenizer_path"] = tokenizer_hf_name_or_path
    config_dict["model"    ]["model_path"] = hf_path
    config_dict["method"   ]["gen_kwargs"]["eos_token_id"
    ] = reward_tokenizer.cls_token_id
    
    if RANK == 0:
        LOGGER.info("[bold blue]Config:")
        rich.print(config_dict)

    config = TRLConfig.from_dict(config_dict)
    return config


def train(
    *,
    reward_model_hf_name_or_path: Optional[str] = DEFAULT_REWARD_MODEL,
    main_model_hf_name_or_path: Optional[str]   = DEFAULT_MAIN_MODEL,
    tokenizer_hf_name_or_path: Optional[str]    = DEFAULT_TOKENIZER_MODEL,
    model_class_name: str                       = ModelClassChoices.SEQ2SEQ,
    trlx_config_path: Union[Path, str]          = DEFAULT_PPO_CONFIG_PATH,
    val_subset_size: Optional[int]              = None,
    dataset_to_use: str                         = DEFAULT_DATASET_TO_USE,
    do_single_proc: int                         = DEFAULT_DO_SINGLE_PROC,
    deterministic: bool                         = DEFAULT_DETERMINISTIC,
    log_level: str                              = "INFO",
):
    # -------------------------------------------------------------------------
    # Logging stuff.
    # -------------------------------------------------------------------------
    args = locals().copy()
    _logging(args=args, log_level=log_level)

    if RANK == 0:
        rich.print(
            f"[bold green]MODELS:"
            f"\t- reward: {reward_model_hf_name_or_path}\n"
            f"\t- main:   {main_model_hf_name_or_path}\n"
        )

    if do_single_proc:
        assert deterministic, (
            "Must be deterministic if single proc, useless otherwise."
        )

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["CUDA_LAUNCH_BLOCKING"   ] = "true"
        torch.backends.cudnn.deterministic    =  True
        torch.backends.cudnn.    benchmark    =  False
        torch.use_deterministic_algorithms      (True)

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    _sanity_check_model_type(model_class_name, reward_model_hf_name_or_path)
    _sanity_check_model_type(model_class_name, main_model_hf_name_or_path)
    config_dict = yaml.safe_load(Path(trlx_config_path).read_text())
    ds_train_obj, ds_eval_obj, reward_tokenizer = _build_dataset(
        tokenizer_hf_name_or_path  = tokenizer_hf_name_or_path,
        val_subset_size            = val_subset_size,
        dataset_to_use             = dataset_to_use,
        config_dict                = config_dict, 
    )
    
    # -------------------------------------------------------------------------
    # Setup Config.
    # -------------------------------------------------------------------------
    config = _setup_config(
        main_model_hf_name_or_path = main_model_hf_name_or_path,
        tokenizer_hf_name_or_path  = tokenizer_hf_name_or_path,
        reward_tokenizer           = reward_tokenizer,
        config_dict                = config_dict,
    )
    config.train.do_single_process = do_single_proc

    # -------------------------------------------------------------------------
    # Modify TOKENIZERS_PARALLELISM
    # -------------------------------------------------------------------------
    # It needs to be enabled before this point to
    # tokenize the whole dataset, but it's not needed after that.
    # -------------------------------------------------------------------------
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # -------------------------------------------------------------------------
    # Metric and Reward.
    # -------------------------------------------------------------------------
    # The metric and the reward need access to the labe.
    # -------------------------------------------------------------------------
    # The metric needs 
    merger = MergedExtraInfo(
        ds_train_obj = ds_train_obj,
        ds_eval_obj  = ds_eval_obj,
    )
    
    metric_accuracy = lib_metric.ScratchpadAnswerAccuracy(
        extra_info_engine = merger.merged_get_extra_info,
    )

    # Afaik the eval should not need the extra info engine.
    scratchpad_reward_fn = lib_reward.ScratchpadRewardFn(
        reward_model_hf_name_or_path = reward_model_hf_name_or_path,
        get_extra_info_fn            = merger.merged_get_extra_info,
        reward_tokenizer             = reward_tokenizer,
        do_single_proc               = do_single_proc,
        metric_fn                    = metric_accuracy,
    )

    # -------------------------------------------------------------------------
    # Training.
    # -------------------------------------------------------------------------
    model = trlx.train(
        eval_prompts  = list(ds_eval_obj),
        model_path    = main_model_hf_name_or_path,
        metric_fn     = metric_accuracy,
        reward_fn     = scratchpad_reward_fn,
        prompts       = list(ds_train_obj),
        config        = config,
    )


if __name__ == "__main__":
    fire.Fire(train)
