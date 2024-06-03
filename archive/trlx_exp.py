#!/usr/bin/env python
# coding: utf-8


"""
Training script for the RL part of the project.

There are wrappers for GSM8K and for the ASDiv datasets.

By default, we support the GPT2 model.


"""

import collections
import contextlib
import enum
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import sys
import tempfile
from pathlib import Path
from typing import *

import datasets
import fire
import numpy as np
import rich
import torch
import transformers
import yaml

sys.path.append("/home/mila/g/gagnonju/Marg-Li-CoT/with_trlx/trlx_repo")
import general_utils

from trlx import trlx
from trlx.trlx.data.configs import TRLConfig

import lib_data

print("Done with imports")

PPO_CONFIG_PATH = (
    "/home/mila/g/gagnonju/Marg-Li-CoT/our_scratchpad/configs/ppo_config.yml"
)
MAIN_MODEL = "google/flan-t5-small"
REWARD_MODEL = "google/flan-t5-small"
TOKENIZER_MODEL = "google/flan-t5-small"
DATASET_TO_USE = "asdiv"


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
    do_load_from_hf_name_or_path,
):

    if do_load_from_hf_name_or_path:

        assert reward_model is None, f"{reward_model = }"
        assert model is None, f"{type(model    ) = }"
        assert tokenizer is None, f"{type(tokenizer) = }"

        rich.print("[bold red]Loading from HF name or path")
        rich.print(f"[bold red]{main_model_hf_name_or_path = }")
        rich.print(f"[bold red]{reward_model_hf_name_or_path = }")
        rich.print(f"[bold red]{tokenizer_hf_name_or_path = }")

        hf_path = main_model_hf_name_or_path

        # We do some modifications to the tokenizer, so even if we load from a HF name or path, we still need to save it
        # and have the trained reload it intenally
        reward_model = model_class.from_pretrained(reward_model_hf_name_or_path).cuda()

        if reward_model.config.model_type == "gpt2":
            assert False
            reward_tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_hf_name_or_path, padding_side="left"
            )
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
            reward_tokenizer.cls_token = reward_tokenizer.eos_token

        elif reward_model.config.model_type == "t5":
            reward_tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_hf_name_or_path, padding_side="right"
            )
        else:
            raise ValueError(f"{reward_model.config.model_type = }")

        print("done loading first model")
        tmp_dir = tempfile.TemporaryDirectory()
        tok_path = tmp_dir.name
        reward_tokenizer.save_pretrained(tok_path)

    else:
        ###########################################################################
        # Save the model and the tokenizer to a temporary directory
        # so the trlx code can load it
        ###########################################################################
        rich.print("[bold red]Not Loading From HF Name or Path.")

        assert (
            reward_model is not None
        ), f"`reward_model` should not be None. {reward_model = }"
        assert model is not None, f"`model` should not be None.{type(model    ) = }"
        assert tokenizer is not None, f"`tokenizer` should not be None. {tokenizer = }"

        tmp_dir = tempfile.TemporaryDirectory()
        model.save_pretrained(tmp_dir.name)
        tokenizer.save_pretrained(tmp_dir.name)
        hf_path = tmp_dir.name
        tok_path = tmp_dir.name
        reward_model.cuda()

        for param in reward_model.parameters():
            param.requires_grad = False

        reward_tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_path, padding_side="left"
        )
        check_tokenizer(reward_tokenizer)

    with tmp_dir:
        yield (
            hf_path,
            tok_path,
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

    rich.print(f"\n[bold blue]{field}:")
    rich.print(f"input max  = {int(max_)}")
    rich.print(f"input min  = {int(min_)}")
    rich.print(f"input mean = {mean:0.3}")
    rich.print(f"input std  = {std :0.3}")

    # plt.title(field)
    # plt.hist(keys, bins=10, weights=values)
    # plt.gca().xaxis.set_major_locator(
    #     ticker.MaxNLocator(integer=True))
    # plt.show()


class ModelClassChoices(str, enum.Enum):
    seq2seq = "seq2seq"
    causal_lm = "causal_lm"


MODEL_CLASS_CHOICES = {
    ModelClassChoices.causal_lm: transformers.AutoModelForCausalLM,
    ModelClassChoices.seq2seq: transformers.AutoModelForSeq2SeqLM,
}

MODEL_TYPE_CHECKS = {
    ModelClassChoices.causal_lm: {"gpt2"},
    ModelClassChoices.seq2seq: {"bart", "t5"},
}


def sanity_check_model_type(model_class_name: str, hf_name_or_path: str):
    """
    Check that the model type is compatible with the model class.
    Basically checks that we're not trying to instantiate a seq2seq gpt2 model or something like that.
    """
    config = transformers.AutoConfig.from_pretrained(hf_name_or_path)
    assert (
        config.model_type in MODEL_TYPE_CHECKS[model_class_name]
    ), f"Model type {config.model_type} is not compatible with model class {model_class_name}. "


def train(
    do_load_from_hf_name_or_path: bool = True,
    # If we want the method to receive models directly:
    # (This is if do_load_from_hf_name_or_path is False)
    main_model: Optional[transformers.PreTrainedModel] = None,
    reward_model: Optional[transformers.PreTrainedModel] = None,
    tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
    # If we want instead the model to use hf_names_or_paths:
    # (This is if do_load_from_hf_name_or_path is True)
    main_model_hf_name_or_path: Optional[str] = MAIN_MODEL,
    reward_model_hf_name_or_path: Optional[str] = REWARD_MODEL,
    tokenizer_hf_name_or_path: Optional[str] = TOKENIZER_MODEL,
    model_class_name: str = ModelClassChoices.seq2seq,  # One of ModelClassChoices
    # ds_eval: Optional[torch.data.Dataset] = None,
    # ds_train: Optional[torch.data.Dataset] = None,
    trlx_config_path: Union[Path, str] = PPO_CONFIG_PATH,
    dataset_to_use: str = DATASET_TO_USE,
):

    args = locals().copy()
    rich.print("[bold blue]Arguments:")
    general_utils.print_dict(args)
    print("")

    assert dataset_to_use in list(
        lib_data.DatasetChoices
    ), f"{dataset_to_use = } not in {list(lib_data.DatasetChoices)}"

    sanity_check_model_type(model_class_name, main_model_hf_name_or_path)
    sanity_check_model_type(model_class_name, reward_model_hf_name_or_path)
    model_base_class = MODEL_CLASS_CHOICES[model_class_name]

    # The setup makes use of a tempfile.TemporaryDirectory to go around the peculiarities of TRLX.
    # This is why setup is a context manager.
    with setup(
        model=main_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        main_model_hf_name_or_path=main_model_hf_name_or_path,
        reward_model_hf_name_or_path=reward_model_hf_name_or_path,
        tokenizer_hf_name_or_path=tokenizer_hf_name_or_path,
        model_class=model_base_class,
        do_load_from_hf_name_or_path=do_load_from_hf_name_or_path,
    ) as (hf_path, tok_path, reward_model, reward_tokenizer):

        ds_train_obj = lib_data.GSM8KLMDataset(
            datasets.load_dataset("gsm8k", "main", split="train"),
            tokenizer=reward_tokenizer,
        )
        ds_eval_obj = lib_data.GSM8KLMDataset(
            datasets.load_dataset("gsm8k", "main", split="train"),
            tokenizer=reward_tokenizer,
        )

        config_dict = yaml.safe_load(Path(trlx_config_path).read_text())

        config_dict["model"]["model_path"] = hf_path
        config_dict["model"]["tokenizer_path"] = tok_path
        config_dict["method"]["gen_kwargs"][
            "eos_token_id"
        ] = reward_tokenizer.cls_token_id

        rich.print(config_dict)
        config = TRLConfig.from_dict(config_dict)

        # stats_for_key(ds_train_obj, "input", reward_tokenizer)
        # stats_for_key(ds_train_obj, "value", reward_tokenizer)
        # stats_for_key(ds_eval_obj , "input", reward_tokenizer)
        # stats_for_key(ds_eval_obj , "value", reward_tokenizer)

        scratchpad_reward_fn = ScratchpadRewardFn(
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            ds_train_obj=ds_train_obj,
        )

        model = trlx.train(
            model_path=hf_path,
            config=config,
            prompts=ds_train_obj,
            eval_prompts=ds_eval_obj,
            reward_fn=scratchpad_reward_fn,
            model_base_class=model_base_class,
        )


def remove_special_token_ids(
    input_ids: list[int], tokenizer: transformers.PreTrainedTokenizer
):
    """
    Remove special tokens from the input_ids
    """
    all_special_ids = set(tokenizer.all_special_ids)
    filtered_input_ids = [x for x in input_ids if x not in all_special_ids]
    return filtered_input_ids


class ScratchpadRewardFn:
    def __init__(
        self, *, reward_model, reward_tokenizer, ds_train_obj,
    ):

        if reward_model.config.is_encoder_decoder:
            assert (
                reward_tokenizer.padding_side == "right"
            ), f"{reward_tokenizer.padding_side = }"
        else:
            assert (
                reward_tokenizer.padding_side == "left"
            ), f"{reward_tokenizer.padding_side = }"

        self._reward_model = reward_model
        self._reward_tokenizer = reward_tokenizer
        self._ds_train_obj = ds_train_obj

    def __call__(self, batch, question_ids, scratchpad_ids, meta_info):
        # The idea is to:
        # 1. Extract the associated answers & tokenize the answers
        # 2. Create a mask for the answers
        # 3. Tokenize the samples
        # 4. Concate the samples & answers
        # 5. Run the reward model on the concatenated samples & answers
        # 6. Extract the logp for the answers
        # 7. Return the logp for the answers

        rand_id = random.randint(0, len(question_ids) - 1)
        question = self._reward_tokenizer.decode(question_ids[rand_id])
        scratchpad = self._reward_tokenizer.decode(scratchpad_ids[rand_id])
        question_filtered = question.replace(self._reward_tokenizer.pad_token, "")
        scratchpad_filtered = scratchpad.replace(
            self._reward_tokenizer.eos_token, ""
        ).replace(self._reward_tokenizer.pad_token, "")

        rich.print(f'[bold blue]meta_info:[/]  "{meta_info}"\n')
        rich.print(f'[bold blue]question:[/]   "{question}"\n')
        rich.print(f'[bold blue]scratchpad:[/] "{scratchpad}"\n')
        rich.print(f'[bold blue]question_filtered:[/]   "{question_filtered}"\n')
        rich.print(f'[bold blue]scratchpad_filtered:[/] "{scratchpad_filtered}"\n')

        zipped_batch = general_utils.dict_zip(batch)
        # check_tokenizer(self._reward_tokenizer)

        ids_outputs = []
        training_mask_ids_outputs = []
        if self._reward_model.config.is_encoder_decoder:
            ids_inputs = []
            training_mask_ids_inputs = []

        for input_, question_ids_indiv, scratchpad_ids_indiv in zip(
            zipped_batch, question_ids, scratchpad_ids
        ):

            # 1.a Extract the associated answers
            question_ids_indiv = remove_special_token_ids(
                question_ids_indiv.tolist(), self._reward_tokenizer
            )
            scratchpad_ids_indiv = remove_special_token_ids(
                scratchpad_ids_indiv.tolist(), self._reward_tokenizer
            )

            labels = [x.item() for x in input_["labels"] if x >= 0]
            answer = self._reward_tokenizer.decode(labels, skip_special_tokens=True)

            for k, v in self._reward_tokenizer.special_tokens_map.items():
                if k != "additional_special_tokens":
                    answer = answer.replace(v, "")

            answer_tokens = self._reward_tokenizer(answer, add_special_tokens=False)[
                "input_ids"
            ]

            if self._reward_model.config.is_encoder_decoder:
                tokens = (
                    scratchpad_ids_indiv
                    + answer_tokens
                    + [self._reward_tokenizer.eos_token_id]
                )
                ids_inputs.append(question_ids_indiv)
                ids_outputs.append(tokens)
                label_pad = 0
                training_mask_ids_inputs.append([label_pad] * len(question_ids_indiv))
                training_mask_ids_outputs.append(
                    [label_pad] * len(scratchpad_ids_indiv)
                    + [1] * len(answer_tokens)
                    + [1]
                )
            else:
                tokens = (
                    question_ids_indiv
                    + scratchpad_ids_indiv
                    + answer_tokens
                    + [self._reward_tokenizer.eos_token_id]
                )
                ids_outputs.append(tokens)

                label_pad = 0
                training_mask_ids_outputs.append(
                    [label_pad] * len(question_ids_indiv)
                    + [label_pad] * len(scratchpad_ids_indiv)
                    + [1] * len(answer_tokens)
                    + [1]
                )
                assert len(tokens) == len(training_mask_ids_outputs[-1]), (
                    "{len(tokens) = }",
                    "{len(training_mask_ids_outputs[-1]) = }",
                )

        ###########################################################################
        # 1.b Tokenize the answers
        ###########################################################################
        full_seq = self._reward_tokenizer.pad(
            dict(input_ids=ids_outputs), return_tensors="pt", padding=True
        )
        full_seq = {k: v.cuda() for k, v in full_seq.items()}
        if self._reward_model.config.is_encoder_decoder:
            full_seq_inputs = self._reward_tokenizer.pad(
                dict(input_ids=ids_inputs), return_tensors="pt", padding=True
            )
            full_seq_inputs = {k: v.cuda() for k, v in full_seq_inputs.items()}

        ###########################################################################
        # 2. Compute the logp for the answers
        ###########################################################################
        # THE PROBLEM IS THAT THERE ARE IDS THAT ARE NOT IN THE VOCAB
        # THEY SHOULD BE IN THE CHECKPOINT HOWEVER.
        # THIS IS WHAT I WILL DO TOMORROW - INTEGRATION OF THIS SOLUTION WITH THE
        # OTHER CHECKPOINT SHIT
        if not self._reward_model.config.is_encoder_decoder:
            logits = self._reward_model(
                full_seq["input_ids"], attention_mask=full_seq["attention_mask"]
            ).logits
        else:
            logits = self._reward_model(
                input_ids=full_seq_inputs["input_ids"],
                attention_mask=full_seq_inputs["attention_mask"],
                decoder_input_ids=full_seq["input_ids"],
                decoder_attention_mask=full_seq["attention_mask"],
            ).logits

        reward_model_outputs = logits.softmax(-1)

        ###########################################################################
        # 3. Only keep the logp for the actual values used
        ###########################################################################
        logp = reward_model_outputs.gather(
            dim=-1, index=full_seq["input_ids"].unsqueeze(-1)
        ).squeeze(-1)

        ###########################################################################
        # 4. Mask the logits of everything that is not the answer
        ###########################################################################
        full_seq_input_masks = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x) for x in training_mask_ids_outputs],
            batch_first=True,
            padding_value=-100,
        ).to(logp.device)
        logp[full_seq_input_masks] = 1
        logp_per_seq = logp.prod(-1)
        average_logp = logp_per_seq.mean()
        return average_logp


if __name__ == "__main__":
    fire.Fire(train)
