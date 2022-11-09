#!/usr/bin/env python
# coding: utf-8

import collections
import os
import itertools
import re
import time

import datasets
import fire
import editdistance
import numpy as np
import rich
import torch
from tqdm import tqdm
import transformers
import trlx
from trlx.data.configs import TRLConfig
from tqdm import tqdm

import general_utils

print("Done with ipmorts")


CONFIG_PATH = "/home/mila/g/gagnonju/Marg-Li-CoT/our_scratchpad/configs/ppo_config.yml"

class GSM8KLMDataset(torch.utils.data.Dataset):
    _int_patt = re.compile(r"\-?\d+")

    def __init__(self, ds, tokenizer, ):
        self._ds = ds
        self._inputs_key  = "question"
        self._outputs_key = "answer"
        self._tokenizer   = tokenizer
        self._targets     = {}

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        assert isinstance(idx, int), f"{type(idx) = }"
        assert idx >= 0, f"{idx = }"

        input_ = self._ds[idx][self._inputs_key ]
        output = self._ds[idx][self._outputs_key].rsplit("####", 1)[1].strip()
        
        return {
            "inputs": input_ + self._tokenizer.cls_token,
            "labels": output + self._tokenizer.cls_token,
            }


def check_tokenizer(tokenizer):
    assert tokenizer.pad_token != tokenizer.eos_token, f"{tokenizer.pad_token = }, {tokenizer.eos_token = }" 
    assert tokenizer.pad_token != tokenizer.cls_token, f"{tokenizer.pad_token = }, {tokenizer.cls_token = }"
    assert tokenizer.eos_token != tokenizer.cls_token, f"{tokenizer.eos_token = }, {tokenizer.cls_token = }"

    assert tokenizer.pad_token_id != tokenizer.eos_token_id, f"{tokenizer.pad_token_id = }, {tokenizer.eos_token_id = }" 
    assert tokenizer.pad_token_id != tokenizer.cls_token_id, f"{tokenizer.pad_token_id = }, {tokenizer.cls_token_id = }"
    assert tokenizer.eos_token_id != tokenizer.cls_token_id, f"{tokenizer.eos_token_id = }, {tokenizer.cls_token_id = }"


def train(model, tokenizer, ds_train, ds_eval, config):

    from_scratch = False
    if from_scratch:
        # ds_train = datasets.load_dataset("gsm8k", "main", split="train")
        # ds_train_socratic = datasets.load_dataset("gsm8k", "socratic", split="train")
        # ds_eval  = datasets.load_dataset("gigaword", split="validation")

        reward_model_model_name = "gpt2"
        reward_tokenizer = transformers.AutoTokenizer       .from_pretrained(reward_model_model_name)
        reward_tokenizer.add_special_tokens({"pad_token": "<|pad|>", "cls_token": "<|cls|>",})
        check_tokenizer(reward_tokenizer)

    ds_config = dict(tokenizer=reward_tokenizer)
    ds_train_obj = GSM8KLMDataset(ds_train, **ds_config)
    # ds_eval_obj = GSM8KLMDataset(ds_eval , **ds_config)


    ###################################################################################################
    # Stats for the dataset
    ###################################################################################################
    shortest = []
    field_options = ("inputs", "labels")
    def stats_for_key(ds: GSM8KLMDataset, field: str):
        """
        Evaluate stats on the number of tokens per sample
        """
        stuff = collections.Counter()
        for entry in ds:
            # 1. Extract the text of the inputs or of the labels 
            
            assert field in field_options, (
                f"inputs_or_outputs should be in {field_options}, "
                f"got `{field}`"
            )
            target = entry[field]
            
            # 2. Tokenize the text
            assert target.endswith("<|cls|>"), f"{target = }"
            target = target.removesuffix("<|cls|>")

            input_ids = reward_tokenizer(target)["input_ids"]
            if len(input_ids) <= 7:
                shortest.append((target, input_ids))
                
            stuff.update([len(input_ids)])
            

        keys = np.fromiter(stuff.keys(), dtype=float)
        values = np.fromiter(stuff.values(), dtype=float)
        
        mean = np.average(keys, weights=values)
        std  = np.sqrt(np.average((keys - mean) ** 2, weights=values))
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


    stats_for_key(ds_train_obj, "inputs")
    stats_for_key(ds_train_obj, "labels")



    reward_model     = transformers.AutoModelForCausalLM.from_pretrained(reward_model_model_name).cuda()

    for param in reward_model.parameters():
        param.requires_grad = False

    prompt_end     = "<|cls|>"
    scratchpad_end = "<|cls|>"

    CONFIG_PATH = "/home/mila/g/gagnonju/Marg-Li-CoT/our_scratchpad/configs/ppo_config.yml"

    scratchpad_reward_fn = ScratchpadRewardFn(
        reward_model     = reward_model,
        reward_tokenizer = reward_tokenizer,
        prompt_end       = prompt_end,
        scratchpad_end   = scratchpad_end,
        ds_train_obj     = ds_train_obj,
    )

    model = trlx.train(
        "distilgpt2", 
        config=TRLConfig.load_yaml(CONFIG_PATH),
        prompts      = ds_train_obj,
        eval_prompts = ds_train_obj,
        reward_fn    = scratchpad_reward_fn,
    )


class ScratchpadRewardFn:
    def __init__(
        self, 
        *, 
        reward_model, 
        reward_tokenizer, 
        prompt_end, 
        scratchpad_end, 
        ds_train_obj,
    ):
        self._reward_model     = reward_model
        self._reward_tokenizer = reward_tokenizer
        self._prompt_end       = prompt_end
        self._scratchpad_end   = scratchpad_end
        self._ds_train_obj     = ds_train_obj

    def __call__(self, samples, batch):
        # The idea is to:
        # 1. Extract the associated answers & tokenize the answers
        # 2. Create a mask for the answers
        # 3. Tokenize the samples
        # 4. Concate the samples & answers
        # 5. Run the reward model on the concatenated samples & answers
        # 6. Extract the logp for the answers
        # 7. Return the logp for the answers

        reward_model_inputs  = []

        zipped_batch = general_utils.dict_zip(batch)
        check_tokenizer(self._reward_tokenizer)
        ids_outputs = []
        training_mask_ids_outputs = []

        for sample, input_ in zip(samples, zipped_batch):
            # 1.a Extract the associated answers 
            question, scratchpad = sample.split(self._prompt_end, 1)

            labels = [x.item() for x in input_["labels"] if x >= 0]
            answer = self._reward_tokenizer.decode(labels, skip_special_tokens=True)
            for v in self._reward_tokenizer.special_tokens_map.values():
                answer = answer.replace(v, "")

            answer_tokens     = self._reward_tokenizer(answer    )["input_ids"]
            question_tokens   = self._reward_tokenizer(question  )["input_ids"]
            scratchpad_tokens = self._reward_tokenizer(scratchpad)["input_ids"]

            tokens = (
                question_tokens   + [self._reward_tokenizer.cls_token_id] +
                scratchpad_tokens + [self._reward_tokenizer.cls_token_id] +
                answer_tokens     + [self._reward_tokenizer.eos_token_id]
            )
            ids_outputs.append(tokens)
            
            label_pad = 0
            training_mask_ids_outputs.append(
                [label_pad] * len(question_tokens  ) + [label_pad] +
                [label_pad] * len(scratchpad_tokens) + [label_pad] + 
                [1        ] * len(answer_tokens    ) + [1]
            )

        ###########################################################################
        # 1.b Tokenize the answers
        ###########################################################################
        full_seq = self._reward_tokenizer.pad(dict(input_ids=ids_outputs), return_tensors="pt", padding=True)
        full_seq = {k: v.cuda() for k, v in full_seq.items()}

        ###########################################################################
        # 2. Compute the logp for the answers
        ###########################################################################
        # THE PROBLEM IS THAT THERE ARE IDS THAT ARE NOT IN THE VOCAB
        # THEY SHOULD BE IN THE CHECKPOINT HOWEVER.
        # THIS IS WHAT I WILL DO TOMORROW - INTEGRATION OF THIS SOLUTION WITH THE 
        # OTHER CHECKPOINT SHIT
        reward_model_outputs = self._reward_model(full_seq["input_ids"], attention_mask=full_seq["input_ids"]).logsoftmax(-1)
        
        ###########################################################################
        # 3. Only keep the logp for the actual values used
        ###########################################################################
        logp = reward_model_outputs.gather(dim=-1, index=full_seq["input_ids"]).squeeze(-1)

        ###########################################################################
        # 4. Mask the logits of everything that is not the answer
        ###########################################################################
        full_seq_input_masks = torch.nn.utils.rnn.pad_sequence(training_mask_ids_outputs, batch_first=True, padding_value=-100)
        logp *= full_seq_input_masks

        logp = logp.sum(-1)
        assert logp.shape == (len(samples),), logp.shape

        output = logp.mean()
        assert output.shape == (), output.shape
        return output



if __name__ == "__main__":
    fire.Fire(main)