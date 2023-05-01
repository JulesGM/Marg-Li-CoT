# You can see that we load a GPT2 model called `gpt2_imdb`. 
# This model was additionally fine-tuned on the IMDB dataset
# for 1 epoch with the huggingface 
# [script](https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py) (no special settings). The other parameters are mostly taken from the original paper ["Fine-Tuning Language Models from Human Preferences"](
# https://arxiv.org/pdf/1909.08593.pdf). 
# This model as well as the BERT model is available in the 
# Huggingface model zoo [here](https://huggingface.co/models). 
# The following code should automatically download the models.

# ## Load data and models

# ### Load pre-trained GPT2 language models

# We load the GPT2 model with a value head and the tokenizer. 
# We load the model twice; the first model is optimized while 
# the second model serves as a reference to calculate the 
# KL-divergence from the starting point. This serves as an 
# additional reward signal in the PPO training to make sure 
# the optimized model does not deviate too much from the 
# original language model.

# ### Load IMDB dataset
# The IMDB dataset contains 50k movie review annotated with 
# "positive"/"negative" feedback indicating the sentiment.  
# We load the IMDB dataset into a DataFrame and filter for 
# comments that are at least 500 characters long and take 
# the first 1000 characters of each comment. The first filter 
# we apply to avoid comments that are less than `txt_in_len` 
# token long and the second to avoid tokenizing way more text 
# than we actually need.

# ### Training progress
# If you are tracking the training progress with Weights&Biases you should see a plot similar to the following:
# 
# <div style="text-align: center">
# <img src='https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/gpt2-ctrl-training-stats.png' width='800'>
# <p style="text-align: center;"> <b>Figure:</b> Reward mean and distribution evolution during training. </p>
# </div>
# 
# One can observe how the model starts to generate more positive outputs after a few optimisation steps.
# 
# > Note: Investigating the KL-divergence will probably show that at this point the model has not converged to the target KL-divergence, yet. To get there would require longer training or starting with a higher inital coefficient.

# ## Model inspection

# ### Reward distribution
# First, we can have a look at the reward distribution. Both the negative and positive rewards are clearly shifted to high rewards. The neutral rewards, however, are still centered around zero. There are a few possible explanations for this. There could be a bug in the code and the way the neutral rewards are calculated. Another problem could be that sentence sometimes start with a strong sentiment and it is hard for the model shift the sentiment towards neutral.

# ## Optimize model

# **Steps**
# 
# The training loop consists of the following steps:
# 1. Get a batch of queries and create random controls
# 2. Get the query responses from the policy
# 3. Join query and responses and tokenize for BERT analysis
# 4. Get sentiments for query/responses from BERT
# 5. Optimize policy with PPO using the (query, response, reward) triplet
# 6. Log all the training statistics
# 
# **Training time**
# 
# This step takes **~2h** on a P6000 GPU with the above specified settings.



import time
import os
import random

import fire
import torch
import wandb
from tqdm import tqdm
import numpy as np
import pandas as pd
from random import choices
import matplotlib.pyplot as plt

import datasets 
import transformers
import trl


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def extract_pipe_output(outputs):
    positive_logits = []
    for out in outputs:
        for element in out:
            if element["label"] == "POSITIVE":
                positive_logits.append(torch.tensor(element["score"]))
    return positive_logits


def pos_logit_to_reward(logit, task):
    """
    Take the positive sentiment logit and scale it for the task.
        task [negative]: reward = -logit
        task [neutral]: reward = -2*abs(logit)+4
        task [positive]: reward = logit
    """
    for i in range(len(logit)):
        if task[i] == "[negative]":
            logit[i] = -logit[i]
        elif task[i] == "[neutral]":
            logit[i] = -2 * torch.abs(logit[i]) + 4
        elif task[i] == "[positive]":
            pass
        else:
            raise ValueError("task has to be in [0, 1, 2]!")
    return logit


def prep_dataset(tokenizer_model_name, txt_in_len):
    
    gpt2_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model_name)

    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    dataset = datasets.load_dataset("imdb", split="train")
    dataset = dataset.rename_columns({"text": "review", "label": "sentiment"})
    dataset = dataset.filter(lambda x: len(x["review"]) > 500, batched=False)
    dataset = dataset.map(lambda x: {"review": x["review"][:1000]}, batched=False)

    dataset = dataset.map(
        lambda x: {"input_ids": gpt2_tokenizer.encode(
            " " + x["review"], return_tensors="pt"
        )[0, :txt_in_len]},
        batched=False,
    )
    dataset = dataset.map(
        lambda x: {"query": gpt2_tokenizer.decode(x["input_ids"])}, 
        batched=False,
    )
    dataset = dataset[:20480]
    dataset = datasets.Dataset.from_dict(dataset)
    dataset.set_format("pytorch")
    return dataset, gpt2_tokenizer

def make_reward(
    *,
    pipeline_model_name, 
    accelerator_num_process, 
    accelerator_device,
):
    if accelerator_num_process == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    else:
        device = accelerator_device
    sentiment_pipe = transformers.pipeline(
        "sentiment-analysis", 
        pipeline_model_name,
        device=device)
    return sentiment_pipe


def compute_reward(query_no_ctrl, generated, reward_fn, reward_kwargs, task_list):
    texts = [q + r for q, r in zip(query_no_ctrl, generated)]
    logits = extract_pipe_output(reward_fn(texts, **reward_kwargs))
    return pos_logit_to_reward(logits, task_list)


def unroll(
    *,
    query_tensors, 
    ppo_trainer, 
    generation_kwargs, 
    gpt2_tokenizer, 
    txt_out_len,
):
    response_tensors = []
    for query in tqdm(query_tensors, desc="generating one by one"):
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-txt_out_len:])
        
    return [
        gpt2_tokenizer.decode(r.squeeze()) 
        for r in response_tensors
    ], response_tensors


def per_ctrl_stats(ctrl_str, rewards, task_list):
    stats = {}
    for cs in ctrl_str:
        key = "env/reward_" + cs.strip("[]")
        stats[key] = np.mean([
            r.cpu().numpy() for r, t 
            in zip(rewards, task_list) if t == cs
        ])
    return stats


def main(
    pipeline_model_name = "lvwerra/distilbert-imdb",
    learning_rate       = "1.41e-5",
    txt_out_len         = 20,
    txt_in_len          = 5,
    model_name          = "lvwerra/gpt2-imdb",
    num_steps           = 51200,
    seed                = 0,
    name                = None,
):
    script_kwargs = locals().copy()
    np.random.seed(seed)
    reward_kwargs = {
        "function_to_apply": "none",
        "top_k":              None, 
    }
    ppo_trainer_config_dict = dict(
        remove_unused_columns = False, 
        learning_rate         = learning_rate,
        model_name            = model_name, 
        log_with              = "wandb",
        steps                 = num_steps,
    )
    config = trl.PPOConfig(**ppo_trainer_config_dict)
    dataset, gpt2_tokenizer = prep_dataset(config.model_name, txt_in_len)
    generation_kwargs = {
        "max_new_tokens": txt_out_len,
        "pad_token_id":   gpt2_tokenizer.eos_token_id,
        "eos_token_id":   -1,
        "min_length":     -1,
        "do_sample":      True,
        "top_k":          0.0,
        "top_p":          1.0,
    }

    wandb.init(
        config=dict(
            generation_kwargs = generation_kwargs,
            reward_kwargs     = reward_kwargs,
            script_kwargs     = script_kwargs,
            ppo_config        = ppo_trainer_config_dict,
        ),
        project = "new_sentiment",
        entity  = "julesgm",
        name    = name,
    )

    gpt2_model     = trl.AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    gpt2_model_ref = trl.create_reference_model(gpt2_model)

    ppo_trainer = trl.PPOTrainer(
        data_collator = collator,
        tokenizer     = gpt2_tokenizer, 
        ref_model     = gpt2_model_ref, 
        dataset       = dataset, 
        model         = gpt2_model, 
        **ppo_trainer_config_dict
    )

    reward_model = make_reward(
        accelerator_num_process = ppo_trainer.accelerator.num_processes, 
        pipeline_model_name     = pipeline_model_name, 
        accelerator_device      = ppo_trainer.accelerator.device,
    )


    ctrl_str = ["[negative]", "[neutral]", "[positive]"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    assert device.type == "cuda", device.type
    ctrl_tokens = {
        "s": 
        gpt2_tokenizer.encode(s, return_tensors="pt").squeeze().to(device) 
        for s in ctrl_str
    }

    for epoch in range(2):
        for batch_idx, batch in enumerate(
            tqdm(ppo_trainer.dataloader), 
            desc=f"Epoch {epoch} - Training Loop"
        ):
            
            logs = dict()
            game_data = dict()
            
            #### prepend a random control token
            task_list = choices(ctrl_str, k=config.batch_size)
            game_data["query"] = [t + q for t, q in zip(task_list, batch["query"])]
            query_tensors = [
                torch.cat((ctrl_tokens[t], input_ids)) 
                for t, input_ids in zip(task_list, batch["input_ids"])
            ]

            #### get response from gpt2
            game_data["response"], response_tensors = unroll(
                generation_kwargs = generation_kwargs, 
                gpt2_tokenizer    = gpt2_tokenizer, 
                query_tensors     = query_tensors, 
                ppo_trainer       = ppo_trainer, 
                txt_out_len       = txt_out_len,
            )

            #### sentiment analysis
            rewards = compute_reward(
                query_no_ctrl = batch["query"], 
                reward_kwargs = reward_kwargs, 
                generated     = game_data["response"], 
                reward_fn     = reward_model,
                task_list     = task_list,
            )

            #### Run PPO training
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            #### Add per ctrl stats
            stats = dict(**stats, **per_ctrl_stats(
                ctrl_str  = ctrl_str,
                rewards   = rewards,
                task_list = task_list,
            ))
                
            ppo_trainer.log_stats(stats, game_data, rewards)


    for ctrl_s in ctrl_str:
        plt.hist(
            [r for r, t in zip(logs["env/reward_dist"], task_list) if t == ctrl_s], 
            density=True, alpha=0.5, label=ctrl_s
        )

    plt.legend(loc="best")
    plt.title("reward distribution")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)