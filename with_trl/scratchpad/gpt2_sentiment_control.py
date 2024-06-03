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


import sys
from pathlib import Path
from random import choices

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import trl
import wandb
from tqdm import tqdm

try:
    import pretty_traceback

    pretty_traceback.install()
except ImportError:
    pass


SCRIPT_DIR = Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR.parent))
import lib_sentiment_specific
import lib_trl_utils

# def unroll(
#     *,
#     generation_kwargs,
#     gpt2_tokenizer,
#     query_tensors,
#     txt_out_len,
#     ppo_trainer,
#     log_header,
# ):

#     response_tensors = []
#     for query in tqdm(query_tensors, desc=f"{log_header} Generating one by one"):
#         response = ppo_trainer.generate(query, **generation_kwargs)
#         response_tensors.append(response.squeeze()[-txt_out_len:])

#     return (
#         gpt2_tokenizer.batch_decode(response_tensors),
#         response_tensors,
#     )


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def per_ctrl_stats(ctrl_str, rewards, task_list):
    stats = {}
    for cs in ctrl_str:
        key = "env/reward_" + cs.strip("[]")
        stats[key] = np.mean(
            [r.cpu().numpy() for r, t in zip(rewards, task_list) if t == cs]
        )
    return stats


def main(
    pipeline_model_name="lvwerra/distilbert-imdb",
    learning_rate=1.41e-5,
    txt_out_len=20,
    txt_in_len=5,
    model_name="lvwerra/gpt2-imdb",
    num_steps=51200,
    seed=0,
    name=None,
    qty_print=6,
    use_new_unroll=True,
):
    script_kwargs = locals().copy()
    np.random.seed(seed)

    ppo_trainer_config_dict = dict(
        remove_unused_columns=False,
        learning_rate=learning_rate,
        model_name=model_name,
        log_with="wandb",
        steps=num_steps,
    )
    config = trl.PPOConfig(**ppo_trainer_config_dict)
    dataset, gpt2_tokenizer = lib_sentiment_specific.prep_dataset_rl(
        config.model_name, txt_in_len
    )
    generation_kwargs = {
        "max_new_tokens": txt_out_len,
        "pad_token_id": gpt2_tokenizer.eos_token_id,
        "eos_token_id": -1,
        "min_length": -1,
        "do_sample": True,
        "top_k": 0.0,
        "top_p": 1.0,
    }

    wandb.init(
        config=dict(
            generation_kwargs=generation_kwargs,
            script_kwargs=script_kwargs,
            ppo_config=ppo_trainer_config_dict,
        ),
        project="new_sentiment",
        entity="julesgm",
        name=name,
    )

    gpt2_model = trl.AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name
    )
    gpt2_model_ref = trl.create_reference_model(gpt2_model)

    ppo_trainer = trl.PPOTrainer(
        data_collator=collator,
        tokenizer=gpt2_tokenizer,
        ref_model=gpt2_model_ref,
        dataset=dataset,
        model=gpt2_model,
        config=config,
    )

    reward_fn = lib_sentiment_specific.SentimentRewardFn(
        ppo_trainer,
    )

    ctrl_str = ["[negative]", "[neutral]", "[positive]"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", device.type
    ctrl_tokens = {
        s: gpt2_tokenizer.encode(s, return_tensors="pt").squeeze().to(device)
        for s in ctrl_str
    }

    for epoch in range(2):
        for model in [ppo_trainer.model, ppo_trainer.ref_model]:
            if model:
                for param_name, param in model.named_parameters():
                    assert (
                        param.device.type == "cuda"
                    ), f"{param_name} {param.device.type}"

        for batch_idx, batch in enumerate(
            tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch} - Training Loop")
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
            output, scores = lib_trl_utils.batched_unroll(
                generation_kwargs=generation_kwargs,
                query_tensors=query_tensors,
                ppo_trainer=ppo_trainer,
                tokenizer=gpt2_tokenizer,
            )

            #### sentiment analysis
            rewards = reward_fn(
                queries=batch["query"],
                responses=output["response_texts"],
                task_list=task_list,
            )

            #### print table
            lib_trl_utils.print_table(
                log_header=f"(e{epoch}b{batch_idx})",
                queries=batch["query"],
                response=game_data["response"],
                tasks=None,
                name=name,
                qty=qty_print,
                rewards=rewards,
            )

            #### Run PPO training
            stats = ppo_trainer.step(
                queries=query_tensors,
                responses=output["response_tensors"],
                scores=rewards,
            )

            #### Add per ctrl stats
            stats = dict(
                **stats,
                **per_ctrl_stats(
                    ctrl_str=ctrl_str,
                    rewards=rewards,
                    task_list=task_list,
                ),
            )

            ppo_trainer.log_stats(stats, game_data, rewards)

    for ctrl_s in ctrl_str:
        plt.hist(
            [r for r, t in zip(logs["env/reward_dist"], task_list) if t == ctrl_s],
            density=True,
            alpha=0.5,
            label=ctrl_s,
        )

    plt.legend(loc="best")
    plt.title("reward distribution")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
