import itertools
import random
from dataclasses import dataclass, field
from typing import Optional

import fire
import numpy as np
import peft
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from trl import (AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer,
                 set_seed)
from trl.core import LengthSampler

set_seed(0)
random.seed(1)
np.random.seed(2)
torch.manual_seed(3)
torch.cuda.manual_seed_all(4)


DEFAULT_GRADIENT_ACCUMULATION_STEPS = 3
DEFAULT_MINI_BATCH_SIZE             = 16
DEFAULT_LEARNING_RATE               = 5e-5
DEFAULT_BATCH_SIZE                  = 3
DEFAULT_MODEL_NAME                  = "google/flan-t5-small"
DEFAULT_PEFT_CONFIG                 = dict(
    r              = 8,
    lora_alpha     = 32,
    task_type      = peft.TaskType.SEQ_2_SEQ_LM,
    lora_dropout   = 0,
    inference_mode = False,
)

def main(
    *,
    gradient_accumulation_steps = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    mini_batch_size             = DEFAULT_MINI_BATCH_SIZE,
    learning_rate               = DEFAULT_LEARNING_RATE,
    peft_config                 = DEFAULT_PEFT_CONFIG,
    model_name                  = DEFAULT_MODEL_NAME,
    batch_size                  = DEFAULT_BATCH_SIZE,
):

    config = PPOConfig(
        gradient_accumulation_steps = gradient_accumulation_steps,
        mini_batch_size             = mini_batch_size,
        learning_rate               = learning_rate,
        model_name                  = model_name,
        batch_size                  = batch_size,
        log_with                    = "wandb",
    )
    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}


    # Below is an example function to build the dataset. In our case, we use the IMDB dataset
    # from the `datasets` library. One should customize this function to train the model on
    # its own dataset.
    def build_imdb_dataset(tokenizer, input_min_text_length=2, input_max_text_length=8):
        # load imdb with datasets
        ds = load_dataset("imdb", split="train")
        ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()] + [tokenizer.eos_token_id]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds


    def collater(data):
        return dict((key, [d[key] for d in data]) for key in data[0])


    # set seed before initializing value head for deterministic eval
    
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    peft.LoraConfig(**peft_config)
    model = peft.


    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_imdb_dataset(tokenizer)

    query = tokenizer("I really liked this movie because", return_tensors="pt")["input_ids"]

    generation_kwargs = {"top_k": 0.0, "top_p": 1.0, "do_sample": True, "eos_token_id": -1}


    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config, 
        model, 
        ref_model, 
        tokenizer, 
        dataset=dataset, data_collator=collater)

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    sentiment_pipe = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=device)

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    output_min_length = 16
    output_max_length = 32
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from t5
        response_tensors = ppo_trainer.generate(
            query_tensors, 
            **generation_kwargs,
        )
        response_tensors = [r[1:] for r in response_tensors]
        batch["response"] = tokenizer.batch_decode(response_tensors)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]).to(device) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)


if __name__ == "__main__":
    fire.Fire(main)