# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
import os
from typing import Optional
import rich
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


RANK       = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])


########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################

########################################################################
# NOTE for to train with a 8-bit model a more recent version of
# transformers is required, full dependecies for this example:
# pip install  bitsandbytes datasets accelerate loralib
# pip install  git+https://github.com/huggingface/transformers.git@main
# pip install peft
########################################################################

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.

    model_name: Optional[str]      = "edbeeching/gpt-neo-125M-imdb-lora-adapter-merged"
    log_with: Optional[str]        = None
    learning_rate: Optional[float] = 1.41e-5
    mini_batch_size: Optional[int] = 16
    batch_size: Optional[int]      = 32
    gradient_accumulation_steps    = 1


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    config, 
    *,
    input_min_text_length = 2, 
    input_max_text_length = 8,
    dataset_name          = "imdb", 
):
    """

    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.

    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds



def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param        = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    rich.print(
        f"[bold blue]({RANK}/{WORLD_SIZE}):[/]"
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {trainable_params / all_param:0.5%}"
    )


def main():
    parser      = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    config = PPOConfig(
        gradient_accumulation_steps = script_args.gradient_accumulation_steps,
        mini_batch_size             = script_args.mini_batch_size,
        learning_rate               = script_args.learning_rate,
        batch_size                  = script_args.batch_size,
        model_name                  = script_args.model_name,
        log_with                    = script_args.log_with,
    )

    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = dict(
        return_all_scores = True, 
        function_to_apply = "none", 
        batch_size        = config.mini_batch_size,
    )

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(config)

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    dmap_keys = ["transformer", "lm_head"]
    dmap      = {
        k: LOCAL_RANK 
        for k in dmap_keys
    }
    print("pretrained_model = AutoModelForCausalLM.from_pretrained ...")
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        load_in_8bit=True, 
        device_map=dmap,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    """### Apply LoRA
    Here comes the magic with `peft`! Let's load a `PeftModel` and 
    specify that we are going to use low-rank adapters (LoRA) using 
    `get_peft_model` utility function from `peft`.
    """


    print("pretrained_model = prepare_model_for_int8_training ...")
    pretrained_model = prepare_model_for_int8_training(
        pretrained_model, 
        output_embedding_layer_name="embed_out"
    )
    
    target_modules = None
    if "gpt-neox" in script_args.model_name:
        # workaround to use 8bit training on this model
        # hacky workaround due to issues with "EleutherAI/gpt-neox-20b"
        target_modules = ["query_key_value", "xxx"]  

        for name, param in pretrained_model.named_parameters():
            # freeze base model's layers
            param.requires_grad = False

            if getattr(pretrained_model, "is_loaded_in_8bit", False):
                # cast layer norm in fp32 for stability for 8bit models
                if param.ndim == 1 and "layer_norm" in name:
                    param.data = param.data.to(torch.float16)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,  # handled automatically by peft
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print("pretrained_model = get_peft_model(pretrained_model, lora_config)")
    pretrained_model = get_peft_model(pretrained_model, lora_config)

    print("AutoModelForCausalLMWithValueHead.from_pretrained")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

    model.gradient_checkpointing_disable = model.pretrained_model.gradient_checkpointing_disable
    model.gradient_checkpointing_enable  = model.pretrained_model.gradient_checkpointing_enable

    print_trainable_parameters(model)

    # GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.learning_rate
    )

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config, 
        model, 
        data_collator = collator, 
        ref_model     = None,
        tokenizer     = tokenizer,
        optimizer     = optimizer,
        dataset       = dataset,
    )

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    print("pipeline(...)")
    sentiment_pipe = pipeline(
        "sentiment-analysis", 
        model="lvwerra/distilbert-imdb", 
        device=device,
    )

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": -1,
        "min_length":   -1,
        "do_sample":    True,
        "top_k":        0.0,
        "top_p":        1.0,
    }
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(
        output_min_length, 
        output_max_length,
    )


    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), desc="Epoch"):
        query_tensors = batch["input_ids"]

        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True
        # Get response from Causal LM
        response_tensors = []

        for query in tqdm(query_tensors, "Unrolling."):
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[- gen_len:])

        batch["response"] = [
            tokenizer.decode(r.squeeze()) 
            for r in response_tensors
        ]

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # Run PPO step
        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False

        rich.print(f"[blue bold]Epoch {epoch}:[/] [white bold]ppo_trainer.step")
        stats = ppo_trainer.step(
            responses = response_tensors, 
            queries   = query_tensors, 
            scores    = rewards,
        )

        ppo_trainer.log_stats(
            rewards = rewards,
            stats   = stats, 
            batch   = batch, 
        )



if __name__ == "__main__":
    main()
