import accelerate
import datasets
import transformers
import torch
import numpy as np

import logging
import rich.logging
import fire

import itertools
import os

import general_utils


def info(message):
    general_utils.parallel_log(LOGGER, level=logging.INFO, message=message)


LOGGER = logging.getLogger(__name__)


def make_dataloader(*, tokenizer, batch_size, num_workers):
    all_data = datasets.load_dataset("gsm8k", "main", split="train")["question"]
    return torch.utils.data.DataLoader(
        all_data,
        collate_fn=lambda batch: tokenizer(batch, return_tensors="pt", padding=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

def main(
        hf_model_name = "google/flan-t5-small",
        batch_size = 3,
        num_workers = 0,
        gen_kwargs = {
            "max_new_tokens": 100,
        },
        log_level=logging.INFO,
    ):

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 0))

    logging.basicConfig(
        level=log_level,
        format=f"[bright_black][{rank}/{world_size}] %(name)s -[/] %(message)s ",
        handlers=[rich.logging.RichHandler(markup=True)],
    )

    model_1 = transformers.AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
    model_2 = transformers.AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
    tok = transformers.AutoTokenizer.from_pretrained(hf_model_name)
    dataloader = make_dataloader(tokenizer=tok, batch_size=batch_size, num_workers=num_workers)

    single_accel = accelerate.Accelerator()

    lol = torch.ones(3) * rank
    lol = lol.to(single_accel.device)
    all_lols = [
        torch.empty(3, dtype=lol.dtype).to(single_accel.device) 
        for _ in range(world_size)
    ]
    torch.distributed.all_gather(all_lols, lol)
    # info(str(all_lols))
    output = single_accel.gather(lol)
    info(str(output))

    exit()

    two_models = [model_1, model_2]
    *two_models, dataloader = single_accel.prepare(*two_models, dataloader)
    model_1 = two_models[0]
    model_2 = two_models[1]
    assert model_1.device == model_2.device == single_accel.device, (
        f"model_1.device {model_1.device} != model_2.device {model_2.device} != single_accel.device {single_accel.device}"
    )

    batch = next(iter(dataloader))

    with torch.no_grad():
        model_1.eval()
        model_2.eval()

        model_1_gen = model_1.generate(**batch, **gen_kwargs)
        model_2_gen = model_2.generate(**batch, **gen_kwargs)

    for i, model_1_gen_i in enumerate(model_1_gen):
        model_1_gen_i_str = tok.decode(model_1_gen_i, skip_special_tokens=True)
        info(f"model_1_gen_i {i}:\n{model_1_gen_i_str}")

    for i, model_2_gen_i in enumerate(model_2_gen):
        model_2_gen_i_str = tok.decode(model_2_gen_i, skip_special_tokens=True)
        info(f"model_2_gen_i {i}:\n{model_2_gen_i_str}")

    info("Done.")




if __name__ == "__main__":
    fire.Fire(main)