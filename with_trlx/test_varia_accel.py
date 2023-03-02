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
    ), all_data

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

    single_accel = accelerate.Accelerator()

    if single_accel.is_main_process:
        _, all_data = make_dataloader(
            tokenizer=tok, 
            batch_size=batch_size, 
            num_workers=num_workers,
        )
        size = len(all_data) // world_size
        assert isinstance(all_data   , list), f"{type(all_data   ).mro() = }"
        assert isinstance(all_data[0], str ), f"{type(all_data[0]).mro() = }"
        all_data = [all_data[i * size:(i + 1) * size] for i in range(world_size)]
    else:
        all_data = [None] * world_size
    
    output_list = [None]
    torch.distributed.scatter_object_list(
        scatter_object_output_list=output_list,
        scatter_object_input_list=all_data,
        src=0,
    )

    LOGGER.info(f"{[(len(x) if x is not None else x) for x in all_data]}")
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