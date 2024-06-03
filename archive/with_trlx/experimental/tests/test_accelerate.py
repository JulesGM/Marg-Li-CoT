print("Doing imports.")
import contextlib
import logging
import os
import random

import accelerate
import datasets
import fire
import rich
import rich.logging
import time
import torch
import transformers

from general_utils import parallel_log
print("Done with imports.")


LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def one_by_one(accelerator):

    rank = accelerator.process_index
    world_size = accelerator.num_processes

    for i in range(world_size):
        if i == rank:
            yield
        accelerator.wait_for_everyone()

def main(
    qty=1000, 
    batch_size=3, 
    log_level=logging.INFO, 
    hf_name="google/flan-t5-small",
    hf_class=transformers.T5ForConditionalGeneration,
    max_new_tokens=100,
    num_workers=0,
    tokenizers_parallelism=False,
    fixed_seed=True,
):
    # Set some constants
    args = locals().copy()
    a9r = accelerate.Accelerator()
    # Init logging
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    logging.basicConfig(
        level=log_level,
        format=f"[{rank}/{world_size}] %(name)s - %(message)s ",
        handlers=[rich.logging.RichHandler(markup=True)],
    )

    def info(message):
        parallel_log(LOGGER, level=logging.INFO, message=message)

    info(f"Args: {args}")

    for k, v in os.environ.items():
        if "accelerate" in k.lower():
            info(f"{k} = {v}")

    os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizers_parallelism else "false"
    if fixed_seed:
        random.seed(0)

    # Init accelerate
    assert rank       == a9r.process_index, (world_size, a9r.num_processes)
    assert world_size == a9r.num_processes, (world_size, a9r.num_processes)

    # Init model, tokenizer and data
    info(f"Loading model `{hf_name}`.")
    model = hf_class.from_pretrained(hf_name)
    info("Done loading model.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_name)
    all_data = datasets.load_dataset("gsm8k", "main", split="train")["question"][:qty]
    all_tokenized = tokenizer(all_data, return_tensors="pt", padding=True)
    list_of_dicts = []

    for inputs, masks in zip(all_tokenized["input_ids"], all_tokenized["attention_mask"]):
        list_of_dicts.append(dict(input_ids=inputs, attention_mask=masks))

    dataloader = torch.utils.data.DataLoader(
        all_data,
        collate_fn=lambda batch: tokenizer(batch, return_tensors="pt", padding=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model, dataloader = a9r.prepare(model, dataloader)
    batch = next(iter(dataloader))
    info(f"{batch['input_ids'].shape = }")

    info(str(type(dataloader).__name__))
    assert all(len(v) == batch_size for v in batch.values()), (len(batch), batch_size)
    assert "input_ids"      in batch, "{rank = } {world_size = }"
    assert "attention_mask" in batch, "{rank = } {world_size = }"
    info({k: v.shape for k, v in batch.items()})
    assert batch["input_ids"].shape[0] == batch_size, (batch["input_ids"].shape[0], batch_size)    
    info(str(batch.keys()))    
    info(f"{max_new_tokens = }")
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        synced_gpus = int(int(os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "0"))) == 3

        outputs = model.generate(
            **batch, 
            max_length=max_new_tokens,
            synced_gpus=synced_gpus,
        )
        torch.cuda.synchronize()
        delta = time.perf_counter() - start

    info(f"{delta = :0.4}")
    info(f"{outputs.shape = }")

    text = []
    for question, output in zip(batch.input_ids, outputs):
        question_text = tokenizer.decode(question).replace("<pad>", "") 
        question_output = tokenizer.decode(output).replace("<pad>", "") 
        text.append(
            "[bold blue]Question:[white]\n" +
            f"{question_text.strip()}\n" +
            "[/][bold blue]Answer:[white]\n" +
            f"{question_output.strip()}\n[/]"
        )
    info("\n" + "\n".join(text))
    

if __name__ == "__main__":
    fire.Fire(main)
