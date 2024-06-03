import contextlib

import accelerate
import fire
import rich
import rich.logging
import logging
import torch


import transformers


LOGGER = logging.getLogger(__name__)

@contextlib.contextmanager
def one_by_one(accelerator):
    for _ in range(accelerator.process_index):
        accelerator.wait_for_everyone()
    yield
    for _ in range(accelerator.num_processes - accelerator.process_index - 1):
        accelerator.wait_for_everyone()


class Container(torch.nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(hf_model)

        def generate(self, **kwargs):
            return self.model.generate(**kwargs)


def main(hf_model: str = "google/flan-t5-large"):
    kwargs = dict(max_new_tokens=200)
    accelerator = accelerate.Accelerator()
    logging.basicConfig(
        handlers=[rich.logging.RichHandler(markup=True)],
        level=logging.INFO,
        format=f"[{accelerator.process_index + 1}/{accelerator.num_processes}]: %(message)s",
    )
    LOGGER.info("Starting up.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
    dataloader = torch.utils.data.DataLoader(
        [
            "Question: What do you think about ADHD? Answer:",
            "Is limp bizkit still good? ",
        ],
        collate_fn=lambda x: tokenizer(x, padding=True, return_tensors="pt"),
        batch_size=2,

    )

    distributed, dl = accelerator.prepare(Container(hf_model), dataloader)
    
    LOGGER.info("Generating.")

    for batch in dl:
        generated = distributed.generate(**(batch | kwargs))
        LOGGER.info(f"generated {type(distributed).mro()}")

        for line in generated:
            LOGGER.info(tokenizer.decode(line))


if __name__ == "__main__":
    fire.Fire(main)