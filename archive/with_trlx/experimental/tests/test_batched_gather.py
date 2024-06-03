import accelerate
import torch
import fire
import logging
import general_utils

LOGGER = logging.getLogger(__name__)



def main(batch_size=4):
    accelerator = accelerate.Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    tensor = torch.ones(10, 100) * rank
    gathered = general_utils.batched_gather(tensor, batch_size, accelerator)
    print(f"[{rank}/{world_size}]{gathered.shape = }")
    print(f"[{rank}/{world_size}]{gathered[:, 0] = }")


if __name__ == "__main__":
    fire.Fire(main)