import itertools
import logging
import os
import sys

import accelerate
import fire
import rich.logging 
import torch


LOGGER = accelerate.logging.get_logger(
    __name__, 
    log_level="INFO",
)

def main(
    dataset_size=22, 
    dataloader_batch_size=5, 
    n_batches_to_take=3, 
    nccl_debug_level="ERROR",
):
    args = locals().copy()

    # Make nccl less verbose
    os.environ["NCCL_DEBUG"] = nccl_debug_level


    accelerator = accelerate.Accelerator()
    dataloader = torch.utils.data.DataLoader(
        range(dataset_size), batch_size=dataloader_batch_size, shuffle=False)
    accelerated_dataloader = accelerator.prepare_data_loader(dataloader)
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    logging.basicConfig(
        level="INFO", 
        format=f"[{rank} / {world_size}]%(message)s", 
        handlers=[rich.logging.RichHandler(markup=True)]
    )
    # Could definitely do just rank 0 in this case
    LOGGER.info(f"Args: {args}", main_process_only=True)

    a_few_batches = list(itertools.islice(accelerated_dataloader, n_batches_to_take))
    LOGGER.info(f"{a_few_batches}", main_process_only=False)



if __name__ == "__main__":
    print(sys.argv)
    fire.Fire(main)
