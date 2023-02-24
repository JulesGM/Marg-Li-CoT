import fire
import os
import rich
import torch
import general_utils


def main(backend="nccl"):
    in_keys = {}
    in_values = {}

    for k, v in os.environ.items():
        if backend in k.lower():
            in_keys[k] = v

        if backend in v.lower():
            in_values[k] = v

    
    # rich.print(list(in_keys.keys()))

    """
    rich.print("[bold blue]in_keys:")
    if in_keys:
        general_utils.print_dict(in_keys)
    else:
        print("<None>")
    print("")

    rich.print("[bold blue]in_values:")
    if in_values:
        general_utils.print_dict(in_values)
    else:
        print("<None>")
    print("")
    """
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    print(f"{rank = } / {world_size = }")
    stuff = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(stuff, [rank] * 3)
    print(rank, world_size, stuff)

    stuff = [torch.zeros(4).cuda() for _ in range(world_size)]
    torch.distributed.all_gather(stuff, torch.ones(4).cuda() * rank + 1)
    print(stuff)
    


if __name__ == "__main__":
    fire.Fire(main)
