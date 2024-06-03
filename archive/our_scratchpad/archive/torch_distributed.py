#!/usr/bin/env python
from pytorch_lightning.plugins.environments import SLURMEnvironment
import torch
import os

print("ENV")
env = SLURMEnvironment()
print("COMMS")

os.environ["MASTER_ADDR"] = env.main_address
os.environ["MASTER_PORT"] = str(env.main_port)

print(os.environ["MASTER_ADDR"])
print(os.environ["MASTER_PORT"])
print(f"{env.world_size()  = }")
print(f"{env.global_rank() = }")

torch.distributed.init_process_group(
    "nccl", 
    init_method="env://", 
    world_size=env.world_size(), 
    rank=env.global_rank(),
)
print("SEND")

output = [None for _ in range(int(env.world_size()))]
torch.distributed.all_gather_object(output, torch.tensor(torch.distributed.get_rank()))

if torch.distributed.get_rank() == 0:
    print(output)