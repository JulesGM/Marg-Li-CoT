#!/usr/bin/env bash
accelerate launch --no_python python -c 'import accelerate; a = accelerate.Accelerator(); import torch; rank = a.process_index; world_size = torch.distributed.get_world_size(); receptor = [None for _ in range(world_size)]; torch.distributed.all_gather_object(receptor, str(rank) * 50); print(f"[{rank}/{world_size}] {receptor = }")'
