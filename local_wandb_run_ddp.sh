#/usr/bin/env bash
srun bash -c 'WANDB_MODE=dryrun python bin_refine.py --wandb_run_id=None --distribute_strategy=ddp'