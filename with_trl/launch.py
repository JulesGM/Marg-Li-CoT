#!/usr/bin/env python
import os
import shlex
import subprocess
from pathlib import Path

import fire

BIN_PATH = "/home/mila/g/gagnonju/.main/bin/python"
MODULE = "/home/mila/g/gagnonju/.main/bin/accelerate"



def _kill_wandb_servers():
    subprocess.call(
        "pgrep wandb | xargs kill -9",
        shell=True, 
        # stdout=subprocess.STDOUT, 
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
    )


def _kill_other_python_processes():
    subprocess.call(
        f"pgrep python | grep -v {os.getpid()} | xargs kill -9",
        shell=True,
        # stdout=subprocess.STDOUT,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
    )


def check_exists(path):
    path = Path(path)
    assert path.exists(), path


def main(one=False, config_file="accelerate_ddp_no.yaml"):
    if config_file:
        check_exists(config_file)

    check_exists(BIN_PATH)
    check_exists(MODULE)
    _kill_wandb_servers()
    _kill_other_python_processes()

    if one:
        num_processes = 1
    else:
        # Casting it seems useless, but enforces that it should
        # be castable to int, which is a way to constrain what
        # we want
        num_processes = int(os.environ["SLURM_GPUS_ON_NODE"])
    
    command = [
        BIN_PATH, 
        MODULE,
        "launch",
        "--num_processes", str(num_processes),
        "--config_file", config_file, 
        "bin_gptx-neo.py",
    ]

    print(shlex.join(command))
    os.execv(BIN_PATH, command)

if __name__ == "__main__":
    fire.Fire(main)