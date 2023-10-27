#!/usr/bin/env python
from pathlib import Path
import os
import shlex
import subprocess

import fire
import nvgpu
import rich
import rich.panel

MODULE = "accelerate"
SCRIPT_DIR = Path(__file__).absolute().parent
SCRIPT_PATH = SCRIPT_DIR / "bin_main.py"

def kill_wandb_servers():
    print(subprocess.check_output(
        "kill_wandb",
        shell=True,
        universal_newlines=True,
    ))


def main(name, one=False, config_file=SCRIPT_DIR / "accelerate_ddp_no.yaml"):
    config_file = Path(config_file)
    assert config_file.exists(), config_file
    # kill_wandb_servers()

    num_processes = 1 if one else len(nvgpu.gpu_info())
    
    command = [
        "accelerate",
        "launch",
        "--num_processes", num_processes,
        "--config_file", config_file,
    ]
    if one:
        command += [
            "--no_python",
            "python",
            "-m",
            "ipdb",
            "-c",
            "continue",
        ]

    command += [
        SCRIPT_PATH,
        name,
    ]
    
    command = list(map(str, command))

    rich.print(rich.panel.Panel(
        shlex.join(command), 
        title="[bold]Running Command:",
        title_align="left",
    ))
    os.execvp("accelerate", command)
    

if __name__ == "__main__":
    fire.Fire(main)
