#!/usr/bin/env python3

import shlex
import os
from pathlib import Path

import fire
import nvgpu
import rich.console
import rich.panel
import rich.traceback

rich.traceback.install(console=rich.console.Console(force_terminal=True))

SCRIPT_DIR = Path(__file__).absolute().parent
TARGET_SCRIPT = SCRIPT_DIR / "bin_value_pretrain.py"


def main(
    name, 
    one=False, 
    config=SCRIPT_DIR.parent / "accelerate_ddp_no.yaml",
):
    config = Path(config)
    assert config.exists(), config
    assert TARGET_SCRIPT.exists(), TARGET_SCRIPT

    num_processes = 1 if one else len(nvgpu.gpu_info())

    command = [
        "accelerate",
        "launch",
        "--num_processes", num_processes,
        "--config_file", config,
        TARGET_SCRIPT,
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