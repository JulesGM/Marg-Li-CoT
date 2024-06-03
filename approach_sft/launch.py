#!/usr/bin/env python3
""" 
Launches your script with a yaml config & Huggingface Accelerate.


"""
import shlex
import os
import pathlib

import fire
import nvgpu
import rich
import rich.panel
import rich.console
import rich.traceback
from typing import Union

rich.traceback.install(console=rich.console.Console(force_terminal=True))
SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
TARGET_SCRIPT = SCRIPT_DIR / "bin_sft.py"


def main(
    name: str,
    one: bool = False,
    config_path: Union[str, pathlib.Path] = SCRIPT_DIR.parent / "accelerate_configs" / "accelerate_ddp_no.yaml",
):
    config_path = pathlib.Path(config_path)
    assert config_path.exists(), config_path
    assert TARGET_SCRIPT.exists(), TARGET_SCRIPT

    num_processes = 1 if one else len(nvgpu.gpu_info())

    command = [
        "accelerate",
        "launch",
        "--num_processes", num_processes,
        "--config_file", config_path,
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
