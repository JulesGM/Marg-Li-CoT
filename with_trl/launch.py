#!/usr/bin/env python
import enum
import os
from pathlib import Path
import shlex
import subprocess

import fire
import nvgpu
import rich
import rich.markup
import rich.panel

import rich.table

import subprocess
import nvgpu


MODULE = "accelerate"
SCRIPT_DIR = Path(__file__).absolute().parent
SCRIPT_PATH = SCRIPT_DIR / "bin_main.py"


class MixedPrecision(enum.Enum):
    NO = "no"
    BF16 = "bf16"
    FP16 = "fp16"


class ConfigName(enum.Enum):
    SENTIMENT = "sentiment"
    ARITHMETIC = "arithmetic"
    GSM8K = "gsm8k"


def count_gpus():
    try:
        return len(nvgpu.gpu_info())
    except subprocess.CalledProcessError:
        return 0
    assert False


def kill_wandb_servers():
    print(subprocess.check_output(
        "kill_wandb",
        shell=True,
        universal_newlines=True,
    ))


def main(
    *,
    one=False, 
    port=29511,
    config_name=ConfigName.ARITHMETIC, 
    accelerate_config_file=SCRIPT_DIR / "accelerate_ddp_no.yaml",
    overloads=None,
    mixed_precision=None,
    **rest,
):

    args = locals().copy()
    assert not rest, rest
    assert mixed_precision is not None, mixed_precision

    config_name = ConfigName(config_name)
    mixed_precision = MixedPrecision(mixed_precision)
    accelerate_config_file = Path(accelerate_config_file)
    assert accelerate_config_file.exists(), accelerate_config_file
    port = int(port)
    assert isinstance(one, bool), one

    table = rich.table.Table.grid()
    for k, v in args.items():
        table.add_row(
            "[bold]" + rich.markup.escape(str(k)) + ":",
            " " + rich.markup.escape(str(v)),
        )
        
    rich.print(
        rich.panel.Panel(
            table,
            title="[bold]CLI Arguments:",
            title_align="left",
            highlight=True
        )
    )
    
    gpu_count = count_gpus()
    if not gpu_count:
        rich.print("[red bold on white] No gpus. Cancelling.")
        return

    accelerate_config_file = Path(accelerate_config_file)
    assert accelerate_config_file.exists(), accelerate_config_file
    num_processes = 1 if one else len(nvgpu.gpu_info())
    
    command = [
        "accelerate",
        "launch",
        "--main_process_port", str(port),
        "--num_processes",     num_processes,
        "--config_file",       accelerate_config_file,
        "--mixed_precision",   mixed_precision.value,
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
        f"--config-name={shlex.quote(config_name.value)}",
    ]

    if overloads:
        command += shlex.split(overloads)
    
    command = list(map(str, command))

    rich.print(rich.panel.Panel(
        shlex.join(command), 
        title="[bold]Running Command:",
        title_align="left",
        highlight=True,
    ))

    os.execvp("accelerate", command)
    

if __name__ == "__main__":
    fire.Fire(main)
