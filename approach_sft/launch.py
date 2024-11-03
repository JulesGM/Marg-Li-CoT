#!/usr/bin/env python3

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name="<name>"
#SBATCH --partition=main

""" 
Launches your script with a yaml config & Huggingface Accelerate.

python launch.py answer_only

python launch.py outlines

"""
import subprocess as sp
import shlex
import os
import pathlib

import fire
import more_itertools as mit
import pynvml
import rich
import rich.panel
import rich.console
import rich.traceback
import sys
from typing import Union

import os
import json
import subprocess as sp

rich.traceback.install(console=rich.console.Console(markup=True))


def scontrol_show(job_id=None):
    if job_id is None:
        assert "SLURM_JOB_ID" in os.environ, (
            "Either need a value for the argument job_id or a value for the argument $SLURM_JOB_ID"
        )
        job_id = os.environ["SLURM_JOB_ID"]

    output = sp.check_output([
        "scontrol", 
        "--json", 
        "show", 
        "job", 
        str(job_id)
    ], text=True).strip()
    
    parsed = mit.one(json.loads(output)["jobs"])
    
    return parsed


def get_script_dir():

    if "SLURM_JOB_ID" in os.environ:
        scontrol_output = scontrol_show()        
        if scontrol_output["batch_flag"]:            
            return pathlib.Path(scontrol_output["current_working_directory"])
    
    return pathlib.Path(__file__).absolute().parent


# Add general to the path, & import find_experiment
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent # pathlib.Path(get_script_dir())
TARGET_SCRIPT = SCRIPT_DIR / "bin_sft.py"
GENERAL_DIR = SCRIPT_DIR.parent / "general"
assert GENERAL_DIR.exists(), GENERAL_DIR
sys.path.append(str(GENERAL_DIR))
import find_experiment


def main(
    experiment=os.environ.get("EXPERIMENT", None),
    one: bool = False,
    config_path: Union[str, pathlib.Path] = (
        SCRIPT_DIR.parent / "accelerate_configs" / "most_recent_ddp_bf16.yaml" # "accelerate_ddp_no.yaml" # "ddp_bf16.yaml"
    ),
    test_mode: bool = False,
    wandb_id: str = None,
):
    rich.print(locals())

    experiment = find_experiment.check_experiment_and_suggest(
        experiment, SCRIPT_DIR / "config" / "experiment"
    )

    config_path = pathlib.Path(config_path)
    assert config_path.exists(), config_path
    assert TARGET_SCRIPT.exists(), TARGET_SCRIPT

    if experiment.startswith("experiment="):
        experiment = experiment.split("=", 1)[1].strip()

    name = f"{experiment}_{os.environ['SLURM_JOB_ID']}"

    pynvml.nvmlInit()
    # compute the number of gpus
    assert isinstance(one, bool), one
    num_processes = 1 if one else pynvml.nvmlDeviceGetCount()

    import random
    command = [
        "accelerate",
        "launch",
        "--num_processes", num_processes,
        "--config_file", config_path,
        "--num_machines", 1,
        "--main_process_port", 
        29500 + (int(os.environ["SLURM_JOB_ID"]) + random.randint(0, 1234)) % 2000,
        TARGET_SCRIPT,
        f"run_name={name}",
        f"experiment={experiment}",
        f"test_mode={test_mode}",
    ] 
    command = [str(x) for x in command]
    
    rich.print(rich.panel.Panel(
        shlex.join(command),
        title="[bold]Running Command:",
        title_align="left",
        expand=True,
    ))

    if not "SLURM_TMPDIR" in os.environ:
        os.environ["SLURM_TMPDIR"] = str(SCRIPT_DIR / "outputs")

    if wandb_id:
        os.environ["WANDB_RUN_ID"] = wandb_id

    os.execvp("accelerate", command)


if __name__ == "__main__":
    fire.Fire(main)
