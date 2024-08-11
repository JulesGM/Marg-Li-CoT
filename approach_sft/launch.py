#!/usr/bin/env python3

#SBATCH --gres=gpu:a100l:1
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
from typing import Union

import os
import json
import subprocess as sp


def scontrol_show(job_id=None):
  if job_id is None:
    assert "SLURM_JOB_ID" in os.environ, (
        "Either need a value for the argument job_id or a value for the argument $SLURM_JOB_ID"
    )
    job_id = os.environ["SLURM_JOB_ID"]

  output = sp.check_output(["scontrol", "--json", "show", "job", str(job_id)], text=True).strip()
  
  parsed = mit.one(json.loads(output)["jobs"])
  
  return parsed


rich.traceback.install(console=rich.console.Console(force_terminal=True))

def get_script_dir():

    if "SLURM_JOB_ID" in os.environ:
        scontrol_output = scontrol_show()        
        if scontrol_output["batch_flag"]:            
            return pathlib.Path(scontrol_output["current_working_directory"])
    
    return pathlib.Path(__file__).absolute().parent

SCRIPT_DIR = pathlib.Path(get_script_dir())

TARGET_SCRIPT = SCRIPT_DIR / "bin_sft.py"

def main(
    experiment=os.environ.get("EXPERIMENT", None),
    one: bool = False,
    config_path: Union[str, pathlib.Path] = (
        SCRIPT_DIR.parent / "accelerate_configs" / "accelerate_ddp_no.yaml"
    ),
    test_mode: bool = False,
):
    rich.print(locals())

    assert experiment

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

    command = [
        "accelerate",
        "launch",
        "--num_processes", num_processes,
        "--config_file", config_path,
        TARGET_SCRIPT,
        f"run_name={name}",
        f"experiment={experiment}",
        f"test_mode={test_mode}",
    ] 
    
    command = list(map(str, command))
    rich.print(rich.panel.Panel(
        shlex.join(command),
        title="[bold]Running Command:",
        title_align="left",
        expand=True,
    ))

    if not "SLURM_TMPDIR" in os.environ:
        os.environ["SLURM_TMPDIR"] = str(SCRIPT_DIR / "outputs")

    os.execvp("accelerate", command)


if __name__ == "__main__":
    fire.Fire(main)
