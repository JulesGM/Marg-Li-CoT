#!/usr/bin/env python

"""
Accelerate launcher of the main script.

```bash

python launch.py --config_name=<dataset_name>

```

Dataset names:
    - arithmetic
    - gsm8k



"""

import enum
import os
from pathlib import Path
import shlex
import subprocess

import fire
import more_itertools as mit
import nvgpu
import rich
import rich.markup
import rich.panel

import rich.table

import subprocess
import nvgpu

try:
    import edit_distance
except ImportError:
    edit_distance = None


MODULE = "accelerate"
SCRIPT_DIR = Path(__file__).absolute().parent
SCRIPT_PATH = SCRIPT_DIR / "bin_main.py"


class MixedPrecision(enum.Enum):
    NO   = "no"
    BF16 = "bf16"
    FP16 = "fp16"


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

def list_experiments(experiment_folder: str | Path) -> set[str]:
    experiment_folder = Path(experiment_folder)
    assert experiment_folder.exists(), experiment_folder
    list_experiments_files = experiment_folder.glob("**/*.yaml")
    return set(
        str(x.relative_to(experiment_folder).parent / x.stem) 
        for x in list_experiments_files
    )


def check_experiment_and_suggest(experiment: str, folder: str | Path, top_n=10) -> None:
    assert isinstance(experiment, str), experiment
    experiments = list_experiments(folder)

    if not experiment in experiments: 
        rich.print(f"Experiment [bold]{experiment}[/] not found.")

        if edit_distance:
            exps_with_distances = [
                (possible_experiment, edit_distance.edit_distance(possible_experiment, experiment))
                for possible_experiment in experiments
            ]
            
            exps_with_distances.sort(key=lambda x: x[1])
            sorted_exps, sorted_dists = mit.zip_equal(*exps_with_distances)

            rich.print(f"Did you mean: [bold]{sorted_exps[0]}")
            print(f"Options are:")
            print("\t- " + "\n\t- ".join(sorted_exps[:top_n]))

            if len(sorted_exps) > top_n:
                print(f"\t... and {len(sorted_exps) - top_n} more")

        else:
            print("Options are:")
            print("\n".join(experiments[:top_n]))
            if len(experiments) > top_n:
                print(f"\t... and {len(experiments) - top_n} more")

        exit(-1)


def main(
    experiment,
    *,
    debug=False,
    one=False, 
    port=29511,
    accelerate_config_file=SCRIPT_DIR.parent / "accelerate_configs" / "accelerate_ddp_no.yaml",
    overloads=None,
    mixed_precision="bf16",
    **rest,
):
    """
    Arguments:
        experiment: str 
            The hydra experiment name

        overloads: list[str]
            The hydra overloads.

        one: bool
            If True, run the script in one process. Also tries to use ipdb.
        
        accelerate_config_file:
            The accelerate config file to use.

        mixed_precision: str
            The mixed precision to use with accelerate.
            
        port: int
            The port for the main process.
        

    Last updated: 9/26/2024 by @julesgm
    """
    ###########################################################################
    # Copy and log args.
    ###########################################################################
    args = locals().copy()
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
            highlight=True,
            title_align="left",
        )
    )
    
    check_experiment_and_suggest(experiment, SCRIPT_DIR / "config" / "experiment")

    ###########################################################################
    # Verify and cast args.
    ###########################################################################
    assert not rest, rest
    assert mixed_precision is not None, mixed_precision
    mixed_precision = MixedPrecision(mixed_precision)
    accelerate_config_file = Path(accelerate_config_file)
    assert accelerate_config_file.exists(), accelerate_config_file
    port = int(port)
    assert isinstance(one, bool), one
    
    gpu_count = count_gpus()
    if not gpu_count:
        rich.print("[red bold on white] No gpus. Cancelling.")
        return

    accelerate_config_file = Path(accelerate_config_file)
    assert accelerate_config_file.exists(), accelerate_config_file
    
    ###########################################################################
    # Create the command and run it.
    ###########################################################################
    num_processes = 1 if one else len(nvgpu.gpu_info())
    if debug:
        # Description: Debug mode is just running the script with ipdb.
        # Difference with --one: --one doesn't seem to work with ipdb.
        command = [
            "python",
            "-m",
            "ipdb",
            "-c",
            "continue",
        ]
    else:
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
        f"experiment={shlex.quote(experiment)}",
    ]

    if overloads:
        assert isinstance(overloads, list), overloads
        assert all(isinstance(x, str) for x in overloads), overloads
        command += overloads
    
    command = [str(x) for x in command]

    rich.print(rich.panel.Panel(
        shlex.join(command), 
        title="[bold]Running Command:",
        title_align="left",
        highlight=True,
    ))

    # Replace the current process with the following command.
    os.execvp("accelerate", command)
    

if __name__ == "__main__":
    fire.Fire(main)
