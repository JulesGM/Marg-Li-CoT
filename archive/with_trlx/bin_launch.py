#!/usr/bin/env python
import os
import itertools
import fire
import rich
import subprocess
import pretty_traceback
pretty_traceback.install()


DEFAULT_ONE                     = False
DEFAULT_SERVER_PORT             = 29505
DEFAULT_VAL_SUBSET_SIZE         = 3
DEFAULT_ACCELERATE_CONFIG       = "accelerate_ddp_no.yaml"
DEFAULT_MIXED_PRECISION_DEFAULT = "no"


def _check_mixed_precision_compatibility(default):
    """
    If any of the GPUs is not an A100, disable bf16.
    """

    # Extract the GPU models
    gpu_models = subprocess.check_output(
        "nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq", 
        shell=True, universal_newlines=True
    ).strip().split("\n")

    # If any of the GPUs is not a A100, disable bf16
    for model in gpu_models:
        if "a100" not in model:
            rich.print(f"[bold blue]Disabling bf16 because of GPU model: {model}")
            return "no"

    return default

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

def dict_to_command_list(d):
    return itertools.chain.from_iterable(
        [[f"--{k}", v] for k, v in d.items()]
    )

def _build_command(
        *,
        bin_path, 
        script_config,
        accelerate_bin, 
        accelerate_config, 
        accel_config,
    ):
    if not os.path.exists(bin_path):
        raise RuntimeError(f"Expected bin path to exist: {bin_path}")
    
    command = accelerate_bin + ["launch"]                   # Adds srun if multi node
    command.extend(dict_to_command_list(accelerate_config)) # Adds nodes & procs info
    command.extend(["--config_file", accel_config]) # Adds specific args
    command.append(bin_path)                                # Adds script path 
    command.extend(dict_to_command_list(script_config))     # Adds script args
    command = [str(c) for c in command]

    return command

def _build_accelerate_config(one, server_port, mixed_precision):

    accelerate_path = subprocess.check_output(
        "which accelerate",
        shell              = True,
        universal_newlines = True,
    ).strip()

    srun_path = subprocess.check_output(
        "which srun",
        shell              = True, 
        universal_newlines = True,
    ).strip()

    total_processes = (
        int(os.environ["SLURM_NNODES"      ]) * 
        int(os.environ["SLURM_GPUS_ON_NODE"])
    )
    server_hostname = os.environ["SLURMD_NODENAME"]
    num_nodes       = int(os.environ["SLURM_NNODES"])

    conditional_args_single_node = {}
    conditional_args_multi_node  = {"deepspeed_multinode_launcher": "standard"}
    accelerate_bin_single_node   = [accelerate_path]
    accelerate_bin_multi_node    = [srun_path, accelerate_path]

    if num_nodes > 1 and not one:
        accelerate_bin              = accelerate_bin_multi_node
        accelerate_conditional_args = conditional_args_multi_node
    else:
        accelerate_bin              = accelerate_bin_single_node
        accelerate_conditional_args = conditional_args_single_node

    if one:
        rich.print("[bold yellow]Only using one process, to debug.")
        num_nodes        = 1
        accelerate_bin   = accelerate_bin_single_node
        total_processes  = 1
        server_hostname  = ""
        accelerate_conditional_args = conditional_args_single_node

    accelerate_config = {
        "main_process_port": server_port,
        "mixed_precision":   mixed_precision,
        "main_process_ip":   server_hostname,
        "num_processes":     total_processes,
        "num_machines":      num_nodes,
    }

    accelerate_config = dict(
        **accelerate_config, 
        **accelerate_conditional_args,
    )
    
    return accelerate_bin, accelerate_config


def main(
        one                     = DEFAULT_ONE,
        bin_path                = f"{os.getcwd()}/bin_exp.py",
        server_port             = DEFAULT_SERVER_PORT,
        val_subset_size         = DEFAULT_VAL_SUBSET_SIZE, 
        default_accel_config    = DEFAULT_ACCELERATE_CONFIG,
        mixed_precision_default = DEFAULT_MIXED_PRECISION_DEFAULT,
    ):

    # Check mixed precision compatibility
    mixed_precision = _check_mixed_precision_compatibility(
        mixed_precision_default)
    del mixed_precision_default
    
    # Build accelerate_bin and accelerate_config
    accelerate_bin, accelerate_config = _build_accelerate_config(
        one=one, server_port=server_port, mixed_precision=mixed_precision)
    del one, server_port, mixed_precision
    
    # Build script_config
    if val_subset_size is not None:
        rich.print(f"[bold red]>>> Using a subset! Of size: {val_subset_size}")
    script_config = {"val_subset_size": val_subset_size}
    del val_subset_size

    # Kill all wandb servers
    _kill_wandb_servers()
    
    # Kill all python processes
    _kill_other_python_processes()

    # Run the training
    command = _build_command(
        bin_path             = bin_path,
        script_config        = script_config,
        accelerate_bin       = accelerate_bin,
        accelerate_config    = accelerate_config, 
        accel_config         = default_accel_config,
    )
    del bin_path, accelerate_bin, accelerate_config, script_config

    rich.print("[bold green]>>> Running command:")
    rich.print(f"[bold]{command}")
    os.execv(command[0], command)


if __name__ == "__main__":
    fire.Fire(main)