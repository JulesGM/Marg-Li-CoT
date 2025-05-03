"""
Reads from a glob pattern, and runs lighteval on each of the paths, with SLURM (or locally, which is deprecated).

- Does test the glob pattern, and only runs the command if there are matches.
- Filters the input paths to only include those with the correct safetensors or pt extension (safetensors is the default).

"""

import concurrent.futures
import enum
import json
import logging
import os
import pathlib
import queue
import re
import shlex
import shutil
import subprocess
import typing

import fire
import nvgpu
import rich


LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = pathlib.Path(__file__).absolute().parent


class DispatchStyle(str, enum.Enum):
    LOCAL = "local"
    SLURM = "slurm"


class SafeTensorsOrPT(str, enum.Enum):
    SAFETENSORS = "safetensors"
    PT = "pt"


def build_command(
        *,
        custom_tasks: str,
        input_path: str | pathlib.Path, 
        output_dir: str | pathlib.Path, 
        task_key: str,
        max_model_length: int = 2048,
    ) -> list[str]:

    return [
        "uv",
        "run",
        "lighteval",
        "accelerate", 
        "--model_args", 
        f"pretrained={shlex.quote(str(input_path))},revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length={max_model_length}", 
        "--tasks", task_key, 
        "--output_dir", str(output_dir), 
        "--use_chat_template", 
        "--custom_tasks", str(custom_tasks), 
        "--save_details"
    ]


def slurm_dispatcher(
        *, 
        custom_tasks: str, 
        input_paths: typing.Sequence[str | pathlib.Path], 
        output_dir: str | pathlib.Path, 
        task_key: str, 
        dry_run: bool,
        max_model_length: int,
        duration: None | str,
    ):

    # Start the work
    for input_path in input_paths:
        
        input_path = pathlib.Path(input_path).expanduser().absolute()
        predicted_results_path = output_dir / "results" / str(input_path).replace("/", "_")
        predicted_results_path.mkdir(parents=True, exist_ok=True)

        # Only run if there are no results yet
        if not list(predicted_results_path.glob("results_*.json")):
            copy_meta_info_json(input_path=input_path, predicted_results_path=predicted_results_path)
            script_command = build_command(
                custom_tasks      = custom_tasks,
                input_path        = input_path, 
                output_dir        = output_dir, 
                task_key          = task_key, 
                max_model_length  = max_model_length,
            )

            slurm_command = [
                    "sbatch",
                    "--gres=gpu:l40s:1", 
                    "--cpus-per-task", 
                    "8", 
                    "--mem", 
                    "40GB", 
                    "--partition", 
                    "long",
                    "--output", f"{output_dir}/slurm_logs/%j.out",
                    "--error", f"{output_dir}/slurm_logs/%j.err",
                    f"--wrap", f"cd {SCRIPT_DIR}; {shlex.join(script_command)}",
                ]
            
            if duration:
                slurm_command.insert(1, f"--time={duration}")

            rich.print(f"[bold green]Submitting:[/] {shlex.join(slurm_command)}")
            if not dry_run:
                subprocess.check_output(slurm_command)
            else:
                rich.print(f"[bold orange1]Dry run:[/] {shlex.join(slurm_command)}")
        else:
            rich.print(f"[bold orange1]Skipping:[/] {input_path} because it already has results")
        print()

    print("Done!")



def extract_input_paths(input_path, glob_pattern, safetensors_or_pt: SafeTensorsOrPT):
    input_path = pathlib.Path(input_path).expanduser().absolute()
    assert input_path.is_dir(), f"Path does not exist: {input_path}"
    input_paths = list(input_path.glob(glob_pattern))

    if not input_paths:
        raise ValueError(f"No input paths found for glob pattern: {input_path}/{glob_pattern}")
    # else:
    #     for path in input_paths:
    #         if not path.is_dir():
    #             raise ValueError(f"Path is not a directory: {path}")
    #         LOGGER.debug(f"- {path}")

    input_paths.sort(key=lambda x: int(re.findall("\d+", str(x))[-1]))

    good_paths = []
    for step_path in input_paths:
        if safetensors_or_pt == SafeTensorsOrPT.SAFETENSORS:
            if list(step_path.glob("model.safetensors")):
                good_paths.append(step_path)
        elif safetensors_or_pt == SafeTensorsOrPT.PT:
            raise NotImplementedError("PT is not implemented")
        else:
            raise ValueError(f"Unknown safetensors_or_pt: {safetensors_or_pt}")

    if not good_paths:
        raise ValueError(f"No paths with {safetensors_or_pt} found for glob pattern: {input_path}/{glob_pattern}")

    return input_paths


def copy_meta_info_json(input_path: pathlib.Path, predicted_results_path: pathlib.Path):
    input_path = pathlib.Path(input_path).expanduser().absolute()
    json_path = input_path.parent / "meta_info.json"

    target_name = "meta_info.json"
    if not json_path.is_file():
        json_path = input_path / "hydra_config.json"
        target_name = "hydra_config.json"
        if not json_path.is_file():
            raise ValueError(f"Neither meta_info.json nor hydra_config.json exists: {input_path}")
    shutil.copy(json_path, predicted_results_path / target_name)


def main(
        *,
        input_path, 
        task_key,
        output_dir,
        custom_tasks,
        max_model_length: int,
        glob_pattern,
        dispatch_style: DispatchStyle = DispatchStyle.SLURM,
        safetensors_or_pt: SafeTensorsOrPT = SafeTensorsOrPT.SAFETENSORS,
        logger_level: int | str = os.environ.get("PYTHON_LOG_LEVEL", logging.DEBUG),
        dry_run: bool = False,
        duration: None | str = None,
    ):

    logging.basicConfig(level=logger_level)

    custom_tasks = pathlib.Path(custom_tasks).expanduser().absolute()
    assert custom_tasks.is_file(), f"Custom tasks file does not exist: {custom_tasks}"

    safetensors_or_pt = SafeTensorsOrPT(safetensors_or_pt)

    assert len(task_key.split("|")) == 4, "Task key must have 4 parts separated by |"
    output_dir = pathlib.Path(output_dir).expanduser().absolute()
    assert output_dir.is_dir(), f"Output directory does not exist: {output_dir}"

    input_paths = extract_input_paths(input_path, glob_pattern, safetensors_or_pt)

    slurm_dispatcher(
        custom_tasks      = custom_tasks,
        input_paths       = input_paths,
        output_dir        = output_dir,
        task_key          = task_key,
        dry_run           = dry_run,
        max_model_length  = max_model_length,
        duration          = duration,
    )


if __name__ == "__main__":
    fire.Fire(main)