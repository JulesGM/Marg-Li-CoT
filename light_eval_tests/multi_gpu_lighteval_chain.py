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

class DispatchStyle(str, enum.Enum):
    LOCAL = "local"
    SLURM = "slurm"



class SafeTensorsOrPT(str, enum.Enum):
    SAFETENSORS = "safetensors"
    PT = "pt"


def build_command(input_path: typing.Union[str, pathlib.Path], task_key: str, output_dir: typing.Union[str, pathlib.Path], custom_tasks: str):

    return [
        "/home/mila/g/gagnonju/.mambaforge/bin/lighteval", 
        "accelerate", 
        "--model_args", 
        f"pretrained={shlex.quote(str(input_path))},revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048", 
        "--tasks", task_key, 
        "--output_dir", str(output_dir), 
        "--use_chat_template", 
        "--custom_tasks", str(custom_tasks), 
        "--save_details"
    ]


def slurm_dispatcher(*, input_paths, output_dir, task_key, custom_tasks):
    # Start the work
    for input_path in input_paths:
        
        input_path = pathlib.Path(input_path).expanduser().resolve()
        predicted_results_path = output_dir / "results" / str(input_path).replace("/", "_")
        predicted_results_path.mkdir(parents=True, exist_ok=True)

        # Only run if there are no results yet
        if not list(predicted_results_path.glob("results_*.json")):
            copy_meta_info_json(input_path=input_path, predicted_results_path=predicted_results_path)
            script_command = build_command(input_path=input_path, task_key=task_key, output_dir=output_dir, custom_tasks=custom_tasks)
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
                    f"--wrap={shlex.join(script_command)}",
                ]
            rich.print(f"[bold green]Submitting:[/] {shlex.join(slurm_command)}")
            subprocess.check_output(slurm_command)
        else:
            rich.print(f"[bold orange1]Skipping:[/] {input_path} because it already has results")
        print()

    print("Done!")


def local_worker(
        gpu_id: int, 
        gpu_queue: queue.Queue[int], 
        output_dir: str | pathlib.Path,
        path: str | pathlib.Path, 
        task_key: str, 
        verbose: bool,
        custom_tasks: str,
    ):
    raise NotImplementedError("Deprecated; not tested in a while.")

    print(f"[{gpu_id}] Starting with: {path}")
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Blocks here.
    subprocess.run(
        command = build_command(path=path, task_key=task_key, output_dir=output_dir, custom_tasks=custom_tasks), 
        env=env_vars,
        # **extra_args,
    )
    
    print(f"[{gpu_id}] Done with: {path}")
    gpu_queue.put(gpu_id)


def local_dispatcher(*, input_paths, output_dir, task_key, custom_tasks, threads_verbose):
    raise NotImplementedError("Deprecated; not tested in a while.")
    # Put the gpu ids in the queue
    num_gpus = len(nvgpu.gpu_info())
    print(f"Number of GPUs: {num_gpus}")

    # Prepare the gpu ids
    if not gpu_ids:
        gpu_ids = list(range(num_gpus))

    assert all(isinstance(i, int) for i in gpu_ids), (
        f"All gpu ids must be integers, got {gpu_ids} {[type(i) for i in gpu_ids]}"
    )
    print(f"Using GPUs: {gpu_ids}")

    # Queue the gpu ids
    gpu_queue = queue.Queue()
    for i in gpu_ids:
        gpu_queue.put(i)
    print()


    # Start the work
    with concurrent.futures.ThreadPoolExecutor(num_gpus) as executor:
        for path in input_paths:
            gpu_id = gpu_queue.get()
            executor.submit(
                local_worker,
                gpu_id=gpu_id, 
                gpu_queue=gpu_queue, 
                output_dir=output_dir,
                path=path, 
                task_key=task_key,
                custom_tasks=custom_tasks,
                verbose=threads_verbose,
            )

    print("Done!")


def extract_input_paths(input_path, glob_pattern, safetensors_or_pt: SafeTensorsOrPT):
    input_path = pathlib.Path(input_path).expanduser().resolve()
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
    input_path = pathlib.Path(input_path).expanduser().resolve()
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
        glob_pattern="step_*",
        threads_verbose=True,
        dispatch_style: DispatchStyle = DispatchStyle.SLURM,
        safetensors_or_pt: SafeTensorsOrPT = SafeTensorsOrPT.SAFETENSORS,
        logger_level: int | str = os.environ.get("PYTHON_LOG_LEVEL", logging.DEBUG),
    ):

    logging.basicConfig(level=logger_level)

    custom_tasks = pathlib.Path(custom_tasks).expanduser().resolve()
    assert custom_tasks.is_file(), f"Custom tasks file does not exist: {custom_tasks}"

    safetensors_or_pt = SafeTensorsOrPT(safetensors_or_pt)
    dispatch_style = DispatchStyle(dispatch_style)

    assert len(task_key.split("|")) == 4, "Task key must have 4 parts separated by |"
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    assert output_dir.is_dir(), f"Output directory does not exist: {output_dir}"

    input_paths = extract_input_paths(input_path, glob_pattern, safetensors_or_pt)

    if dispatch_style == DispatchStyle.LOCAL:
        local_dispatcher(
            input_paths=input_paths,
            output_dir=output_dir,
            task_key=task_key,
            custom_tasks=custom_tasks,
            threads_verbose=threads_verbose,
        )
    elif dispatch_style == DispatchStyle.SLURM:
        slurm_dispatcher(
            input_paths=input_paths,
            output_dir=output_dir,
            task_key=task_key,
            custom_tasks=custom_tasks,
        )
    else:
        raise ValueError(f"Unknown dispatch style: {dispatch_style}")


if __name__ == "__main__":
    fire.Fire(main)