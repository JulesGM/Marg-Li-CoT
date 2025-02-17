# In real life, this should probably be done with SLURM
import concurrent.futures
import enum
import os
import pathlib
import queue
import re
import shlex
import subprocess
import typing

import fire
import nvgpu
import rich


class DispatchStyle(str, enum.Enum):
    LOCAL = "local"
    SLURM = "slurm"


def build_command(path: typing.Union[str, pathlib.Path], task_key: str, output_dir: typing.Union[str, pathlib.Path]):
    return [
        "/home/mila/g/gagnonju/.mambaforge/bin/lighteval", 
        "accelerate", 
        "--model_args", 
        f"pretrained={shlex.quote(str(path))},revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048", 
        "--tasks", task_key, 
        "--output_dir", str(output_dir), 
        "--use_chat_template", 
        "--custom_tasks", "/home/mila/g/gagnonju/marglicot/with_open-instruct/light_eval_tests/util_code/tasks.py", 
        "--save_details"
    ]


def slurm_dispatcher(*, input_paths, output_dir, task_key):
    # Start the work
    for path in input_paths:
        script_command = build_command(path=path, task_key=task_key, output_dir=output_dir)
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
        rich.print(f"Submitting: {shlex.join(slurm_command)}")
        output = subprocess.check_output(slurm_command)
        print(output)

    print("Done!")



def local_worker(
        gpu_id: int, 
        gpu_queue: queue.Queue[int], 
        output_dir: str | pathlib.Path,
        path: str | pathlib.Path, 
        task_key: str, 
        verbose: bool,
    ):


    print(f"[{gpu_id}] Starting with: {path}")
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Blocks here.
    subprocess.run(
        command = build_command(path=path, task_key=task_key, output_dir=output_dir), 
        env=env_vars,
        # **extra_args,
    )
    
    print(f"[{gpu_id}] Done with: {path}")
    gpu_queue.put(gpu_id)


def local_dispatcher(*, input_paths, output_dir, task_key, threads_verbose):
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
                verbose=threads_verbose,
            )

    print("Done!")


# /home/mila/g/gagnonju/scratch/lambdal_marglicot_openinstruct/open_instruct_output/2025-02-10_19-33-46_rlvr_math_only_smollm2_instruct_checkpoints
# /home/mila/g/gagnonju/scratch/lambdal_marglicot_openinstruct/open_instruct_output/2025-02-10_19-35-25_rlvr_gsm8k_math_smollm2_instruct_checkpoints
# /home/mila/g/gagnonju/scratch/lambdal_marglicot_openinstruct/open_instruct_output/2025-02-10_19-32-40_rlvr_gsm8k_only_smollm2_instruct_checkpoints


def extract_input_paths(input_path):
    input_path = pathlib.Path(input_path).expanduser().resolve()
    assert input_path.is_dir(), f"Path does not exist: {input_path}"
    input_paths = list(input_path.glob("step_*"))
    input_paths.sort(key=lambda x: int(re.findall("\d+", x.name)[-1]))
    for step_path in input_paths:
        print(f"- {step_path}")
    print()
    return input_paths


def main(
        input_path, 
        task_key,
        output_dir,
        threads_verbose=True,
        dispatch_style: DispatchStyle = DispatchStyle.SLURM,
    ):
    dispatch_style = DispatchStyle(dispatch_style)

    assert len(task_key.split("|")) == 4, "Task key must have 4 parts separated by |"
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    assert output_dir.is_dir(), f"Output directory does not exist: {output_dir}"

    input_paths = extract_input_paths(input_path)

    if dispatch_style == DispatchStyle.LOCAL:
        local_dispatcher(
            input_paths=input_paths, 
            output_dir=output_dir, 
            task_key=task_key, 
            threads_verbose=threads_verbose, 
        )
    elif dispatch_style == DispatchStyle.SLURM:
        slurm_dispatcher(
            input_paths=input_paths, 
            output_dir=output_dir, 
            task_key=task_key, 
        )
    else:
        raise ValueError(f"Unknown dispatch style: {dispatch_style}")


if __name__ == "__main__":
    fire.Fire(main)