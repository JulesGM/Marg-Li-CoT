import fire
import concurrent.futures
import pathlib
import subprocess
import queue
import nvgpu
import shlex
import re
import os

# CUDA_VISIBLE_DEVICES=3 lighteval accelerate \
# --model_args="pretrained=${CHECKPOINT_PATH},revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048" \
# --tasks="custom|math|5|0" \
# --output_dir=./outputs_math/ \
# --use_chat_template \
# --custom_tasks "./tasks.py" \
# --save_details


def work(
        gpu_id: int, 
        gpu_queue: queue.Queue[int], 
        output_dir: str | pathlib.Path,
        path: str | pathlib.Path, 
        task_key: str, 
        verbose: bool,
    ):
    extra_args = {}
    if not verbose:
        extra_args = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, 
        )

    print(f"[{gpu_id}] Starting with: {path}")
    proc = subprocess.run([
        "lighteval", "accelerate", 
        "--model_args", 
        f"pretrained={shlex.quote(str(path))},revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048", 
        "--tasks", task_key, 
        "--output_dir", str(output_dir), 
        "--use_chat_template", 
        "--custom_tasks", "./tasks.py", 
        "--save_details"
    ], 
    env=dict(**os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id)),
    check=True,
    **extra_args,
    )

    print(f"[{gpu_id}] Done with: {path}")
    gpu_queue.put(gpu_id)


def main(
        input_path="/network/scratch/g/gagnonju/open_instruct_output/2024-12-31_21-22-51_rlvr_gsm8k_only_smollm2_instruct_checkpoints/", 
        task_key="custom|math|5|0",
        output_dir="./outputs_math/",
        threads_verbose=True,
    ):
    assert len(task_key.split("|")) == 4, "Task key must have 4 parts separated by |"

    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    assert output_dir.is_dir(), f"Output directory does not exist: {output_dir}"

    input_path = pathlib.Path(input_path).expanduser().resolve()
    assert input_path.is_dir(), f"Path does not exist: {input_path}"
    input_paths = list(input_path.glob("step_*"))
    input_paths.sort(key=lambda x: int(re.findall("\d+", x.name)[-1]))
    for step_path in input_paths:
        print(f"- {step_path}")
    print()

    num_gpus = len(nvgpu.gpu_info())
    print(f"Number of GPUs: {num_gpus}")
    gpu_queue = queue.Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)
    print()

    with concurrent.futures.ThreadPoolExecutor(num_gpus) as executor:
        for path in input_paths:
            print(path.name)

            gpu_id = gpu_queue.get()
            executor.submit(
                work, 
                gpu_id=gpu_id, 
                gpu_queue=gpu_queue, 
                output_dir=output_dir,
                path=path, 
                task_key=task_key,
                verbose=threads_verbose,
            )
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)