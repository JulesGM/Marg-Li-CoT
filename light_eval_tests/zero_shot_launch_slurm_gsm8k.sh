#!/usr/bin/env bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task 8
#SBATCH --mem 40GB
#SBATCH --partition long
#SBATCH --output "./all_eval_outputs_important/zero_shot_outputs_gsm8k_important/slurm_logs/%j.out"
#SBATCH --error "./all_eval_outputs_important/zero_shot_outputs_gsm8k_important/slurm_logs/%j.err"

MODEL_NAME="HuggingFaceTB/SmolLM2-1.7B-Instruct"
OUTPUT_DIR=./all_eval_outputs_important/zero_shot_outputs_gsm8k_important/
TASK_PATH=./util_code/tasks.py
mkdir -p "${OUTPUT_DIR}"

ACTIVE_PATH="/home/mila/g/gagnonju/marglicot/light_eval_tests/"


cd ${ACTIVE_PATH} && uv run lighteval \
accelerate \
--model_args \
pretrained=${MODEL_NAME},revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048 \
--tasks 'custom|gsm8k|0|0' \
--output_dir ${OUTPUT_DIR} \
--use_chat_template \
--custom_tasks "$(realpath ${TASK_PATH})" \
--save_details
