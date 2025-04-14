#!/usr/bin/env bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task 8
#SBATCH --mem 40GB
#SBATCH --partition long
#SBATCH --output "./all_eval_outputs_important/zero_shot_outputs_gsm8k_important/slurm_logs/%j.out"
#SBATCH --error "./all_eval_outputs_important/zero_shot_outputs_gsm8k_important/slurm_logs/%j.err"

set -euo pipefail

N_SHOTS=5
MODEL_NAME="HuggingFaceTB/SmolLM2-1.7B-Instruct"
OUTPUT_DIR="./all_eval_outputs_important/few_shot_outputs_gsm8k_important/"
ACTIVE_PATH="/home/mila/g/gagnonju/marglicot/metrics_debug_lighteval"
TASK_PATH=./tasks.py
SEQ_LEN=4096
ALLOW_TRUNCATION=1
MEMORY_UTILISATION=0.7
TASK="gsm8k"

mkdir -p "${OUTPUT_DIR}"
cd ${ACTIVE_PATH} || exit 1
unset VIRTUAL_ENV
uv run lighteval accelerate \
--model_args pretrained="${MODEL_NAME}",revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation="${MEMORY_UTILISATION}",max_model_length="${SEQ_LEN}" \
--tasks "custom|${TASK}|${N_SHOTS}|${ALLOW_TRUNCATION}" \
--output_dir "${OUTPUT_DIR}" \
--use_chat_template \
--custom_tasks "$(realpath ${TASK_PATH})" \
--save_details

