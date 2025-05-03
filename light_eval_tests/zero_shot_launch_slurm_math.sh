#!/usr/bin/env bash
#SBATCH --gres=gpu:l40s:1 
#SBATCH --cpus-per-task 8 
#SBATCH --mem 40GB 
#SBATCH --partition long 
#SBATCH --output "./all_eval_outputs_important/zero_shot_outputs_math/slurm_logs/%j.out" 
#SBATCH --error "./all_eval_outputs_important/zero_shot_outputs_math/slurm_logs/%j.err" 

MODEL_NAME="HuggingFaceTB/SmolLM2-1.7B-Instruct"
OUTPUT_DIR=./all_eval_outputs_important/zero_shot_outputs_math/
TASK_PATH=./util_code/tasks_ref.py
mkdir -p "${OUTPUT_DIR}"

ACTIVE_PATH="/home/mila/g/gagnonju/marglicot/light_eval_tests/"
MAX_SEQ_LENGTH=4096
NUM_SHOTS=0
ALLOW_TRUNCATE_SHOTS=1

cd "${ACTIVE_PATH}" && uv run lighteval \
accelerate \
--model_args \
pretrained="${MODEL_NAME},revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=${MAX_SEQ_LENGTH}" \
--tasks "custom|math|${NUM_SHOTS}|${ALLOW_TRUNCATE_SHOTS}" \
--output_dir "${OUTPUT_DIR}" \
--use_chat_template \
--custom_tasks "$(realpath ${TASK_PATH})" \
--save_details