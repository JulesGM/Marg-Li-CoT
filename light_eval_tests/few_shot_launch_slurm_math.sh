#!/usr/bin/env bash

MODEL_NAME="HuggingFaceTB/SmolLM2-1.7B-Instruct"
OUTPUT_DIR=./all_eval_outputs_important/few_shot_outputs_math/
TASK_PATH=./util_code/tasks.py

mkdir -p ${OUTPUT_DIR}

sbatch \
--gres=gpu:l40s:1 \
--cpus-per-task 8 \
--mem 40GB \
--partition long \
--output "${OUTPUT_DIR}/slurm_logs/%j.out" \
--error "${OUTPUT_DIR}/slurm_logs/%j.err" \
--wrap="/home/mila/g/gagnonju/.mambaforge/bin/lighteval \
accelerate \
--model_args \
pretrained=${MODEL_NAME},revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048 \
--tasks 'custom|math|5|0' \
--output_dir ${OUTPUT_DIR} \
--use_chat_template \
--custom_tasks $(realpath ${TASK_PATH}) \
--save_details"

