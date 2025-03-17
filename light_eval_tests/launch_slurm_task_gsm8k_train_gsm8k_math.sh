#!/usr/bin/env bash

OUTPUT_DIR=./all_eval_outputs_important/lambdal_outputs_task_gsm8k_train_gsm8k_math/
mkdir -p "${OUTPUT_DIR}"

python multi_gpu_lighteval_chain.py \
--input_path=/home/mila/g/gagnonju/scratch/lambdal_marglicot_openinstruct/open_instruct_output/2025-02-10_19-35-25_rlvr_gsm8k_math_smollm2_instruct_checkpoints \
--task_key="custom|gsm8k|8|0" \
--output_dir="${OUTPUT_DIR}" \
--dispatch_style=slurm \
--custom_tasks "$(realpath ./util_code/tasks.py)"
