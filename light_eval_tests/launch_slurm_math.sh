#!/usr/bin/env bash

OUTPUT_DIR=./all_eval_outputs_important/lambdal_outputs_math/
mkdir -p "${OUTPUT_DIR}"

python multi_gpu_lighteval_chain.py \
--input_path=/home/mila/g/gagnonju/scratch/lambdal_marglicot_openinstruct/open_instruct_output/2025-02-10_19-33-46_rlvr_math_only_smollm2_instruct_checkpoints \
--task_key="custom|math|5|0" \
--output_dir="${OUTPUT_DIR}" \
--dispatch_style=slurm \
--custom_tasks "$(realpath ./util_code/tasks.py)"
