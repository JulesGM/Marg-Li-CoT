#!/usr/bin/env bash

# Runs all of the lighteval tests on all of the directories 
# that match "$INPUT_DIR/$GLOB_PATTERN" and that have a checkpoint in that directory.

set -euo pipefail

NUM_SHOTS=0
OUTPUT_DIR="./all_eval_outputs_important/sft_outputs_gsm8k/${NUM_SHOTS}_shot"
INPUT_DIR="$HOME/scratch/marglicot_saves/sft_saves/"
mkdir -p "${OUTPUT_DIR}"
TASK_PATH=./util_code/tasks.py
GLOB_PATTERN='*gsm8k*/*/model/'

python multi_gpu_lighteval_chain.py \
--task_key="custom|gsm8k|${NUM_SHOTS}|0" \
--input_path="${INPUT_DIR}" \
--output_dir="${OUTPUT_DIR}" \
--glob_pattern="${GLOB_PATTERN}" \
--dispatch_style=slurm \
--custom_tasks "$(realpath ${TASK_PATH})"
