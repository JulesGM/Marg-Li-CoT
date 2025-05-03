#!/usr/bin/env bash

# Runs all of the lighteval tests on all of the directories 
# that match "$INPUT_DIR/$GLOB_PATTERN" and that have a checkpoint in that directory.

set -euo pipefail

NUM_SHOTS_TRAINED=0

NUM_SHOTS_EVAL=0
OUTPUT_DIR="./all_eval_outputs_important/rejection_sampling_outputs_math_${NUM_SHOTS_TRAINED}/${NUM_SHOTS_EVAL}_shot"
INPUT_DIR="$HOME/scratch/rejection_sampling_saves/hendrycks_math/"
mkdir -p "${OUTPUT_DIR}"
TASK_PATH=./util_code/tasks.py
GLOB_PATTERN='*/*/'

python multi_gpu_lighteval_chain.py \
--task_key="custom|math|${NUM_SHOTS_EVAL}|1" \
--input_path="${INPUT_DIR}" \
--output_dir="${OUTPUT_DIR}" \
--glob_pattern="${GLOB_PATTERN}" \
--dispatch_style=slurm \
--custom_tasks "$(realpath ${TASK_PATH})"
