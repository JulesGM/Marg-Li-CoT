#!/usr/bin/env bash
# Runs all of the lighteval tests on all of the directories 
# that match "$INPUT_DIR/$GLOB_PATTERN" and that have a checkpoint in that directory.

set -euo pipefail

NUM_SHOTS=4
ALLOW_TRUNCATE_SHOTS=1
# DATE="2025-04-06"
DATE="2025-04-13"
OUTPUT_DIR="./all_eval_outputs_important/sft_outputs_math/${NUM_SHOTS}_shot"
INPUT_DIR="$HOME/scratch/marglicot_saves/sft_saves/"
MAX_MODEL_LENGTH=4096


# Fixed -----------------------------------------------------------------------
TASK="math"
mkdir -p "${OUTPUT_DIR}"
TASK_PATH=./util_code/tasks.py
GLOB_PATTERN='*'"$TASK"'*'"${DATE}"'*/*/model/'
# -----------------------------------------------------------------------------

python multi_gpu_lighteval_chain.py \
--task_key="custom|${TASK}|${NUM_SHOTS}|${ALLOW_TRUNCATE_SHOTS}" \
--input_path="${INPUT_DIR}" \
--output_dir="${OUTPUT_DIR}" \
--glob_pattern="${GLOB_PATTERN}" \
--dispatch_style=slurm \
--max_model_length="${MAX_MODEL_LENGTH}" \
--custom_tasks "$(realpath ${TASK_PATH})"
