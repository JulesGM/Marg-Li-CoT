#!/usr/bin/env bash
#
# Runs all of the lighteval tests on all of the directories 
# that match "$INPUT_DIR/$GLOB_PATTERN" and that have a checkpoint in that directory.
# This script evaluates models trained with rejection sampling on GSM8K
#
# Arguments:
#   $1: Number of shots used during training (e.g. 0, 8)
#   $2: Number of shots to use during evaluation (optional, defaults to same as training)
#
# The script will:
# 1. Look for model checkpoints in INPUT_DIR matching GLOB_PATTERN
# 2. Run lighteval on each checkpoint using the specified number of evaluation shots
# 3. Save results to OUTPUT_DIR with format rejection_sampling_outputs_gsm8k_{train_shots}/{eval_shots}_shot
#
# Example usage:
#   ./rejection_sampling_launch_slurm_gsm8k.sh 0    # Train with 0 shots, eval with 0 shots
#   ./rejection_sampling_launch_slurm_gsm8k.sh 8 0  # Train with 8 shots, eval with 0 shots
#


set -euo pipefail

NUM_SHOTS_TRAINED=5
NUM_SHOTS_EVAL=5
MODEL_MAX_LENGTH=4096
ALLOW_FEW_SHOT_TRUNCATION=1
TASK="gsm8k"
DURATION=3:00:00

# Not likely to change ------------------------------------------------------------
GLOB_PATTERN='*/'
OUTPUT_DIR="./all_eval_outputs_important/rejection_sampling_outputs_gsm8k_${NUM_SHOTS_TRAINED}/${NUM_SHOTS_EVAL}_shot"
INPUT_DIR="$HOME/scratch/rejection_sampling_saves/gsm8k_${NUM_SHOTS_TRAINED}/2025-04-18_18-58-10"
TASK_PATH=./util_code/tasks.py

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory ${INPUT_DIR} does not exist"
    exit 1
fi

# Check if any matching directories exist
if ! ls -d "${INPUT_DIR}/"${GLOB_PATTERN} 1> /dev/null 2>&1; then
    echo "Error: No directories found matching pattern ${INPUT_DIR}/${GLOB_PATTERN}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"


uv run multi_gpu_lighteval_chain.py \
--task_key="custom|${TASK}|${NUM_SHOTS_EVAL}|${ALLOW_FEW_SHOT_TRUNCATION}" \
--input_path="${INPUT_DIR}" \
--output_dir="${OUTPUT_DIR}" \
--glob_pattern="${GLOB_PATTERN}" \
--dispatch_style=slurm \
--max_model_length="${MODEL_MAX_LENGTH}" \
--custom_tasks "$(realpath ${TASK_PATH})" \
--duration="${DURATION}"
