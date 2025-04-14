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

GLOB_PATTERN='*/*/'

if [ $# -eq 1 ]; then
    NUM_SHOTS_TRAINED=$1
    NUM_SHOTS_EVAL=$1
elif [ $# -eq 2 ]; then
    NUM_SHOTS_TRAINED=$1
    NUM_SHOTS_EVAL=$2
else
    echo "Usage: $0 <num_shots_trained> [num_shots_eval]"
    echo "If num_shots_eval is not provided, it will use the same value as num_shots_trained"
    exit 1
fi

OUTPUT_DIR="./all_eval_outputs_important/rejection_sampling_outputs_gsm8k_${NUM_SHOTS_TRAINED}/${NUM_SHOTS_EVAL}_shot"
INPUT_DIR="$HOME/scratch/rejection_sampling_saves/gsm8k_${NUM_SHOTS_TRAINED}/"

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
TASK_PATH=./util_code/tasks.py

uv run multi_gpu_lighteval_chain.py \
--task_key="custom|gsm8k|${NUM_SHOTS_EVAL}|0" \
--input_path="${INPUT_DIR}" \
--output_dir="${OUTPUT_DIR}" \
--glob_pattern="${GLOB_PATTERN}" \
--dispatch_style=slurm \
--custom_tasks "$(realpath ${TASK_PATH})" 

# uv run lighteval accelerate --model_args pretrained=/home/mila/g/gagnonju/scratch/rejection_sampling_saves/gsm8k_0/2025-03-24_01-32-35/epoch_3,revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048 
# --tasks 'custom|gsm8k|0|0' 
# --output_dir /home/mila/g/gagnonju/marglicot/light_eval_tests/all_eval_outputs_important/rejection_sampling_outputs_gsm8k_0/0_shot 
# --use_chat_template --custom_tasks /home/mila/g/gagnonju/marglicot/light_eval_tests/util_code/tasks.py --save_details