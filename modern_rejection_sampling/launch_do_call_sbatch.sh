#!/usr/bin/env bash
#SBATCH --partition long
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task 16
#SBATCH --mem 140GB
#SBATCH --job-name=math_4
#SBATCH --array=0-3

# Long 4:  --gres=gpu:l40s:4  --cpus-per-task 48 --mem 1000GB --partition long
# Long 2:  --gres=gpu:l40s:2  --cpus-per-task 24 --mem 500GB  --partition long
# Long 2:  --gres=gpu:l40s:3  --cpus-per-task 36 --mem 750GB  --partition long
# Main 2:  --gres=gpu:l40s:2  --cpus-per-task  8 --mem 48GB   --partition main-gpu


set -eu -o pipefail

# Define configs array
config=hendrycks_math_4

LEARNING_RATE_ARRAY=(0.00001 0.00005 0.0001 0.0005)
LEARNING_RATE=${LEARNING_RATE_ARRAY[${SLURM_ARRAY_TASK_ID:-0}]}

# Check if experiment config exists
if [ ! -f "config/experiment/$config.yaml" ]; then
    echo "Error: Experiment config 'config/experiment/$config.yaml' not found"
    exit 1
fi

echo "Got config: $config and learning rate: $LEARNING_RATE"
scontrol update JobId="$SLURM_JOB_ID" JobName=rejection_sampling_"${config}_${LEARNING_RATE}"

OUTPUT_DIR="/network/scratch/g/gagnonju/rejection_sampling_saves/${config}/$(date +%Y-%m-%d_%H-%M-%S)_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/"
# export NCCL_DEBUG=DEBUG
cd /home/mila/g/gagnonju/marglicot/modern_rejection_sampling || exit 1

export ACCELERATE_CONFIG_FILE="./config/accelerate.yaml"
uv run ray_train.py experiment="$config" training.learning_rate="$LEARNING_RATE" output_dir="$OUTPUT_DIR"
