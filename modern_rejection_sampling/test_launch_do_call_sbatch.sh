#!/usr/bin/env bash
#SBATCH --gres=gpu:l40s:4 
#SBATCH --cpus-per-task 32 
#SBATCH --mem 160GB 
#SBATCH --partition long
#SBATCH --job-name=text_config_sweep
#SBATCH --output=output_text_config_sweep_%A_%a.out
#SBATCH --error=error_text_config_sweep_%A_%a.err
#SBATCH --array=0-1

# Define configs array
configs=(test_gsm8k_8 test_gsm8k_0)


LEARNING_RATE=0.0001


# Validate experiment argument is not provided (since we're using array)
if [ ! -z "$1" ]; then
    echo "Error: No arguments needed for array job"
    echo "This script will run all configs automatically"
    exit 1
fi

# Get config for this array job
config=${configs[${SLURM_ARRAY_TASK_ID}]}

# Check if experiment config exists
if [ ! -f "config/experiment/$config.yaml" ]; then
    echo "Error: Experiment config 'config/experiment/$config.yaml' not found"
    exit 1
fi

scontrol update JobId="$SLURM_JOB_ID" JobName=config_sweep_"${config}_${LEARNING_RATE}"

OUTPUT_DIR="/network/scratch/g/gagnonju/rejection_sampling_saves/${config}/$(date +%Y-%m-%d_%H-%M-%S)/"

# export NCCL_DEBUG=DEBUG

uv run ray_train.py experiment="$config" training.learning_rate="$LEARNING_RATE" output_dir="$OUTPUT_DIR"
