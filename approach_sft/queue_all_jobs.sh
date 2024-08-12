#!/usr/bin/env bash
set -euo pipefail

GPU_TYPE=${GPU_TYPE:-rtx8000}
PARTITION=${PARTITION:-long}

echo "GPU_TYPE: $GPU_TYPE"
echo "PARTITION: $PARTITION"


declare -a runs=(
    answer_only_gsm8k 
    cot_gsm8k 
    answer_only_arithmetic 
    cot_arithmetic
) ;


for EXPERIMENT in "${runs[@]}" ; 
do
    COMMAND=(
        --gres="gpu:${GPU_TYPE}:1" 
        --partition="${PARTITION}" 
        --cpus-per-task=4 
        --mem=32G 
        --job-name="$EXPERIMENT" 
        --output=slurm_logs/slurm_"$EXPERIMENT"_%j.out
        launch.py 
        experiment="$EXPERIMENT" 
    )

    echo sbatch "${COMMAND[@]}"
    sbatch "${COMMAND[@]}"
done
