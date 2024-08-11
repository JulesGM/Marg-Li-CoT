#!/usr/bin/env bash
set -euo pipefail

declare -a runs=(
    answer_only_gsm8k 
    answer_only_arithmetic 
    cot_gsm8k 
    cot_arithmetic
) ;

for EXPERIMENT in "${runs[@]}" ; 
do
    sbatch --job-name="$EXPERIMENT" launch.py experiment="$EXPERIMENT"
done
