#!/usr/bin/env bash
# /home/mila/g/gagnonju/scratch/open_instruct_output/2024-12-27_22-43-27_rlvr_8b_checkpoints/step_1200
# HuggingFaceTB/SmolLM2-1.7B-Instruct
set -x

# CHECKPOINT_PATH="/network/scratch/g/gagnonju/open_instruct_output/2024-12-31_21-52-49_rlvr_gsm8k_only_smollm2_instruct_checkpoints/step_800"
# CHECKPOINT_PATH="/network/scratch/g/gagnonju/open_instruct_output/2024-12-30_17-51-43_rlvr_8b_checkpoints/step_200/"
# CHECKPOINT_PATH="HuggingFaceTB/SmolLM2-1.7B-Instruct"

CHECKPOINT_PATH="/network/scratch/g/gagnonju/open_instruct_output/2024-12-31_21-22-51_rlvr_gsm8k_only_smollm2_instruct_checkpoints/step_200/"

lighteval accelerate \
--model_args="pretrained=${CHECKPOINT_PATH},revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048" \
--tasks="custom|gsm8k|8|0" \
--output_dir=./outputs/ \
--use_chat_template \
--custom_tasks "./tasks.py" \
--save_details