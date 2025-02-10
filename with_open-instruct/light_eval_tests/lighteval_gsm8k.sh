#!/usr/bin/env bash
############################################################################################################
# Launches an individual checkpoint evaluation on the GSM8k dataset 
############################################################################################################
set -x

CHECKPOINT_PATH="/network/scratch/g/gagnonju/open_instruct_output/2024-12-31_21-22-51_rlvr_gsm8k_only_smollm2_instruct_checkpoints/step_1600"

CUDA_VISIBLE_DEVICES=3 lighteval accelerate \
--model_args="pretrained=${CHECKPOINT_PATH},revision=main,dtype=bfloat16,vllm,gpu_memory_utilisation=0.8,max_model_length=2048" \
--tasks="custom|gsm8k|8|0" \
--output_dir=./outputs_gsm8k/ \
--use_chat_template \
--custom_tasks "./util_code/tasks.py" \
--save_details