#!/usr/bin/env bash
#SBATCH --gres=gpu:l40s:1 
#SBATCH --cpus-per-task 8  
#SBATCH --mem 40GB 
#SBATCH --partition long

/home/mila/g/gagnonju/.mambaforge/bin/python open-instruct/open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer '{"ai2-adapt-dev/gsm8k_ground_truth": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"ai2-adapt-dev/gsm8k_ground_truth": 1.0}' \
    --dataset_eval_splits test \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --reward_model_path HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template ORIGINAL_FORMAT \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 10000000 \
    --penalty_reward_value -10.0 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 32 \
    --local_rollout_batch_size 32 \
    --actor_num_gpus_per_node 3 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.05 \
    --apply_verifiable_reward true \
    --output_dir "$SCRATCH"/open_instruct_output/"$(date +"%Y-%m-%d_%H-%M-%S")"_rlvr_gsm8k_only_smollm2_instruct \
    --seed 3 \
    --num_evals 3 \
    --save_freq 100 \
    --reward_model_multiplier 0.0 \
    --with_tracking \
    --wandb_project_name open_instruct_rl \
    --deepspeed_stage 3 \
    --wandb_entity julesgm

# --gradient_checkpointing \
# --dataset_mixer '{"ai2-adapt-dev/gsm8k_math_ifeval_ground_truth_mixed": 1.0}' \
# --dataset_train_splits train \
# --dataset_eval_mixer '{"ai2-adapt-dev/gsm8k_math_ground_truth": 1.0}' \
# --dataset_eval_splits test \