# Using Reinforcement Learning to Guide Chains of Thought
- Uses [Hugging Face TRL](https://github.com/lvwerra/trl) for PPO
- Uses [Hugging Face Peft](https://github.com/huggingface/peft) for [LoRA](https://arxiv.org/abs/2106.09685).
- Uses [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes) internally for 4bits and 8bits reference model modes.
- Uses [our QLora standalone lib](https://github.com/JulesGM/peft_qlora) for [QLora](https://arxiv.org/abs/2305.14314).


## Sets of TRL and/or SFT jobs:
Launch jobs with

    ./job_sets/launch_sets.py <job_set_name>   

Check the status with:

    ./job_sets/check_status.py

## With TRL:
Where the reinforcement learning is located.

    ./with_trl/launch.py <experiment_name>

## Approach SFT:

    ./approach_sft/launch.py <experiment_name>

