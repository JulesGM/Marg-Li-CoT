# Using Reinforcement Learning to Guide Chains of Thought
- Uses [Hugging Face TRL](https://github.com/lvwerra/trl) for PPO
- Uses [Hugging Face Peft](https://github.com/huggingface/peft) for [LoRA](https://arxiv.org/abs/2106.09685).
- Uses [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes) internally for 4bits and 8bits reference model modes.
- Uses [our QLora standalone lib](https://github.com/JulesGM/peft_qlora) for [QLora](https://arxiv.org/abs/2305.14314).


## With TRL:
Where the reinforcement learning is located.


## Approach SFT:
There, one finds the supervised baselines:

- Generate, then learn, masked.

Launch all SFT jobs with 

```bash

cd approach_sft
./queue_all_jobs.sh

```

