import logging

import accelerate
import fire
import rich
import rich.logging
import torch
import transformers

from rl4lms.envs.text_generation.registry import PolicyRegistry
import rl4lms.envs.text_generation.policy.seq2seq_policy as rl4lms_seq2seq_policy

LOGGER = logging.getLogger(__name__)

class A:
    def __init__(self, model_name):
        self.accel_a = accelerate.Accelerator()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model_a = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        self.dl_a = torch.utils.data.DataLoader(list(range(10000)), batch_size=2)

    def generate(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model_a.device) for k, v in inputs.items()}
        generated = self.model_a.generate(
            **inputs, 
            min_length=0, 
            max_new_tokens=200, 
            num_beams=8, 
            do_sample=True
        )
        LOGGER.info(self.tokenizer.batch_decode(generated))


    __call__ = generate



def main(
    hf_model_a = "google/flan-t5-base",
    hf_model_b = "google/flan-t5-base",
):
    
    acc        = accelerate.Accelerator()
    transformers.AutoTokenizer.from_pretrained(hf_model_a)
    policy_cls = PolicyRegistry.get("seq2seq_lm_actor_critic_policy")

    policy = rl4lms_seq2seq_policy.Seq2SeqLMActorCriticPolicy(
        observation_space=policy_cls["observation_space"],
        action_space=policy_cls["action_space"],
        lr_schedule=policy_cls["lr_schedule"],
        model_name=hf_model_a,
    )

    acc.prepare(
        policy, 
        torch.utils.data.DataLoader(list(range(10000)), batch_size=2),
    )

    return


    a = A(hf_model_a)
    a, a.accel_dl_a = a.accel_a.prepare(a, a.dl_a)

    
    b = A(hf_model_b)
    b, b.accel_dl_a = b.accel_a.prepare(b, b.dl_a)


    logging.basicConfig(
        handlers=[rich.logging.RichHandler(markup=True)],
        level=logging.INFO,
        format=f"[{a.accel_a.process_index + 1}/{a.accel_a.num_processes}]: %(message)s",
    )

    texts = [
        "Give me an advanced definition of ADHD.",
    ]



    a(texts)
    b(texts)
    a(texts)
    b(texts)



if __name__ == "__main__":
    fire.Fire(main)
