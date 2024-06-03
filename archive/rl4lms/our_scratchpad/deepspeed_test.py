import logging

import deepspeed
import fire
import rich
import rich.logging
import transformers

LOGGER = logging.getLogger(__name__)


def main(
    hf_model_a = "google/flan-t5-base",
    hf_model_b = "google/flan-t5-base",
):
    
    class A:
        def __init__(self, model_name):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_a)
            self.model_a = transformers.T5ForConditionalGeneration.from_pretrained(hf_model_a)
            # self.dl_a = torch.utils.data.DataLoader(list(range(10000)), batch_size=2)
            self.accelerate_model_a = deepspeed.initialize(
                model=self.model_a, 
                model_parameters=self.model_a.parameters(),
                config="/home/mila/g/gagnonju/Marg-Li-CoT/rl4lms/config_ds_json/deepspeed.json"
            )

        def generate(self, texts):
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.accelerate_model_a.device) for k, v in inputs.items()}
            LOGGER.info(self.tokenizer.batch_decode(self.accelerate_model_a.generate(
                **inputs, 
                min_length=0, 
                max_new_tokens=200, 
                num_beams=8, 
                do_sample=True
    
            )))


        __call__ = generate


    a = A(hf_model_a)
    b = A(hf_model_b)

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
