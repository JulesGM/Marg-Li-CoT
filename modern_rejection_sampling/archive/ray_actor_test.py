from open_instruct import vllm_utils2
import ray
import transformers
import vllm
import torch

print(vllm_utils2.__file__)

@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, model_name):
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.forward_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.inference_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left")

    def train(self, text_list):
        data = self.forward_tokenizer(
            text_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.model.device)

        return self.model(**data, labels=data.input_ids).loss.item()
    

def main():
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    trainer = Trainer.remote(model_name)
    print(ray.get(trainer.train.remote(["Hello, how are you today?"])))

    # actor = vllm_utils2.create_vllm_engines(
    #     num_engines=1, 
    #     tensor_parallel_size=1,
    #     pretrain=model_name,
    #     revision=None,
    #     seed=0,
    #     enable_prefix_caching=True,
    #     max_model_len=2048,
    # )[0]

    # print(ray.get(actor.generate.remote(
    #     tokenizer.apply_chat_template(
    #     [{"role": "user", "content": "Hello, how are you today?"}],
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     sampling_params=vllm.SamplingParams(max_tokens=1000)
    # )))[0].outputs[0].text.strip())




if __name__ == "__main__":
    main()