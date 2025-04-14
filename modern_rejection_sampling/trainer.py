# Standard library imports
import gc
import json
import os
import pathlib

# Third party imports
import ray
import rich
import torch
import transformers

# Local imports
from open_instruct import vllm_utils2



def json_default(obj):
    if isinstance(obj, pathlib.Path):
        return str(obj)
    elif isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")



@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, *, model_name: str | pathlib.Path, learning_rate: float, num_warmup_steps: int, total_steps: int, init_method_internal: str, model_max_length: int):
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}, {learning_rate = }, {num_warmup_steps = }, {total_steps = }, {init_method_internal = }, {model_max_length = }")
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
        ).to(0)

        self.forward_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=model_max_length,
        )
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )
        self.model_update_group = None
        torch.distributed.init_process_group(
            backend="nccl", 
            init_method=init_method_internal, 
            world_size=1, 
            rank=0
        )

    def init_process_group(self, *args, **kwargs):
        self.model_update_group = vllm_utils2.init_process_group(*args, **kwargs)

    def train(self, tokenized_batch_of_conversations: list[torch.Tensor]) -> float:
        assert isinstance(tokenized_batch_of_conversations, list), f"{type(tokenized_batch_of_conversations) = }"
        assert all(isinstance(t, torch.Tensor) for t in tokenized_batch_of_conversations), f"{[type(t) for t in tokenized_batch_of_conversations] = }"

        gc.collect()
        torch.cuda.empty_cache()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            data = self.forward_tokenizer.pad(
                dict(input_ids=tokenized_batch_of_conversations), # The inputs are not padded, so we don't need to provide an attention mask
                padding="max_length",
                return_tensors="pt",
            )

            if not isinstance(data, transformers.BatchEncoding):
                raise ValueError(
                    f"\n{tokenized_batch_of_conversations = }\n" +
                    f"{len(tokenized_batch_of_conversations) = }\n" +
                    f"{type(tokenized_batch_of_conversations).mro() = }\n" +
                    f"{type(data).mro() = }\n{data = }"
                )

            data = data.to(self.model.device)
            print(f"{data.input_ids.shape = }", flush=True)

            rich.print(f"[blue]Tokenized, now calling forward: {data.input_ids.shape = }")
            self.optimizer.zero_grad()
            loss = self.model(**data, labels=data.input_ids).loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()

        return loss.item()
    
    def broadcast_to_vllm(self, *, vllm_engines, gather_whole_model):
        assert self.model_update_group
        print("Broadcasting to VLLM", flush=True)

        # avoid OOM
        torch.cuda.empty_cache()
        model = self.model
        count, num_params = 0, len(list(model.named_parameters()))
        refss = []

        if gather_whole_model:
            for name, param in model.named_parameters():
                count += 1  # empty_cache at last param
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape 
                    refs = [
                        engine.update_weight.remote(
                            name, 
                            dtype=param.dtype, 
                            shape=shape, 
                            empty_cache=count == num_params,
                        )
                        for engine in vllm_engines
                    ]
                    refss.extend(refs)
                assert torch.distributed.get_rank() == 0
                torch.distributed.broadcast(param.data, 0, group=self.model_update_group)

        else:  # broadcast each parameter independently
            for name, param in model.named_parameters():
                assert torch.distributed.get_rank() == 0
                count += 1
                shape = param.shape
                refs = [
                    engine.update_weight.remote(
                        name, dtype=torch.bfloat16, shape=shape, empty_cache=count == num_params
                    )
                    for engine in vllm_engines
                ]
                assert torch.distributed.get_rank() == 0
                torch.distributed.broadcast(param.data.bfloat16(), 0, group=self.model_update_group)
        assert torch.distributed.get_rank() == 0
        ray.get(refss)
        print("Done broadcasting", flush=True)

    def save_pretrained(self, *, output_dir, cfg_container, wandb_run_id, wandb_url):
        self.model.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "hydra_config.json"), "w") as f:
            json.dump(cfg_container, f, indent=4, sort_keys=True, default=json_default)
        with open(os.path.join(output_dir, "wandb_info.json"), "w") as f:
            json.dump({"run_id": wandb_run_id, "url": wandb_url}, f, indent=4, sort_keys=True, default=json_default)
        self.forward_tokenizer.save_pretrained(output_dir)

