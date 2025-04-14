import os
import re
import time

import accelerate
import datasets
import deepspeed
import hydra
import more_itertools
import omegaconf
import rich
import torch
import torch.utils.data
import transformers
import vllm
import wandb
from open_instruct.vllm_utils2 import create_vllm_engines, init_process_group


def is_correct_gsm8k(candidate: str, ground_truth: str) -> bool:
    extracted = re.findall(r"\-?\d+", candidate).strip().lower()
    formatted_gt = ground_truth.strip().lower()
    print(f"Extracted: {extracted}, GT: {formatted_gt}")
    return  extracted == formatted_gt

# We are debugging with gsm8k
is_correct = is_correct_gsm8k

def broadcast_to_vllm(*, accelerator, vllm_engines, hf_model, gather_whole_model, deepspeed_stage):
    # avoid OOM
    torch.cuda.empty_cache()
    model = accelerator.unwrape(hf_model)
    count, num_params = 0, len(list(model.named_parameters()))
    refss = []

    if gather_whole_model:
        with deepspeed.zero.GatheredParameters(model.parameters(), enabled=deepspeed_stage == 3):
            for name, param in model.named_parameters():
                count += 1  # empty_cache at last param
                # Fire all vllm engines for broadcast
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if deepspeed_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                        )
                        for engine in vllm_engines
                    ]
                    refss.extend(refs)
                if torch.distributed.get_rank() == 0:
                    torch.distributed.broadcast(param.data, 0)

    else:  # broadcast each parameter independently
        for name, param in model.named_parameters():
            count += 1
            if torch.distributed.get_rank() == 0:
                shape = param.shape if deepspeed_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(
                        name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                    )
                    for engine in vllm_engines
                ]
                refss.extend(refs)
            with deepspeed.zero.GatheredParameters([param], enabled=deepspeed_stage == 3):
                if torch.distributed.get_rank() == 0:
                    torch.distributed.broadcast(param.data, 0)


def get_deepspeed_stage(accelerator):
    return accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", 0)


@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(cfg: omegaconf.DictConfig) -> None:
    # Print configuration for verification.
    print("Configuration:\n", omegaconf.OmegaConf.to_yaml(cfg))
    rich.print("[red bold]CUDA_VISIBLE_DEVICES = ", os.environ.get("CUDA_VISIBLE_DEVICES"))
    # Set GPU IDs for accelerate from config.
    # Initialize Accelerator using the module name.
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    # Initialize wandb on rank 0 only.
    if accelerator.is_main_process:
        wandb.init(
            project=cfg.wandb.project, 
            entity=cfg.wandb.entity,
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        )
    
    # Unpack configuration parameters.
    model_name = cfg.model.name
    batch_size = cfg.training.batch_size
    learning_rate = cfg.training.learning_rate
    num_epochs = cfg.training.num_epochs
    max_new_tokens = cfg.training.max_new_tokens

    temperature = cfg.vllm_sampling.temperature
    num_candidates = cfg.vllm_sampling.num_candidates
    top_p = cfg.vllm_sampling.top_p  # May be null

    dataset_name = cfg.dataset.name
    dataset_split = cfg.dataset.split
    q_field = cfg.dataset.question_field
    a_field = cfg.dataset.answer_field
    load_dataset_args = cfg.dataset.load_dataset_args

    # --- Load training dataset ---
    train_dataset = datasets.load_dataset(dataset_name, *load_dataset_args, split=dataset_split)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    total_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * total_steps)

    # --- Load model and tokenizer ---
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    rich.print(f"[red bold]ACCELERATE DEVICE IS {device}")
    rich.print(f"[green bold]VLLM DEVICE IS {cfg.vllm.gpu_id}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Set up optimizer and scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # --- Initialize vLLM inference engine without tensor parallelism ---
    inference_llm = create_vllm_engines(
        model=model_name,
        trust_remote_code=True,
        device=cfg.vllm.gpu_id,
    )

    # --- Load evaluation dataset (once, outside the loop) ---
    eval_dataset_full = datasets.load_dataset(dataset_name, split=dataset_split)
    eval_subset_size = cfg.evaluation.eval_subset_size
    eval_batch_size = cfg.evaluation.eval_batch_size
    if eval_subset_size is not None:
        eval_dataset = eval_dataset_full.select(range(eval_subset_size))
    else:
        eval_dataset = eval_dataset_full
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

    global_step = 0

    # --- Training loop over epochs ---
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            questions = batch[q_field]
            answers_gt = batch[a_field]

            selected_candidates = []
            # For each question, only rank 0 calls vLLM; others wait.
            for q, gt in more_itertools.zip_equal(questions, answers_gt):
                candidate_result = None
                num_attempts = 0
                while not candidate_result:
                    if accelerator.is_main_process:
                        # Generate multiple candidates at once.
                        num_attempts += 1
                        gen_kwargs = {"max_new_tokens": max_new_tokens, "temperature": temperature, "num_candidates": num_candidates}
                        if top_p is not None:
                            gen_kwargs["top_p"] = top_p
                        
                        print(f"Generating candidates... Attempt # {num_attempts}")
                        start = time.perf_counter()
                        results = inference_llm.generate(q, **gen_kwargs)
                        print(f"Generation took {time.perf_counter() - start:.1f} seconds for {len(results)} candidates")

                        # Loop over candidates and pick the first correct one.
                        for candidate_obj in results:
                            candidate_text = candidate_obj.outputs[0].text
                            if is_correct(candidate_text, gt):
                                candidate_result = candidate_text
                                break
                        # If none correct, default to first candidate.
                        if candidate_result is None:
                            candidate_result = results[0].outputs[0].text
                    else:
                        raise NotImplementedError(f"We should just have one global rank at this time. {accelerator.process_index = } / {accelerator.process_count = }")

                accelerator.wait_for_everyone()
                # Broadcast the candidate result from rank 0 to all processes.
                candidate_result = accelerator.broadcast_object(candidate_result, src=0)
                selected_candidates.append(candidate_result)

            inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            targets = tokenizer(selected_candidates, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            print("Broadcasting model weights...")
            start = time.perf_counter()
            broadcast_to_vllm(
                accelerator=accelerator,
                vllm_engines=inference_llm,
                hf_model=model,
                gather_whole_model=True,
                deepspeed_stage=get_deepspeed_stage(accelerator),
            )
            print(f"Broadcasting took {time.perf_counter() - start:.1f} seconds")

            epoch_loss += loss.item()
            global_step += 1
            num_batches += 1

            if accelerator.is_main_process and global_step % cfg.wandb.log_interval == 0:
                wandb.log({"train/loss": loss.item(), "step": global_step})

            if global_step >= total_steps:
                break

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")
        print(f"Epoch {epoch + 1} complete. Average Loss: {avg_loss:.4f}")
        if accelerator.is_main_process:
            wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})

        # --- Evaluation run at end of epoch ---
        print("Starting evaluation run...")
        model.eval()
        local_correct = 0
        local_total = 0
        for eval_batch in eval_loader:
            q = eval_batch[q_field][0]
            gt = eval_batch[a_field][0]
            # Only rank 0 calls vLLM generation.
            candidate_result = None
            if accelerator.is_main_process:
                gen_kwargs = {"max_new_tokens": max_new_tokens, "temperature": temperature, "num_candidates": num_candidates}
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p
                results = inference_llm.generate(q, **gen_kwargs)
                for candidate_obj in results:
                    candidate_text = candidate_obj.outputs[0].text
                    if is_correct(candidate_text, gt):
                        candidate_result = candidate_text
                        break
                if candidate_result is None:
                    candidate_result = results[0].outputs[0].text
            else:
                candidate_result = None

            accelerator.wait_for_everyone()
            candidate_result = accelerator.broadcast_object(candidate_result, src=0)
            if is_correct(candidate_result, gt):
                local_correct += 1
            local_total += 1

        correct_tensor = torch.tensor([local_correct], dtype=torch.float32, device=device)
        total_tensor = torch.tensor([local_total], dtype=torch.float32, device=device)
        gathered_correct = accelerator.gather(correct_tensor)
        gathered_total = accelerator.gather(total_tensor)
        if accelerator.is_main_process:
            total_correct = gathered_correct.sum().item()
            total_count = gathered_total.sum().item()
            eval_accuracy = total_correct / total_count if total_count > 0 else 0.0
            print(f"Evaluation Accuracy after Epoch {epoch + 1}: {eval_accuracy * 100:.2f}%")
            wandb.log({"eval/accuracy": eval_accuracy, "epoch": epoch + 1})
        model.train()

    if accelerator.is_main_process:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
