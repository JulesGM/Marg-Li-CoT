import concurrent.futures as cf
import enum
import os
import queue
import re
import threading
import time

import accelerate
import datasets
import deepspeed
import hydra
import more_itertools
import omegaconf
import rich
import rich.panel
import rich.rule
import rich.traceback
import torch
import torch.utils.data
import tqdm
import transformers
from open_instruct.ground_truth_utils import (verify_gsm8k_sample,
                                              verify_ifeval_sample,
                                              verify_math_sample)

import wandb

rich.traceback.install()

def __gsm8k(*, model_output: str, ground_truth_answer: str) -> bool:
    assert isinstance(model_output, str), "Candidate must be a string."
    assert isinstance(ground_truth_answer, str), "Ground truth must be a string."


    extracted_list = re.findall(r"\-?\d+", model_output)
    formatted_gt = ground_truth_answer.strip().lower().rsplit("####")[-1].strip()

    if extracted_list:
        extracted = extracted_list[-1].strip().lower()
        # print(f"Extracted: {extracted}, GT: {formatted_gt}")

        return  extracted == formatted_gt
    # print(f"Couldn't extract any number, GT: {formatted_gt}")
    return False


def is_correct_gsm8k(*, model_output: str, ground_truth_answer: str) -> bool:
    assert isinstance(model_output, str), "Candidate must be a string."
    assert isinstance(ground_truth_answer, str), "Ground truth must be a string."

    formatted_gt = ground_truth_answer.strip().lower().rsplit("####")[-1].strip()

    return verify_gsm8k_sample(model_output=model_output, ground_truth_answer=formatted_gt)

# We are debugging with gsm8k
is_correct = is_correct_gsm8k


class ReturnState(str, enum.Enum):
    END_EPOCH = "end_epoch"
    SUCCESS = "success"
    SKIP = "skip"


def generate_one(
        *, 
        loader, 
        loader_lock,
        q_field, 
        a_field, 
        generation_tokenizer, 
        max_new_tokens, 
        temperature, 
        num_candidates,
        model_queue,
        top_p=None,
    ):

    try:
        # start = time.perf_counter()
        with loader_lock:
            batch = next(loader)
        # print(f"Getting batch took {time.perf_counter() - start:.1f} seconds")
    except StopIteration:
        return ReturnState.END_EPOCH, None
    
    # start = time.perf_counter()
    model, gpu_id = model_queue.get()
    # print(f"Getting model took {time.perf_counter() - start:.1f} seconds")

    header = f"(GPU {gpu_id}) "

    q = more_itertools.one(batch[q_field])
    gt = more_itertools.one(batch[a_field])

    num_skipped = 0
    candidate_result = None
    num_attempts = 0
    # rich.print(rich.rule.Rule(
    #     f"[bold blue]New question ({len(train_batch)} / {batch_size}) " +
    #     f"of batch ({num_batches} of ?) " + 
    #     f"({samples_attempted} attempted of {len(train_loader)}, " + 
    #     f"{samples_successful} successful, " + 
    #     f"{samples_successful / samples_attempted:.1%} success rate) " +
    #     f"epoch {epoch + 1}/{num_epochs}",
    #     align="left",
    # ))

    while not candidate_result:
        num_attempts += 1
        if num_attempts > 1:
            candidate_result = None
            num_skipped += 1
            rich.print(
                rich.rule.Rule(
                    f"[bold red]SKIPPING ONE, {num_skipped}", 
                    align="left", style="bold red"
                )
            )
            break

        # rich.print(rich.rule.Rule(f"{header}Attempt # {num_attempts}", align="left"))
        # rich.print(rich.panel.Panel(q, title=f"{header}[bold]Question:", title_align="left"))

        gen_kwargs = {
            "max_new_tokens": max_new_tokens, 
            "temperature": temperature, 
            "num_return_sequences": num_candidates,
            "do_sample": True,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        
        tok_q = generation_tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], 
            return_tensors="pt",
            add_generation_prompt=True,
        )

        start = time.perf_counter()
        assert tok_q.shape[0] == 1, "Only one question at a time."
        results = generation_tokenizer.batch_decode(
            model.generate(tok_q.to(model.device), **gen_kwargs)
        )
        rich.print(f"{header}Generation took {time.perf_counter() - start:.1f} seconds for {len(results)} candidates")

        # Loop over candidates and pick the first correct one.
        are_good = []
        
        for candidate_text in results:
            unit = is_correct(model_output=candidate_text, ground_truth_answer=gt) 
            are_good.append(unit)
        
        # print(f"Correctness: {are_good}")

        for candidate_text in results:
            if is_correct(model_output=candidate_text, ground_truth_answer=gt):
                candidate_result = candidate_text
                break
    
    # start = time.perf_counter()
    model_queue.put((model, gpu_id))
    # print(f"Putting model back took {time.perf_counter() - start:.1f} seconds")

    # Broadcast the candidate result from rank 0 to all processes.
    if candidate_result: 
        return ReturnState.SUCCESS, candidate_result
    
    return ReturnState.SKIP, None

def sync_models(models):
    for gpu_id, model in enumerate(models[1:], start=1):
        model.load_state_dict(models[0].state_dict())
        assert model.device.index == gpu_id, (
            f"Model should be on the correct device. {model.device}, cuda:{gpu_id}"
        )
        assert model.device.type == "cuda", model.device.type
        model.to(gpu_id)

def generate_train_batch(
        *,
        loader,
        loader_lock,
        q_field,
        a_field,
        generation_tokenizer,
        max_new_tokens,
        temperature,
        num_candidates,
        top_p,
        model_queue,
        pool,
        batch_size,
        samples_attempted,
        samples_successful,
        finished_epoch,
    ):

    torch.cuda.empty_cache()

    train_batch = []
    futures = set()

    submit_args = (generate_one,)
    submit_kwargs = dict(
        loader=loader,
        loader_lock=loader_lock,
        q_field=q_field,
        a_field=a_field,
        generation_tokenizer=generation_tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_candidates=num_candidates,
        top_p=top_p,
        model_queue=model_queue,
    )

    for _ in range(batch_size):
        samples_attempted += 1
        futures.add(pool.submit(*submit_args, **submit_kwargs))

    # As futures complete:
    while (
        len(train_batch) < batch_size and # There shouldn't be any more futures
        not (finished_epoch and not futures) # Done with epoch and no more futures to process
    ):
        # We call copy to avoid modifying the list while iterating.

        # for f in cf.as_completed(futures_copy):
        while futures:
            print(f"{len(futures) = }")
            done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            for f in done:
                status, candidate = f.result()
                if status == ReturnState.END_EPOCH:
                    finished_epoch = True
                    # Finished the epoch, don't resubmit
                elif status == ReturnState.SUCCESS:
                    train_batch.append(candidate)
                    samples_successful += 1
                elif status == ReturnState.SKIP:
                    assert len(train_batch) < batch_size
                    if not finished_epoch:
                        samples_attempted += 1
                        futures.add(pool.submit(*submit_args, **submit_kwargs))

    assert not futures, "Shouldn't have any futures left to process."
    return train_batch, samples_attempted, samples_successful, finished_epoch


@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(cfg: omegaconf.DictConfig) -> None:
    # Print configuration for verification.
    print("Configuration:\n", omegaconf.OmegaConf.to_yaml(cfg))
    # Set GPU IDs for accelerate from config.
    # Initialize Accelerator using the module name.

    wandb.init(
        project=cfg.wandb.project, 
        entity=cfg.wandb.entity,
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        dir=os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp")),
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
    valid_dataset_split = cfg.dataset.valid_split
    q_field = cfg.dataset.question_field
    a_field = cfg.dataset.answer_field
    load_dataset_args = cfg.dataset.load_dataset_args

    # --- Load training dataset ---
    viz_load_dataset_args = (dataset_name, *load_dataset_args), dict(split=dataset_split)
    print(f"Loading dataset {dataset_name} with args {viz_load_dataset_args}")
    train_dataset = datasets.load_dataset(dataset_name, *load_dataset_args, split=dataset_split)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    total_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * total_steps)

    # --- Load model and tokenizer ---
    num_gpus = torch.cuda.device_count()
    models = [
        transformers.AutoModelForCausalLM.from_pretrained(model_name).to(i) 
        for i in range(num_gpus)
    ]
    
    model = models[0]
    assert model.device.index == 0, model.device

    forward_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if forward_tokenizer.pad_token is None:
        forward_tokenizer.pad_token = forward_tokenizer.eos_token
    generation_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if generation_tokenizer.pad_token is None:
        generation_tokenizer.pad_token = generation_tokenizer.eos_token

    # --- Set up optimizer and scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    # --- Load evaluation dataset (once, outside the loop) ---
    eval_dataset_full = datasets.load_dataset(dataset_name, *load_dataset_args, split=valid_dataset_split)
    eval_subset_size = cfg.evaluation.eval_subset_size
    eval_batch_size = cfg.evaluation.eval_batch_size
    if eval_subset_size is not None:
        eval_dataset = eval_dataset_full.select(range(eval_subset_size))
    else:
        eval_dataset = eval_dataset_full
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

    global_step = 0
    pool = cf.ThreadPoolExecutor(max_workers=num_gpus)

    model_queue = queue.Queue()
    for gpu_id, model_it in enumerate(models):
        model_queue.put((model_it, gpu_id))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # --- Training loop over epochs ---
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1} / {num_epochs}")
            epoch_loss = 0.0
            num_batches = 0
            
            finished_epoch = False
            loader = iter(train_loader)
            loader_lock = threading.Lock()

            samples_attempted = 0
            samples_successful = 0

            while not finished_epoch:
                rich.print(f"[red bold]BEGIN TRAIN BATCH >>>>>>")
                rich.print(f"Syncing models...")
                sync_models(models)
                rich.print(f"[green bold]BEGIN CREATING BATCH >>>>>>")
                start = time.perf_counter()
                train_batch, samples_attempted, samples_successful, finished_epoch = generate_train_batch(
                    loader=loader,
                    loader_lock=loader_lock,
                    q_field=q_field,
                    a_field=a_field,
                    generation_tokenizer=generation_tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    num_candidates=num_candidates,
                    top_p=top_p,
                    model_queue=model_queue,
                    pool=pool,
                    batch_size=batch_size,
                    samples_attempted=samples_attempted,
                    samples_successful=samples_successful,
                    finished_epoch=finished_epoch,
                )
                torch.cuda.empty_cache()

                rich.print(f"[green bold]DONE CREATING BATCH. TOOK {time.perf_counter() - start} seconds >>>>>>")
                rich.print(rich.rule.Rule("[bold red]Got a batch >>>>>", align="left"))
                targets = forward_tokenizer(train_batch, return_tensors="pt", padding=True).to(0)

                assert model.device.index == 0, model.device
                assert targets.input_ids.device.index == 0, targets.input_ids.device
                assert targets.attention_mask.device.index == 0, targets.attention_mask.device
                outputs = model(**targets, labels=targets.input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += len(train_batch)
                num_batches += 1

                if global_step % cfg.wandb.log_interval == 0:
                    wandb.log({"train/loss": loss.item(), "step": global_step})

                # --- Evaluation run at end of epoch ---
                print("Starting evaluation run...")
                model.eval()
                local_correct = 0

            rich.print(f"[red bold]FINISHED EPOCH {epoch} ##################################")

            for eval_batch in tqdm.tqdm(eval_loader, desc="Evaluating"):
                q = eval_batch[q_field]
                gt = eval_batch[a_field]

                gen_kwargs = {
                    "max_new_tokens": max_new_tokens, 
                    "do_sample": False,
                }
                
                tok_q = [
                    generation_tokenizer.apply_chat_template(
                        [{"role": "user", "content": q}], 
                        return_tensors="pt",
                        add_generation_prompt=True,
                    ) for q in eval_batch[q_field]
                ]
                
                tok_q = generation_tokenizer.pad(
                    tok_q, return_tensors="pt", 
                    padding="longest",
                ).to(0)

                results = forward_tokenizer.batch_decode(
                    model.generate(**tok_q, **gen_kwargs)[:, tok_q.input_ids.shape[1]:]
                )

                for candidate_result, gt in more_itertools.zip_equal(results, gt):
                    local_correct += is_correct(candidate_result, gt)
                
                local_total = len(gt)

            eval_accuracy = total_correct / total_count if total_count > 0 else 0.0
            print(f"Evaluation Accuracy after Epoch {epoch + 1}: {eval_accuracy * 100:.2f}%")
            wandb.log({"eval/accuracy": eval_accuracy, "epoch": epoch + 1})
            model.train()

    if accelerator.is_main_process:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
