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
from open_instruct import vllm_utils2
import ray
import vllm
import logging
import tqdm

rich.traceback.install()
print(vllm_utils2.__file__)


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
        max_tokens, 
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
            break

        # rich.print(rich.rule.Rule(f"{header}Attempt # {num_attempts}", align="left"))
        # rich.print(rich.panel.Panel(q, title=f"{header}[bold]Question:", title_align="left"))

        gen_kwargs = vllm.SamplingParams(**{
            "max_tokens": max_tokens, 
            "temperature": temperature, 
            "n": num_candidates,
        })
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        
        start = time.perf_counter()
        outputs = ray.get(
            model.chat.remote(
                [{"role": "user", "content": q}], 
                sampling_params=gen_kwargs,
                use_tqdm=False,
            )
            )[0].outputs
        results = [output.text for output in outputs]
        # rich.print(f"{header}Generation took {time.perf_counter() - start:.1f} seconds for {len(results)} candidates")

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


def generate_train_batch(
        *,
        loader,
        loader_lock,
        q_field,
        a_field,
        max_tokens,
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

    train_batch = []
    futures = set()

    submit_args = (generate_one,)
    submit_kwargs = dict(
        loader=loader,
        loader_lock=loader_lock,
        q_field=q_field,
        a_field=a_field,
        max_tokens=max_tokens,
        temperature=temperature,
        num_candidates=num_candidates,
        top_p=top_p,
        model_queue=model_queue,
    )

    for _ in range(batch_size):
        futures.add(pool.submit(*submit_args, **submit_kwargs))

    # As futures complete:
    while (
        len(train_batch) < batch_size and # There shouldn't be any more futures
        not (finished_epoch and not futures) # Done with epoch and no more futures to process
    ):
        # We call copy to avoid modifying the list while iterating.

        # for f in cf.as_completed(futures_copy):
        while futures:
            done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            for f in done:
                status, candidate = f.result()
                if status == ReturnState.END_EPOCH:
                    finished_epoch = True
                    # Finished the epoch, don't resubmit
                elif status == ReturnState.SUCCESS:
                    train_batch.append(candidate)
                    samples_successful += 1
                    samples_attempted += 1
                elif status == ReturnState.SKIP:
                    samples_attempted += 1
                    rich.print(
                        rich.rule.Rule(
                            f"[bold red]Skipping one. " + 
                            f"{samples_successful / samples_attempted : .1%} success rate. " +
                            f"{samples_successful}/{samples_attempted}", 
                            align="left", style="bold red"
                        )
                    )
                    assert len(train_batch) < batch_size
                    if not finished_epoch:
                        futures.add(pool.submit(*submit_args, **submit_kwargs))
                else:
                    raise ValueError(f"Unknown status: {status}")

    assert not futures, "Shouldn't have any futures left to process."
    return train_batch, samples_attempted, samples_successful, finished_epoch



@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, *, model_name, learning_rate, num_warmup_steps, total_steps, init_method_internal, model_max_length):
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(0)
        self.forward_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, model_max_length=model_max_length)
        
        # --- Set up optimizer and scheduler ---
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

    def train(self, text_list):
        torch.cuda.empty_cache()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            data = self.forward_tokenizer(
                text_list, 
                return_tensors="pt", 
                padding="max_length", 
            ).to(self.model.device)
            
            rich.print(f"[blue]Tokenized, now calling forward: {data.input_ids.shape = }")
            loss = self.model(**data, labels=data.input_ids).loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return loss.item()
    
    def broadcast_to_vllm(self, *, vllm_engines, gather_whole_model):
            assert self.model_update_group

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

def run_eval(
        *,
        epoch,
        eval_loader,
        q_field,
        a_field,
        vllm_engine,
        in_epoch,
):
    
    total_correct = 0
    total_seen = 0

    for eval_batch in tqdm.tqdm(eval_loader, desc="Evaluating"):
        gt = eval_batch[a_field]

        gen_kwargs = vllm.SamplingParams(**{
            "max_tokens": 2048,
            "temperature": 0,
            "n": 1,
        })
        
        # results_raw = ray.get(vllm_engines[0].generate.remote(preped, sampling_params=gen_kwargs))
        results_raw = ray.get(
            vllm_engine.chat.remote(
                [[{"role": "user", "content": q}] for q in eval_batch[q_field]], 
                sampling_params=gen_kwargs
            )
        )
        results_text = [output.outputs[0].text for output in results_raw]

        for candidate_result, gt in more_itertools.zip_equal(results_text, gt):
            total_correct += is_correct(model_output=candidate_result, ground_truth_answer=gt)
        total_seen += len(gt)

    eval_accuracy = total_correct / total_seen if total_seen > 0 else 0.0
    print("\n\n")
    rich.print(rich.panel.Panel(f"Evaluation Accuracy after Epoch {epoch + 1}: {eval_accuracy:.1%}"))
    wandb.log({("eval_in_epoch/accuracy" if in_epoch else f"eval/accuracy"): eval_accuracy,})


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
    max_tokens = cfg.training.max_length
    gather_whole_model = False

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
    

    init_proc_group_refs = []
    master_address = "localhost"
    master_port = 29512
    backend = "nccl"
    init_method = f"tcp://{master_address}:{master_port}"
    group_name = "update_group"

    init_method_internal = "tcp://localhost:25323" # This is for the trainer's (eventual) internal DDP

    trainer = Trainer.remote(
        model_name=model_name,
        learning_rate=learning_rate,
        num_warmup_steps=num_warmup_steps,
        total_steps=total_steps,
        init_method_internal=init_method_internal,
        model_max_length=max_tokens,
    )
    
    init_proc_group_refs.append(trainer.init_process_group.remote(
        backend=backend,
        init_method=init_method,
        world_size=num_gpus,
        rank=0,
        group_name=group_name,
    ))
    
    vllm_engines = vllm_utils2.create_vllm_engines(
        num_engines=num_gpus - 1, 
        tensor_parallel_size=1,
        pretrain=model_name,
        revision=None,
        seed=0,
        enable_prefix_caching=True,
        max_model_len=2048,
    )

    for i, engine in enumerate(vllm_engines):
        init_proc_group_refs.append(engine.init_process_group.remote(
            backend=backend,
            world_size=num_gpus,
            master_address=master_address,
            master_port=master_port,
            rank_offset=i + 1,
            group_name=group_name,
        ))

    rich.print("[blue]Initializing process groups")
    ray.get(init_proc_group_refs)
    rich.print("[bold green]Initialized process groups")

    model_queue = queue.Queue()
    for gpu_id, engine in enumerate(vllm_engines, start=1):
        model_queue.put((engine, gpu_id))

    
    # --- Load evaluation dataset (once, outside the loop) ---
    eval_dataset_full = datasets.load_dataset(dataset_name, *load_dataset_args, split=valid_dataset_split)
    eval_batch_size = cfg.evaluation.eval_batch_size

    eval_loader_full = torch.utils.data.DataLoader(
        eval_dataset_full, 
        batch_size=eval_batch_size, 
        shuffle=False,
    )

    global_step = 0
    pool = cf.ThreadPoolExecutor(max_workers=num_gpus)

    # --- Training loop over epochs ---
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1} / {num_epochs}")
        epoch_loss = 0.0
        num_batches = 0
        
        finished_epoch = False
        loader = iter(tqdm.tqdm(train_loader))
        loader_lock = threading.Lock()
        samples_attempted = 0
        samples_successful = 0

        while not finished_epoch:
            rich.print(f"[bold]Syncing model weights.")
            ray.get(trainer.broadcast_to_vllm.remote(
                vllm_engines=vllm_engines,
                gather_whole_model=gather_whole_model,
            ))
            rich.print(f"[bold]Begin creating batch")
            start = time.perf_counter()

            train_batch, samples_attempted, samples_successful, finished_epoch = generate_train_batch(
                loader=loader,
                loader_lock=loader_lock,
                q_field=q_field,
                a_field=a_field,
                max_tokens=max_tokens,
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

            rich.print(rich.rule.Rule(
                f"[green bold]Done creating batch. Took {time.perf_counter() - start:0.1f} seconds.",
                align="left"
            ))
            
            loss = ray.get(trainer.train.remote(train_batch))
            rich.print(rich.rule.Rule(
                f"[green bold]Done with step. Loss: {loss :.1e}", 
                align="left"
            ))

            epoch_loss += loss
            global_step += len(train_batch)
            num_batches += 1

            # Eval Every 10%
            separator = int(cfg.evaluation.eval_percentage * (len(train_loader) // batch_size))
            eval_test = num_batches % separator == 0
            print(
                f"Num Batches: {num_batches}, " +
                f"Separator: {separator}, " + 
                f"{num_batches % separator}, " +
                f"Eval Test: {eval_test}"
            )
            if eval_test:
                run_eval(
                    epoch=epoch,
                    eval_loader=eval_loader_full,
                    q_field=q_field,
                    a_field=a_field,
                    vllm_engine=vllm_engines[0],
                    in_epoch=True,
                )

            if global_step % cfg.wandb.log_interval == 0:
                wandb.log({
                    "train/loss": loss, 
                    "step": global_step, 
                    "attempted": samples_attempted,
                    "successful": samples_successful / samples_attempted if samples_attempted > 0 else 0.0,
                })

        rich.print(f"[red bold]FINISHED EPOCH {epoch} ##################################")
        
        run_eval(
            epoch=epoch,
            eval_loader=eval_loader_full,
            q_field=q_field,
            a_field=a_field,
            vllm_engine=vllm_engines[0],
            in_epoch=False,
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
