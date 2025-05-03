"""
Online rejection sampling based training.

VLLM engine on N - 1 GPUs.
Model training on 1 GPU.

This is because generation with rejection sampling takes much longer than training. 
The weights are broadcasted from the training GPU to the VLLM engines after each batch.

cfg.training.forward_max_length is the max length of the complete, formatted chat of tokens.
cfg.training.generation_max_length is the max length of the generated tokens for VLLM.

We pad forward to max length to have more predictable memory usage and fail fast, as the forward pass is so much faster than generation.

We dynamically make sure that the generated candidates don't exceed forward_max_length, right at generation time, & reject candidates that exceed it.



"""

import sys

# Standard library imports
import concurrent.futures as cf
import dataclasses
import enum
import gc
import json
import os
import pathlib
import queue
import re
import threading
import time
import typing

# Set environment variables
os.environ["OPENINSTRUCT_PARSE_LATEX_BACKEND"] = "lark" # Necessary for compatibility reasons. Needs to be done before importing open_instruct

# Third party imports
import datasets
import hydra
import more_itertools as mit
import numpy as np
import omegaconf
import ray
import ray.exceptions
import rich
import rich.panel
import rich.rule
import rich.traceback
import torch
import torch.utils.data
import transformers
import tqdm
import vllm
import wandb

from open_instruct.math_utils import (
    last_boxed_only_string,
    remove_boxed,
    get_unnormalized_answer,
    normalize_final_answer,
    is_equiv,
    hendrycks_is_equiv
)
from open_instruct import vllm_utils2

# Local imports
import config
import trainer as trainer_lib


rich.traceback.install()


def verify_gsm8k_sample(model_output, ground_truth_answer, verbose=False):
    # model_output = model_output.split("<|assistant|>\n")[-1].strip()
    # gsm is easy: extract numbers, and then just compare last number with answer.
    # matches how we do eval.
    predictions = None
    # replace numbers like `x,xxx` with `xxxx`
    response = re.sub(r"(\d),(\d)", r"\1\2", model_output)
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", response)
    if numbers:
        predictions = numbers[-1]
    else:
        predictions = response
    if verbose:
        print(f"predictions: {predictions}, ground_truth_answer: {ground_truth_answer}")
    return str(predictions).lower() == str(ground_truth_answer).lower(), predictions



def verify_math_sample(model_output, ground_truth_answer):
    ground_truth_answer = last_boxed_only_string(ground_truth_answer)
    if ground_truth_answer is not None:
        try:
            ground_truth_answer = remove_boxed(ground_truth_answer)
        except AssertionError:
            ground_truth_answer = None
    if ground_truth_answer is None:
        raise NotImplementedError(f"Bad ground truth: {ground_truth_answer}")

    raw_answer = model_output
    # for math, more complex. We will try a few different ways to extract the answer.
    # this roughly follows 'flex em' in oe-eval-internal
    all_answers = []
    # First, try find answer in \boxed{}.
    boxed_answer = last_boxed_only_string(raw_answer)
    if boxed_answer is not None:
        try:
            boxed_answer = remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None
    if boxed_answer is not None:
        all_answers.append(boxed_answer)
    # Second, try to extract via minerva format.
    minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)
    # If nothing still, try to find the last latex-formatted answer
    if len(all_answers) == 0:
        dollars = [m.start() for m in re.finditer("\\$", raw_answer)]
        if len(dollars) > 1:
            # Add the answer between the second to last and last dollar sign
            answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
            all_answers.append(answer)
    # otherwise, just take the full output. Probably wont work, bit of a yolo.
    if len(all_answers) == 0:
        all_answers.append(normalize_final_answer(model_output))
    # now, compare all answers to ground truth.
    matched = False

    for answer in all_answers:
        if is_equiv(answer, ground_truth_answer):
            matched = True
            break
        elif hendrycks_is_equiv(answer, ground_truth_answer):
            matched = True
            break
    # if we got any match, we are good.
    return matched

def is_correct_gsm8k(*, model_output: str, ground_truth_answer: str) -> bool:
    assert isinstance(model_output, str), "Candidate must be a string."
    assert isinstance(ground_truth_answer, str), "Ground truth must be a string."

    formatted_gt = ground_truth_answer.strip().lower().rsplit("####")[-1].strip().replace(",", "")

    return verify_gsm8k_sample(model_output=model_output, ground_truth_answer=formatted_gt)


# We are debugging with gsm8k
IS_CORRECT = {
    "gsm8k": is_correct_gsm8k,
    "hendrycks/competition_math": verify_math_sample,
}


def tokenize_conversations(*, forward_tokenizer, conversations: list[dict[str, str]]) -> torch.Tensor:
    messages = forward_tokenizer.apply_chat_template(conversations, tokenize=False)

    tokenized_conversation = forward_tokenizer(
        messages,
        return_tensors="pt",
    )

    return tokenized_conversation.input_ids


class ReturnState(str, enum.Enum):
    END_EPOCH = "end_epoch"
    SUCCESS = "success"
    SKIP = "skip"


@dataclasses.dataclass(kw_only=True, frozen=True)
class GeneratedConversation:
    tokenized_conversation: torch.Tensor
    question_text: str
    generated_answer_text: str
    reference_answer_text: str


def generate_one(
        *, 
        loader: torch.utils.data.DataLoader, 
        loader_lock: threading.Lock,
        q_field: str, 
        a_field: str, 
        generation_max_length: int, 
        temperature: float, 
        num_candidates: int,
        model_queue: queue.Queue[tuple[trainer_lib.Trainer, int]],
        is_correct: typing.Callable[[str, str], bool],
        use_few_shots: bool,
        few_shot_examples: list[dict[str, str]],
        top_p: float | None,
        forward_tokenizer: transformers.AutoTokenizer,
        forward_max_length: int,
        model_name: str,
        training_agent,
        gather_whole_model: bool,
        max_retries: int,
        num_gpus: int,
        master_address: str,
        master_port: int,
        group_name: str,
        backend: str,
    ) -> tuple[ReturnState, GeneratedConversation | None]:
    """
    Threaded worker that generates a single batch of candidates.
    """
    try:
        
        with loader_lock:
            batch = next(loader)
    except StopIteration:
        return ReturnState.END_EPOCH, None
    
    model, gpu_id = model_queue.get()

    q: str = mit.one(batch[q_field])
    gt: str = mit.one(batch[a_field])

    num_skipped = 0
    candidate_result: GeneratedConversation | None = None
    num_attempts = 0

    if use_few_shots:
        assert few_shot_examples, "few_shot_examples must be a non-empty list"

        messages = create_few_shot_prompt(
            question=q, 
            few_shot_examples=few_shot_examples,
            q_field=q_field,
            a_field=a_field
        )
    else:
        messages = [{"role": "user", "content": q}]

    while not candidate_result:
        num_attempts += 1
        if num_attempts > 1:
            candidate_result = None
            num_skipped += 1
            break

        gen_kwargs = vllm.SamplingParams(**{
            "max_tokens": generation_max_length,  # This is max_new_tokens not max_tokens. Important.
            "temperature": temperature, 
            "n": num_candidates,
        })
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        
        outputs = None
        for _ in range(max_retries):
            try:
                outputs = ray.get(
                    model.chat.remote(
                        messages, 
                        sampling_params=gen_kwargs,
                        use_tqdm=False,
                    )
                )[0].outputs

                break
            except ray.exceptions.RayActorError as e:
                rich.print(f"Error: {e}")
                rich.print(f"Retrying...")
                model = mit.one(vllm_utils2.create_vllm_engines(
                    num_engines=1,
                    tensor_parallel_size=1,
                    pretrain=model_name,
                    revision=None,
                    seed=0,
                    enable_prefix_caching=True,
                    max_model_len=forward_max_length, # Might cause issues .. but this is the logical way to do it
                ))
            
                ray.get(model.init_process_group.remote(
                    backend=backend,
                    world_size=num_gpus,
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=gpu_id,
                    group_name=group_name,
                ))

                ray.get(training_agent.broadcast_to_vllm.remote(
                    vllm_engines=[model],
                    gather_whole_model=gather_whole_model,
                ))

                continue
        if not outputs:
            raise ValueError(f"Failed to generate any outputs after {max_retries} retries")

        results = [output.text for output in outputs]
        # Loop over candidates and pick the first correct one.
        are_good = []
        
        for candidate_text in results:
            unit = is_correct(model_output=candidate_text, ground_truth_answer=gt) 
            are_good.append(unit)
        
        # print(f"Correctness: {are_good}")

        for candidate_text in results:
            if is_correct(model_output=candidate_text, ground_truth_answer=gt):
                candidate_messages = [
                    {"role": "user", "content": q}, 
                    {"role": "assistant", "content": candidate_text}
                ]
                tokenized_conversation = tokenize_conversations(
                    forward_tokenizer=forward_tokenizer,
                    conversations=[candidate_messages]
                )
                assert isinstance(tokenized_conversation, torch.Tensor), f"{type(tokenized_conversation).mro() = }"
                if tokenized_conversation.shape[1] > forward_max_length:
                    rich.print(f"Candidate text is too long: {tokenized_conversation.shape[1]} > {forward_max_length}")
                    continue
                
                candidate_result = GeneratedConversation(
                    tokenized_conversation=tokenized_conversation,
                    question_text=q,
                    reference_answer_text=gt,
                    generated_answer_text=candidate_text, # The last generated candidate_text
                )
                break
    
    model_queue.put((model, gpu_id))

    if candidate_result: 
        return ReturnState.SUCCESS, candidate_result
    
    return ReturnState.SKIP, None


def generate_train_batch(
        *,
        a_field: str,
        batch_size: int,
        few_shot_examples: list[dict[str, str]],
        finished_epoch: bool,
        generation_max_length: int,
        is_correct: typing.Callable[[str, str], bool],
        loader: torch.utils.data.DataLoader,
        loader_lock: threading.Lock,
        model_queue: queue.Queue,
        num_candidates: int,
        pool: cf.ThreadPoolExecutor,
        q_field: str,
        samples_attempted: int,
        samples_successful: int,
        temperature: float,
        top_p: float | None,
        use_few_shots: bool,
        forward_tokenizer: transformers.AutoTokenizer,
        forward_max_length: int,
        model_name: str,
        training_agent: trainer_lib.Trainer,
        gather_whole_model: bool,
        max_retries: int,
        num_gpus: int,
        master_address: str,
        master_port: int,
        group_name: str,
        backend: str,
    ) -> tuple[list[GeneratedConversation], int, int, bool]:

    train_batch = []
    futures = set()

    submit_args = (generate_one,)
    submit_kwargs = dict(
        loader=loader,
        loader_lock=loader_lock,
        q_field=q_field,
        a_field=a_field,
        temperature=temperature,
        num_candidates=num_candidates,
        top_p=top_p,
        model_queue=model_queue,
        is_correct=is_correct,
        use_few_shots=use_few_shots,
        few_shot_examples=few_shot_examples,
        forward_tokenizer=forward_tokenizer,
        forward_max_length=forward_max_length,
        model_name=model_name,
        training_agent=training_agent,
        gather_whole_model=gather_whole_model,
        generation_max_length=generation_max_length,
        max_retries=max_retries,
        num_gpus=num_gpus,
        master_address=master_address,
        master_port=master_port,
        group_name=group_name,
        backend=backend,
    )

    for i in range(batch_size):
        futures.add(pool.submit(*submit_args, **submit_kwargs))

    # As futures complete:
    while (
        len(train_batch) < batch_size and # There shouldn't be any more futures
        not (finished_epoch and not futures) # Done with epoch and no more futures to process
    ):
        # We call copy to avoid modifying the list while iterating.

        # for f in cf.as_completed(futures_copy):
        while futures:
            print(f"len(futures): {len(futures)}")
            done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            for f in done:
                status, generated_conversation = f.result()
                if status == ReturnState.END_EPOCH:
                    finished_epoch = True
                    # Finished the epoch, don't resubmit
                elif status == ReturnState.SUCCESS:
                    train_batch.append(generated_conversation)
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


def create_few_shot_prompt(
        *, 
        question: str, 
        few_shot_examples: list[dict[str, str]], 
        q_field: str, 
        a_field: str,
    ) -> list[dict[str, str]]:
    """
    Creates a few-shot prompt using chat format by prepending examples to the question.
    
    Args:
        question: The current question to answer
        few_shot_examples: List of example data points (dict with q_field and a_field)
        q_field: Field name for questions
        a_field: Field name for answers
        
    Returns:
        List of message dictionaries in chat format
    """
    if not few_shot_examples:
        raise ValueError("create_few_shot_prompt: Few shot examples are required")
    
    messages = []
    
    # Add example conversations
    for example in few_shot_examples:
        messages.append({"role": "user", "content": example[q_field]})
        messages.append({"role": "assistant", "content": example[a_field]})
    
    # Add the current question
    messages.append({"role": "user", "content": question})
    return messages

def save_pretrained(*, trainer: trainer_lib.Trainer, cfg_container: dict, epoch: int, output_dir: str):
    ray.get([trainer.save_pretrained.remote(
        output_dir=f"{output_dir}/epoch_{epoch}", 
        cfg_container=cfg_container, 
        wandb_run_id=wandb.run.id, 
        wandb_url=wandb.run.url
    )])


@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(cfg: omegaconf.DictConfig) -> None:
    # Print configuration for verification.
    print("Configuration:\n", omegaconf.OmegaConf.to_yaml(cfg))
    # Set GPU IDs for accelerate from config.
    # Initialize Accelerator using the module name.

    num_gpus = torch.cuda.device_count()
    rich.print(f"Number of GPUs: {num_gpus}")

    cfg_container = omegaconf.OmegaConf.to_container(cfg, resolve=True)

    wandb.init(
        project=cfg.wandb.project, 
        entity=cfg.wandb.entity,
        config=cfg_container,
        dir=os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp")),
    )

    cfg = config.Config(**cfg)

    # Unpack configuration parameters.
    model_name = cfg.model.name
    batch_size = cfg.training.batch_size
    learning_rate = cfg.training.learning_rate
    num_epochs = cfg.training.num_epochs
    gather_whole_model = False

    temperature = cfg.vllm_sampling.temperature
    num_candidates = cfg.vllm_sampling.num_candidates
    top_p = cfg.vllm_sampling.top_p  # May be null

    dataset_name = cfg.dataset.name
    dataset_split = cfg.dataset.split
    # valid_dataset_split = cfg.dataset.valid_split
    q_field = cfg.dataset.question_field
    a_field = cfg.dataset.answer_field
    load_dataset_args = cfg.dataset.load_dataset_args
    # evaluation_subset_size = cfg.evaluation.eval_subset

    is_correct = IS_CORRECT[dataset_name]

    forward_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        max_length=cfg.training.forward_max_length,
    )

    # --- Load training dataset ---
    viz_load_dataset_args = (dataset_name, *load_dataset_args), dict(split=dataset_split)
    print(f"Loading dataset {dataset_name} with args {viz_load_dataset_args}")
    train_dataset = datasets.load_dataset(dataset_name, *load_dataset_args, split=dataset_split)

    if cfg.few_shot_qty > 0:
        few_shot_examples = train_dataset.select(range(cfg.few_shot_qty))
    else:
        few_shot_examples = None

    if cfg.training.train_subset_mode:
        assert isinstance(cfg.training.train_subset_size, int), (
            f"{cfg.training.train_subset_size = } {type(cfg.training.train_subset_size).mro() = }")
        rich.print(f"[red bold]Training on subset of {cfg.training.train_subset_size} examples")
        train_dataset = torch.utils.data.Subset(
            train_dataset, 
            range(cfg.training.train_subset_size)
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=True, 
    )


    total_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * total_steps)

    # --- Load model and tokenizer ---
    num_gpus = torch.cuda.device_count()
    
    init_proc_group_refs = []
    master_address = "localhost"
    master_port = int(cfg.master_port)
    backend = "nccl"
    init_method = f"tcp://{master_address}:{master_port}"
    group_name = "update_group"

    init_method_internal = f"tcp://localhost:{int(cfg.internal_master_port)}" # This is for the trainer's (eventual) internal DDP


    trainer = trainer_lib.Trainer.remote(
        model_name=model_name,
        learning_rate=learning_rate,
        num_warmup_steps=num_warmup_steps,
        total_steps=total_steps,
        init_method_internal=init_method_internal,
        model_max_length=cfg.training.forward_max_length,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
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
        max_model_len=cfg.training.forward_max_length, # Might cause issues .. but this is the logical way to do it
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

    rich.print("[blue bold]Initializing process groups")
    ray.get(init_proc_group_refs)
    rich.print("[bold green]Initialized process groups")

    model_queue = queue.Queue()
    for gpu_id, engine in enumerate(vllm_engines, start=1):
        model_queue.put((engine, gpu_id))

    rich.print("[blue bold]Initializing model queue")
    global_step = 0
    pool = cf.ThreadPoolExecutor(max_workers=num_gpus)


    # Save the initial model, mostly to fail fast if something is wrong
    rich.print("[blue bold]Saving initial model")
    save_pretrained(
        trainer=trainer,
        cfg_container=cfg_container,
        epoch=0,
        output_dir=cfg.output_dir
    )
    rich.print("[blue bold]Saved initial model")

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1} / {num_epochs}")
        epoch_loss = 0.0
        num_batches = 0
        
        finished_epoch = False
        loader = iter(tqdm.tqdm(train_loader))
        loader_lock = threading.Lock()
        samples_attempted = 0
        samples_successful = 0

        rich.print("[blue bold]Starting epoch")
        while not finished_epoch:
            gc.collect()
            if cfg.training.train_subset_mode:
                rich.print(f"[red bold]Training on subset of {cfg.training.train_subset_size} examples")

            ray.get(trainer.broadcast_to_vllm.remote(
                vllm_engines=vllm_engines,
                gather_whole_model=gather_whole_model,
            ))
            
            start = time.perf_counter()

            train_batch, samples_attempted, samples_successful, finished_epoch = generate_train_batch(
                a_field=a_field,
                batch_size=batch_size,
                few_shot_examples=few_shot_examples,
                finished_epoch=finished_epoch,
                forward_max_length=cfg.training.forward_max_length,
                forward_tokenizer=forward_tokenizer,
                is_correct=is_correct,
                loader=loader,
                loader_lock=loader_lock,
                generation_max_length=cfg.training.generation_max_length,
                model_queue=model_queue,
                num_candidates=num_candidates,
                pool=pool,
                q_field=q_field,
                samples_attempted=samples_attempted,
                samples_successful=samples_successful,
                temperature=temperature,
                top_p=top_p,
                use_few_shots=cfg.few_shot_qty > 0,
                training_agent=trainer,
                gather_whole_model=gather_whole_model,
                model_name=model_name,
                max_retries=cfg.max_retries_vllm_crash,
                
                num_gpus=num_gpus,
                master_address=master_address,
                master_port=master_port,
                group_name=group_name,
                backend=backend,
            )
            torch.cuda.empty_cache()

            table = rich.table.Table(
                "Text", 
                "Lenght in Tokens",
                "Reference Answer Text",
                title="Train Batch", 
                title_style="bold green", 
                show_lines=True
            )
            for entry in train_batch:
                tokenized_entry = mit.one(entry.tokenized_conversation)
                decoded_text = forward_tokenizer.decode(tokenized_entry, skip_special_tokens=False)
                table.add_row(f"'{rich.markup.escape(decoded_text)}'", f"{len(tokenized_entry)}", f"{entry.reference_answer_text}")
            rich.print(table)
            

            # Make a few sanity checks about the batch. 
            batch_of_conversations = [mit.one(entry.tokenized_conversation) for entry in train_batch]
            breakpoint()

            if not batch_of_conversations:
                assert finished_epoch, "Batch is empty. This should only happen at the end of the epoch."
                rich.print("[bold orange2]Got an empty batch, but it was the end of the epoch, which makes this ok.")
                continue

            assert batch_of_conversations, "Batch is empty."
            print(f"{[len(x) for x in batch_of_conversations] = }", flush=True)
            assert all(isinstance(x, torch.Tensor) for x in batch_of_conversations), f"Each tokenized conversation must be a tensor. {[type(x) for x in batch_of_conversations] = }"
            assert all(x.ndim == 1 for x in batch_of_conversations), f"Each tensor must be 1D. {[x.ndim for x in batch_of_conversations] = }"
            assert all(x.dtype == torch.int64 for x in batch_of_conversations), f"Each tensor must be of type int64. {[x.dtype for x in batch_of_conversations] = }"
            
            loss = ray.get(trainer.train.remote(
                tokenized_batch_of_conversations=batch_of_conversations
            ))
            
            rich.print(rich.rule.Rule(
                f"[green bold]Done with step. Loss: {loss :.1e}", 
                align="left"
            ))

            epoch_loss += loss
            global_step += len(train_batch)
            num_batches += 1

            # Eval Every 10%
            # separator = int(cfg.evaluation.eval_percentage * (len(train_loader) // batch_size))
            # eval_test = num_batches % separator == 0
            # print(
            #     f"Num Batches: {num_batches}, " +
            #     f"Separator: {separator}, " + 
            #     f"{num_batches % separator}, " +
            #     f"Eval Test: {eval_test}"
            # )
 

            if global_step % cfg.wandb.log_interval == 0:
                wandb.log({
                    "train/loss": loss, 
                    "step": global_step, 
                    "attempted": samples_attempted,
                    "successful": samples_successful / samples_attempted if samples_attempted > 0 else 0.0,
                })

        rich.print(f"[red bold]FINISHED EPOCH {epoch} ##################################")
        save_pretrained(
            trainer=trainer,
            cfg_container=cfg_container,
            epoch=epoch + 1,
            output_dir=cfg.output_dir
        )
    print("Training complete.")


if __name__ == "__main__":
    
    main()
