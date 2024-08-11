#!/usr/bin/env python
# coding: utf-8


import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import collections
import contextlib
import logging
import random

import accelerate
import mlc_datasets
import fire
import numpy as np
import peft
import rich
import rich.status
import torch
from tqdm.rich import tqdm
import transformers
import torch

from general_utils import parallel_print as pprint


mlc_datasets.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


@contextlib.contextmanager
def pstatus(message):
    if os.environ["RANK"] == "0":
        with rich.status.Status(message) as status:
            yield status
    else:
        yield


random    .seed(0)
np.random .seed(0)
torch     .manual_seed(0)
torch.cuda.manual_seed(0)

device = 0
model_name_or_path     = "google/flan-ul2"
tokenizer_name_or_path = model_name_or_path

train_batch_size = 1
eval_batch_size  = 1

label_column     = "text_label"
text_column      = "sentence"

input_max_length_tokenizer = 200
outout_max_length_tokenizer = 3
padding_strategy = True
tokenizer_truncate = True

num_epochs       = 3
lr               = 1e-3

peft_config = peft.LoraConfig(
    inference_mode = False, 
    lora_dropout   = 0.1,
    lora_alpha     = 32, 
    task_type      = peft.TaskType.SEQ_2_SEQ_LM, 
    r              = 8, 
)

print_eval_every_x_step = 500


def preprocess_function(
        examples, tokenizer, text_column, label_column
    ):

    inputs       = examples[text_column]
    targets      = examples[label_column]
    prompt       = (
        "Answer if the sentiment of the following "
        "sentence is positive, negative or neutral: "
    )
    inputs       = [prompt + x for x in inputs]

    model_inputs = tokenizer(
        inputs, 
        return_tensors = "pt",
        max_length     = input_max_length_tokenizer,
        truncation     = tokenizer_truncate,
        padding        = padding_strategy, 
    )
    labels       = tokenizer(
        targets, 
        return_tensors = "pt",
        max_length     = outout_max_length_tokenizer, 
        truncation     = tokenizer_truncate,
        padding        = padding_strategy, 
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels

    return model_inputs

def make_dataset(model):
    # loading dataset
    dataset = mlc_datasets.load_dataset(
        "financial_phrasebank", "sentences_allagree"
    )
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset["validation"] = dataset["test"]
    del dataset["test"]

    classes = dataset["train"].features["label"].names
    dataset = dataset.map(
        lambda x: {"text_label": [
            classes[label] for label in x["label"]
        ]},
        batched=True,
        num_proc=1,
    )

    # data preprocessing
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path)

    processed_datasets = dataset.map(
        lambda examples: preprocess_function(
            label_column = label_column,
            text_column  = text_column, 
            tokenizer    = tokenizer, 
            examples     = examples, 
        ),
        load_from_cache_file = False,
        remove_columns       = dataset["train"].column_names,
        num_proc             = 1,
        batched              = True,
        desc                 = "Running tokenizer on dataset",
    )

    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, 
        return_tensors     = "pt",
        max_length         = (
            input_max_length_tokenizer if 
            padding_strategy =="max_length" else None
        ),
        padding            = padding_strategy,
        model              = model,
    )

    train_dataloader = torch.utils.data.DataLoader(
        processed_datasets["train"], 
        collate_fn = collator,
        batch_size = train_batch_size, 
        pin_memory = True,
        shuffle    = True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        processed_datasets["validation"], 
        collate_fn = collator, 
        batch_size = eval_batch_size, 
        pin_memory = True,
        shuffle    = False,
    )

    return (
        train_dataloader,
        eval_dataloader,
        dataset,
        tokenizer
    )

def calc_acc(
        *, 
        dataset,
        preds, 
    ):
    
    correct = 0
    total = 0
    incorrect = collections.Counter()

    assert len(preds) == len(dataset["text_label"]), (
        f"{len(preds)                        = } != "
        f"{len(dataset['text_label']) = }\n"
    )
    for pred, true in zip(preds, dataset["text_label"]):
        if pred.strip() == true.strip():
            correct += 1
        else:
            incorrect.update([pred])
        total += 1
    accuracy = correct / total

    pprint(
        f"\n"
        f"{accuracy = :0.2%} on the evaluation dataset\n"
    )

def eval_epoch(
        *, 
        eval_dataloader,
        eval_dataset,
        accelerator,
        tokenizer, 
        fn_model, 
    ):

    eval_preds = []
    eval_loss  = torch.tensor(0, dtype=torch.float32)

    for step, batch in enumerate(tqdm(
        eval_dataloader, 
        desc="Evaluating", 
        disable=os.environ["RANK"] != "0",
    )):
        accelerator.unwrap_model(fn_model).eval()
        
        # We test `is False` because it can't be None.
        assert accelerator.unwrap_model(fn_model).training is False, (
            f"{accelerator.unwrap_model(fn_model).training = }",
            f"{fn_model.training = }",
        )

        batch    = {
            k: v.to(fn_model.device) 
            for k, v in batch.items()
        }

        with torch.no_grad():
            outputs = fn_model(**batch)

        loss       = outputs.loss # Hard to split for gather, annoying.
        eval_loss += loss.detach().float().item()
        pred = torch.argmax(outputs.logits, -1).detach()
        padded_eval_pred = accelerator.pad_across_processes(
            pred, 
            pad_index = tokenizer.pad_token_id,
            dim       = 1, 
        )
        gathered_padded_eval_pred = accelerator.gather_for_metrics(
            padded_eval_pred
        )
        eval_preds.append(gathered_padded_eval_pred)


    eval_loss  = accelerator.gather(
        eval_loss.to(accelerator.device)
    ).sum()

    catted_eval_preds = torch.cat(
        [x.cpu() for x in eval_preds], 
        dim=0,
    ).numpy()

    assert len(eval_dataset) == len(catted_eval_preds), (
        f"{len(eval_dataset) = } != {len(catted_eval_preds) = }"
    )

    decoded_catted_eval_preds = tokenizer.batch_decode(
        catted_eval_preds, 
        skip_special_tokens=True,
    )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl   = torch.exp(eval_epoch_loss)

    return (
        eval_epoch_loss.item(), 
        eval_ppl.item(), 
        decoded_catted_eval_preds,
    )

def train_epoch(
        *,
        train_dataloader, 
        eval_dataloader, 
        lr_scheduler, 
        accelerator,
        eval_every, 
        tokenizer,
        optimizer, 
        fn_model, 
        dataset,
        epoch, 
):
    total_loss = torch.tensor(0, dtype=torch.float32)

    for step, batch in enumerate(tqdm(
        train_dataloader, 
        desc="Training", 
        disable=os.environ["RANK"] != "0",
    )):
        
        optimizer.zero_grad()
        accelerator.unwrap_model(fn_model).train()

        assert accelerator.unwrap_model(fn_model).training, (
            f"\nExpected True, got:"
            f"\t{accelerator.unwrap_model(fn_model).training = }\n"
            f"\t{fn_model.training = }\m"
        )

        batch       = {k: v.to(device) for k, v in batch.items()}
        outputs     = fn_model(**batch)
        loss        = outputs.loss
        total_loss += loss.detach().float().cpu()

        accelerator .backward(loss)
        optimizer   .step()        
        lr_scheduler.step()

        if (
            eval_dataloader is not None and 
            step % eval_every == 0      and 
            step > 0
        ):
            _, _, eval_preds = eval_epoch(
                eval_dataloader = eval_dataloader,
                eval_dataset    = dataset["validation"],
                accelerator     = accelerator,
                tokenizer       = tokenizer, 
                fn_model        = fn_model, 
            )
            pprint(f"[bold green]{epoch} - {step}:[/] ")
            calc_acc(
                dataset = dataset["validation"],
                preds   = eval_preds, 
            )
    
    # TODO(@julesgm): If we really wanted the accurate loss,
    # we would need to use gather_for_metrics like in eval with the preds.
    # The issue is that model.forward().loss already averages over the batch,
    # which prevents us from using gather_for_metrics, which excludes members
    # of the batch that are repeated to make constant batch sizes.
    total_loss = accelerator.gather(
        total_loss.to(accelerator.device)
    ).mean().cpu()

    return total_loss.item()


def main():
    accelerator = accelerate.Accelerator()

    dmap_keys = ["encoder", "lm_head", "shared", "decoder"]
    dmap = {k: int(os.environ["LOCAL_RANK"]) for k in dmap_keys}

    with pstatus("[bold green]Loading model..."):
        frozen_model = transformers.T5ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            load_in_8bit = True,
            torch_dtype  = torch.float16,
            device_map   = dmap,
        )
        for name, param in frozen_model.named_parameters():
            param.requires_grad = False
        peft.PeftModel.print_trainable_parameters(frozen_model)

    with pstatus("[bold green]Doing PEFT stuff..."):
        model = peft.get_peft_model(frozen_model, peft_config)
        model.print_trainable_parameters()

    with pstatus("[bold green]Making the dataset objects ..."):
        (
            train_dataloader, eval_dataloader, dataset, tokenizer
        ) = make_dataset(model)

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        num_training_steps = (len(train_dataloader) * num_epochs),
        num_warmup_steps   = 0,
        optimizer          = optimizer,
    )

    assert model is not None, "model is None"

    with pstatus("[bold green]accelerator.prepare(...) ..."):
        (
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        ) = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

    assert model is not None, "model is None"
    assert frozen_model is not None, "frozen_model is None"
    
    eval_epoch_loss, eval_ppl, eval_preds = eval_epoch(
        eval_dataloader = eval_dataloader,
        eval_dataset    = dataset["validation"],
        accelerator     = accelerator,
        tokenizer       = tokenizer,
        fn_model        = frozen_model, 
    )
    pprint(
        f"[bold blue]Zero shot frozen:[/] " 
        f"epoch = -1: "
        f"{eval_ppl        = :0.3} "
        f"{eval_epoch_loss = :0.3} "
    )

    calc_acc(
        dataset = dataset["validation"],
        preds   = eval_preds, 
    )

    eval_epoch_loss, eval_ppl, eval_preds = eval_epoch(
        eval_dataloader = eval_dataloader,
        eval_dataset    = dataset["validation"],
        accelerator     = accelerator,
        tokenizer       = tokenizer, 
        fn_model        = model, 
    )
    pprint(
        f"[bold green]Peft zero-shot:[/] "
        f"{eval_ppl        = :0.3} "
        f"{eval_epoch_loss = :0.3}"
    )
    calc_acc(
        dataset = dataset["validation"],
        preds   = eval_preds, 
    )

    for epoch in range(num_epochs):
        total_loss = train_epoch(
            train_dataloader = train_dataloader, 
            eval_dataloader  = eval_dataloader,
            lr_scheduler     = lr_scheduler,
            accelerator      = accelerator,
            eval_every       = print_eval_every_x_step,
            optimizer        = optimizer, 
            tokenizer        = tokenizer,
            fn_model         = model,
            dataset          = dataset,
            epoch            = epoch, 
        )
        
        eval_epoch_loss, eval_ppl, eval_preds = eval_epoch(
            eval_dataloader = eval_dataloader,
            eval_dataset    = dataset["validation"],
            accelerator     = accelerator,
            tokenizer       = tokenizer, 
            fn_model        = model, 
        )

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(torch.tensor(train_epoch_loss))
        train_ppl = accelerator.gather(
            train_ppl.to(accelerator.device)
        ).mean().item()

        pprint(
            f"[bold blue]"
            f"{epoch            = }:[/] "
            f"{train_ppl        = :0.3} "
            f"{train_epoch_loss = :0.3} "
            f"{eval_ppl         = :0.3} "
            f"{eval_epoch_loss  = :0.3} "
        )
        
        calc_acc(
            dataset = dataset["validation"],
            preds   = eval_preds, 
        )

    eval_epoch_loss, eval_ppl, eval_preds = eval_epoch(
        eval_dataloader = eval_dataloader,
        eval_dataset    = dataset["validation"],
        accelerator     = accelerator,
        tokenizer       = tokenizer, 
        fn_model        = model, 
    )
    pprint(
        f"[bold green]Peft zero-shot:[/] "
        f"{eval_epoch_loss = :0.3} "
        f"{eval_ppl        = :0.3} "
    )
    calc_acc(
        dataset = dataset["validation"],
        preds   = eval_preds, 
    )


if __name__ == "__main__":
    fire.Fire(main)
