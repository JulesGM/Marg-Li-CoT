"""
Supervised Trainer.

"""
import collections
import enum
import pathlib
import os
import sys

os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["DATASETS_VERBOSITY"] = "warning"
os.environ["WANDB_SILENT"] = "true"
os.environ["NCCL_DEBUG"] = "WARN"

import itertools as it
import logging

import accelerate
import datasets
import fire
import more_itertools as mi
import numpy as np
import peft
import peft_qlora
import rich
import rich.markup
import rich.table
import torch
import torch.backends
import torch.backends.cuda
import torch.utils
import torch.utils.data
import transformers
import transformers.utils
import trl
import trl.trainer.utils
import wandb
from tqdm import tqdm

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR.parent))

import lib_base_classes
import lib_trl_utils
import lib_metric
import lib_utils
import libs_extraction.lib_multiple_choice

import approach_sft.lib_dataloaders as lib_dataloaders
import approach_sft.lib_constants as lib_constants
import approach_sft.lib_tables as lib_tables
import approach_sft.lib_utils as stf_lib_utils

datasets.disable_caching()
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("deepspeed").setLevel(logging.WARNING)
datasets.logging.set_verbosity_warning()
transformers.logging.set_verbosity_warning()  # type: ignore
torch.backends.cuda.matmul.allow_tf32 = True

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
LOGGER = logging.getLogger(__name__)


########################################################################
# 🏎️ Change a lot
########################################################################
# DEFAULT_MODEL_NAME_OR_PATH = "EleutherAI/gpt-neo-125M"
# DEFAULT_MODEL_NAME_OR_PATH = "EleutherAI/pythia-1.4b-deduped" #
# DEFAULT_MODEL_NAME_OR_PATH = "EleutherAI/pythia-12b-deduped"

DEFAULT_DATA_DIRECTORY = pathlib.Path(
    "/network/scratch/g/gagnonju/saved_scratchpad_gen_outputs/chatgpt-3.5-commonsenseqa-scratchpads/cond-on-answers"
)
DEFAULT_MODEL_NAME_OR_PATH = "EleutherAI/gpt-j-6B"
DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_EVAL_BATCH_SIZE = 1

DEFAULT_OUTPUT_TYPE = lib_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_GEN_KWARGS = dict(
    repetition_penalty=3.0,
    early_stopping=True,
    max_new_tokens=200,
    min_new_tokens=4,
    
    do_sample=False,

    synced_gpus=os.getenv("ACCELERATE_DEEPSPEED_ZERO_STAGE", "") == "3",
)

########################################################################
# ✋ Regularization
########################################################################
DEFAULT_LORA_DROPOUT = 0.1

########################################################################
# 🛑 Never change
########################################################################
DEFAULT_LM_MODE = lib_constants.LMModes.CAUSAL_FULL
DEFAULT_MAX_TOTAL_LENGTH_TOK = 300
DEFAULT_BASE_PRECISION = lib_utils.ValidPrecisions.bfloat16
DEFAULT_N_BATCHES_PREDICT_TRAIN = 10
DEFAULT_TASK = lib_utils.Task.GSM8K
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "sft_output"
DEFAULT_WANDB_PROJECT_NAME = "sft"
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_BATCH_TABLE_PRINT_QTY = 2
DEFAULT_PREDICT_QTY_PRINT = 2
DEFAULT_PEFT_CONFIG = dict(
    inference_mode=False,
    lora_dropout=0.,
    lora_alpha=16,
    r=16,
    bias="none",
)

########################################################################


def predict(
    *,
    model: peft.PeftModel,
    batch: lib_base_classes.DataListContainer,
    gen_kwargs,
    predict_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    split: str,
    epoch: int,
    accelerator: accelerate.Accelerator,
    metrics: dict[str, lib_base_classes.Metric],
    qty_print: int,
) -> dict[str, list[float]]:
    ###########################################################################
    # Preprocess & Generate
    ###########################################################################
    query = predict_tokenizer.pad(
        dict(input_ids=batch.tok_ref_query),
        return_tensors="pt",
        padding=True,
    )

    lib_utils.not_last_token(
        tensor=query.input_ids,
        predict_tokenizer=predict_tokenizer,
    )

    predictions = accelerator.unwrap_model(model).generate(
        **query.to(LOCAL_RANK), **gen_kwargs
    )  # type: ignore

    ###########################################################################
    # Compute metrics
    ###########################################################################
    predictions_batch_obj = lib_base_classes.BatchedUnrollReturn(
        response_tensors=predictions,
        any_tokenizer=predict_tokenizer,
    )

    metric_outputs = {}
    local_metric_outputs = collections.defaultdict(list)
    for name, metric in metrics.items():
        local_metric_output = metric(
            responses=predictions_batch_obj, # type: ignore
            batch=batch,
        )

        local_metric_outputs[name] = local_metric_output # type: ignore
        pre_gather = np.fromiter((x for x in local_metric_output.values if x is not None), dtype=float).mean()
        pre_gather = torch.tensor(pre_gather).to(LOCAL_RANK)  # type: ignore
        metric_outputs[name] = accelerator.gather(pre_gather).mean().item()  # type: ignore

    ###########################################################################
    # Print some outputs
    ###########################################################################
    if RANK == 0:
        lib_tables.predict_table(
            predictions_batch_obj=predictions_batch_obj,
            local_metric_outputs=local_metric_outputs,
            predict_tokenizer=predict_tokenizer,
            tok_predict_query=query,
            predictions=predictions,
            qty_print=qty_print,
            batch=batch,
            split=split,
            epoch=epoch,
        )

    return metric_outputs


def iter_all_equal(iterable, key):
    iterator = iter(iterable)
    first = mi.first(iterator)
    return all(key(x) == key(first) for x in iterator)


def main(
    run_name,
    *,
    n_batches_predict_train=DEFAULT_N_BATCHES_PREDICT_TRAIN,
    batch_table_print_qty=DEFAULT_BATCH_TABLE_PRINT_QTY,
    wandb_project_name=DEFAULT_WANDB_PROJECT_NAME,
    model_name_or_path=DEFAULT_MODEL_NAME_OR_PATH,
    train_batch_size=DEFAULT_TRAIN_BATCH_SIZE,
    eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
    output_type=DEFAULT_OUTPUT_TYPE,
    num_epochs=DEFAULT_NUM_EPOCHS,
    precision=DEFAULT_BASE_PRECISION,
    lm_mode=DEFAULT_LM_MODE,
    task=DEFAULT_TASK,
    predict_qty_print=DEFAULT_PREDICT_QTY_PRINT,
    gen_kwargs=DEFAULT_GEN_KWARGS,
    data_directory=DEFAULT_DATA_DIRECTORY,
    peft_config_dict=DEFAULT_PEFT_CONFIG,

    # max_total_length_tok=DEFAULT_MAX_TOTAL_LENGTH_TOK,
    # output_dir=DEFAULT_OUTPUT_DIR,
    # lora_dropout=DEFAULT_LORA_DROPOUT,
):
    args = locals().copy()

    # We convert the enums to their values so they can be displayed in wandb.
    for k, v in args.items():
        if isinstance(v, enum.Enum):
            args[k] = v.value

    ###########################################################################
    # 🔍 Checks.
    ###########################################################################
    lm_mode = lib_constants.LMModes(lm_mode)
    task = lib_utils.Task(task)
    precision = lib_utils.ValidPrecisions(precision)
    is_encoder_decoder = stf_lib_utils.get_is_encoder_decoder(
        model_name_or_path)

    if is_encoder_decoder:
        assert (
            lm_mode == lib_constants.LMModes.SEQ2SEQ
        ), "Encoder-decoder models can only be trained in seq2seq mode."
    else:
        assert (
            lm_mode != lib_constants.LMModes.SEQ2SEQ
        ), "Causal models are not compatible with LMModes.SEQ2SEQ."

    if RANK == 0:
        wandb.init(
            name=run_name,
            project=wandb_project_name,
            config=dict(args=args, gen_kwargs=gen_kwargs),
        )

    metrics = dict(
        exact_match=lib_metric.ScratchpadAnswerAccuracy(
            libs_extraction.lib_multiple_choice.MultipleChoiceRfindExtractor(
            ["(A)", "(B)", "(C)", "(D)", "(E)"])
        ),
    )

    ###########################################################################
    # 🏗️ Load Model and Build Optimizer.
    ###########################################################################
    if RANK == 0:
        print(f"Loading model {model_name_or_path}")

    model = lib_trl_utils.load_then_peft_ize_model(
        prediction_tokenizer=prediction_tokenizer,
        forward_tokenizer=forward_tokenizer,
        peft_config_dict=peft_config_dict,
        model_name=model_name_or_path,
        precision=precision,
        just_device_map=False,
        use_peft=True,
    )

    if RANK == 0:
        print("Model loaded.")

    optimizer = torch.optim.Adam(model.parameters())
    model.print_trainable_parameters()

    ###########################################################################
    # 🏗️ Load Tokenizer and Data.
    ###########################################################################
    tokenizers = lib_utils.make_tokenizers_sft(
        model_name_or_path, 
        model=model,
        is_encoder_decoder=is_encoder_decoder
    )

    forward_tokenizer = tokenizers["forward_tokenizer"] 
    prediction_tokenizer = tokenizers["prediction_tokenizer"]
    del tokenizers

    assert not is_encoder_decoder
    dataloaders = lib_dataloaders.get_dataloaders(
        # max_total_length_tok=max_total_length_tok,
        # is_encoder_decoder=is_encoder_decoder,
        # task=task,
        data_directory=data_directory,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        output_type=output_type,
        forward_tokenizer=forward_tokenizer,
        prediction_tokenizer=prediction_tokenizer,
        lm_mode=lm_mode,
    )

    ###########################################################################
    # 🏎️ Accelerator business
    ###########################################################################
    accelerator = accelerate.Accelerator()
    model, optimizer, dataloaders = accelerator.prepare(model, optimizer, dataloaders)
    # dataloaders = {
    #     split: accelerator.prepare_data_loader(dataloader)
    #     for split, dataloader in dataloaders.items()
    # }

    ###########################################################################
    # Main Loop
    ###########################################################################
    for epoch in tqdm(range(num_epochs), disable=RANK != 0, desc="Epochs"):
        accelerator.unwrap_model(model).print_trainable_parameters()  # type: ignore

        # Train
        for i, batch in enumerate(
            tqdm(dataloaders["train"], disable=RANK != 0, desc="Train Batches")
        ):
            # Predict on Training
            train_metrics_outputs = None
            if i < n_batches_predict_train:
                lib_tables.batch_table(
                    batch=batch["forward"],
                    forward_tokenizer=forward_tokenizer,
                    print_qty=batch_table_print_qty,
                    epoch_idx=epoch,
                    num_epochs=num_epochs,
                    batch_idx=i,
                    num_batches=len(dataloaders["train"]),
                    is_forward=True,
                )

                model.eval()
                train_metrics_outputs = predict(
                    predict_tokenizer=prediction_tokenizer,
                    accelerator=accelerator,
                    gen_kwargs=gen_kwargs,
                    metrics=metrics, # type: ignore
                    split="train",
                    model=model,
                    batch=batch["predict"],
                    epoch=epoch,
                    qty_print=predict_qty_print,
                )

            ###################################################################
            # Training Step
            ###################################################################
            optimizer.zero_grad()
            model.train()
            lib_utils.not_first_token(
                forward_tokenizer=forward_tokenizer,
                tensor=batch["forward"]["input_ids"],
            )
            lib_utils.not_first_token(
                forward_tokenizer=forward_tokenizer,
                tensor=batch["forward"]["labels"],
            )
            loss = model(
                **{k: v.to(LOCAL_RANK) for k, v in batch["forward"].items()}
            ).loss

            accelerator.backward(loss)
            optimizer.step()

            # Training Logging Logging
            loss_logging = accelerator.gather(loss.detach()).mean() # type: ignore

            if RANK == 0:
                wandb.log({"train/loss": loss_logging})
                if train_metrics_outputs:
                    wandb.log(
                        {
                            f"train/metrics/{k}": np.mean(v)
                            for k, v in train_metrics_outputs.items()
                        }
                    )

        #######################################################################
        # Eval on the whole dataset.
        # 1. Generate predictions to see the model's performance.
        # 2. Compute the perplexity on the correct answer.
        #######################################################################
        eval_losses = []
        eval_metrics_outputs = collections.defaultdict(
            list
        )  
        for i, batch in enumerate(
            tqdm(dataloaders["eval"], disable=RANK != 0, desc="Eval Batches")
        ):
            model.eval()
            if i == 0:
                lib_tables.batch_table(
                    forward_tokenizer=forward_tokenizer,
                    num_batches=len(dataloaders["eval"]),
                    num_epochs=num_epochs,
                    is_forward=True,
                    print_qty=batch_table_print_qty,
                    batch_idx=i,
                    epoch_idx=epoch,
                    batch=batch["forward"],
                )

            #######################################################################
            # Predict & Stack metrics
            #######################################################################
            metrics_outputs_batch = predict(
                predict_tokenizer=prediction_tokenizer,
                accelerator=accelerator,
                gen_kwargs=gen_kwargs,
                qty_print=predict_qty_print,
                metrics=metrics, # type: ignore
                model=model,
                batch=batch["predict"],
                split="eval",
                epoch=epoch,
            )

            for k, v in metrics_outputs_batch.items():
                eval_metrics_outputs[k].append(v)  # type: ignore

            assert iter_all_equal(eval_metrics_outputs.values(), lambda v: len(v)), {
                k: len(v) for k, v in eval_metrics_outputs.items()
            }

            #######################################################################
            # Eval Loss, Gather & Stack
            #######################################################################
            lib_utils.not_first_token(
                tensor=batch["forward"]["input_ids"],
                forward_tokenizer=forward_tokenizer,
            )

            lib_utils.not_first_token(
                tensor=batch["forward"]["labels"],
                forward_tokenizer=forward_tokenizer,
            )

            model.eval()
            with torch.no_grad():
                loss = model(
                    **{k: v.to(LOCAL_RANK) for k, v in batch["forward"].items()}
                ).loss

            # Eval Logging
            loss_logging = accelerator.gather(loss.detach()).mean()  # type: ignore
            eval_losses.append(loss_logging.item())

        #######################################################################
        # Log the Eval Metrics
        #######################################################################
        if RANK == 0:
            wandb.log({"eval/loss": np.mean(eval_losses)})
            wandb.log(
                {
                    f"eval/metrics/{k}": np.mean(v)  # type: ignore
                    for k, v in eval_metrics_outputs.items()
                }
            )


if __name__ == "__main__":
    fire.Fire(main)
