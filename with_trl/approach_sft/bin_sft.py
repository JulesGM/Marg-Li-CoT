"""
Supervised Trainer.

"""
import collections
import enum
import pathlib
import os
import sys
from typing import Any, Optional, Union

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
import rich
import rich.markup
import rich.table
import rich.traceback
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

import approach_sft.lib_sft_dataloaders as lib_sft_dataloaders
import approach_sft.lib_sft_constants as lib_sft_constants
import approach_sft.lib_sft_tables as lib_sft_tables
import approach_sft.lib_sft_utils as lib_sft_utils

rich.traceback.install()
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
# DEFAULT_MODEL_NAME_OR_PATH = "EleutherAI/pythia-410m"; DEFAULT_TRAIN_BATCH_SIZE = 64; DEFAULT_EVAL_BATCH_SIZE = 128
DEFAULT_MODEL_NAME_OR_PATH = "EleutherAI/gpt-j-6B"; DEFAULT_TRAIN_BATCH_SIZE = 8; DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_USE_PEFT = True


DEFAULT_OUTPUT_TYPE = lib_sft_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=200,
    min_new_tokens=4,
    
    do_sample=False,
    early_stopping=True,
    repetition_penalty=1,
    synced_gpus=os.getenv("ACCELERATE_DEEPSPEED_ZERO_STAGE", "") == "3",
    use_cache=True,
)

########################################################################
# ✋ Regularization
########################################################################
DEFAULT_LORA_DROPOUT = 0.1

########################################################################
# 🛑 Never change
########################################################################
DEFAULT_LM_MODE = lib_sft_constants.LMModes.CAUSAL_FULL
DEFAULT_MAX_TOTAL_LENGTH_TOK = 300
DEFAULT_BASE_PRECISION = lib_utils.ValidPrecisions.bfloat16
DEFAULT_N_BATCHES_PREDICT_TRAIN = 20

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "sft_output"
DEFAULT_WANDB_PROJECT_NAME = "sft"
DEFAULT_WANDB_ENTITY = "julesgm"
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_QTY_EVAL_SMALL = 200
DEFAULT_BATCH_TABLE_PRINT_QTY = 2
DEFAULT_PREDICT_QTY_PRINT = 2
DEFAULT_PEFT_CONFIG = dict(
    inference_mode=False,
    lora_dropout=0.,
    lora_alpha=16,
    r=16,
    bias="none",
    task_type=peft.TaskType.CAUSAL_LM,

)
########################################################################

def predict(
    *,
    accelerator: accelerate.Accelerator,
    batch: lib_base_classes.DataListContainer,
    epoch: int,
    gen_kwargs,
    metrics: dict[str, lib_base_classes.Metric],
    model: peft.PeftModel,
    predict_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    qty_print: int,
    split: str,
) -> dict[str, list[float]]:
    ###########################################################################
    # Preprocess & Generate
    ###########################################################################
    query = batch["predict"]

    lib_utils.not_last_token(
        tensor=query.input_ids,
        predict_tokenizer=predict_tokenizer,
    )

    predictions = accelerator.unwrap_model(model).generate(
        **query.to(LOCAL_RANK), **gen_kwargs
    )[:, query["input_ids"].shape[1]:] # type: ignore

    ###########################################################################
    # Compute metrics
    ###########################################################################
 
    metric_outputs = {}
    local_metric_outputs = collections.defaultdict(list)
    for name, metric in metrics.items():
        local_metric_output = metric(
            responses=predict_tokenizer.batch_decode(predictions), # type: ignore
            batch=lib_base_classes.DataListContainer(
                tok_ref_query=None,
                tok_ref_answer=None,
                tok_ref_scratchpad=None,
                detok_ref_query=None,
                detok_ref_answer=batch["extra_info"]["ref_qa_answer"],
                detok_ref_scratchpad=None,
                obj_ref_equations=None,
            ),
        )

        local_metric_outputs[name] = local_metric_output # type: ignore
        pre_gather = np.fromiter(
            (x for x in local_metric_output.values if x is not None), dtype=float).mean()
        pre_gather = torch.tensor(pre_gather).to(LOCAL_RANK)  # type: ignore
        metric_outputs[name] = accelerator.gather(pre_gather).mean().item()  # type: ignore

    ###########################################################################
    # Print some outputs
    ###########################################################################
    if RANK == 0:
        lib_sft_tables.predict_table(
            predictions_batch_obj=lib_base_classes.BatchedUnrollReturn(
                response_tensors=predictions,
                raw_response_tensors=None, 
                any_tokenizer=predict_tokenizer,
                ), # type: ignore
            local_metric_outputs=local_metric_outputs,
            predict_tokenizer=predict_tokenizer,
            tok_predict_query=query,
            predictions=predictions,
            qty_print=qty_print,
            batch=lib_base_classes.DataListContainer(
                tok_ref_query=None,
                tok_ref_answer=None,
                tok_ref_scratchpad=None,
                detok_ref_query=predict_tokenizer.batch_decode(query["input_ids"]),
                detok_ref_answer=None,
                detok_ref_scratchpad=None,
            ),
            split=split,
            epoch=epoch,
        )

    return metric_outputs


def iter_all_equal(iterable, key):
    iterator = iter(iterable)
    first = mi.first(iterator)
    return all(key(x) == key(first) for x in iterator)


def step(
    *, 
    accelerator: accelerate.Accelerator,
    batch,
    cv_set: lib_sft_constants.CVSet,
    forward_tokenizer: transformers.PreTrainedTokenizerBase,
    model: peft.peft_model.PeftModelForCausalLM,
    optimizer: torch.optim.Optimizer,
):
    if cv_set:
        optimizer.zero_grad()
        model.train()

    lib_utils.not_first_token(
        forward_tokenizer=forward_tokenizer,
        tensor=batch["forward"]["input_ids"],
    )

    loss = model(
        **{k: v.to(LOCAL_RANK) for k, v in batch["forward"].items()},
        labels=batch["forward"]["input_ids"],
    ).loss

    if cv_set == lib_sft_constants.CVSet.TRAIN:
        accelerator.backward(loss)
        optimizer.step()

    # Training Logging Logging
    loss_logging = accelerator.gather(loss.detach()).mean() # type: ignore

    if RANK == 0:
        wandb.log({f"{cv_set.value}/loss": loss_logging.item()})
        

    return loss_logging


def evaluate(
    *,
    accelerator: accelerate.Accelerator,
    batch,
    batch_idx: int,
    batch_table_print_qty: int,
    cv_split: lib_sft_constants.CVSet,
    dataloaders,
    epoch_idx: int,
    forward_tokenizer: transformers.PreTrainedTokenizerBase,
    gen_kwargs,
    max_num_epochs: int,
    metrics: dict[str, lib_base_classes.Metric],
    model: peft.peft_model.PeftModelForCausalLM,
    predict_qty_print: int,
    prediction_tokenizer: transformers.PreTrainedTokenizerBase,
):
    # Predict on Training
    lib_sft_tables.batch_table(
        batch=batch["forward"],
        forward_tokenizer=forward_tokenizer,
        print_qty=batch_table_print_qty,
        epoch_idx=epoch_idx,
        num_epochs=max_num_epochs,
        batch_idx=batch_idx,
        num_batches=len(dataloaders[cv_split]),
        is_forward=True,
    )

    model.eval()
    metrics_outputs = predict(
        predict_tokenizer=prediction_tokenizer,
        accelerator=accelerator,
        gen_kwargs=gen_kwargs,
        metrics=metrics, # type: ignore
        split=cv_split,
        model=model,
        batch=batch,
        epoch=epoch_idx,
        qty_print=predict_qty_print,
    )
    
    if RANK == 0:
        wandb.log(
            {
                f"{cv_split}/metrics/{k}": np.mean(v)
                for k, v in metrics_outputs.items()
            }
        )
    


def main(
    run_name,
    *,
    batch_table_print_qty=DEFAULT_BATCH_TABLE_PRINT_QTY,
    data_directory=DEFAULT_DATA_DIRECTORY,
    eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
    gen_kwargs=DEFAULT_GEN_KWARGS,
    lm_mode=DEFAULT_LM_MODE,
    max_num_epochs=DEFAULT_NUM_EPOCHS,
    model_name_or_path=DEFAULT_MODEL_NAME_OR_PATH,
    n_batches_predict_train=DEFAULT_N_BATCHES_PREDICT_TRAIN,
    output_type=DEFAULT_OUTPUT_TYPE,
    peft_config_dict=DEFAULT_PEFT_CONFIG,
    precision=DEFAULT_BASE_PRECISION,
    predict_qty_print=DEFAULT_PREDICT_QTY_PRINT,
    qty_eval_small=DEFAULT_QTY_EVAL_SMALL,
    train_batch_size=DEFAULT_TRAIN_BATCH_SIZE,
    use_peft=DEFAULT_USE_PEFT,
    wandb_entity=DEFAULT_WANDB_ENTITY,
    wandb_project_name=DEFAULT_WANDB_PROJECT_NAME,
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
    data_mode = lib_sft_constants.DataModes(data_mode)
    lm_mode = lib_sft_constants.LMModes(lm_mode)
    precision = lib_utils.ValidPrecisions(precision)
    is_encoder_decoder = lib_sft_utils.get_is_encoder_decoder(
        model_name_or_path)

    assert not is_encoder_decoder, "Encoder decoder not supported yet."
    
    if RANK == 0:
        if "SLURM_TMPDIR" not in os.environ:
            job_id = os.environ["SLURM_JOB_ID"]
            tmp_dir = pathlib.Path(f"/Tmp/slurm.{job_id}.0")
            assert tmp_dir.exists(), f"{tmp_dir} does not exist."
        else:
            tmp_dir = pathlib.Path(os.environ["SLURM_TMPDIR"])

        wandb.init(
            name=run_name,
            project=wandb_project_name,
            entity=wandb_entity,
            config=dict(args=args, gen_kwargs=gen_kwargs),
            dir=tmp_dir / "wandb",
        )

    metrics = dict(
        exact_match=lib_metric.ScratchpadAnswerAccuracy(
            libs_extraction.lib_multiple_choice.MultipleChoiceRfindExtractor(
            ["(A)", "(B)", "(C)", "(D)", "(E)"])
        ),
    )
    
    ###########################################################################
    # 🏗️ Load Tokenizer and Data.
    ###########################################################################
    tmp_tokenizers = lib_trl_utils.load_tokenizers(
        model_name=model_name_or_path, 
        config=transformers.AutoConfig.from_pretrained(model_name_or_path),
    )

    forward_tokenizer = tmp_tokenizers["forward_tokenizer"]
    prediction_tokenizer = tmp_tokenizers["prediction_tokenizer"]
    del tmp_tokenizers

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
        use_peft=use_peft,
        adapter_name="default",
    )

    if RANK == 0:
        print("Model loaded.")

    optimizer = torch.optim.Adam(model.parameters())
    model.print_trainable_parameters()

    assert not is_encoder_decoder
    dataloaders, small_eval_dataloader = lib_sft_dataloaders.get_dataloaders(
        data_directory=data_directory,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        output_type=output_type,
        forward_tokenizer=forward_tokenizer,
        prediction_tokenizer=prediction_tokenizer,
        lm_mode=lm_mode,
        qty_eval_small=qty_eval_small,
    )

    ###########################################################################
    # 🏎️ Accelerator business
    ###########################################################################
    accelerator = accelerate.Accelerator()
    model, optimizer, dataloaders, small_eval_dataloader = accelerator.prepare(
        model, optimizer, dataloaders, small_eval_dataloader)

    ###########################################################################
    # Main Loop
    ###########################################################################
    for epoch_idx in tqdm(range(max_num_epochs), disable=RANK != 0, desc="Epochs"):
        accelerator.unwrap_model(model).print_trainable_parameters()  # type: ignore
        train_dataset_iterator = iter(dataloaders["train"])

        at_least_one = True
        while at_least_one:
            at_least_one = False

            # Train
            for batch_idx, batch in enumerate(
                tqdm(
                    it.islice(train_dataset_iterator, n_batches_predict_train), 
                    disable=RANK != 0, desc="Train Batches",
                )
            ):
                at_least_one = True
                cv_split = lib_sft_constants.CVSet.TRAIN
                    
                step(
                    accelerator=accelerator,
                    batch=batch,
                    cv_set=cv_split,
                    forward_tokenizer=forward_tokenizer,
                    model=model,
                    optimizer=optimizer,
                )

                if batch_idx % 10 == 0:
                    evaluate(
                        accelerator=accelerator,
                        batch_idx=batch_idx,
                        batch=batch,
                        cv_split=cv_split,
                        dataloaders=dataloaders,
                        epoch_idx=epoch_idx,
                        forward_tokenizer=forward_tokenizer,
                        gen_kwargs=gen_kwargs,
                        predict_qty_print=predict_qty_print,
                        prediction_tokenizer=prediction_tokenizer,
                        batch_table_print_qty=batch_table_print_qty,
                        max_num_epochs=max_num_epochs,
                        model=model,
                        metrics=metrics,
                    )

            for batch_idx, batch in enumerate(
                tqdm(small_eval_dataloader, disable=RANK != 0, desc="Eval Batches")
            ):
                cv_split = lib_sft_constants.CVSet.VALIDATION

                with torch.no_grad():
                    step(
                        accelerator=accelerator,
                        batch=batch,
                        cv_set=cv_split,
                        forward_tokenizer=forward_tokenizer,
                        model=model,
                        optimizer=optimizer,
                    )
                    
                    evaluate(
                        accelerator=accelerator,
                        batch_idx=batch_idx,
                        batch=batch,
                        cv_split=cv_split,
                        dataloaders=dataloaders,
                        epoch_idx=epoch_idx,
                        forward_tokenizer=forward_tokenizer,
                        gen_kwargs=gen_kwargs,
                        predict_qty_print=predict_qty_print,
                        prediction_tokenizer=prediction_tokenizer,
                        batch_table_print_qty=batch_table_print_qty,
                        max_num_epochs=max_num_epochs,
                        model=model,
                        metrics=metrics,
                    )

    for batch in tqdm(
        dataloaders[lib_sft_constants.CVSet.VALIDATION], 
        disable=RANK != 0, 
        desc="Test Batches"
    ):
        with torch.no_grad():
            step(
                accelerator=accelerator,
                batch=batch,
                cv_set=cv_split,
                forward_tokenizer=forward_tokenizer,
                model=model,
                optimizer=optimizer,
            )
            
            evaluate(
                accelerator=accelerator,
                batch_idx=batch_idx,
                batch=batch,
                cv_split=cv_split,
                dataloaders=dataloaders,
                epoch_idx=epoch_idx,
                forward_tokenizer=forward_tokenizer,
                gen_kwargs=gen_kwargs,
                predict_qty_print=predict_qty_print,
                prediction_tokenizer=prediction_tokenizer,
                batch_table_print_qty=batch_table_print_qty,
                max_num_epochs=max_num_epochs,
                model=model,
                metrics=metrics,
            )

if __name__ == "__main__":
    fire.Fire(main)
