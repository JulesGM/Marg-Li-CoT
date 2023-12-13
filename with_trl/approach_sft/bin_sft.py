"""
Supervised Trainer.

"""
import collections
import enum
import pathlib
import os
import random
import sys
from typing import Any, Optional, Union

os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["DATASETS_VERBOSITY"] = "warning"
# os.environ["WANDB_SILENT"] = "true"
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
import rich.console
import rich.markup
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
import lib_constant
import lib_trl_utils
import lib_metric
import lib_utils
import libs_extraction.lib_multiple_choice

import approach_sft.lib_sft_dataloaders as lib_sft_dataloaders
import approach_sft.lib_sft_constants as lib_sft_constants
import approach_sft.lib_sft_tables as lib_sft_tables
import approach_sft.lib_sft_utils as lib_sft_utils

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

rich.traceback.install(console=rich.console.Console(force_terminal=True))
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
wandb.Table.MAX_ARTIFACTS_ROWS = 10000000

########################################################################
# 🏎️ Change a lot
########################################################################
DEFAULT_DATASET = lib_utils.Datasets.ARITHMETIC

DEFAULT_DATA_DIRECTORY = pathlib.Path(
    # "/network/scratch/g/gagnonju/saved_scratchpad_gen_outputs/chatgpt-3.5-commonsenseqa-scratchpads/cond-on-answers"
    # "/network/scratch/g/gagnonju/saved_scratchpad_gen_outputs/chatgpt-3.5-commonsenseqa-scratchpads/not-cond-on-answers"
    # "/network/scratch/g/gagnonju/saved_scratchpad_gen_outputs/chatgpt-4-commonsenseqa-scratchpads/not-cond-on-answers"
    "/home/mila/g/gagnonju/Marg-Li-CoT/with_trl/libs_data/arithmetic/outputs"
)

ANSWER_ONLY_MODE = lib_sft_constants.OutputTypes.ANSWER_ONLY
SCRATCHPAD_MODE = lib_sft_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER

# DEFAULT_MODEL_NAME_OR_PATH = "EleutherAI/pythia-410m"; DEFAULT_OUTPUT_TYPE = lib_sft_constants.OutputTypes.CHAIN_OF_THOUGHT_THEN_ANSWER; DEFAULT_TRAIN_BATCH_SIZE = 64; DEFAULT_EVAL_BATCH_SIZE = 128; DEFAULT_USE_PEFT = True; 
DEFAULT_MODEL_NAME_OR_PATH = "EleutherAI/gpt-j-6B"; DEFAULT_OUTPUT_TYPE = ANSWER_ONLY_MODE; DEFAULT_TRAIN_BATCH_SIZE = 8; DEFAULT_EVAL_BATCH_SIZE = DEFAULT_TRAIN_BATCH_SIZE * 3; DEFAULT_USE_PEFT = True; 
# DEFAULT_MODEL_NAME_OR_PATH = "EleutherAI/gpt-j-6B"; DEFAULT_OUTPUT_TYPE = SCRATCHPAD_MODE; DEFAULT_TRAIN_BATCH_SIZE = 8; DEFAULT_EVAL_BATCH_SIZE = DEFAULT_TRAIN_BATCH_SIZE; DEFAULT_USE_PEFT = True; 
DEFAULT_N_BATCHES_PREDICT_TRAIN = 50


DEFAULT_PEFT_DO_ALL_LIN_LAYERS = True
DEFAULT_MASK_QUERY = False
DEFAULT_FILTER_OUT_BAD = True
DEFAULT_LEARNING_RATE = 10 ** -4

DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=20 if DEFAULT_OUTPUT_TYPE == lib_sft_constants.OutputTypes.ANSWER_ONLY else 300,
    min_new_tokens=1,
    
    do_sample=False,
    # early_stopping=True,
    repetition_penalty=1,
    synced_gpus=os.getenv("ACCELERATE_DEEPSPEED_ZERO_STAGE", "") == "3",
    temperature=1,
    use_cache=True,
    
)

########################################################################
# 🛑 Never change
########################################################################
DEFAULT_JUST_DEVICE_MAP = False
DEFAULT_LM_MODE = lib_sft_constants.LMModes.CAUSAL_FULL
DEFAULT_BASE_PRECISION = lib_utils.ValidPrecisions.bfloat16

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "sft_output"
DEFAULT_WANDB_PROJECT_NAME = "sft_arithmetic"
DEFAULT_WANDB_ENTITY = "julesgm"
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_QTY_EVAL_SMALL = 150
DEFAULT_BATCH_TABLE_PRINT_QTY = 2
DEFAULT_PREDICT_QTY_PRINT = 2
DEFAULT_PEFT_CONFIG = dict(
    inference_mode=False,
    lora_dropout=0.1,
    lora_alpha=16,
    r=16,
    bias="none",
    task_type=peft.TaskType.CAUSAL_LM,
)


########################################################################
def empty_cache(accelerator):
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    accelerator.free_memory()


def predict(
    *,
    accelerator: accelerate.Accelerator,
    batch: lib_base_classes.DataListContainer,
    epoch: int,
    gen_kwargs,
    global_step: int,
    metrics: dict[str, lib_base_classes.Metric],
    model: peft.PeftModel,
    predict_table: lib_utils.WandbTableRepair,
    predict_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    qty_print: int,
    split: str,
) -> dict[str, list[float]]:
    
    ###########################################################################
    # Generate
    ###########################################################################
    query = batch["predict"]
    lib_utils.not_last_token(
        predict_tokenizer = predict_tokenizer,
        tensor            = query.input_ids,
    )
    model.eval()
    query_seq_len = query["input_ids"].shape[1]
    predictions = accelerator.unwrap_model(model).generate(
            **query.to(accelerator.local_process_index), 
            **gen_kwargs
        )[:, query_seq_len:]
    response_text_for_metrics = predict_tokenizer.batch_decode(
            predictions, 
            skip_special_tokens=True,
        )

    ###########################################################################
    # Compute the metrics
    ###########################################################################
    #######################################
    # Prepare the inputs for the metrics, and the containers for the outputs
    #######################################
    metric_outputs = {}
    local_metric_outputs = collections.defaultdict(list)
    batch_for_metrics = lib_base_classes.DataListContainer(
        tok_ref_query        = None,
        tok_ref_answer       = None,
        tok_ref_scratchpad   = None,
        detok_ref_query      = batch["extra_info"]["ref_qa_question"],
        detok_ref_answer     = batch["extra_info"]["ref_qa_answer"],
        detok_ref_scratchpad = batch["extra_info"].get("ref_qa_scratchpad", None),
        obj_ref_equations    = None,
    )
    
    for name, metric in metrics.items():
        #######################################
        # Actually compute metrics
        #######################################
        local_metric_output = metric(
            batch     = batch_for_metrics,
            responses = response_text_for_metrics,)
        local_metric_outputs[name] = local_metric_output 
        
        #######################################
        # Gather metrics
        #######################################
        pre_gather = [x for x in local_metric_output.values if x is not None]
        pre_gather = torch.tensor(pre_gather).to(accelerator.local_process_index)
        metric_outputs[name] = accelerator.gather_for_metrics(
            pre_gather).mean().item()

    ###########################################################################
    # Print some outputs
    ###########################################################################
    if RANK == 0:
        prediction_batch_obj = lib_base_classes.BatchedUnrollReturn(
                response_tensors     = predictions,
                raw_response_tensors = None, 
                any_tokenizer        = predict_tokenizer,
            )
        
        lib_sft_tables.predict_table(
            batch                 = batch_for_metrics,
            epoch                 = epoch,
            global_step           = global_step,
            local_metric_outputs  = local_metric_outputs,
            predict_tokenizer     = predict_tokenizer,
            predictions           = predictions,
            predictions_batch_obj = prediction_batch_obj,
            qty_print             = qty_print,
            split                 = split,
            wandb_and_rich_table  = predict_table,
        )
    
    empty_cache(accelerator)
    return metric_outputs


def iter_all_equal(iterable, key):
    iterator = iter(iterable)
    first = mi.first(iterator)
    return all(key(x) == key(first) for x in iterator)


def step(
    *, 
    accelerator: accelerate.Accelerator,
    batch,
    cv_set: lib_utils.CVSets,
    epoch: int,
    forward_logger: "ForwardLogger",
    forward_tokenizer: transformers.PreTrainedTokenizerBase,
    mask_query: bool,
    model: peft.peft_model.PeftModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    log: bool,
    global_step: int,
):
    if cv_set:
        model.train()
        optimizer.zero_grad()

    lib_utils.not_first_token(
        forward_tokenizer=forward_tokenizer,
        tensor=batch["forward"]["input_ids"],
    )

    if mask_query: 
        labels = batch["masked_forward"]["input_ids"]
    else:
        labels = batch["forward"]["input_ids"]

    gpu_batch = {k: v.to(accelerator.local_process_index) for k, v in batch["forward"].items()}
    loss = model(**gpu_batch, labels=labels).loss
    forward_logger.log(
        batch       = batch["forward"]["input_ids"], 
        epoch       = epoch,
        global_step = global_step
    )

    if cv_set == lib_utils.CVSets.TRAIN:
        accelerator.backward(loss)
        optimizer.step()

    # Training Logging Logging
    loss_logging = accelerator.gather(loss.detach()).mean() # type: ignore

    if RANK == 0 and log:
        wandb.log(
            {f"{lib_constant.WANDB_NAMESPACE}/{cv_set.value}/loss": loss_logging.item()}, 
            step=global_step, 
        )
        
    empty_cache(accelerator)
    return loss_logging


class Evaluator:
    def __init__(
        self,
        *,
        cv_split: lib_utils.CVSets,
        accelerator: accelerate.Accelerator,
        batch_table_print_qty,
        forward_tokenizer,
        gen_kwargs,
        max_num_epochs,
        metrics,
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
        predict_qty_print: int,
    ):
        
        self._cv_split              = cv_split
        self._accelerator           = accelerator
        self._batch_table_print_qty = batch_table_print_qty
        self._forward_tokenizer     = forward_tokenizer
        self._gen_kwargs            = gen_kwargs
        self._max_num_epochs        = max_num_epochs
        self._metrics               = metrics
        self._prediction_tokenizer  = prediction_tokenizer
        self._predict_qty_print     = predict_qty_print
        
        if RANK == 0:
            rich_kwargs = dict(show_lines = True, title = f"{cv_split.value} - Predictions")
            columns = [
                "Epoch",       "Question:", 
                "Prediction:", "Extracted Gen A:", 
                "Ref A:",      "Qty Toks:",
            ]
            
            self._predict_table = lib_utils.WandbAndRichTable(
                columns     = columns, 
                rich_kwargs = rich_kwargs,
            )
        else:
            self._predict_table = None


    def evaluate_one(
        self,
        *,
        batch,
        batch_idx: int,
        epoch_idx: int,
        log: bool,
        model: peft.peft_model.PeftModelForCausalLM,
        global_step: int,
        total_num_batches: int,
    ):
        # Predict on Training
        lib_sft_tables.batch_table(
            batch             = batch["forward"],
            batch_idx         = batch_idx,
            epoch_idx         = epoch_idx,
            forward_tokenizer = self._forward_tokenizer,
            is_forward        = True,
            print_qty         = self._batch_table_print_qty,
            num_batches       = total_num_batches,
            num_epochs        = self._max_num_epochs,
        )

        metrics_outputs = predict(
            accelerator       = self._accelerator,
            batch             = batch,
            epoch             = epoch_idx,
            gen_kwargs        = self._gen_kwargs,
            global_step       = global_step,
            metrics           = self._metrics,
            model             = model,
            predict_table     = self._predict_table,
            predict_tokenizer = self._prediction_tokenizer,
            qty_print         = self._predict_qty_print,
            split             = self._cv_split,
        )
        
        if RANK == 0 and log:
            dict_to_log = {
                f"{lib_constant.WANDB_NAMESPACE}/{self._cv_split.value}/{metric_name}": 
                np.mean(metric_value) 
                for metric_name, metric_value in metrics_outputs.items()
            }

            wandb.log(dict_to_log, step=global_step)
        
        return metrics_outputs
        

    def evaluate(
            self, 
            *, 
            dataloader, 
            epoch_idx, 
            global_step, 
            model, 
            stepper,
        ):

        losses = []
        metrics = collections.defaultdict(list)

        for batch_idx, batch in enumerate(tqdm(
            dataloader, 
            disable = RANK != 0, 
            desc    = "Eval Batches"
        )):
            
            with torch.no_grad():
                losses.append(
                    stepper(
                        batch       = batch,
                        epoch_idx   = epoch_idx,
                        global_step = global_step,
                        log         = False,
                    ).cpu().item()
                )
                
                metrics_outputs = self.evaluate_one(
                        batch             = batch,
                        batch_idx         = batch_idx,
                        epoch_idx         = epoch_idx,
                        global_step       = global_step,
                        model             = model,
                        log               = False,
                        total_num_batches = len(dataloader),
                    )
                
                for metrics_name, metrics_values in metrics_outputs.items():
                    metrics[metrics_name].append(metrics_values)

        if RANK == 0:
            assert "loss" not in metrics
            dict_to_log = {
                f"{lib_constant.WANDB_NAMESPACE}/{self._cv_split.value}/{metric_name}": np.mean(metric_values)
                for metric_name, metric_values in metrics.items()
            }
            
            dict_to_log[f"{lib_constant.WANDB_NAMESPACE}/{self._cv_split.value}/loss"] = np.mean(losses)
            wandb.log(dict_to_log, step=global_step)


class ForwardLogger:
    def __init__(self, *, any_tokenizer, cv_set):
        if RANK == 0:
            self._table = lib_utils.WandbTableRepair(
                wandb_kwargs=dict(columns=["epoch", "input"])
            )
            self._any_tokenizer = any_tokenizer
            self._cv_set = cv_set


    def log(self, epoch, batch, global_step):
        if RANK == 0:
            idx = random.randint(0, len(batch) - 1)
            self._table.add_data(epoch, self._any_tokenizer.decode(batch[idx]))
            wandb.log(
                {f"{lib_constant.WANDB_NAMESPACE}/{self._cv_set.value}_forward_logger": 
                 self._table.get_loggable_object()}, 
                step=global_step,
            )




def main(
    run_name,
    *,
    batch_table_print_qty   = DEFAULT_BATCH_TABLE_PRINT_QTY,
    dataset_choice          = DEFAULT_DATASET,
    data_directory          = DEFAULT_DATA_DIRECTORY,
    eval_batch_size         = DEFAULT_EVAL_BATCH_SIZE,
    filter_out_bad          = DEFAULT_FILTER_OUT_BAD,
    gen_kwargs              = DEFAULT_GEN_KWARGS,
    learning_rate           = DEFAULT_LEARNING_RATE,
    lm_mode                 = DEFAULT_LM_MODE,
    just_device_map         = DEFAULT_JUST_DEVICE_MAP,
    mask_query              = DEFAULT_MASK_QUERY,
    max_num_epochs          = DEFAULT_NUM_EPOCHS,
    model_name_or_path      = DEFAULT_MODEL_NAME_OR_PATH,
    n_batches_predict_train = DEFAULT_N_BATCHES_PREDICT_TRAIN,
    output_type             = DEFAULT_OUTPUT_TYPE,
    peft_config_dict        = DEFAULT_PEFT_CONFIG,
    peft_do_all_lin_layers  = DEFAULT_PEFT_DO_ALL_LIN_LAYERS,
    predict_qty_print       = DEFAULT_PREDICT_QTY_PRINT,
    precision               = DEFAULT_BASE_PRECISION,
    qty_eval_small          = DEFAULT_QTY_EVAL_SMALL,
    stop_at_line_return     = False,
    train_batch_size        = DEFAULT_TRAIN_BATCH_SIZE,
    use_peft                = DEFAULT_USE_PEFT,
    wandb_entity            = DEFAULT_WANDB_ENTITY,
    wandb_project_name      = DEFAULT_WANDB_PROJECT_NAME,
):
    args = locals().copy()

    # We convert the enums to their values so they can be displayed in wandb.
    for k, v in args.items():
        if isinstance(v, enum.Enum):
            args[k] = v.value

    ###########################################################################
    # 🔍 Checks, Wandb then Metrics
    ###########################################################################
    lm_mode = lib_sft_constants.LMModes(lm_mode)
    precision = lib_utils.ValidPrecisions(precision)
    is_encoder_decoder = lib_sft_utils.get_is_encoder_decoder(
        model_name_or_path)
    assert not is_encoder_decoder, "Encoder decoder not supported yet."
    
    if RANK == 0:
        wandb_dir = lib_utils.get_tmp_dir() / "wandb"
        wandb_dir.mkdir(exist_ok=True)
        wandb.init(
            name    = run_name,
            project = wandb_project_name,
            entity  = wandb_entity,
            config  = dict(args=args, gen_kwargs=gen_kwargs),
            dir     = wandb_dir,
        )

    if dataset_choice == lib_utils.Datasets.COMMONSENSE_QA:
        metrics = dict(
            exact_match=lib_metric.ScratchpadAnswerAccuracy(
                libs_extraction.lib_multiple_choice.MultipleChoiceRfindExtractor(
                ["(A)", "(B)", "(C)", "(D)", "(E)"])
            ),
            # exact_match_STAR=lib_metric.ScratchpadAnswerAccuracy(
            #     libs_extraction.lib_multiple_choice.MultipleChoiceSTARExtractor(
            #     ["(A)", "(B)", "(C)", "(D)", "(E)"])
            # ),
        )
    elif dataset_choice == lib_utils.Datasets.ARITHMETIC:
        metrics = dict(
            exact_match = lib_metric.ScratchpadAnswerAccuracy(
                libs_extraction.lib_final_line.FinalLineExtractor()
            )
        )
    else:
        raise NotImplementedError(dataset_choice)
        

    ###########################################################################
    # 🏗️ Load Tokenizer and Data.
    ###########################################################################
    tmp_tokenizers = lib_trl_utils.load_tokenizers(
        config     = transformers.AutoConfig.from_pretrained(model_name_or_path),
        model_name = model_name_or_path, 
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
        adapter_name           = "default",
        forward_tokenizer      = forward_tokenizer,
        just_device_map        = just_device_map,
        model_name             = model_name_or_path,
        peft_config_dict       = peft_config_dict,
        peft_do_all_lin_layers = peft_do_all_lin_layers,
        prediction_tokenizer   = prediction_tokenizer,
        precision              = precision,
        use_peft               = use_peft,
    )

    if RANK == 0:
        print("Model loaded.")

    optimizer = torch.optim.Adam([
        x for x in model.parameters() if x.requires_grad],
        lr=learning_rate,
    )

    ###########################################################################
    # Set EOS to line return
    ###########################################################################
    if stop_at_line_return:
        line_return_tok = lib_utils.line_return_token(
            any_tokenizer=prediction_tokenizer)
        assert "eos_token_id" not in gen_kwargs, gen_kwargs
        gen_kwargs["eos_token_id"] = line_return_tok

    ###########################################################################
    # Dataloaders
    ###########################################################################
    assert not is_encoder_decoder
    dataloaders, small_eval_dl = lib_sft_dataloaders.get_dataloaders(
        data_directory       = data_directory,
        dataset_choice       = dataset_choice,
        eval_batch_size      = eval_batch_size,
        filter_bads          = filter_out_bad,
        forward_tokenizer    = forward_tokenizer,
        lm_mode              = lm_mode,
        output_type          = output_type,
        prediction_tokenizer = prediction_tokenizer,
        qty_eval_small       = qty_eval_small,
        train_batch_size     = train_batch_size,
    )

    ###########################################################################
    # 🏎️ Accelerator business
    ###########################################################################
    accelerator = accelerate.Accelerator()
    model, optimizer, dataloaders, small_eval_dl = accelerator.prepare(
        model, optimizer, dataloaders, small_eval_dl)

    ###########################################################################
    # 🔁 Main Loop
    ###########################################################################
    train_evaluator = Evaluator(
        accelerator           = accelerator,
        batch_table_print_qty = batch_table_print_qty,
        cv_split              = lib_utils.CVSets.TRAIN,
        forward_tokenizer     = forward_tokenizer,
        gen_kwargs            = gen_kwargs,
        max_num_epochs        = max_num_epochs,
        metrics               = metrics,
        prediction_tokenizer  = prediction_tokenizer,
        predict_qty_print     = predict_qty_print,
    )

    validation_evaluator = Evaluator(
        accelerator           = accelerator,
        batch_table_print_qty = batch_table_print_qty,
        cv_split              = lib_utils.CVSets.VALID,
        forward_tokenizer     = forward_tokenizer,
        gen_kwargs            = gen_kwargs,
        max_num_epochs        = max_num_epochs,
        metrics               = metrics,
        prediction_tokenizer  = prediction_tokenizer,
        predict_qty_print     = predict_qty_print,
    )
    
    train_forward_logger = ForwardLogger(
        any_tokenizer = forward_tokenizer, 
        cv_set        = lib_utils.CVSets.TRAIN,
    )
    validation_forward_logger = ForwardLogger(
        any_tokenizer = forward_tokenizer,
        cv_set        = lib_utils.CVSets.VALID,
    )

    def training_stepper(
        *, 
        batch,
        epoch_idx: int,
        global_step: int,
        log,
    ):
        return step(
            accelerator       = accelerator,
            batch             = batch,
            cv_set            = lib_utils.CVSets.TRAIN,
            epoch             = epoch_idx,
            forward_logger    = train_forward_logger,
            forward_tokenizer = forward_tokenizer,
            global_step       = global_step,
            mask_query        = mask_query,
            model             = model,
            optimizer         = optimizer,
            log               = log,
        )

    def validation_stepper(
        *,
        batch,
        epoch_idx,
        global_step,
        log,
    ):
        return step(
            accelerator       = accelerator,
            batch             = batch,
            cv_set            = lib_utils.CVSets.VALID,
            epoch             = epoch_idx,
            forward_logger    = validation_forward_logger,
            forward_tokenizer = forward_tokenizer,
            global_step       = global_step,
            log               = log,
            mask_query        = mask_query,
            model             = model,
            optimizer         = optimizer,
        )
    

    global_step = 0
    for epoch_idx in tqdm(
        range(max_num_epochs), 
        disable = RANK != 0, 
        desc    = "Epochs",
    ):
        
        train_dataset_iterator = iter(dataloaders[lib_utils.CVSets.TRAIN])

        if RANK == 0:
            wandb.log(
                {f"{lib_constant.WANDB_NAMESPACE}/epoch": epoch_idx}, 
                step=global_step,
            )

        """
        We want to evaluate every few batches. We:
            - Do a number of steps over a training iterator
            - Validate
            - If we had done at least one training step, the 
              iterator is not done, so the loop should repeat
        """

        at_least_one = True
        while at_least_one:
            at_least_one = False

            # Train
            for batch_idx, batch in enumerate(tqdm(it.islice(
                    train_dataset_iterator, 
                    n_batches_predict_train,
                ), 
                disable = RANK != 0, 
                desc    = "Train Batches",
            )):
                
                at_least_one = True
                global_step += len(batch["forward"]["input_ids"]) * WORLD_SIZE
                
                training_stepper(
                    batch       = batch,
                    epoch_idx   = epoch_idx,
                    global_step = global_step,
                    log         = True,
                )

                if batch_idx % 10 == 0:
                    with torch.no_grad():
                        model.eval()
                        train_evaluator.evaluate_one(
                            batch             = batch,
                            batch_idx         = batch_idx,
                            epoch_idx         = epoch_idx,
                            global_step       = global_step,
                            model             = model,
                            log               = True,
                            total_num_batches = len(dataloaders[lib_utils.CVSets.TRAIN])
                        )
            
            validation_evaluator.evaluate(
                dataloader  = small_eval_dl, 
                epoch_idx   = epoch_idx, 
                global_step = global_step, 
                model       = model, 
                stepper     = validation_stepper
            )



    validation_evaluator.evaluate(
        dataloader  = dataloaders[lib_utils.CVSets.VALID], 
        epoch_idx   = epoch_idx, 
        global_step = global_step + 1,
        model       = model, 
        stepper     = validation_stepper
    )

if __name__ == "__main__":
    fire.Fire(main)
