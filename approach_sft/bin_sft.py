#!/usr/bin/env python3
"""
Supervised Trainer.

"""
import abc
import collections
import datetime
import functools
import itertools as it
import os
import pathlib
import random
import re
import tempfile
from typing import Optional

import more_itertools as mit
import outlines
import outlines.generate
import outlines.models
import outlines.models.transformers
import outlines.samplers

os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["DATASETS_VERBOSITY"] = "warning"
# os.environ["WANDB_SILENT"] = "true"
os.environ["NCCL_DEBUG"] = "WARN"

import logging

import accelerate
import datasets
import fire
import hydra
import hydra.core.hydra_config
import git
import more_itertools as mi
import numpy as np
import omegaconf
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
import wandb
from tqdm import tqdm
from with_trl import (lib_base_classes, lib_constant, lib_metric,
                      lib_trl_utils, lib_utils)
from with_trl.libs_extraction import lib_final_line, lib_multiple_choice

from approach_sft import (lib_sft_constants, lib_sft_dataloaders,
                          lib_sft_tables, lib_sft_utils)
import lib_sft_multi_regexes

rich.traceback.install(
    console=rich.console.Console(
        force_terminal=True, 
        force_interactive=True, 
        markup=True,
    )
)

torch.set_float32_matmul_precision("high")

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

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
LOGGER = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(git.Repo(
    __file__, 
    search_parent_directories=True
).working_tree_dir)



def repo_path(input_path):
    """ Helper for Hydra """
    return REPO_ROOT / input_path



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
    # Prepare the inputs for the metrics, 
    # and the containers for the outputs
    #######################################
    metric_outputs = {}
    local_metric_outputs = collections.defaultdict(list)
    # batch_for_metrics = lib_base_classes.DataListContainer(
    #     detok_ref_query      = batch["extra_info"]["ref_qa_question"],
    #     detok_ref_answer     = batch["extra_info"]["ref_qa_answer"],
    #     detok_ref_scratchpad = batch["extra_info"].get("ref_qa_scratchpad", None),

    #     # tok_ref_query        = None,
    #     # tok_ref_answer       = None,
    #     # tok_ref_scratchpad   = None,

    #     difficulty_level     = None,
    #     extra_information    = None,
    # )
    
    for name, metric in metrics.items():
        #######################################
        # Actually compute metrics
        #######################################
        local_metric_output = metric(
            batch     = batch["extra_info"],
            responses = response_text_for_metrics,)
        local_metric_outputs[name] = local_metric_output 
        
        #######################################
        # Gather metrics
        #######################################
        pre_gather = [x for x in local_metric_output.values if x is not None]
        pre_gather = torch.tensor(pre_gather).to(
            accelerator.local_process_index)
        metric_outputs[name] = accelerator.gather_for_metrics(
            pre_gather).mean().item()

    ###########################################################################
    # Print some outputs
    ###########################################################################
    if RANK == 0:
        prediction_batch_obj = lib_base_classes.BatchedUnrollReturn(
                response_tensors = predictions,
                any_tokenizer    = predict_tokenizer,
            )
        
        lib_sft_tables.predict_table(
            batch                 = batch["extra_info"],
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


class OutlinesContextABC(abc.ABC):
    @abc.abstractmethod
    def data_collator(self, batch):
        pass

    @abc.abstractmethod
    def outlines_forward(self, batch):
        pass


class ArithmeticOutlinesContext(OutlinesContextABC):
    def __init__(self, model, forward_tokenizer, predict_tokenizer, sampler):
        self._forward_tokenizer = forward_tokenizer
        self._predict_tokenizer = predict_tokenizer
        self._outlines_model = outlines.models.Transformers(
            model=model, 
            tokenizer=self._forward_tokenizer,
        )
        self._multi_fsm_gen = lib_sft_multi_regexes.MultiFSMSequenceGenerator(
            model=self._outlines_model,
            sampler=sampler,
            device=model.device,
        )
        self._pattern_format = r"\n<scratch>\n.{{5,200}}\n</scratch>\nA:\n{escaped_answer}\n!"

    def data_collator(self, batch):
        batch["fsms"] = []
        batch["fsm_regex_str"] = []
        for i in range(len(batch["ref_qa_question"])):    
            escaped_answer = re.escape(batch["ref_qa_answer"][i]).strip()
            pattern = self._pattern_format.format(
                escaped_answer=escaped_answer
            )
            batch["fsms"].append(outlines.fsm.guide.RegexGuide(
                pattern, 
                self._outlines_model.tokenizer,
            ))
            batch["fsm_regex_str"].append(pattern)
        return batch

    def outlines_forward(
        self, batch,
    ):  
        with torch.no_grad():
            samples, states = self._multi_fsm_gen(
                [f"What is {x}" for x in batch["ref_qa_question"]],
                batch["fsms"],
                max_tokens=200,
            )

        #######################################################################
        # Print samples
        #######################################################################
        table = rich.table.Table(
            "Prompt", 
            "Sample", 
            "Pattern", 
            show_lines=True,
        )

        for prompt, pat, sample in mit.zip_equal(
            batch["ref_qa_question"],
            batch["fsm_regex_str"],
            samples,
        ):
            table.add_row(
                rich.markup.escape("\"" + prompt + "\""), 
                rich.markup.escape("\"" + sample + "\""),
                rich.markup.escape("\"" + pat    + "\""), 
                rich.markup.escape(str(len(sample))),
            )

        rich.print(table)
        real_batch = {}
        real_batch["forward"] = self._forward_tokenizer(
            samples, padding=True, return_tensors="pt")
        real_batch["predict"] = self._predict_tokenizer(
            batch["ref_qa_question"], padding=True, return_tensors="pt")
        real_batch["extra_info"] = batch | {"patterns": batch["fsm_regex_str"]}
        
        return real_batch

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
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    predict_tokenizer,
    optimizer: torch.optim.Optimizer,
    log: bool,
    global_step: int,
    do_train: bool,
    output_type: lib_sft_constants.OutputTypes,
):
    assert not (cv_set == lib_utils.CVSets.VALID and do_train), (
        cv_set)

    if cv_set == lib_utils.CVSets.TRAIN:
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

    gpu_batch = {
        k: v.to(accelerator.local_process_index) 
        for k, v in batch["forward"].items()
    }

    loss = model(**gpu_batch, labels=labels).loss
    forward_logger.log(
        batch       = batch["forward"]["input_ids"], 
        epoch       = epoch,
        global_step = global_step
    )

    assert do_train == (optimizer is not None), (
        do_train, optimizer)
    
    if do_train:
        assert optimizer is not None
        assert cv_set == lib_utils.CVSets.TRAIN, cv_set
        accelerator.backward(loss)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # Training Logging Logging
    loss_logging = accelerator.gather(loss.detach()).mean() # type: ignore

    if RANK == 0 and log:
        wandb.log(
            {f"{cv_set.value}/loss": loss_logging.item()}, 
            step=global_step, 
        )
        
    empty_cache(accelerator)
    return loss_logging


class Evaluator:
    """
    ... Does eval?
    For each sample:
        1. Calls predict
        2. Logs the output..

    """
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
        output_type,
        outlines_context,
    ):
        
        self._outlines_context      = outlines_context
        self._cv_split              = cv_split
        self._accelerator           = accelerator
        self._batch_table_print_qty = batch_table_print_qty
        self._forward_tokenizer     = forward_tokenizer
        self._gen_kwargs            = gen_kwargs
        self._max_num_epochs        = max_num_epochs
        self._metrics               = metrics
        self._prediction_tokenizer  = prediction_tokenizer
        self._predict_qty_print     = predict_qty_print
        self._output_type           = output_type
        
        if RANK == 0:
            rich_kwargs = dict(show_lines = True, title = f"{cv_split.value} - Predictions")
            columns = [
                "Epoch",       "Question:", 
                "Prediction:", "Extracted Gen A:", 
                "Ref A:",      "Qty Toks:",
            ]

            self._predict_table = lib_utils.WandbAndRichTable(
                columns         = columns, 
                rich_kwargs     = rich_kwargs,
                table_name      = "Main table",
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
        wandb_key,
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
                f"{wandb_key}/{metric_name}": 
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
            wandb_key,
        ):

        losses = []
        metrics = collections.defaultdict(list)

        # For each batch in eval
        for batch_idx, batch in enumerate(tqdm(
            dataloader, 
            disable = RANK != 0, 
            desc    = "Eval Batches"
        )):
            # 
            if self._output_type == lib_sft_constants.OutputTypes.OUTLINES:
                batch = self._outlines_context.outlines_forward(
                    batch=batch,
                )

            assert "forward" in batch, batch.keys()
            assert "predict" in batch, batch.keys()

            with torch.no_grad():
                # Call forward on the batch to compute the cross entropy loss.
                losses.append(
                    stepper(
                        batch       = batch,
                        epoch       = epoch_idx,
                        global_step = global_step,
                        log         = False,
                        do_train    = False,
                    ).cpu().item()
                )
                
                # Call predict on the batch, 
                metrics_outputs = self.evaluate_one(
                        batch             = batch,
                        batch_idx         = batch_idx,
                        epoch_idx         = epoch_idx,
                        global_step       = global_step,
                        model             = model,
                        log               = False,
                        total_num_batches = len(dataloader),
                        wandb_key         = wandb_key,
                    )
                
                for metrics_name, metrics_values in metrics_outputs.items():
                    metrics[metrics_name].append(metrics_values)

        if RANK == 0:
            assert "loss" not in metrics
            dict_to_log = {
                f"{wandb_key}/{metric_name}": 
                np.mean(metric_values)
                for metric_name, metric_values in metrics.items()
            }
            
            dict_to_log[f"{self._cv_split.value}/loss"] = np.mean(losses)
            wandb.log(dict_to_log, step=global_step)

class ForwardLogger:
    def __init__(self, *, any_tokenizer, cv_set):
        if RANK == 0:
            self._table = lib_utils.WandbTableRepair(
                wandb_kwargs=dict(columns=["epoch", "input"]),
                columns=["epoch", "input"],
                table_name="ForwardLogger",
            )
            self._any_tokenizer = any_tokenizer
            self._cv_set = cv_set


    def log(self, epoch, batch, global_step):
        if RANK == 0:
            idx = random.randint(0, len(batch) - 1)
            self._table.add_data(
                epoch, self._any_tokenizer.decode(batch[idx])
            )


OUTLINES_CLASSES = {
    lib_utils.Datasets.ARITHMETIC: ArithmeticOutlinesContext,
}

@hydra.main(
    version_base="1.3.2", 
    config_path="config", 
    config_name="config",
)
def main(
    cfg: omegaconf.DictConfig,
):
    cfg = hydra.utils.instantiate(cfg)

    # We convert the enums to their values so they can be displayed in wandb.
    # for k, v in args.items():
    #     if isinstance(v, enum.Enum):
    #         args[k] = v.value

    ###########################################################################
    # 🔍 Checks, Wandb then Metrics
    ###########################################################################
    
    is_encoder_decoder = lib_sft_utils.get_is_encoder_decoder(
        cfg.model_name_or_path)
    assert not is_encoder_decoder, "Encoder decoder not supported yet."
    
    experiment = hydra.core.hydra_config.HydraConfig.get().runtime.choices["experiment"]
    
    # Assign a printable timestamp
    if RANK == 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        timestamp = None
    
    if WORLD_SIZE > 1:
        container = [None for _ in range(WORLD_SIZE)]
        torch.distributed.all_gather_object(container, timestamp)
        timestamp = container[0]

    save_path = pathlib.Path(cfg.save_path) / f"{experiment}-{timestamp}"
    if RANK == 0:
        os.makedirs(save_path, exist_ok=False)

        wandb_dir_obj = tempfile.TemporaryDirectory()
        wandb_dir = wandb_dir_obj.name
        
        wandb.init(
            name     = cfg.run_name,
            project  = cfg.wandb_project_name,
            entity   = cfg.wandb_entity,
            config   = dict(
                args=omegaconf.OmegaConf.to_container(cfg, resolve=True), 
                gen_kwargs=cfg.gen_kwargs,
                save_dir = save_path,
            ),
            dir      = wandb_dir,
            mode     = "disabled" if cfg.test_mode else "online",
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # if cfg.dataset_choice == lib_utils.Datasets.COMMONSENSE_QA:
    #     metrics = dict(
    #         exact_match=lib_metric.ScratchpadAnswerAccuracy(
    #             lib_multiple_choice.MultipleChoiceRfindExtractor(
    #                 ["(A)", "(B)", "(C)", "(D)", "(E)"]),
    #             pad_token=tokenizer.pad_token,
    #         ),
    #     )

    if cfg.dataset_choice == lib_utils.Datasets.ARITHMETIC:
        metrics = dict(
            exact_match = lib_metric.ScratchpadAnswerAccuracy(
                extractor=lib_final_line.FinalLineExtractor(
                    ignore_one_line = cfg.extractor_ignore_one_line,
                    pad_token=tokenizer.pad_token,
                ),
                pad_token=tokenizer.pad_token,
            )
        )
    elif cfg.dataset_choice == lib_utils.Datasets.GSM8K:
        from with_trl.libs_extraction import lib_numerical
        metrics = dict(
            exact_match = lib_metric.ScratchpadAnswerAccuracy(
                extractor=lib_numerical.ConvToNum(),
                pad_token=tokenizer.pad_token,
            )
        )
    else:
        raise NotImplementedError(cfg.dataset_choice)
        

    ###########################################################################
    # 🏗️ Load Tokenizer and Data.
    ###########################################################################
    tmp_tokenizers       = lib_trl_utils.load_tokenizers(cfg.model_name_or_path)
    forward_tokenizer    = tmp_tokenizers["forward_tokenizer"   ]
    prediction_tokenizer = tmp_tokenizers["prediction_tokenizer"]
    del tmp_tokenizers

    ###########################################################################
    # 🏗️ Load Model and Build Optimizer.
    ###########################################################################
    if RANK == 0:
        print(f"Loading model {cfg.model_name_or_path}")

    model = lib_trl_utils.load_then_peft_ize_model(
        adapter_name           = "default",
        forward_tokenizer      = forward_tokenizer,
        just_device_map        = cfg.just_device_map,
        model_name             = cfg.model_name_or_path,
        peft_config            = peft.LoraConfig(**cfg.peft_config_dict),
        peft_do_all_lin_layers = cfg.peft_do_all_lin_layers,
        precision              = cfg.precision,
        prediction_tokenizer   = prediction_tokenizer,
        trust_remote_code      = True,
        use_peft               = cfg.use_peft,
    ).to(LOCAL_RANK)

    if RANK == 0:
        print("Model loaded.")

    optimizer = torch.optim.Adam(
        [x for x in model.parameters() if x.requires_grad],
        lr=cfg.learning_rate,
    )

    ###########################################################################
    # Set EOS to line return
    ###########################################################################
    if cfg.stop_at_line_return:
        line_return_tok = lib_utils.line_return_token(
            any_tokenizer=prediction_tokenizer
        )
        assert "eos_token_id" not in cfg.gen_kwargs, cfg.gen_kwargs
        cfg.gen_kwargs["eos_token_id"] = line_return_tok

    ###########################################################################
    # Dataloaders
    ###########################################################################
    assert not is_encoder_decoder
    if cfg.output_type.enum == lib_sft_constants.OutputTypes.OUTLINES:
        outlines_context = OUTLINES_CLASSES[cfg.dataset_choice](
            model              = model,
            forward_tokenizer  = forward_tokenizer,
            predict_tokenizer  = prediction_tokenizer,
            # sampler            = outlines.samplers.GreedySampler(),
            sampler            = outlines.samplers.BeamSearchSampler(10),
        )

    else:
        outlines_context = None 

    dataloaders, small_eval_dl = lib_sft_dataloaders.get_dataloaders(
        answer_only               = False, # Doesn't do anything for sft
        data_directory            = cfg.data_directory,
        dataset_choice            = cfg.dataset_choice,
        eval_batch_size           = cfg.output_type.eval_batch_size * WORLD_SIZE,
        extractor_ignore_one_line = cfg.extractor_ignore_one_line,
        filter_bads               = cfg.filter_out_bad,
        forward_tokenizer         = forward_tokenizer,
        lm_mode                   = cfg.lm_mode,
        output_type               = cfg.output_type.enum,
        outlines_context          = outlines_context,
        prediction_tokenizer      = prediction_tokenizer,
        qty_eval_small            = cfg.qty_eval_small,
        seed                      = 0,
        train_batch_size          = cfg.output_type.train_batch_size * WORLD_SIZE,
        subset_data               = cfg.subset_data,
        use_workers               = cfg.use_workers,
    )

    total_num_steps = len(dataloaders[lib_utils.CVSets.TRAIN]) * cfg.max_num_epochs
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=total_num_steps * 0.1,
        num_training_steps=len(dataloaders[lib_utils.CVSets.TRAIN]) * cfg.max_num_epochs,
    )

    ###########################################################################
    # 🏎️ Accelerator business
    ###########################################################################
    accelerator = accelerate.Accelerator()
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    ###########################################################################
    # 🔁 Main Loop
    ###########################################################################
    train_evaluator = Evaluator(
        accelerator           = accelerator,
        batch_table_print_qty = cfg.batch_table_print_qty,
        cv_split              = lib_utils.CVSets.TRAIN,
        forward_tokenizer     = forward_tokenizer,
        gen_kwargs            = cfg.gen_kwargs,
        max_num_epochs        = cfg.max_num_epochs,
        metrics               = metrics,
        prediction_tokenizer  = prediction_tokenizer,
        predict_qty_print     = cfg.predict_qty_print,
        output_type           = cfg.output_type.enum,
        outlines_context      = outlines_context,
    )

    validation_evaluator = Evaluator(
        accelerator           = accelerator,
        batch_table_print_qty = cfg.batch_table_print_qty,
        cv_split              = lib_utils.CVSets.VALID,
        forward_tokenizer     = forward_tokenizer,
        gen_kwargs            = cfg.gen_kwargs,
        max_num_epochs        = cfg.max_num_epochs,
        metrics               = metrics,
        prediction_tokenizer  = prediction_tokenizer,
        predict_qty_print     = cfg.predict_qty_print,
        output_type           = cfg.output_type.enum,
        outlines_context      = outlines_context,
    )
    
    train_forward_logger = ForwardLogger(
        any_tokenizer = forward_tokenizer, 
        cv_set        = lib_utils.CVSets.TRAIN,
    )
    validation_forward_logger = ForwardLogger(
        any_tokenizer = forward_tokenizer,
        cv_set        = lib_utils.CVSets.VALID,
    )

    training_stepper = functools.partial(
        step,
        accelerator       = accelerator,
        cv_set            = lib_utils.CVSets.TRAIN,
        forward_logger    = train_forward_logger,
        forward_tokenizer = forward_tokenizer,
        mask_query        = cfg.mask_query,
        model             = model,
        optimizer         = optimizer,
        scheduler         = scheduler,
        output_type       = cfg.output_type.enum,
        predict_tokenizer = prediction_tokenizer,
    )

    validation_stepper = functools.partial(
        step,
        accelerator       = accelerator,
        cv_set            = lib_utils.CVSets.VALID,
        forward_logger    = validation_forward_logger,
        forward_tokenizer = forward_tokenizer,
        mask_query        = cfg.mask_query,
        model             = model,
        optimizer         = None,
        scheduler         = None,
        output_type       = cfg.output_type.enum,
        predict_tokenizer = prediction_tokenizer,
    )


    if cfg.test_mode:
        sizes = []

        for dataloader_name, dataloader in it.chain(
            dataloaders.items(), 
            dict(small_eval_dl=small_eval_dl).items(),
        ):
            for batch in tqdm(
                dataloader, 
                desc=f"Testing {dataloader_name}", 
            ):
                forward_ids = batch["forward"]["input_ids"]
                predict_ids = batch["predict"]["input_ids"]

                for mask in batch["forward"]["attention_mask"]:
                    sizes.append(mask.int().sum())

                assert forward_ids.shape[0] == predict_ids.shape[0], (
                    f"Shapes do not match: "
                    f"forward_ids.shape={forward_ids.shape}, "
                    f"predict_ids.shape={predict_ids.shape}"
                )
                assert forward_ids.shape[1] > predict_ids.shape[1], (
                    f"Shape mismatch: "
                    f"forward_ids.shape={forward_ids.shape}, "
                    f"predict_ids.shape={predict_ids.shape}"
                )
                
                for entry_forward, entry_predict in mit.zip_equal(forward_ids, predict_ids):
                    entry_forward = torch.tensor([
                        x for x in entry_forward 
                        if x != forward_tokenizer.pad_token_id
                    ])
                    entry_predict = torch.tensor([
                        x for x in entry_predict 
                        if x != prediction_tokenizer.pad_token_id
                    ])
                    assert (entry_forward[:len(entry_predict)] == entry_predict).all(), (
                        entry_forward, 
                        entry_predict, 
                        forward_tokenizer.decode(entry_forward), 
                        prediction_tokenizer.decode(entry_predict),
                    )

        sizes = sorted(sizes)
        breakpoint()
        exit(0)

    full_valid_wandb_key = "full_valid"
    small_valid_wandb_key = "small_valid"
    single_train_wandb_key = "single_train"
    full_train_wandb_key = "full_train"

    # validation_evaluator.evaluate(
    #     dataloader  = dataloaders[lib_utils.CVSets.VALID], 
    #     epoch_idx   = 0, 
    #     global_step = 0,
    #     model       = model, 
    #     stepper     = validation_stepper,
    #     wandb_key   = full_valid_wandb_key,
    # )
    global_step = 0

    for epoch_idx in tqdm(
        range(cfg.max_num_epochs), 
        disable = RANK != 0, 
        desc    = "Epochs",
    ):
        
        """
        We want to evaluate every few batches. We:
            - Do a number of steps over a training iterator
            - Validate
            - If we had done at least one training step, the 
                iterator is not done, so the loop should repeat
        """

        # Train
        for batch_idx, batch in enumerate(tqdm(
            dataloaders[lib_utils.CVSets.TRAIN],
            disable = RANK != 0, 
            desc    = f"[Epoch {epoch_idx}] Train Batches",
        )):
            if LOCAL_RANK == 0:
                wandb.log(dict(
                    lr=mit.one(optimizer.param_groups)["lr"],
                    epoch=epoch_idx,
                ), step=global_step)

            if cfg.output_type.enum == lib_sft_constants.OutputTypes.OUTLINES:
                batch = outlines_context.outlines_forward(
                    batch=batch,
                )

            global_step += len(batch["forward"]["input_ids"]) * WORLD_SIZE

            training_stepper(
                batch       = batch,
                epoch       = epoch_idx,
                global_step = global_step,
                log         = True,
                do_train    = True,
            )

            # Eval just one batch every 10 samples:
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
                        total_num_batches = None,
                        wandb_key         = single_train_wandb_key,
                    )
        
            # Do a small eval subset every `cfg.n_batches_predict_train` batches:
            if batch_idx % cfg.n_batches_predict_train == 0 and batch_idx >= 0:
                validation_evaluator.evaluate(
                    global_step = global_step,
                    dataloader  = small_eval_dl, 
                    epoch_idx   = epoch_idx,
                    stepper     = validation_stepper,
                    model       = model,
                    wandb_key   = small_valid_wandb_key,
                )
        
        ################################################################################
        # End of epoch
        ################################################################################
        if RANK == 0:
                ckpt_path = save_path / f"ep_{epoch_idx}_model.pt"
                torch.save(dict(
                    cfg=omegaconf.OmegaConf.to_container(cfg, resolve=True),
                    optimizer=optimizer.state_dict(),
                    epoch=epoch_idx,
                    global_step=global_step,
                    wandb_run_id=wandb.run.id,
                    wandb_url=wandb.run.get_url(),
                ), str(ckpt_path))
                pretrained_save_dir = save_path / f"ep_{epoch_idx}_model"
                model.module.save_pretrained(str(pretrained_save_dir))
                
        accelerator.wait_for_everyone()

        validation_evaluator.evaluate(
            dataloader  = dataloaders[lib_utils.CVSets.VALID], 
            epoch_idx   = epoch_idx, 
            global_step = global_step + 1,
            model       = model, 
            stepper     = validation_stepper,
            wandb_key   = full_valid_wandb_key,
        )

    ################################################################################
    # End of training
    ################################################################################
    train_evaluator.evaluate(
        dataloader  = dataloaders[lib_utils.CVSets.TRAIN],
        epoch_idx   = epoch_idx,
        global_step = global_step + 1,
        model       = model, 
        stepper     = training_stepper,
        wandb_key   = full_train_wandb_key,
    )

    wandb.finish()

if __name__ == "__main__":
    main()
