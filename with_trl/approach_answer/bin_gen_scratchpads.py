""" 
Generates scratchpads for the fed-answer problem.


Things to do:
- Dataloaders
- Metrics.. accuracy conditioning on the scratchpad
"""

import os

###############################################################################
# We are required to do these before the imports for them to take effect
###############################################################################
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["DATASETS_VERBOSITY"] = "warning"
os.environ["WANDB_SILENT"] = "true"
os.environ["NCCL_DEBUG"] = "WARN"
###############################################################################

import json
import logging
import pathlib
import sys
from typing import Any, Optional, Union

import accelerate
import fire
import more_itertools
import peft
import rich
import rich.markup
import rich.traceback
import rich.table
import torch
import tqdm
import transformers
import wandb

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR.parent))

import lib_trl_utils
import lib_utils
from approach_answer import lib_data_commonsense_qa

rich.traceback.install()

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
if LOCAL_RANK == 0:
    print("Done with imports")


###############################################################################
# Defaut Hyperparameter Values, can be changed with CLI
###############################################################################
DEFAULT_BATCH_SIZE = 4
DEFAULT_MODEL_NAME = "EleutherAI/pythia-410m"
DEFAULT_OUTPUT_PATH = "/network/scratch/g/gagnonju/scratchpad_gen_outputs/output_ds"
DEFAULT_USE_PEFT = True
DEFAULT_PEFT_CONFIG = dict(
    inference_mode=False,
    lora_dropout=0.,
    lora_alpha=16,
    bias="none",
    r=16,
    task_type=peft.TaskType.CAUSAL_LM,
)
DEFAULT_GENERATE_KWARGS = dict(
    max_new_tokens=200,
    min_new_tokens=4,
    num_beams=4,
    do_sample=False,
    synced_gpus=True,
)
DEFAULT_WANDB_PROJECT = "fed-answer-gen"
DEFAULT_WANDB_ENTITY = "julesgm"
###############################################################################


def data_collator(batch, prediction_tokenizer):
    assert prediction_tokenizer.padding_side == "left", prediction_tokenizer.padding_side
    input_text = [x["ref_fs_scratchpad_gen_query"] for x in batch]
    
    return dict(
        inputs=prediction_tokenizer(input_text, return_tensors="pt", padding=True).to(LOCAL_RANK), 
        other_info=batch,
    )

def convert_args_for_wandb(args):
    ok_collection_types = (dict, list, tuple)
    ok_indiv_types = (int, float, str, bool)
    ok_types_all = ok_collection_types + ok_indiv_types

    if isinstance(args, dict):
        for k, v in args.items():
            if not isinstance(v, ok_types_all):
                args[k] = str(v)
            elif isinstance(v, ok_collection_types):
                args[k] = convert_args_for_wandb(v)
            else:
                # Logically this should only happen if in ok_indiv_types, the check is unnecessary
                assert isinstance(v, ok_indiv_types), f"Unexpected type {type(v).mro()} for {k}: {v}"

    elif isinstance(args, (list, tuple)):
        is_tuple = isinstance(args, tuple)
        args = list(args)
        for i, v in enumerate(args):
            if not isinstance(v, ok_types_all):
                args[i] = str(v)
            elif isinstance(v, ok_collection_types):
                args[i] = convert_args_for_wandb(v)
            else:
                # Logically this should only happen if in ok_indiv_types, the check is unnecessary
                assert isinstance(v, ok_indiv_types), f"Unexpected type {type(v).mro()} for {i}: {v}"

        if is_tuple:
            args = tuple(args)

    else:
        raise ValueError(f"Unexpected type {type(args).mro()} for {args}")

    return args
            

class OutputSampleWriter:
    def __init__(
        self, 
        *, 
        output_path: str, 
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
        split: str,
    ):
        assert split in ("train", "validation", "test")
        
        self._output_path = pathlib.Path(output_path)
        self._output_file = (self._output_path / f"samples.{split}.jsonl").open("w")
        self._prediction_tokenizer = prediction_tokenizer
        self._split = split

    def close(self):
        self._output_file.close()

    def __call__(
        self, 
        *, 
        batch,
        raw_output_sample_ids,
        clean_output_sample_ids,
    ) -> Any:
        
        clean_batch_gen_txt = self._prediction_tokenizer.batch_decode(
            clean_output_sample_ids, 
            skip_special_tokens=True
        )
        raw_batch_gen_txt = self._prediction_tokenizer.batch_decode(
            raw_output_sample_ids, 
            skip_special_tokens=True
        )
        # This format matches the few-shot examples

        for sample, clean_gen_txt, raw_gen_txt in more_itertools.zip_equal(
            batch,
            clean_batch_gen_txt,
            raw_batch_gen_txt,
        ):
            json_txt = json.dumps(dict(
                **sample,
                clean_gen_scratchpad_txt=clean_gen_txt,
                raw_gen_scratchpad_txt=raw_gen_txt,
            ))

            self._output_file.write(json_txt.strip() + "\n")


class WandbLoggingState:
    def __init__(self, any_tokenizer, split, also_print):
        self._any_tokenizer = any_tokenizer
        self._table = wandb.Table(columns=["batch_idx", "ref_inputs", "clean_output", "raw_output"])
        self._table_just_clean_gen = wandb.Table(columns=["clean_output"])
        self._split = split
        self._also_print = also_print
        

    def log(
            self,
            *,
            batch_idx,
            batch,
            raw_output_tokens,
            clean_output_tokens,
        ):
        
        raw_output_text = self._any_tokenizer.batch_decode(raw_output_tokens, skip_special_tokens=False)
        clean_output_text = self._any_tokenizer.batch_decode(clean_output_tokens, skip_special_tokens=True)
        
        if self._also_print:
            self._rich_table = rich.table.Table("batch_idx", "ref_inputs", "clean_output", show_lines=True)

        for clean_text, raw_text, sample in more_itertools.zip_equal(clean_output_text, raw_output_text, batch):
            end_prompt = sample["ref_fs_scratchpad_gen_query_detok"][sample["ref_fs_scratchpad_gen_query_detok"].rfind("Q:"):]

            self._table.add_data(
                batch_idx,
                end_prompt,
                clean_text,
                raw_output_text
            )
            self._table_just_clean_gen.add_data(clean_text)

            if self._also_print:
                self._rich_table.add_row(
                    str(batch_idx),
                    rich.markup.escape(end_prompt),
                    rich.markup.escape(clean_text),
                    # rich.markup.escape(raw_text),
                )

        rich.print(self._rich_table)
        wandb.log({f"{self._split}/batch": self._table})
        wandb.log({f"{self._split}/just_clean_gen": self._table_just_clean_gen})

    __call__ = log


def gathered_batch_gen(*, accelerator, model, batch, generate_kwargs, prediction_tokenizer):
    output_tokens = accelerator.unwrap_model(model).generate(
                **batch["inputs"], **generate_kwargs,)
    before_shape = output_tokens.shape
    output_tokens = output_tokens[:, batch["inputs"]["input_ids"].shape[1]:].contiguous()
    assert output_tokens.shape[1], (output_tokens.shape, before_shape)
    
    # Clean the output
    clean_text = [
        x.strip().split("\n", 1)[0] 
        for x in 
        prediction_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    ]
    clean_output_tokens = prediction_tokenizer(clean_text, return_tensors="pt", padding=True)["input_ids"].to(LOCAL_RANK).contiguous()

    return accelerator.gather(clean_output_tokens).detach().cpu(), accelerator.gather(output_tokens).detach().cpu()


def write_args_to_file(args, output_path):
    with open(output_path / "config.json", "w") as f:
        json.dump(dict(args=args, wandb_url=wandb.run.get_url(),), f, indent=4)


def print_dict_as_table(d, **table_kwargs):
    table = rich.table.Table("Key", "Value", **table_kwargs)
    for k, v in d.items():
        table.add_row(rich.markup.escape(str(k)), rich.markup.escape(str(v)))
    rich.print(table)


def main(
    run_name: str="gen-scratchpad",
    generate_kwargs: dict=DEFAULT_GENERATE_KWARGS,
    model_name: str = DEFAULT_MODEL_NAME,
    use_peft: bool=DEFAULT_USE_PEFT,
    precision: lib_utils.ValidPrecisions=lib_utils.ValidPrecisions.bfloat16,
    output_path: str=DEFAULT_OUTPUT_PATH,
    peft_config_dict=DEFAULT_PEFT_CONFIG,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    args = convert_args_for_wandb(locals().copy())
    print_dict_as_table(args, title="[blue bold]Script Args:")
    
    ###########################################################################
    # Setup the directories for logging / data saving
    ###########################################################################
    precision = lib_utils.ValidPrecisions(precision)
    output_path = pathlib.Path(output_path) 
    assert output_path.parent.exists(), f"Parent directory of {output_path} does not exist."
    output_path.mkdir(exist_ok=False)

    ###########################################################################
    # Setup wandb & save the script config
    ###########################################################################
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config=args,
        name=run_name,
    )
    write_args_to_file(args, output_path)

    ###########################################################################
    # Load the tokenizers, the model, and the peft model, then distribute it.
    ###########################################################################
    config = transformers.AutoConfig.from_pretrained(  # type: ignore
        model_name,
        trust_remote_code=True,
    )

    tmp_tokenizers = lib_trl_utils.load_tokenizers(
        model_name=model_name,
        config=config,
    )

    forward_tokenizer = tmp_tokenizers["forward_tokenizer"]
    prediction_tokenizer = tmp_tokenizers["prediction_tokenizer"]
    del tmp_tokenizers

    pretrained_model = lib_trl_utils.load_then_peft_ize_model(
        peft_config_dict=peft_config_dict,
        model_name=model_name,
        precision=precision,
        use_peft=use_peft,
        forward_tokenizer=forward_tokenizer,
        prediction_tokenizer=prediction_tokenizer,
    )
    
    accelerator = accelerate.Accelerator()
    model = accelerator.prepare_model(pretrained_model)


    ###########################################################################
    # Setup the per split 
    # - Datasets
    # - Dataloaders
    # - Output writers
    # - Wandb sample loggers
    ###########################################################################
    datasets = {}
    dataloaders = {}
    if accelerator.is_main_process:
        output_writers = {}
        wandb_logging_states = {}

    for split in ["train", "validation"]:

        datasets[split] = lib_data_commonsense_qa.CommonSenseScratchpadGenMC(
            any_tokenizer=prediction_tokenizer,
            split=split,
        )

        dataloaders[split] = accelerator.prepare_data_loader(
                torch.utils.data.DataLoader(
                datasets[split], 
                shuffle=False, 
                batch_size=batch_size, 
                collate_fn=lambda samples: data_collator(
                    samples, prediction_tokenizer=prediction_tokenizer,
                ),
            )
        )

        if accelerator.is_main_process:
            output_writers[split] = OutputSampleWriter(
                output_path=output_path, 
                split=split, 
                prediction_tokenizer=prediction_tokenizer,
            )
            wandb_logging_states[split] = WandbLoggingState(
                any_tokenizer=prediction_tokenizer, 
                split=split,
                also_print=True,
            )

    ###########################################################################
    # Main action, generation loop.
    # - Batch generation.
    # - Output writing.
    # - Wandb logging.
    ###########################################################################
    for split in dataloaders:
        for batch_idx, batch in enumerate(tqdm.tqdm(
            dataloaders[split], 
            desc=f"GENERATING FOR {split}", 
            disable=not accelerator.is_main_process
        )):
            
            clean_gathered_output_tokens, raw_gathered_output_tokens = gathered_batch_gen(
                generate_kwargs=generate_kwargs,
                accelerator=accelerator, 
                model=model, 
                batch=batch,
                prediction_tokenizer=prediction_tokenizer,
            )

            if accelerator.is_main_process:
                output_writers[split](
                    batch=batch["other_info"],
                    raw_output_sample_ids=raw_gathered_output_tokens.detach().cpu().numpy(),
                    clean_output_sample_ids=clean_gathered_output_tokens.detach().cpu().numpy(),
                )
                wandb_logging_states[split](
                    batch_idx=batch_idx,
                    batch=batch["other_info"], 
                    raw_output_tokens=raw_gathered_output_tokens,
                    clean_output_tokens=clean_gathered_output_tokens,
                )

        if accelerator.is_main_process:
            output_writers[split].close()

if __name__ == "__main__":
    fire.Fire(main)