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
import datasets
import fire
import h5py
import more_itertools as mit
import numpy as np
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
from approach_answer import lib_approach_utils
from approach_answer import lib_wandb_logger
from approach_answer import lib_output_writer

datasets.disable_caching()
rich.traceback.install()

LOGGER = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


RANK = int(os.environ.get("RANK", 0))
if RANK == 0:
    print("Done with imports")

###############################################################################
# Defaut Hyperparameter Values, can be changed with CLI
###############################################################################
DEFAULT_DO_DISTILLATION = False
DEFAULT_JUST_DEVICE_MAP = True

DEFAULT_MODEL_NAME = "stabilityai/StableBeluga2"; DEFAULT_PRECISION = lib_utils.ValidPrecisions.bfloat16; DEFAULT_BATCH_SIZE = 16

# DEFAULT_MODEL_NAME = "EleutherAI/pythia-410m"; DEFAULT_BATCH_SIZE = 64; DEFAULT_PRECISION = lib_utils.ValidPrecisions.bfloat16
# DEFAULT_MODEL_NAME = "EleutherAI/gpt-j-6B"; DEFAULT_BATCH_SIZE = 8; DEFAULT_PRECISION = lib_utils.ValidPrecisions.bfloat16
DEFAULT_OUTPUT_PATH = "/network/scratch/g/gagnonju/scratchpad_gen_outputs/"
DEFAULT_USE_PEFT = True

DEFAULT_GENERATE_KWARGS = dict(
    max_new_tokens=200,
    min_new_tokens=4,
    do_sample=False,
    synced_gpus=os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "") == "3",
)

DEFAULT_WANDB_PROJECT = "fed-answer-gen"
DEFAULT_WANDB_ENTITY = "julesgm"
###############################################################################


def data_collator(batch, prediction_tokenizer, device):
    assert prediction_tokenizer.padding_side == "left", prediction_tokenizer.padding_side
    
    return dict(
        inputs=prediction_tokenizer.pad(
                dict(input_ids=[x["ref_fs_scratchpad_gen_query_tok"] for x in batch]),
                return_tensors="pt",
            ).to(device),
        other_info=batch,
    )


def gathered_batch_gen(
        *, 
        accelerator: accelerate.Accelerator, 
        model: transformers.PreTrainedModel,
        batch: dict[str, torch.Tensor],
        generate_kwargs: dict[str, Any],
        forward_tokenizer: transformers.PreTrainedTokenizerBase,
        do_distillation: bool,
        just_device_map: bool,
    ):

    line_return_id = mit.last(forward_tokenizer.encode("\n"))
    output = accelerator.unwrap_model(model).generate(
        **batch["inputs"],
        **generate_kwargs,

        use_cache=True,
        output_scores=True,
        early_stopping=True,
        return_dict_in_generate=True,
        eos_token_id=line_return_id,
    )

    scores = output.scores
    output_tokens = output.sequences
    del output

    before_shape = output_tokens.shape
    output_tokens = output_tokens[:, batch["inputs"]["input_ids"].shape[1]:].contiguous()
    scores = torch.stack(scores, dim=1).contiguous()
    assert output_tokens.shape == scores.shape[:2], (output_tokens.shape, scores.shape[:2])
    assert output_tokens.shape[1], (output_tokens.shape, before_shape)

    ###########################################################################
    # Extract the tokens and the logits up to the first line return.
    ###########################################################################
    
    attempt_clean_tokens = []
    
    if do_distillation:
        attempt_clean_logits = []
    
    for batch_id, ids in enumerate(output_tokens):
        """
        text = forwards_tokenizer.decode(ids)
        first_lr_str = text.find("\n")
        if first_lr_str == -1:
            first_lr = None
        else:
            ids = forwards_tokenizer.encode(text)
            assert all(x == y for x, y in mit.zip_equal(ids, output_tokens[batch_id])
            first_lr = ids.char_to_token(first_lr_str)
        """

        try:
            first_lr = ids.tolist().index(line_return_id)
        except ValueError as e:
            # If we don't find a line return, just use the whole thing
            if "is not in list" in str(e):
                first_lr = None
            else:
                raise

        assert ids.ndim == 1, ids.shape
        attempt_clean_tokens.append(ids[:first_lr])

        if do_distillation:
            attempt_clean_logits.append(scores[batch_id, :first_lr].detach())

    ###########################################################################
    # Gather the clean tokens
    ###########################################################################
    # The tokens need to be stored in a tensor that is of the same shape
    # on all the processes first.
    ###########################################################################
    local_padded_clean_tokens = forward_tokenizer.pad(
        dict(input_ids=attempt_clean_tokens),
        return_tensors="pt",
    )["input_ids"].to(accelerator.device)

    global_padded_tokens = accelerator.pad_across_processes(
        local_padded_clean_tokens,
        pad_index=forward_tokenizer.pad_token_id,
        pad_first=False,
        dim=1,
    )

    gathered_attempt_clean_tokens = accelerator.gather(global_padded_tokens).cpu()
    del local_padded_clean_tokens, global_padded_tokens

    ###########################################################################
    # Gather the raw tokens
    ###########################################################################
    # Similar, but the local sizes already match.
    ###########################################################################
    global_padded_raw_ids = accelerator.pad_across_processes(
        output_tokens,
        pad_index=forward_tokenizer.pad_token_id,
        pad_first=False,
        dim=1,
    )
    gathered_raw_output_tokens = accelerator.gather(global_padded_raw_ids).cpu()
    del global_padded_raw_ids

    ###########################################################################
    # Gather logits
    ###########################################################################
    # Logits are a bit more complicated. They need to be padded in 2D.
    # We don't pad locally to save on transfer qties.
    ###########################################################################
    if do_distillation:
        clean_logits = lib_approach_utils.pad_logits_across_processes(
            accelerator=accelerator,
            logits=clean_logits, 
            fill_value=float("nan"), 
        ).to(accelerator.device)
        gathered_attempt_clean_logits = accelerator.gather(
            attempt_clean_logits
        ).cpu()

    return dict(
        clean_output_tokens=gathered_attempt_clean_tokens,
        clean_logits=gathered_attempt_clean_logits if do_distillation else None,
        raw_output_tokens=gathered_raw_output_tokens,
) 


def main(
    run_name: str="gen-scratchpad",
    generate_kwargs: dict=DEFAULT_GENERATE_KWARGS,
    model_name: str = DEFAULT_MODEL_NAME,
    output_path: str = DEFAULT_OUTPUT_PATH,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    do_distillation: bool = DEFAULT_DO_DISTILLATION,
    precision: int = DEFAULT_PRECISION,
    just_device_map: bool = DEFAULT_JUST_DEVICE_MAP,
):
    
    args = lib_approach_utils.convert_args_for_wandb(locals().copy())
    accelerator = accelerate.Accelerator()
    if accelerator.is_main_process:
        lib_approach_utils.print_dict_as_table(args, title="[blue bold]Script Args:")
    DatasetCls = lib_data_commonsense_qa.CommonSenseScratchpadGenMC

    ###########################################################################
    # Setup the directories for logging / data saving
    ###########################################################################
    output_path = pathlib.Path(output_path) 
    output_path /= run_name.replace(" ", "-"
        ).replace("\n", "-"
        ).replace("\t", "-"
        ).replace("/", "-"
    )
    assert output_path.parent.exists(), f"Parent directory of {output_path} does not exist."
    if accelerator.is_main_process:
        output_path.mkdir(exist_ok=False)
        # We don't need to wait because only RANK 0 
        # writes in the directory

    ###########################################################################
    # Setup wandb & save the script config
    ###########################################################################
    config_dict = dict(
        args=args, 
        few_shots_str=DatasetCls.FEW_SHOTS_STR,
        dataset_class_name=DatasetCls.__name__,
    )
    if accelerator.is_main_process:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config=config_dict,
            name=run_name,
        )            
        lib_approach_utils.write_args_to_file(
            config_dict=config_dict,
            output_path=output_path,
        )

    ###########################################################################
    # Load the tokenizers, the model, and the peft model, then distribute it.
    ###########################################################################
    config = transformers.AutoConfig.from_pretrained(  # type: ignore
        model_name,
    )

    tmp_tokenizers = lib_trl_utils.load_tokenizers(
        model_name=model_name,
        config=config,
    )

    forward_tokenizer = tmp_tokenizers["forward_tokenizer"]
    prediction_tokenizer = tmp_tokenizers["prediction_tokenizer"]
    del tmp_tokenizers
    
    if accelerator.is_main_process:
        rich.print("Loading the model...")

    pretrained_model = lib_trl_utils.load_then_peft_ize_model(
        precision=precision,
        use_peft=False,
        peft_config_dict=None,
        model_name=model_name,
        forward_tokenizer=forward_tokenizer,
        prediction_tokenizer=prediction_tokenizer,
        just_device_map=just_device_map,
    )

    if just_device_map:
        model = pretrained_model
    else:
        model = accelerator.prepare_model(pretrained_model)

    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if accelerator.is_main_process:
        rich.print("Model loaded.")

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
        datasets[split] = DatasetCls(
            any_tokenizer=prediction_tokenizer,
            split=split,
        )

        dataloaders[split] = accelerator.prepare_data_loader(
                torch.utils.data.DataLoader(
                datasets[split], 
                shuffle=False, 
                batch_size=batch_size, 
                collate_fn=lambda samples: data_collator(
                    samples, 
                    prediction_tokenizer=prediction_tokenizer,
                    device=accelerator.device,
                ),
            )
        )

        if accelerator.is_main_process:
            output_writers[split] = lib_output_writer.OutputSampleWriter(
                do_distillation=do_distillation,
                forward_tokenizer=forward_tokenizer,
                output_path=output_path, 
                split=split, 
                prediction_tokenizer=prediction_tokenizer,
                dataset_split_size=len(datasets[split]),
                max_gen_len=generate_kwargs["max_new_tokens"],
                few_shots_str=DatasetCls.FEW_SHOTS_STR,
            )
            wandb_logging_states[split] = lib_wandb_logger.WandbLoggingState(
                any_tokenizer=prediction_tokenizer, 
                split=split,
                also_print=True,
            )
        # Lots of stuff happening in OutputSampleWriter.__init__ so we don't
        # want to let the other processes get too far ahead, where the
        # timeouts on the gathers may trigger
        accelerator.wait_for_everyone()

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
            returned_dict = gathered_batch_gen(
                generate_kwargs=generate_kwargs,
                accelerator=accelerator, 
                model=model, 
                batch=batch,
                forward_tokenizer=forward_tokenizer,
                do_distillation=do_distillation,
                just_device_map=just_device_map,
            )
            clean_gathered_output_tokens = returned_dict["clean_output_tokens"]
            raw_gathered_output_tokens   = returned_dict["raw_output_tokens"]
            clean_gathered_logits        = returned_dict["clean_logits"]

            gathered_batch = lib_output_writer.OutputSampleWriter.gather_batch_for_writing(
                batch["other_info"])
            
            if accelerator.is_main_process:

                output_writers[split](
                    gathered_batch=gathered_batch,
                    clean_gathered_output_sample_ids=clean_gathered_output_tokens,
                    clean_gathered_logits=clean_gathered_logits,
                )

                # We only log local samples
                wandb_logging_states[split](
                    batch_idx=batch_idx,
                    batch=batch["other_info"], 
                    # Will always take the first set
                    raw_output_tokens=raw_gathered_output_tokens[
                        batch_size * accelerator.local_process_index:
                        batch_size * (accelerator.local_process_index + 1)
                    ],
                    clean_output_tokens=clean_gathered_output_tokens[
                        batch_size * accelerator.local_process_index:
                        batch_size * (accelerator.local_process_index + 1)
                    ],
                )

        if accelerator.is_main_process:
            output_writers[split].close()


if __name__ == "__main__":
    fire.Fire(main)