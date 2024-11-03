import enum
import os
import pathlib
import sys

import fire
import hydra
import numpy as np
import rich.traceback
import torch
import transformers
rich.traceback.install()

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def add_to_sys_path(target):
    target = pathlib.Path(target)
    assert target.exists(), f"Path {target} does not exist"
    assert target.is_dir(), f"Path {target} is not a directory"
    sys.path.append(str(target))


add_to_sys_path(SCRIPT_DIR.parent / "approach_sft")
add_to_sys_path(SCRIPT_DIR.parent / "with_trl")

from approach_sft import lib_sft_dataloaders

from with_trl import lib_metric, lib_utils
from with_trl.libs_extraction import lib_final_line
from with_trl.libs_extraction import lib_numerical

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

class TrainingType(enum.Enum):
    SFT = "sft"
    RL = "rl"


def load_and_parse_sft(checkpoint_prefix):
    with open(checkpoint_prefix.parent / f"{checkpoint_prefix.name}.pt", "rb") as f:
        config = torch.load(f, weights_only=False)

    model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint_prefix)
    dataset_choice = config["cfg"]["dataset_choice"]
    forward_tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["cfg"]["model_name_or_path"])
    prediction_tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["cfg"]["model_name_or_path"], padding_side="left")

    return dict(
        model=model, 
        dataset_choice=dataset_choice, 
        extractor_ignore_one_line=config["cfg"]["extractor_ignore_one_line"],
        forward_tokenizer=forward_tokenizer,
        prediction_tokenizer=prediction_tokenizer,
        cfg=config["cfg"]
    )


def load_metrics(dataset_choice, extractor_ignore_one_line, forward_tokenizer):
    if dataset_choice == lib_utils.Datasets.ARITHMETIC:
        metrics = dict(
            exact_match = lib_metric.ScratchpadAnswerAccuracy(
                extractor=lib_final_line.FinalLineExtractor(
                    ignore_one_line = extractor_ignore_one_line,
                    pad_token=forward_tokenizer.pad_token,
                ),
                pad_token=forward_tokenizer.pad_token,
            )
        )
    elif dataset_choice == lib_utils.Datasets.GSM8K:
        metrics = dict(
            exact_match = lib_metric.ScratchpadAnswerAccuracy(
                extractor=lib_numerical.ConvToNum(),
                pad_token=forward_tokenizer.pad_token,
            )
        )
    elif dataset_choice == lib_utils.Datasets.MATH:
        metrics = dict(
            exact_match = lib_metric.HendrycksMathScratchpadAnswerAccuracy()
        )

    else:
        raise NotImplementedError(dataset_choice)

    return metrics

def inference(
        from_training_type=TrainingType.SFT, 
        checkpoint_prefix="/network/scratch/g/gagnonju/marglicot_saves/sft_saves/cot_math_qwen-2024-10-20_00-01-21/ep_1_model"
    ):

    from_training_type = TrainingType(from_training_type)
    checkpoint_prefix = pathlib.Path(checkpoint_prefix)
    checkpoint_dir = checkpoint_prefix.parent
    assert checkpoint_dir.exists(), (
        f"Checkpoint path {checkpoint_dir} does not exist"
    )

    if from_training_type == TrainingType.SFT:
        load_and_parse_sft_dict = load_and_parse_sft(checkpoint_prefix)        
    
    metrics = load_metrics(
        dataset_choice=load_and_parse_sft_dict["dataset_choice"],
        extractor_ignore_one_line=load_and_parse_sft_dict["extractor_ignore_one_line"],
        forward_tokenizer=load_and_parse_sft_dict["forward_tokenizer"],
    )

    forward_tokenizer = load_and_parse_sft_dict["forward_tokenizer"]
    prediction_tokenizer = load_and_parse_sft_dict["prediction_tokenizer"]
    cfg = load_and_parse_sft_dict["cfg"]

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
        outlines_context          = None,
        prediction_tokenizer      = prediction_tokenizer,
        qty_eval_small            = cfg.qty_eval_small,
        seed                      = 0,
        train_batch_size          = cfg.output_type.train_batch_size * WORLD_SIZE,
        subset_data               = cfg.subset_data,
        use_workers               = cfg.use_workers,
    )


if __name__ == "__main__":
    fire.Fire(inference)