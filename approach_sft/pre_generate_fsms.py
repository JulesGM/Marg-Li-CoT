"""

THIS IS INCOMPLETE

The idea is to iterate over the dataloaders and generate 
the FSMs for each of the examples, and then save them to an 
arrow file.

This would require the dataloaders to return the index when 
iterating over the dataloaders.

"""
import outlines
import outlines.fsm.guide
import fire
import outlines.models
import outlines.models.transformers
import pandas as pd
import lib_sft_dataloaders
from with_trl import lib_trl_utils
import hydra


@hydra.main(
    version_base="1.3.2", 
    config_path="config", 
    config_name="config",
)
def main(cfg):
    tmp_tokenizers       = lib_trl_utils.load_tokenizers(cfg.model_name_or_path)
    forward_tokenizer    = tmp_tokenizers["forward_tokenizer"   ]
    prediction_tokenizer = tmp_tokenizers["prediction_tokenizer"]
    del tmp_tokenizers

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
        prediction_tokenizer      = prediction_tokenizer,
        qty_eval_small            = cfg.qty_eval_small,
        train_batch_size          = cfg.output_type.train_batch_size * WORLD_SIZE,
        seed                      = 0,
    )
    tokenizer = outlines.models.transformers.Tokenizer(
        cfg.model_name_or_path,)
    

    for batch in dataloaders:
        new_fsm = outlines.fsm.guide.RegexGuide(
            regex_str, tokenizer,
        )

