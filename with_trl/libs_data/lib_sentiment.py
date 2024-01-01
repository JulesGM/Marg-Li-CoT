import os

import torch
import torch.utils.data

import lib_base_classes
import lib_sentiment_specific
import libs_data


LOCAL_RANK = os.environ.get("LOCAL_RANK", 0)


class SentimentData(libs_data.lib_base.Dataset):
    def __init__(self, any_tokenizer, split):
        self._dataset = lib_sentiment_specific.prep_dataset_rl(
            any_tokenizer=any_tokenizer,
            txt_in_len=5,
            split=split,
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        inner_iter = self._dataset[idx]
        return lib_base_classes.DataItemContainer(
            tok_ref_query=inner_iter["input_ids"],
            tok_ref_answer=None,
            tok_ref_scratchpad=None,
            
            detok_ref_query=inner_iter["query"],
            detok_ref_answer=None,
            detok_ref_scratchpad=None,

            difficulty_level=None,
            extra_information=None,
        )
        
    def get_extractor(self) -> None:
        return lambda x: x
    
    @property
    def use_few_shots(self):
        return False
    