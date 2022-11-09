from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import transformers

from trlx.data.ilql_types import ILQLBatch, ILQLElement
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline


@register_datapipeline
class PromptPipeline(BasePipeline):
    """
    Tokenizes texts, and then pads them into batches
    """

    def __init__(self, *, prompts, tokenizer=None):
        super().__init__()
        self._tokenizer = tokenizer
        self._prompts = prompts

    def __getitem__(self, ix: int):
        assert self._prompts[ix].keys() == {"inputs", "labels"}, (
            self._prompts[ix].keys(), {"inputs", "labels"}
        )

        inputs = self._prompts[ix]["inputs"]
        batch = self._tokenizer(inputs)
        label_key = "labels"

        assert label_key in self._prompts[ix], list(self._prompts[ix].keys())

        if label_key in self._prompts[ix]:
            
            outputs              = self._prompts[ix][label_key]
            output_labels        = self._tokenizer(outputs)
            batch[label_key]     = output_labels["input_ids"]
        
        return batch

    def __len__(self) -> int:
        return len(self._prompts)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        assert self._tokenizer
        import rich

        pre_collate_fn = (
            transformers.DataCollatorForSeq2Seq(self._tokenizer, return_tensors="pt") 
            if self._tokenizer else torch.vstack
        )

        def collate_fn(*args, **kwargs):
            rich.print("[bold green]Collate fn called")
            output = pre_collate_fn(*args, **kwargs)
            rich.print("[bold red]Collate fn finished")
            return output

        return DataLoader(
            self, 
            batch_size=batch_size, 
            collate_fn=collate_fn, 
            shuffle=shuffle,
        )


class ILQLRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training ILQL
    """

    def __init__(
        self, input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones
    ):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rewards = rewards
        self.states_ixs = states_ixs
        self.actions_ixs = actions_ixs
        self.dones = dones

    def __getitem__(self, ix: int) -> ILQLElement:
        return ILQLElement(
            self.input_ids[ix],
            self.attention_mask[ix],
            self.rewards[ix],
            self.states_ixs[ix],
            self.actions_ixs[ix],
            self.dones[ix],
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int):
        def collate_fn(elems: Iterable[ILQLElement]):
            return ILQLBatch(
                pad_sequence(
                    [x.input_ids      for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.attention_mask for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.rewards        for x in elems], batch_first=True, padding_value=0.0
                ),
                pad_sequence(
                    [x.states_ixs     for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.actions_ixs    for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.dones          for x in elems], batch_first=True, padding_value=0
                ),
            )

        return DataLoader(
            self, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
