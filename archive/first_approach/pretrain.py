import dataclasses
from pathlib import Path
from typing import *

import numpy as np
import torch
import pytorch_lightning as pl
import transformers

import console
import constants
import general_utils as utils
import train_utils


CONSOLE = console.Console(force_terminal=True, force_interactive=True, width=200)

@dataclasses.dataclass
class MLETrainingCollator:
    _tokenizer: transformers.PreTrainedTokenizer
    _lm_masking_mode: str

    def __call__(self, raw_examples):
        """
        - For perplexity evaluation:
            (The chainer should already be in place for input_ids and input_ids_and_scratchpad_with_value)
            - input_ids: question + chainer (e.g., " -> ") + scratchpad + value
            - attention_mask: the same as above, but with 0s everywhere there is padding
            - labels: -100 except scratchpad + value (so, for the question, the chainer and the padding.)

        """

        examples = train_utils.prep_mle_train_and_valid(
            examples=raw_examples, 
            eos_token_id=self._tokenizer.eos_token_id,
            scratchpad_eos_token_id=self._tokenizer.cls_token_id, 
            pad_token_id=self._tokenizer.pad_token_id,
            lm_masking_mode=self._lm_masking_mode,
        )

        return examples



class PreTrain(pl.LightningModule):
    def __init__(
        self,
        *,
        batch_sizes: Dict[str, int],
        chainer: str,
        datasets: Dict[str, torch.utils.data.Dataset],
        generation_kwargs: dict[str, Any],
        learning_rate: float,
        meta_info: dict,
        model: transformers.GPT2LMHeadModel,
        is_adamw: bool,
        lm_masking_mode: str,
        path_log_results: Path,
        scheduler_type: str,
        tokenizer: transformers.PreTrainedTokenizer,
        wandb_logger: Optional[pl.loggers.WandbLogger],
        weight_decay: Optional[float],
        shuffle_training_data: bool,
        shuffle_validation_data: bool,
        scheduler_fn: callable,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "datasets", "tokenizer", "scheduler_fn"])
        self._chainer:                  Final[str]                                 = chainer
        self._datasets:                 Final[Dict[str, torch.utils.data.Dataset]] = datasets
        self._value_model:              Optional[transformers.GPT2PreTrainedModel] = None
        self._tokenizer:                Final[transformers.PreTrainedTokenizer]    = tokenizer
        self._model:                    Final[transformers.GPT2LMHeadModel]        = model
        self._wandb_logger:             Final[pl.loggers.WandbLogger]              = wandb_logger
        self._generation_kwargs:        Final[dict[str, Any]]                      = generation_kwargs
        self._batch_size:               Final[dict[str, int]]                      = batch_sizes
        self._dataloader_num_workers:   Final[int]                                 = dataloader_num_workers
        self._lm_masking_mode:          Final[str]                                 = lm_masking_mode
        self._meta_info                                                            = meta_info
        self._logging_conf:             Final[dict[str, bool]]                     = dict(
            prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True
        )
        self._scheduler_fn:              Final[Callable]                            = scheduler_fn

        ################################################################################
        # Related to datasets
        ################################################################################
        self._shuffle_train:       Final[bool]           = shuffle_training_data
        self._shuffle_val:         Final[bool]           = shuffle_validation_data
        self._training_collator:  Final[dict[str, Any]] = MLETrainingCollator(self._tokenizer, self._lm_masking_mode)
        

        ################################################################################
        # Rel. to logging results for answer overlap estim.
        ################################################################################
        self._results_to_log:   Optional[dict[str, dict[bool, dict[str, torch.Tensor]]]] = {}
        self._labels_to_log:    dict[str, str] = {}
        self._path_log_results: Final[Path]    = path_log_results

        ################################################################################
        # Specific to the optimizer, its scheduler
        ################################################################################
        self._weight_decay:   Final[Optional[float]] = weight_decay
        self._learning_rate:  Final[float]           = learning_rate
        self._is_adamw:       Final[bool]            = is_adamw
        self._scheduler_type: Final[str]             = scheduler_type
        self._scheduler                              = None


    def get_model(self):
        return self._model


    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)


    def training_step(self, batch, batch_idx):
        assert False
        
        assert "labels" in batch, (
            "Labels must be in batch. We must mask the input section with -100"
        )

        batch = {
            k: v for k, v in batch.items() 
            if k in ["input_ids", "attention_mask", "labels"]
        }
        
        outputs = self(**batch)

        self.log(
            "train_loss", 
            outputs.loss.item(), 
            batch_size=self._batch_size[constants.PipelineModes.MLE_TRAINING], 
            **self._logging_conf
        )

        # TODO: this is costly
        assert not torch.any(torch.isnan(outputs.loss)), "Loss is NaN"

        return outputs.loss


    def _generate(self, *, batch, generation_kwargs, model):
        utils.check_contained("generation_input_ids", batch.keys())
        utils.check_equal(batch["generation_input_ids"].ndim, 2)
        assert torch.all(batch["generation_input_ids"][:, -1] != self._tokenizer.pad_token_id), (
            "Batches need to be padded left for batch "
            "generation. Found a pad token at the end of a sequence."
        )
        
        generation_inputs = batch["generation_input_ids"]        
        generation_attention_mask = batch["generation_attention_mask"]
        
        inputs_outputs = model.generate(
            input_ids=generation_inputs, 
            attention_mask=generation_attention_mask, 
            **generation_kwargs,
        )
    
        return inputs_outputs


    def validation_step(self, batch: Dict[str, torch.LongTensor], batch_idx):  # type: ignore[override]
        return train_utils.shared_validation_step(self, batch, batch_idx, chainer=self._chainer)


    def predict_step(self, batch, batch_idx):
        batch = cast(Dict[str, torch.LongTensor], batch)
        mode = constants.PipelineModes.VALIDATION
        generated_decoded, label = self._generate(self._model, batch, self._generation_kwargs[mode])
        train_utils.print_predictions(generated_decoded, label)


    def on_validation_epoch_end(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        pass


    def configure_optimizers(self):
        """
        See ref
        https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        """
        if self._is_adamw:
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.Adam

        optimizer = optimizer_class(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

        self._scheduler = self._scheduler_fn[self._scheduler_type](
            optimizer, train_utils.compute_steps_per_epoch(self.trainer)
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=self._scheduler,
                interval="step",
                frequency=1,
                name=type(self._scheduler).__name__,
            )
        )


    def train_dataloader(self):        

        return torch.utils.data.DataLoader(
            self._datasets[constants.PipelineModes.MLE_TRAINING],
            collate_fn=self._training_collator,
            batch_size=self._batch_size[constants.PipelineModes.MLE_TRAINING],
            num_workers=self._dataloader_num_workers,
            shuffle=self._shuffle_train,
        )


    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._datasets[constants.PipelineModes.VALIDATION],
            collate_fn=train_utils.ValitationCollator(self._tokenizer, self._lm_masking_mode),
            batch_size=self._batch_size[constants.PipelineModes.VALIDATION],
            num_workers=self._dataloader_num_workers,
            shuffle=self._shuffle_val,
        )

    
    def predict_dataloader(self):
        return self.val_dataloader()


    def on_save_checkpoint(self, ckpt):
        return 

    def inference(self, batch: Dict[str, torch.LongTensor], mode):
        gen_outputs = self._generate(
            model             = self._model,
            batch             = batch, 
            generation_kwargs = self._generation_kwargs[mode], 
        )
        generated_tokens = gen_outputs[:, batch["generation_input_ids"].shape[1]:]

        ###################################################################
        # Compute Scratchpad Accuracy
        ###################################################################
        pos_clss = (generated_tokens == self._tokenizer.cls_token_id).long() * torch.arange(
            generated_tokens.shape[1]).unsqueeze(0).to(generated_tokens.device)
        last_cls_pos = (pos_clss).max(dim=1).values.unsqueeze(-1)
        del pos_clss

        is_scratchpad = torch.arange(generated_tokens.shape[1]).repeat(
            (generated_tokens.shape[0], 1)).to(generated_tokens.device) < last_cls_pos
        gen_scratchpads = train_utils.remove_padding(generated_tokens, is_scratchpad)
        scratchpad_texts = train_utils.get_scratchpad_texts(gen_scratchpads, batch["scratchpad"], self._tokenizer)
        scratchpad_matches = np.fromiter((gen == ref for gen, ref in scratchpad_texts), dtype=bool)
        scratchpads_acc = np.mean(scratchpad_matches)

        ###################################################################
        # Compute Accuracy p(y | x, z) only
        ###################################################################
        gen_values     = train_utils.remove_padding(
            generated_tokens, is_scratchpad.logical_not())
        values_texts   = train_utils.get_values_texts(
            gen_values, batch["value"], tokenizer=self._tokenizer)
        values_matches = np.fromiter((gen == ref for gen, ref in values_texts), dtype=bool)
        values_acc     = np.mean(values_matches)

        gen_outputs = gen_outputs[:, batch["generation_input_ids"].shape[1]:]

        return dict(
            scratchpad_matches = scratchpad_matches, 
            scratchpads_acc    = scratchpads_acc, 
            values_matches     = values_matches, 
            gen_outputs        = gen_outputs,
            values_acc         = values_acc, 
        )