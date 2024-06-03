import dataclasses
import itertools
from pathlib import Path
import random
from typing import *


import numpy as np
import pytorch_lightning as pl
import rich.table as table
import torch
import transformers

import general_utils as utils
import constants
import train_utils

import console
import rl

CONSOLE = console.Console(force_terminal=True, force_interactive=True, width=200)



def create_value_model(original_model, fixed_base: bool):
    ###################################################################
    # Create the value model
    ###################################################################
    model_copy = train_utils.clone_hf_model(original_model).to(original_model.device)
    if fixed_base:
        train_utils.fix_model_params_in_place(model_copy)
    
    model_copy.config.num_labels = 1
    value_model = rl.ppo.GPT2ForTokenClassificationWithActivationOnTop(torch.tanh, model_copy.config)
    value_model.transformer = model_copy.transformer    

    utils.check_isinstance(value_model.transformer, transformers.GPT2Model)
    utils.check_equal(value_model.classifier.out_features, 1)

    return value_model


class RLTraining(pl.LightningModule):
    def __init__(
        self,
        *,
        model: transformers.GPT2LMHeadModel,
        fixed_scratchpad_model,
        fixed_answer_model,
        
        batch_sizes: Dict[str, int],
        chainer,
        datasets: Dict[str, torch.utils.data.Dataset],
        generation_kwargs: dict[str, Any],
        learning_rate: float,
        loss_mode: str,
        meta_info: dict,
        is_adamw: bool,
        lm_masking_mode: str,
        path_log_results: Path,
        scheduler_type: str,
        tokenizer: transformers.PreTrainedTokenizer,
        wandb_logger: Optional[pl.loggers.WandbLogger],
        weight_decay: Optional[float],
        shuffle_training_data,
        shuffle_validation_data,
        scheduler_fn,
        dataloader_num_workers=0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "model", 
            "datasets", 
            "tokenizer", 
            "scheduler_fn", 
            "fixed_answer_model",
            "fixed_scratchpad_model", 
        ])

        utils.check_contained(
            loss_mode, 
            constants.LossModes.__members__.values()
        )
        self._chainer                                                              = chainer                                       
        self._loss_mode                                                            = loss_mode
        self._datasets:                 Final[Dict[str, torch.utils.data.Dataset]] = datasets
        self._value_model:              Optional[transformers.GPT2PreTrainedModel] = None
        self._tokenizer:                Final[transformers.PreTrainedTokenizer]    = tokenizer
        
        self._fixed_model:              Optional[transformers.GPT2LMHeadModel]     = fixed_scratchpad_model
        self._fixed_answer_model:       Optional[transformers.GPT2LMHeadModel]     = fixed_answer_model
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
        self._scheduler_function                                                   = scheduler_fn


        ################################################################################
        # Related to datasets
        ################################################################################
        self._shuffle_train:       Final[bool]           = shuffle_training_data
        self._shuffle_val:         Final[bool]           = shuffle_validation_data
        self._training_collator:   Final[str]            = MarginalLikelihoodTrainingCollator(self._tokenizer, self._lm_masking_mode)

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

        
    def inference(self, batch, mode):
        utils.check_equal("cuda", self._fixed_model.device.type)
        ###################################################################
        # Compute the scratchpad with the learnable model
        ###################################################################
        config_scratchpad                 = self._generation_kwargs[mode].copy()
        config_scratchpad["eos_token_id"] = self._tokenizer.cls_token_id
        gen_outputs = self._generate(
            model             = self._model,
            batch             = batch, 
            generation_kwargs = config_scratchpad, 
        )

        ###################################################################
        # Compute The accuracy of Scratchpads
        ###################################################################
        gen_scratchpads = gen_outputs[:, batch["generation_input_ids"].shape[1]:]
        scratchpad_texts = train_utils.get_scratchpad_texts(gen_scratchpads, batch["scratchpad"], tokenizer=self._tokenizer)
        scratchpad_matches = np.fromiter((gen == ref for gen, ref in scratchpad_texts), dtype=bool)
        scratchpads_acc = np.mean(scratchpad_matches)

        ###################################################################
        # Compute the answer after the scratchpad
        ###################################################################
        config_answer  = self._generation_kwargs[mode].copy()
        unpadded       = train_utils.remove_padding(gen_outputs, gen_outputs != self._tokenizer.pad_token_id)
        padded         = train_utils.pad           (unpadded, pad_token_id=self._tokenizer.pad_token_id, direction="left")
        attention_mask = train_utils.generate_mask (unpadded, "left")

        new_batch = dict(
            generation_input_ids      = padded        .to(self._fixed_model.device),
            generation_attention_mask = attention_mask.to(self._fixed_model.device),
        )
        gen_values = self._generate(
            model             = self._fixed_answer_model,
            generation_kwargs = config_answer, 
            batch             = new_batch, 
        )[:, new_batch["generation_input_ids"].shape[1]:]

        ###################################################################
        # Compute Accuracy p(y | x, z) only
        ###################################################################
        values_texts   = train_utils.get_values_texts(gen_values, batch["value"], tokenizer=self._tokenizer)
        values_matches = np.fromiter((gen == ref for gen, ref in values_texts), dtype=bool)
        values_acc     = np.mean(values_matches)

        gen_outputs = torch.cat([gen_scratchpads, gen_values], dim=1)
        
        return dict(
            scratchpad_matches = scratchpad_matches, 
            scratchpads_acc    = scratchpads_acc, 
            values_matches     = values_matches, 
            gen_outputs        = gen_outputs,
            values_acc         = values_acc, 
        )


    def get_model(self):
        return self._model

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """
        
        p(z|x): <generation>
            generation_input_ids: masked, question, chainer. *Padded left*.
            generation_attention_mask

        p(y, z| x): 
            input_ids: input, 
            labels:
        ---

        p(y, z | x) = p(y | z, x) * p(z | x)

        """

        utils.check_equal(self.      _model.device.type, "cuda")
        utils.check_equal(self._fixed_model.device.type, "cuda")
        utils.check_equal(self._fixed_answer_model.device.type, "cuda")

        #######################################################################
        # Useful constants
        #######################################################################
        mode: Final[str] = constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING
        batch_size = self._batch_size[mode]
        vocab_size = len(self._tokenizer)
        num_scratchpads = self._generation_kwargs[mode]["num_return_sequences"]
        disable_timing = True 
        SHOW_SCRATCHPAD_PADDING_TABLE = False
        SHOW_MULTI_SCRATCHPAD_TABLE = True
        SHOW_SMALL_GENERATION_TABLE = False
        torch.autograd.set_detect_anomaly(True)                


        if not disable_timing:
            CONSOLE.print_zero_rank(f"batch size:           {self._batch_size[mode]}")
            CONSOLE.print_zero_rank(f"num return sequences: {self._generation_kwargs[mode]['num_return_sequences']}")

        #######################################################################
        # Generate the scratchpads
        #######################################################################
        with utils.cuda_timeit("bin_refine.py::Generation", disable=disable_timing):
            self._model.eval()
            # Generating until CLS makes us only generate the scratchpad. 
            # Generating until EOS makes us generate the whole output.
            # That's how the model is pre-trained with the mle objective.
            generate_output_dict = self._model.generate(
                input_ids               = batch["generation_input_ids"],
                attention_mask          = batch["generation_attention_mask"], 
                # Config stuff
                eos_token_id            = self._tokenizer.cls_token_id,  
                output_scores           = True,
                return_dict_in_generate = True,
                **self._generation_kwargs[mode],
            )
            self._model.train()


        #######################################################################
        # Preparations common to the different losses
        #######################################################################
        batch_size = batch["generation_input_ids"].shape[0]
        num_scratchpads = self._generation_kwargs[mode]["num_return_sequences"]
        vocab_size = len(self._tokenizer)
        utils.check_equal(self._batch_size[mode], batch_size)

        input_ids_then_scratchpads = generate_output_dict["sequences"].reshape(
            batch_size * num_scratchpads,
            -1,
        )
        
        scores = torch.stack(generate_output_dict["scores"]).reshape(
            batch_size,
            num_scratchpads,
            len(generate_output_dict["scores"]),
            vocab_size
        )

        label_pad_token_id = self._tokenizer.pad_token_id
        utils.check_equal(label_pad_token_id, self._tokenizer.pad_token_id)
        utils.rich_print_zero_rank(f"\n\n[red bold]loss_mode: [white bold]{self._loss_mode.capitalize()}")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sample preparation for per scratchpad stuff
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        samples_marginal = prep_samples_marginal(
            batch               = batch, 
            batch_size          = batch_size, 
            eos_token_id        = self._tokenizer.eos_token_id, 
            cls_token_id        = self._tokenizer.cls_token_id, 
            disable_timing      = disable_timing, 
            inputs_outputs      = input_ids_then_scratchpads, 
            num_scratchpads     = num_scratchpads, 
            generation_kwargs   = self._generation_kwargs[mode], 
            label_pad_token_id  = label_pad_token_id,
            inputs_pad_token_id = self._tokenizer.pad_token_id,
        ) 

        ITGSWRV_attention_mask = samples_marginal.ITGSWRV_attention_mask
        ITGSWRV_ids            = samples_marginal.ITGSWRV_ids
        ITGS_ids               = samples_marginal.ITGS_ids
        ITGS_attention_mask    = samples_marginal.ITGS_attention_mask
        GS_ids                 = samples_marginal.GS_ids
        GS_attention_mask      = samples_marginal.GS_attention_mask
        
        # MITGSWRV: Masked Input Then Generated Scratchpad With Reference Value
        MITGSWRV_ids            = samples_marginal.MITGSWRV_ids
        y_mask_is_not_pad       = samples_marginal.y_mask_is_not_pad
        z_mask_is_not_pad       = samples_marginal.z_mask_is_not_pad
        shift_y_mask_is_not_pad = y_mask_is_not_pad[:, :, 1:]
        shift_z_mask_is_not_pad = z_mask_is_not_pad[:, :, 1:]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Learning rate
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert torch.all(
            ITGSWRV_ids[:, 0] != self._tokenizer.pad_token_id
        )
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]["lr"]                
        utils.check_equal(len(self.trainer.optimizers), 1)
        utils.check_equal(len(optimizer.param_groups), 1)
        CONSOLE.print_zero_rank(f"\n[bold]Learning rate:[/] {lr:.3}")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute the logits
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert (ITGSWRV_ids[:, 0] != self._tokenizer.pad_token_id).all()
        
        ITGSWRV_logits = self._model(
            input_ids      = ITGSWRV_ids,
            attention_mask = ITGSWRV_attention_mask,
        ).logits

        with torch.no_grad():
            ITGSWRV_logits_fixed_model = self._fixed_model(
                input_ids      = ITGSWRV_ids,
                attention_mask = ITGSWRV_attention_mask,
            ).logits

        # if self._fixed_answer_model is not self._fixed_model:
        with torch.no_grad():
            ITGSWRV_logits_fixed_answer_model = self._fixed_answer_model(
                input_ids      = ITGSWRV_ids,
                attention_mask = ITGSWRV_attention_mask,
            ).logits
        # else:
        #     ITGSWRV_logits_fixed_answer_model = ITGSWRV_logits_fixed_model


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Shift the logits and MITGSWRV: Masked Inputs Then Generated Scratchpads With Reference Value
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        shift_MITGSWRV_log_softmax, shift_MITGSWRV_is_pad, shift_MITGSWRV = prep_logits_and_MITGSWRV(
            vocab_size         = vocab_size,
            ITGSWRV_logits     = ITGSWRV_logits, 
            batch_size         = batch_size, 
            MITGSWRV_ids       = MITGSWRV_ids, 
            num_scratchpads    = num_scratchpads, 
            label_pad_token_id = label_pad_token_id, 
            tokenizer          = self._tokenizer,
        ) 
        shift_seq_len = shift_MITGSWRV_log_softmax.shape[2]

        shift_MITGSWRV_log_softmax_fixed_model, _, _ = prep_logits_and_MITGSWRV(
            vocab_size         = vocab_size,
            ITGSWRV_logits     = ITGSWRV_logits_fixed_model,
            batch_size         = batch_size,
            MITGSWRV_ids       = MITGSWRV_ids,
            num_scratchpads    = num_scratchpads,
            label_pad_token_id = label_pad_token_id,
            tokenizer          = self._tokenizer,
        )
        
        # if ITGSWRV_logits_fixed_answer_model is ITGSWRV_logits_fixed_model:
        #     shift_MITGSWRV_log_softmax_fixed_answer_model = shift_MITGSWRV_log_softmax_fixed_model.clone()
        # else:
        shift_MITGSWRV_log_softmax_fixed_answer_model, _, _ = prep_logits_and_MITGSWRV(
            vocab_size         = vocab_size,
            ITGSWRV_logits     = ITGSWRV_logits_fixed_answer_model,
            batch_size         = batch_size,
            MITGSWRV_ids       = MITGSWRV_ids,
            num_scratchpads    = num_scratchpads,
            label_pad_token_id = label_pad_token_id,
            tokenizer          = self._tokenizer,
        )


        #######################################################################
        # Extract the log-probs for the labels for y and z
        #######################################################################
        utils.check_equal(shift_MITGSWRV_log_softmax .shape, (batch_size, num_scratchpads, shift_seq_len))
        utils.check_equal(shift_MITGSWRV_is_pad      .shape, (batch_size, num_scratchpads, shift_seq_len))
        utils.check_equal(shift_y_mask_is_not_pad    .shape, (batch_size, num_scratchpads, shift_seq_len))
        utils.check_equal(shift_z_mask_is_not_pad    .shape, (batch_size, num_scratchpads, shift_seq_len))

        mask_y = shift_y_mask_is_not_pad * shift_MITGSWRV_is_pad.bool().logical_not().long()
        mask_z = shift_z_mask_is_not_pad * shift_MITGSWRV_is_pad.bool().logical_not().long()

        z_log_probs = shift_MITGSWRV_log_softmax * mask_z
        
        with torch.no_grad():
            y_log_probs_fixed = shift_MITGSWRV_log_softmax_fixed_answer_model * mask_y
            z_log_probs_fixed = shift_MITGSWRV_log_softmax_fixed_model        * mask_z
        

        #######################################################################
        # -> Log-likelihoods for y and for z
        # -> Importance Sampling Ratio
        #######################################################################                
        utils.check_equal(z_log_probs       .shape, (batch_size, num_scratchpads, shift_seq_len))
        utils.check_equal(z_log_probs_fixed .shape, (batch_size, num_scratchpads, shift_seq_len))

        y_log_probs_fixed_per_seq = y_log_probs_fixed.detach().sum(dim=-1)
        z_log_probs_per_seq       = z_log_probs               .sum(dim=-1)

        most_helpful_idx = y_log_probs_fixed_per_seq.argmax(dim=-1).detach()

        ###############################################################
        # COMPUTE THE CROSS-ENTROPY LOSS WITH THE MOST HELPFUL SEQUENCE
        ###############################################################
        MITGSWRV_ids           = MITGSWRV_ids          .reshape(batch_size, num_scratchpads, -1)
        ITGSWRV_ids            = ITGSWRV_ids           .reshape(batch_size, num_scratchpads, -1)
        ITGSWRV_attention_mask = ITGSWRV_attention_mask.reshape(batch_size, num_scratchpads, -1)

        #######################################################################
        # Do MLE on the strongest scratchpad
        #######################################################################
        CONSOLE.print_zero_rank(f"\n[bold blue]{self._loss_mode}\n")
        if self._loss_mode == constants.LossModes.STRONGEST_MLE:
            assert False
            most_helpful_log_probs = z_log_probs_per_seq.gather(
                dim=-1, 
                index=most_helpful_idx.unsqueeze(-1),
            ).squeeze(-1)
            
            ref_final_ITGSWRV_input_ids      = _ref_special_gather(tensor=ITGSWRV_ids,            most_helpful_idx=most_helpful_idx, batch_size=batch_size)
            ref_final_ITGSWRV_attention_mask = _ref_special_gather(tensor=ITGSWRV_attention_mask, most_helpful_idx=most_helpful_idx, batch_size=batch_size)
            ref_final_MITGSWRV_ids           = _ref_special_gather(tensor=MITGSWRV_ids,           most_helpful_idx=most_helpful_idx, batch_size=batch_size)

            final_ITGSWRV_input_ids      = _special_gather(tensor=ITGSWRV_ids,            most_helpful_idx=most_helpful_idx, batch_size=batch_size, num_scratchpads=num_scratchpads)
            final_ITGSWRV_attention_mask = _special_gather(tensor=ITGSWRV_attention_mask, most_helpful_idx=most_helpful_idx, batch_size=batch_size, num_scratchpads=num_scratchpads)
            final_MITGSWRV_ids           = _special_gather(tensor=MITGSWRV_ids,           most_helpful_idx=most_helpful_idx, batch_size=batch_size, num_scratchpads=num_scratchpads)

            assert -100 not in MITGSWRV_ids
            assert (ref_final_ITGSWRV_input_ids      == final_ITGSWRV_input_ids).all()
            assert (ref_final_ITGSWRV_attention_mask == final_ITGSWRV_attention_mask).all()
            assert (ref_final_MITGSWRV_ids           == final_MITGSWRV_ids).all()

            del ref_final_ITGSWRV_input_ids
            del ref_final_ITGSWRV_attention_mask
            del ref_final_MITGSWRV_ids
            
            final_MITGSWRV_ids[final_MITGSWRV_ids == self._tokenizer.pad_token_id] = -100

            assert self._tokenizer.pad_token_id not in final_MITGSWRV_ids
            assert (final_ITGSWRV_input_ids[:, 0] != self._tokenizer.pad_token_id).all()

            loss = self._model(
                input_ids      = final_ITGSWRV_input_ids,
                attention_mask = final_ITGSWRV_attention_mask,
                labels         = final_MITGSWRV_ids,
            ).loss

        elif self._loss_mode == constants.LossModes.MARGINAL_KL_W_FIXED:

            """
            Things to explore:
            - With softmax on p_y
            - Without softmax
            - Beta values
            - KL instead of l2
            """

            z_log_probs_seq  = z_log_probs      .sum(dim=-1)  # (batch_size, num_scratchpads)
            # z_log_probs_seq -= z_log_probs_fixed.sum(dim=-1).detach()  # (batch_size, num_scratchpads)
            # import_ratio_w_fixed_z = seq_z_log_probs - seq_z_log_probs_fixed  # (batch_size, num_scratchpads)
            

            MARGINAL_KL_W_FIXED_BETA = 0
            Y_TERM_BETA = 0

            MARGINAL_KL_W_FIXED_REWARD_TEMPERATURE = 3
            
            y_prob_term_seq = (y_log_probs_fixed_per_seq.detach() * MARGINAL_KL_W_FIXED_REWARD_TEMPERATURE).softmax(-1)
            assert y_prob_term_seq.ndim == 2, y_prob_term_seq.shape

            rl_reward_per_seq = (z_log_probs_seq.exp() * y_prob_term_seq)
            rl_reward_per_question = rl_reward_per_seq.sum(-1)  # Marginal likelihood. Give higher probability to sequences that are more likely to be correct
            assert rl_reward_per_question.ndim == 1, rl_reward_per_question.shape
            rl_requard_per_batch = rl_reward_per_question.mean()
            assert rl_requard_per_batch.ndim == 0, rl_requard_per_batch.shape

            ppo_importance_sampling_penalty_seq = (z_log_probs - z_log_probs_fixed.detach()).sum(-1)
            
            assert z_log_probs_fixed.ndim == 3, z_log_probs_fixed.shape
            assert z_log_probs      .ndim == 3, z_log_probs      .shape
            assert ppo_importance_sampling_penalty_seq.ndim == 2, ppo_importance_sampling_penalty_seq.shape

            ppo_importance_sampling_penalty_per_question = ppo_importance_sampling_penalty_seq.mean(-1)
            assert ppo_importance_sampling_penalty_per_question.ndim == 1, ppo_importance_sampling_penalty_per_question.shape
            ppo_importance_sampling_penalty_per_batch = ppo_importance_sampling_penalty_per_question.mean()

            loss = MARGINAL_KL_W_FIXED_BETA * ppo_importance_sampling_penalty_per_batch - Y_TERM_BETA * rl_requard_per_batch.mean(-1)

            assert not z_log_probs_fixed.requires_grad
            assert rl_requard_per_batch.requires_grad
            assert ppo_importance_sampling_penalty_seq.requires_grad

            self.log("rl_reward",                        rl_requard_per_batch.mean().item(),                          batch_size=self._batch_size[mode],  **self._logging_conf)
            self.log("ppo_importance_sampling_penalty",  ppo_importance_sampling_penalty_seq.mean(-1).mean().item(),  batch_size=self._batch_size[mode],  **self._logging_conf)
            self.log("loss",                             loss.item(),                                                 batch_size=self._batch_size[mode],  **self._logging_conf)

            utils.check_equal(loss.ndim, 0)

            y_prob_term = y_prob_term_seq

        elif self._loss_mode == constants.LossModes.PPO:

            """
            Things to explore:
            - With softmax on p_y
            - Without softmax
            - Beta values
            - KL instead of l2
            """
            assert False
            
            experience = ppo.make_experience(
                all_tokens_input_ids      = ITGS_ids,
                all_tokens_attention_mask = ITGS_attention_mask,
                query_tensors             = torch.repeat_interleave(batch["generation_input_ids"].unsqueeze(1), dim=1, repeats=num_scratchpads),
                response_tensors          = GS_ids,
                scores                    = y_log_probs_fixed_per_seq,
                logprobs_generated        = z_log_probs,
                logprobs_generated_fixed  = z_log_probs_fixed,
                init_kl_coef              = 0.2,                  # from https://github.com/CarperAI/trlx/blob/e96c815dc152e230d9ad2ec10f8b84215df671d0/configs/ppo_config.yml#L39
                value_model               = self._value_model,

                batch_size                = batch_size,
                num_scratchpads           = num_scratchpads,
                
                # Arguments for logging
                batch_idx                 = batch_idx,
                logger                    = self.logger,    
                logger_kwargs             = self._logging_conf, 
            )

            loss = ppo.loss(
                model            = self._model,
                value_model      = self._value_model,
                all_logprobs     = experience["logprobs"],
                all_rewards      = experience["rewards"],
                all_values       = experience["values"],
                query_tensors    = experience["query_tensors"],    
                response_tensors = experience["response_tensors"], 
                vf_coef          = 0.2,  # https://github.com/CarperAI/trlx/blob/e96c815dc152e230d9ad2ec10f8b84215df671d0/configs/ppo_config.yml#L46
                cliprange_value  = 0.2,  # https://github.com/CarperAI/trlx/blob/e96c815dc152e230d9ad2ec10f8b84215df671d0/configs/ppo_config.yml#L44
                cliprange        = 0.2,  # https://github.com/CarperAI/trlx/blob/e96c815dc152e230d9ad2ec10f8b84215df671d0/configs/ppo_config.yml#L45
                gamma            = 1,    # https://github.com/CarperAI/trlx/blob/e96c815dc152e230d9ad2ec10f8b84215df671d0/configs/ppo_config.yml#L42
                lambda_          = 0.95, # https://github.com/CarperAI/trlx/blob/e96c815dc152e230d9ad2ec10f8b84215df671d0/configs/ppo_config.yml#L39
                num_scratchpads  = num_scratchpads,
                batch_size       = self._batch_size[mode],
            )

            self.log("loss",                             loss.item(),                             batch_size=self._batch_size[mode],  **self._logging_conf)
            self.log("log_loss",                         loss.log().item(),                       batch_size=self._batch_size[mode],  **self._logging_conf)

            y_prob_term = y_log_probs_fixed_per_seq.exp().detach()

            utils.check_equal(loss.ndim, 0)
        else:
            raise ValueError(f"Unknown loss mode {self._loss_mode}")

        with torch.no_grad():
            y_part_prob = y_prob_term
            z_part_prob = z_log_probs_per_seq.exp()

        demo_input_idx = random.randint(0, batch_size - 1)
        demo_input_sp =  random.randint(0, num_scratchpads - 1)

        if SHOW_SMALL_GENERATION_TABLE:
            train_utils.show_small_generation_table(
                scores         = scores,
                tokenizer      = self._tokenizer,
                demo_input_sp  = demo_input_sp,
                demo_input_idx = demo_input_idx,
                inputs_outputs = input_ids_then_scratchpads.view(batch_size, num_scratchpads, -1),
            )

        if SHOW_MULTI_SCRATCHPAD_TABLE:                    
            show_multi_scratchpad_table(
                rl_loss         = rl_reward_per_seq,
                y_prob          = y_part_prob,
                z_prob          = z_part_prob,
                labels          = batch["labels"], 
                tokenizer       = self._tokenizer,
                shift_MITGSWRV  = shift_MITGSWRV,
                demo_input_idx  = demo_input_idx,
                num_scratchpads = num_scratchpads,
            )

        if SHOW_SCRATCHPAD_PADDING_TABLE :                    
            train_utils.show_scratchpad_padding_table(
                mask_x=mask_x,
                mask_y=mask_y,
                batch_size=batch_size,
                shift_MITGSWRV=shift_MITGSWRV,
                tokenizer=self._tokenizer,
                demo_input_sp=demo_input_sp,
                demo_input_idx=demo_input_idx, 
                num_scratchpads=num_scratchpads,
                shift_MITGSWRV_is_pad=shift_MITGSWRV_is_pad,
                padded_final_input_ids=final_ITGSWRV_input_ids,
                shift_prob=shift_log_probs.sum(dim=-1).exp() if 
                    "shift_log_probs" in locals() else shift_probs.prod(dim=-1),
            )

        utils.rich_print_zero_rank(f"[{self.trainer.global_rank}] [bold]Loss:[/] {loss}")
        
        return loss


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
        train_utils.shared_predict_step(self, batch, batch_idx)

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

        self._scheduler = self._scheduler_function[self._scheduler_type](
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
            self._datasets[constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING],
            collate_fn=self._training_collator,
            batch_size=self._batch_size[constants.PipelineModes.MARGINAL_LIKELIHOOD_TRAINING],
            num_workers=self._dataloader_num_workers,
            shuffle=self._shuffle_train,
        )


    def val_dataloader(self):
        mode: Final[str] = constants.PipelineModes.VALIDATION
        return torch.utils.data.DataLoader(
            self._datasets[mode],
            collate_fn=train_utils.ValitationCollator(self._tokenizer, self._lm_masking_mode),
            batch_size=self._batch_size[mode],
            num_workers=self._dataloader_num_workers,
            shuffle=self._shuffle_val,
        )

    
    def predict_dataloader(self):
        return self.val_dataloader()


    def on_save_checkpoint(self, ckpt):
        return 


def prep_logits_and_MITGSWRV(
    *,
    vocab_size:         int,
    batch_size:         int, 
    num_scratchpads:    int, 
    label_pad_token_id: int,
    ITGSWRV_logits:     torch.Tensor, 
    MITGSWRV_ids:       torch.Tensor, 
    tokenizer:          transformers.GPT2Tokenizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    assert -100 not in MITGSWRV_ids
    assert num_scratchpads != -100, num_scratchpads
    assert label_pad_token_id != -100, label_pad_token_id

    shift_MITGSWRV = MITGSWRV_ids[..., 1:].contiguous()
    flat_MITGSWRV = shift_MITGSWRV.view(-1).unsqueeze(-1).contiguous()
    bsz_times_num_beams = ITGSWRV_logits.shape[0]

    seq_len = shift_MITGSWRV.shape[1]
    shift_MITGSWRV = shift_MITGSWRV.reshape(batch_size, num_scratchpads, seq_len)
    shift_MITGSWRV_is_pad = (shift_MITGSWRV == label_pad_token_id).to(MITGSWRV_ids.dtype)

    # Shift the logits, prep them for the gather, do the gather
    shift_ITGSWRV_logits = ITGSWRV_logits[..., :-1, :].contiguous()
    flat_shift_ITGSWRV_logits = shift_ITGSWRV_logits.view(-1, shift_ITGSWRV_logits.shape[-1]).contiguous()
    
    flat_shift_ITGSWRV_log_softmax   = flat_shift_ITGSWRV_logits.log_softmax(dim=-1)
    flat_shift_MITGSWRV_log_softmax  = flat_shift_ITGSWRV_log_softmax.gather(dim=1, index=flat_MITGSWRV)
    MITGSWRV_log_softmax = flat_shift_MITGSWRV_log_softmax.reshape(batch_size, num_scratchpads, seq_len)
        
    utils.check_equal(MITGSWRV_log_softmax.shape[-1],         seq_len)
    utils.check_equal(shift_MITGSWRV.shape,                  (batch_size,  num_scratchpads,  seq_len))
    utils.check_equal(shift_MITGSWRV_is_pad.shape,           (batch_size,  num_scratchpads,  seq_len))
    utils.check_equal(flat_shift_MITGSWRV_log_softmax.shape, (batch_size * num_scratchpads * seq_len,     1))
    utils.check_equal(flat_shift_ITGSWRV_logits.shape,       (batch_size * num_scratchpads * seq_len,     vocab_size))
    utils.check_equal(bsz_times_num_beams,                    batch_size * num_scratchpads)
    utils.check_equal(ITGSWRV_logits.shape,                  (batch_size * num_scratchpads,  seq_len + 1, vocab_size))

    return MITGSWRV_log_softmax, shift_MITGSWRV_is_pad, shift_MITGSWRV


@dataclasses.dataclass
class SamplesMarginal:
    y_mask_is_not_pad      : torch.Tensor
    z_mask_is_not_pad      : torch.Tensor
    MITGSWRV_ids           : torch.Tensor   # MITGSWRV: Masked Inputs Then Generated Samples With Reference Values
    ITGSWRV_ids            : torch.Tensor   # ITGSWRV: Inputs Then Generated Samples With Reference Values
    ITGSWRV_attention_mask : torch.Tensor   # ITGSWRV: Inputs Then Generated Samples With Reference Values
    ITGS_ids               : torch.Tensor   # ITG: Inputs Then Generated samples
    ITGS_attention_mask    : torch.Tensor   # ITG: Inputs Then Generated samples
    GS_ids                 : torch.Tensor   # GS: Generated Samples
    GS_attention_mask      : torch.Tensor   # GS: Generated Samples



def prep_samples_marginal(
    *, 
    inputs_outputs:      torch.Tensor, 
    batch:               dict[str, torch.Tensor], 
    generation_kwargs:   dict[str, Any], 
    disable_timing:      bool, 
    eos_token_id:        int, 
    cls_token_id:        int, 
    label_pad_token_id:  int, 
    inputs_pad_token_id: int,
    batch_size:          int, 
    num_scratchpads:     int,   
) -> SamplesMarginal:
    
    utils.check_equal(label_pad_token_id, inputs_pad_token_id)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Unpad and interleave.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # utils.repeat_interleave is just a generator, doesn't not unnecessarily build a tensor.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with utils.cuda_timeit("[bold]bin_refine.py::PSI-A:[/] Unpad, Repeat-interleave", disable=disable_timing):
        with utils.cuda_timeit("[bold]bin_refine.py::PSI-A:[/] unpad then repeat-interleave", disable=disable_timing):
            unpadded_inputs_outputs = train_utils.remove_padding(
                inputs_outputs, 
                inputs_outputs != inputs_pad_token_id,
            )
            unpadded_values = batch["value"]
            unpadded_inputs = train_utils.remove_padding(
                batch["generation_input_ids"], 
                batch["generation_attention_mask"] == 1,
            )

            unpadded_repeated_inputs = utils.repeat_interleave(
                unpadded_inputs, 
                generation_kwargs["num_return_sequences"],
            )
            # Reproduce the structure of the multiple beams per input tensor
            unpadded_repeated_values = utils.repeat_interleave(
                unpadded_values, 
                generation_kwargs["num_return_sequences"],
            ) 

    final_ITGSWRV: list[list[int]] = []
    final_MITGSWRV: list[list[int]] = []
    y_mask: list[list[int]] = []
    z_mask: list[list[int]] = []
    ITG_ids = []
    GS_ids = []

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MITGSWRV: Masked Input Then Generated Scratchpad With Rerefence Values
    # ITGSWRV:  Input Then Generated Scratchpad With Rerefence Values
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with utils.cuda_timeit("[bold]bin_refine.py::PSI-B:[/] Loop", disable=disable_timing):
        for inputs, io, value  in zip(
            unpadded_repeated_inputs,
            unpadded_inputs_outputs ,
            unpadded_repeated_values,
        ):
            # TODO: Future optimization: don't convert to a list. Probably faster.
            value = value.tolist()
            io_list = io.tolist()
            del io

            if not io_list[-1] == cls_token_id:
                io_list.append(cls_token_id)
            
            scratchpad = io_list[len(inputs):]

            final_ITGSWRV_entry   = io_list                                                    + value + [eos_token_id]
            final_MITGSWRV_entry  = len(inputs) * [label_pad_token_id] + scratchpad            + value + [eos_token_id]
            y_mask_entry          = len(inputs) * [0]                  + len(scratchpad) * [0] + (len(value) + 1) * [1]
            z_mask_entry          = len(inputs) * [0]                  + len(scratchpad) * [1] + (len(value) + 1) * [0]

            final_ITGSWRV.append(final_ITGSWRV_entry)
            final_MITGSWRV.append(final_MITGSWRV_entry)
            y_mask.append         (y_mask_entry)
            z_mask.append         (z_mask_entry)

            ITG_ids.append(io_list)
            GS_ids.append(scratchpad)

            utils.check_equal(len(final_MITGSWRV_entry), len(final_ITGSWRV_entry))
            utils.check_equal(len(y_mask_entry),         len(final_ITGSWRV_entry))
            utils.check_equal(len(z_mask_entry),         len(final_ITGSWRV_entry))
            
    with utils.cuda_timeit("[bold]bin_refine.py::PSI-C:[/] Pad", disable=disable_timing):
        utils.check_equal(len(final_ITGSWRV), len(final_MITGSWRV))
        padded_final_ITGSWRV = train_utils.pad(final_ITGSWRV, inputs_pad_token_id, "right").to(inputs_outputs.device)
        padded_final_MITGSWRV = train_utils.pad(final_MITGSWRV, label_pad_token_id , "right").to(inputs_outputs.device)
        padded_final_attention_mask = train_utils.generate_mask(final_ITGSWRV, "right").to(inputs_outputs.device)
        padded_y_mask_is_not_pad = train_utils.pad(y_mask, 0, "right").to(inputs_outputs.device, dtype=padded_final_attention_mask.dtype)
        padded_z_mask_is_not_pad = train_utils.pad(z_mask, 0, "right").to(inputs_outputs.device, dtype=padded_final_attention_mask.dtype)
        padded_y_mask_is_not_pad = padded_y_mask_is_not_pad.reshape(batch_size, num_scratchpads, -1)
        padded_z_mask_is_not_pad = padded_z_mask_is_not_pad.reshape(batch_size, num_scratchpads, -1)
        
        padded_ITG_ids = train_utils.pad(ITG_ids, inputs_pad_token_id, "right").to(inputs_outputs.device)
        padded_ITG_attention_mask = train_utils.generate_mask(ITG_ids, "right").to(inputs_outputs.device)

        padded_GS_ids = train_utils.pad(GS_ids, inputs_pad_token_id, "right").to(inputs_outputs.device)
        padded_GS_attention_mask = train_utils.generate_mask(GS_ids, "right").to(inputs_outputs.device)

        utils.check_equal(padded_final_MITGSWRV       .shape[-1], padded_final_ITGSWRV.shape[-1])
        utils.check_equal(padded_y_mask_is_not_pad    .shape[-1], padded_final_ITGSWRV.shape[-1])
        utils.check_equal(padded_z_mask_is_not_pad    .shape[-1], padded_final_ITGSWRV.shape[-1])
        utils.check_equal(padded_final_attention_mask .shape[-1], padded_final_ITGSWRV.shape[-1])

    with utils.cuda_timeit("bin_refine.py::Score", disable=disable_timing):
        utils.check_equal(padded_final_attention_mask  .shape,       padded_final_MITGSWRV.shape)
        utils.check_equal(padded_final_ITGSWRV         .shape,       padded_final_attention_mask.shape)
        utils.check_equal(padded_y_mask_is_not_pad     .shape[:-1], (batch_size, num_scratchpads,))

    return SamplesMarginal(
        ITGSWRV_attention_mask = padded_final_attention_mask, 
        ITGSWRV_ids            = padded_final_ITGSWRV, 
        ITGS_ids                = padded_ITG_ids,
        ITGS_attention_mask     = padded_ITG_attention_mask, 
        GS_ids                 = padded_GS_ids,
        GS_attention_mask      = padded_GS_attention_mask,
        MITGSWRV_ids           = padded_final_MITGSWRV,
        y_mask_is_not_pad      = padded_y_mask_is_not_pad, 
        z_mask_is_not_pad      = padded_z_mask_is_not_pad,
    )


@dataclasses.dataclass
class MarginalLikelihoodTrainingCollator:
    _tokenizer: transformers.PreTrainedTokenizer
    _lm_masking_mode: str

    def __call__(self, raw_examples):
        """
        - We have the questions, we have the answers. Nothing else.

        Input ids: [question, chainer]
        Labels: [answer]

        loss: likelihoodOf[question, chainer, Generate(question), answer]

        """

        # We can only prepare the inputs for generation. 
        # These need to be padded to the left.
        examples = train_utils.prep_mle_train_and_valid(
            examples=raw_examples, 
            eos_token_id=self._tokenizer.eos_token_id,
            scratchpad_eos_token_id=self._tokenizer.cls_token_id, 
            pad_token_id=self._tokenizer.pad_token_id,
            lm_masking_mode=self._lm_masking_mode,
        )
        
        examples["generation_attention_mask"] = train_utils.generate_mask(examples["input"], "left")
        examples["generation_input_ids"] = train_utils.pad(examples["input"], self._tokenizer.pad_token_id, "left")

        return examples



def _show_multi_scratchpad_table_format_text(ids, tokenizer) -> str:
    return tokenizer.decode(ids
                ).replace("<|pad|>", "").replace("<|endoftext|>", "<eos>"
                ).replace("<|cls|>", "<cls>") 


def show_multi_scratchpad_table(
    *,
    rl_loss,
    labels,
    y_prob, 
    z_prob, 
    tokenizer, 
    shift_MITGSWRV,
    demo_input_idx,
    num_scratchpads, 
):

    rl_loss = rl_loss.clone().detach()    
    y_prob = y_prob.clone().detach()
    z_prob = z_prob.clone().detach()

    table_ = table.Table("Text", "rl_loss", "y score", "z score", "y rank", "z rank")

    label_text = _show_multi_scratchpad_table_format_text([x for x in labels[demo_input_idx] if x > 0], tokenizer)
    table_.add_row(f"[bold magenta]{label_text}[/]", "", "", "", "", end_section=True)

    sort_y = torch.tensor(sorted(range(num_scratchpads), key=lambda i: y_prob[demo_input_idx, i], reverse=True))
    
    y_prob_entry = y_prob[demo_input_idx, sort_y]
    z_prob_entry = z_prob[demo_input_idx, sort_y]
    rl_loss_entry = rl_loss[demo_input_idx, sort_y]

    argsort = sorted(range(num_scratchpads), key=lambda i: z_prob_entry[i], reverse=True)
    ranks_z = {}
    for i, pos in enumerate(argsort):
        ranks_z[pos] = i

    for rank_y in range(num_scratchpads):
        maybe_color = ""
        if rank_y == 0 and ranks_z[rank_y] == 0:
            maybe_color = "[green bold]"
        elif rank_y == 0:
            maybe_color = "[blue bold]"
        elif ranks_z[rank_y] == 0:
            maybe_color = "[yellow bold]"

        maybe_close = ""        
        if maybe_color:
            maybe_close = "[/]"

        y_prob_color_coeff = int(y_prob_entry[rank_y].item() * 255)
        y_prob_color = f"white on #{y_prob_color_coeff:02x}{y_prob_color_coeff:02x}{y_prob_color_coeff:02x}"

        z_prob_color_coeff = int(z_prob_entry[rank_y].item() * 255)
        z_prob_color = f"white on #{z_prob_color_coeff:02x}{z_prob_color_coeff:02x}{z_prob_color_coeff:02x}"

        generated_text = _show_multi_scratchpad_table_format_text(shift_MITGSWRV[demo_input_idx, rank_y], tokenizer) 

        output_str = []
        for lab, gen in itertools.zip_longest(label_text, generated_text, fillvalue=" "):
            if lab == gen:
                output_str.append(gen)
            else:
                output_str.append(f"[red]{gen}[/red]")
        diff_colored = "".join(output_str)
        
        if generated_text == label_text:
            diff_colored = f"[white on green]{generated_text}[/white on green]"


        table_.add_row(
            maybe_color + diff_colored + maybe_close,
            f"{rl_loss_entry[rank_y].item():0.2}",
            f"[{y_prob_color}]{y_prob_entry[rank_y].item():.2}", 
            f"[{z_prob_color}]{z_prob_entry[rank_y].item():.2}",
            str(rank_y),
            str(ranks_z[rank_y])
        )
        
    CONSOLE.print_zero_rank(table_)
