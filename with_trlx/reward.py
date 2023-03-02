import logging

import random
import re
import os
import typing

import accelerate
import more_itertools
import numpy as np
import torch
import transformers

import bisect_tokens
import lib_data
import metric

import general_utils

LOGGER = logging.getLogger(__name__)


def info(message):
    general_utils.parallel_log(LOGGER, logging.INFO, message)


def remove_special_token_ids(
    input_ids: list[int], tokenizer: transformers.PreTrainedTokenizer
):
    """
    Remove special tokens from the input_ids
    """
    all_special_ids = set(tokenizer.all_special_ids)
    filtered_input_ids = [x for x in input_ids if x not in all_special_ids]

    assert len(filtered_input_ids) == len(input_ids) - 1, (
                f"\n"
                f"{tokenizer.decode(input_ids)          = },\n"
                f"{tokenizer.decode(filtered_input_ids) = }.\n"
                f"{len(input_ids)                       = },\n"
                f"{len(filtered_input_ids)              = },\n"
            )

    return filtered_input_ids


def clone_hf_model(hf_model: transformers.PreTrainedModel) -> transformers.PreTrainedModel:
    assert isinstance(hf_model, transformers.PreTrainedModel), type(hf_model)
    copy = type(hf_model)(hf_model.config)
    copy.load_state_dict(hf_model.state_dict())
    return copy


class ScratchpadRewardFn(torch.nn.Module):
    def __init__(
        self, *, 
        reward_model_hf_name_or_path: str,
        reward_tokenizer: transformers.PreTrainedTokenizerBase, 
        ds_train_obj: lib_data.BaseTRLXExtraInfoDataset,
        batch_size: int,
    ):
        super().__init__()

        reward_model           = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            reward_model_hf_name_or_path)
        self._metric           = metric.ScratchpadAnswerAccuracy(
            extra_info_engine=ds_train_obj.get_extra_info)
        self._conv_to_num      = lib_data.ConvToNum()
        self._ds_train_obj     = ds_train_obj
        self._no_answer_rate   = lib_data.MovingAverage(10000)
        self._reward_tokenizer = reward_tokenizer
        self._reward_model     = reward_model.eval()
        
        # We put the cloned model in a list so that it's not 
        # automatically included in the parameters of this module, and distributed
        # by the accelerator.
        self._non_distributed_reward_model = [clone_hf_model(reward_model).eval()]
        self._non_distributed_reward_model[0].parallelize()

        for p in self._reward_model.parameters():
            p.requires_grad = False

        assert all(isinstance(x, torch.nn.parameter.Parameter) for x in self.parameters()), (
            f"{self.parameters() = }"
        )
        assert not any(x.requires_grad for x in self.parameters()), (
            f"{list(self.parameters()) = }"
        )

    
    def __call__(self, prompts, samples, outputs, is_distributed):
        # The idea is to:
        # 1. Extract the associated answers & tokenize the answers
        # 2. Create a mask for the answers
        # 3. Tokenize the samples
        # 4. Concate the samples & answers
        # 5. Run the reward model on the concatenated samples & answers
        # 6. Extract the logp for the answers
        # 7. Return the logp for the answers

        output = self._metric(prompts, samples, outputs)["em_accuracy"][1]
        LOGGER.info(f"[red bold]METRIC ACC: {output = :0.2%}")

        assert isinstance(prompts   , list), f"{type(prompts   ).mro() = }"
        assert isinstance(prompts[0], str ), f"{type(prompts[0]).mro() = }"
        assert isinstance(outputs   , list), f"{type(outputs   ).mro() = }"
        assert isinstance(outputs[0], str ), f"{type(outputs[0]).mro() = }"

        # TODO: this is temporary
        model = self._reward_model if is_distributed else self._non_distributed_reward_model[0]

        questions = prompts
        del prompts

        extra_info = self._ds_train_obj.get_extra_info(questions)
        ref_answers = extra_info["answer"]

        question_tok = self._reward_tokenizer(
            questions, padding=True, return_tensors="pt"
        )
        
        #######################################################################
        # We want to keep the scratchpad and remove the generated answer,
        # then add the reference answer.
        #######################################################################

        scratchpads = []
        
        # Replace the answer by the reference answer in the generated output.

        timer_flags = dict(
            accelerate_sync=False,
            accelerator=None,
            cuda_sync=True,
            disable=True,
            logger=LOGGER,
            log_level=logging.INFO,
        )
        timer = general_utils.ctx_timeit
        
        for output, ref_answer in zip(outputs, ref_answers):
            scratchpads.append(self.replace_answer(
                original_generation=self._conv_to_num(output), ref_answer=ref_answer
            )[0])

        # Find where the ref answer is in the tokenized scratchpads.
        with timer("Extracting the answer tokens", **timer_flags):
            tok_outputs, start_end_outputs, str_matches_outputs = bisect_tokens.extract_match_tokens(
                regexes=[re.escape(ref_answer) for ref_answer in ref_answers], 
                tokenizer=self._reward_tokenizer, 
                strings=scratchpads,
                tokenizer_kwargs=dict(
                    return_tensors="pt", padding=True,
                ),
                verbose=False,
            )

        # Create masks over the answer tokens for the answer perplexity.
        masks = []
        assert len(start_end_outputs) == len(scratchpads), (
            f"{len(start_end_outputs) = }, {len(scratchpads) = }"
        )

        seq_len = tok_outputs["input_ids"].shape[1]
        with timer("Creating the masks", **timer_flags):
            for matches, scratchpad_ids, ref_answer in zip(
                start_end_outputs, tok_outputs["input_ids"], ref_answers
            ):
                start, end = matches[-1]
                mask = [0] * start + [1] * (end + 1 - start) + [0] * (seq_len - end - 1)

                # scratchpad_after = " ".join([
                #     self._reward_tokenizer.decode(token) if m == 0 else " <<REF_ANSWER>>" 
                #     for token, m in zip(scratchpad_ids, mask)
                # ]).replace("<pad>", "")
                # LOGGER.info(
                #     f"[bold blue]ref answer:[white]        {ref_answer}\n"
                #     f"[bold blue]start, end:[white]        {start}, {end}\n"
                #     f"[bold blue]Scratchpad before:[white] {self._reward_tokenizer.decode(scratchpad_ids, skip_special_tokens=True)}\n"
                #     f"[bold blue]Scratchpad after:[white]  {scratchpad_after}\n"
                # )
                masks.append(torch.tensor(mask).to(tok_outputs["input_ids"].device))
                assert len(mask) == seq_len, f"{len(mask) = }, {seq_len = }"

        ###########################################################################
        # 2. Compute the logp for the answers
        ###########################################################################
        with timer("Computing the logits", **timer_flags):
            if self._reward_model.config.is_encoder_decoder:
                assert model.device.type == "cuda", (
                    f"{model.device.type = }"
                )
                LOGGER.info(f"({is_distributed = }):\n> Starting {question_tok['input_ids'].shape = }")
                with torch.no_grad():
                    logits = model(
                        input_ids              = question_tok["input_ids"     ].to(model.device),
                        attention_mask         = question_tok["attention_mask"].to(model.device),
                        decoder_input_ids      = tok_outputs ["input_ids"     ].to(model.device),
                        decoder_attention_mask = tok_outputs ["attention_mask"].to(model.device),
                    ).logits
                LOGGER.info(f"({is_distributed = }):\n> Done with {question_tok['input_ids'].shape = }")

            else:
                assert False
                logits = self._reward_model(
                    full_seq["input_ids"], attention_mask=full_seq["attention_mask"]
                ).logits


        with timer("Computing the softmax", **timer_flags):
            reward_model_outputs_all = logits.softmax(-1)
            idx = tok_outputs["input_ids"].to(reward_model_outputs_all.device)
            reward_model_outputs_scratchpad = torch.gather(reward_model_outputs_all, -1, idx.unsqueeze(-1)).squeeze(-1)

            ###########################################################################
            # 3. Only keep the logp for the actual values used
            ###########################################################################
            masks = torch.stack(masks).to(reward_model_outputs_scratchpad.device)
            logp = masks * reward_model_outputs_scratchpad
            logp[masks == 0] = 1
            logp = logp.log()
            logp_per_seq = logp.sum(-1)
            final_output = logp_per_seq.detach()
        return final_output

    def replace_answer(self, *, original_generation: str, ref_answer: str) -> tuple[str, int, int]:
        answer = self._conv_to_num.extract_answer(original_generation)
        
        # If the answer is None, then we just add the reference answer at the end.
        if answer is None:
            self._no_answer_rate.update(1)
            ratio, (sum_, size) = self._no_answer_rate.get()
            LOGGER.info(f"[red bold]No answer: {ratio:.1%} ({sum_}/{size}) ")

            new_scratchpad = original_generation.strip() + " The answer is: "
            start_pos = len(new_scratchpad)
            end_pos = start_pos + len(ref_answer)
            final = new_scratchpad + ref_answer + "."
            return final, start_pos, end_pos

        self._no_answer_rate.update(0)
        mode = "in_place"

        if mode == "in_place":
            # If the answer is not None, then we replace the answer with the reference answer.
            start = original_generation[:answer.start()]
            end = original_generation[answer.end():]
        elif mode == "remove_end":
            start = original_generation[:answer.start()]
            end = "."
        else:
            raise ValueError(f"{mode = }")

        start_pos = len(start)
        end_pos = len(start) + len(ref_answer)
        final = start + ref_answer + end

        # LOGGER.info(
        #     "\n"
        #     f"[bold blue]original:[/]           {original_generation}\n"
        #     f"[bold blue]ref answer:[/]         {ref_answer}\n"
        #     f"[bold blue]start:[/]              {start}\n"
        #     f"[bold blue]end:[/]                {end}\n"
        #     f"[bold blue]final:[/]              {final}\n"
        # )

        return final, start_pos, end_pos