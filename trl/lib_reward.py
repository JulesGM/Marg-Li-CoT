import collections
import logging
import os
import random
import re
import typing

import accelerate
import general_utils
import more_itertools
import numpy as np
import torch
import transformers

import lib_data
import lib_metric
import lib_bisect_tokens

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.environ["LOCAL_RANK"])


def global_do_checks(model):
    local_rank = LOCAL_RANK
    assert model.device.index == local_rank, (
        f"{model.device.index = }, {local_rank = }"
    )
    assert (torch.cuda.current_device() == local_rank), (
        torch.cuda.current_device(), local_rank)
    assert torch.distributed.get_backend() == "nccl", (
        torch.distributed.get_backend())
    torch.distributed.barrier()


def info(message):
    general_utils.parallel_log(LOGGER, logging.INFO, message)


def remove_special_token_ids(
    input_ids: list[int], tokenizer: transformers.PreTrainedTokenizer
):
    """
    Remove special tokens from the input_ids
    """
    all_special_ids    = set(tokenizer.all_special_ids)
    filtered_input_ids = [x for x in input_ids if x not in all_special_ids]

    assert len(filtered_input_ids) == len(input_ids) - 1, (
                f"\n"
                f"{tokenizer.decode(input_ids)          = },\n"
                f"{tokenizer.decode(filtered_input_ids) = }.\n"
                f"{len(input_ids)                       = },\n"
                f"{len(filtered_input_ids)              = },\n"
            )

    return filtered_input_ids


def _maybe_frozen_head(model, input_dict, use_frozen_head):
    if use_frozen_head:
        with torch.no_grad():
            return model(**input_dict)
    return model(**input_dict)

def _maybe_autocast(model_or_fn, dtype):
    if dtype is None:
        return model_or_fn
    return torch.cuda.amp.autocast(dtype=dtype)(model_or_fn)


def clone_hf_model(
        hf_model: transformers.PreTrainedModel
) -> transformers.PreTrainedModel:
    
    assert isinstance(hf_model, transformers.PreTrainedModel), type(hf_model)
    copy = type(hf_model)(hf_model.config)
    copy.load_state_dict(hf_model.state_dict())
    return copy


class ScratchpadRewardFn:
    def __init__(
        self, *, 
        ref_model: typing.Union[str, transformers.PreTrainedModel],
        tokenizer: transformers.PreTrainedTokenizerBase, 
        uses_peft: bool,
        metric_fn,
    ):
        super().__init__()
        
        #----------------------------------------------------------------
        # Build Models
        #----------------------------------------------------------------
        reward_model = ref_model

        #----------------------------------------------------------------
        # Set Attributes
        #----------------------------------------------------------------
        self._show_answer_replacement = False
        self._reward_tokenizer        = tokenizer
        self._no_answer_rate          = lib_data.MovingAverage(10000)
        self._model                   = reward_model.eval()
        self._conv_to_num             = lib_data.ConvToNum()
        self._metric                  = metric_fn
        self._uses_peft               = uses_peft

    
    def __call__(self, prompts, samples, outputs, ref_answers):
        # The idea is to:
        # 1. Extract the associated answers & tokenize the answers
        # 2. Create a mask for the answers
        # 3. Tokenize the samples
        # 4. Concate the samples & answers
        # 5. Run the reward model on the concatenated samples & answers
        # 6. Extract the logp for the answers
        # 7. Return the logp for the answers

        #######################################################################
        # - Sanity checks 
        # - Varia prep
        #######################################################################
        output = np.mean(self._metric(prompts, samples, outputs)["em_accuracy"])
        LOGGER.info(f"[red bold]REWARD - INSTANT PER BATCH METRIC ACC: {output = :0.2%}")

        # Get the answers.
        question_tok = self._tokenizer(prompts, padding=True, return_tensors="pt")

        timer_flags = dict(
            disable         = True,
            cuda_sync       = False,
            accelerate_sync = False,
            accelerator     = None,
            log_level       = logging.INFO,
            logger          = LOGGER,
        )
        
        #######################################################################
        # - Replace the number words by digits.
        # - Find the answer tokens in the generated output.
        # - Remove it.
        # - Put the ref answer instead.
        # - Create a mask over the not-answer for the perplexity.
        #######################################################################
        # Replace the number words by digits.
        scratchpads = []
        timer = general_utils.ctx_timeit
        with timer("Replacing the number words by digits", **timer_flags):
            for i, (output, ref_answer) in enumerate(
                more_itertools.zip_equal(outputs, ref_answers)
            ):
                scratchpads.append(self.replace_answer(
                    original_generation = self._conv_to_num(output), 
                    ref_answer          = ref_answer,
                )[0])

        # Find the answer tokens.
        with timer("Extracting the answer tokens", **timer_flags):
            tok_outputs, start_end_outputs, str_matches_outputs = (
                lib_bisect_tokens.extract_match_tokens(
                    regexes          = [
                        re.escape(ref_answer) 
                        for ref_answer in ref_answers
                    ],
                    tokenizer        = self._reward_tokenizer, 
                    strings          = scratchpads,
                    tokenizer_kwargs = dict(
                        return_tensors = "pt", 
                        padding        = True,
                    ),
                    verbose=False,
                )
            )
        
        assert (torch.cuda.current_device() == torch.distributed.get_rank()), (
            torch.cuda.current_device(), torch.distributed.get_rank())
        assert len(start_end_outputs) == len(scratchpads), (
            f"{len(start_end_outputs) = }, {len(scratchpads) = }")
        
        # > Replace the answer tokens by the ref answer.
        # > Create the masks.
        masks = []
        seq_len = tok_outputs["input_ids"].shape[1]
        with timer("Creating the masks", **timer_flags):
            for matches, scratchpad_ids, ref_answer in zip(
                start_end_outputs, tok_outputs["input_ids"], ref_answers
            ):
                start, end = matches[-1]
                mask = [0] * start + [1] * (end + 1 - start) + [0] * (seq_len - end - 1)

                if self._show_answer_replacement:
                    scratchpad_before = self._reward_tokenizer.decode(
                        scratchpad_ids, skip_special_tokens=True)
                    scratchpad_after  = " ".join([
                        self._reward_tokenizer.decode(token) 
                        if m == 0 else " <<REF_ANSWER>>" 
                        for token, m in zip(scratchpad_ids, mask)
                    ]).replace("<pad>", "")

                    LOGGER.info(
                        f"[bold blue]ref answer:[white]        {ref_answer}\n"
                        f"[bold blue]start, end:[white]        {start}, {end}\n"
                        f"[bold blue]Scratchpad before:[white] {scratchpad_before}\n"
                        f"[bold blue]Scratchpad after:[white]  {scratchpad_after}\n"
                    )

                masks.append(torch.tensor(mask).to(tok_outputs["input_ids"].device))
                assert len(mask) == seq_len, f"{len(mask) = }, {seq_len = }"

        ###########################################################################
        # 2. Compute the logp for the answers
        ###########################################################################
        if self._reward_model.config.is_encoder_decoder:
            assert self._model.device.type == "cuda", (
                f"{self._model.device.type = }"
            )
            
            # TODO: Maybe we don't have to recompute the logits. 
            # We should get them for generation and for training. 
            with timer("Moving things to GPU", **timer_flags):
                input_dict = dict(
                    input_ids              = question_tok["input_ids"     ].to(self._inputs_device),
                    attention_mask         = question_tok["attention_mask"].to(self._inputs_device),
                    decoder_input_ids      = tok_outputs ["input_ids"     ].to(self._inputs_device),
                    decoder_attention_mask = tok_outputs ["attention_mask"].to(self._inputs_device),
                )

            with timer(
                f"> Computing the logits with the ref model for the reward." +
                f"{question_tok['input_ids'].shape = }"
                , **timer_flags):
                
                with torch.no_grad():                    
                    if self._uses_peft:
                        with self._model.disable_adapter():
                            logits = self._model(**input_dict).logits
                    else:
                        logits = self._model(**input_dict).logits

        else:
            # TODO(@julesgm): Fix this
            assert False
            logits = self._reward_model(
                full_seq["input_ids"], attention_mask=full_seq["attention_mask"]
                ).logits

        with timer("Computing the rest of the reward", **timer_flags):

            reward_model_outputs_all = logits.softmax(-1)
            idx = tok_outputs["input_ids"].to(reward_model_outputs_all.device)
            reward_model_outputs_scratchpad = torch.gather(
                reward_model_outputs_all, -1, idx.unsqueeze(-1)
            ).squeeze(-1)

            ###########################################################################
            # 3. Only keep the logp for the actual values used
            ###########################################################################

            masks        = torch.stack(masks).to(reward_model_outputs_scratchpad.device)
            probs        = reward_model_outputs_scratchpad.clone()
            probs[masks == 0] = 1
            logp         = probs.log()
            logp_per_seq = logp.sum(-1)
            final_output = logp_per_seq.detach()

        return final_output, reward_model_outputs_scratchpad

    def replace_answer(self, *, original_generation: str, ref_answer: str) -> tuple[str, int, int]:
        answer = self._conv_to_num.extract_answer(original_generation)
        
        # If the answer is None, then we just add the reference answer at the end.
        if answer is None:
            self._no_answer_rate.update(1)
            ratio, (sum_, size) = self._no_answer_rate.get()
            LOGGER.info(f"[red bold]No answer: {ratio:.1%} ({sum_}/{size}) ")
            new_scratchpad = original_generation.strip() + " The answer is: "
            start_pos      = len(new_scratchpad)
            end_pos        = start_pos + len(ref_answer)
            final          = new_scratchpad + ref_answer + "."

            return final, start_pos, end_pos

        self._no_answer_rate.update(0)
        mode = "in_place"

        if mode == "in_place":
            # If the answer is not None, then we replace the answer with the reference answer.
            start = original_generation[:answer.start()]
            end   = original_generation[answer.end():]
        elif mode == "remove_end":
            start = original_generation[:answer.start()]
            end   = "."
        else:
            raise ValueError(f"{mode = }")

        start_pos = len(start)
        end_pos   = len(start) + len(ref_answer)
        final     = start + ref_answer + end

        # LOGGER.info(
        #     "\n"
        #     f"[bold blue]original:[/]           {original_generation}\n"
        #     f"[bold blue]ref answer:[/]         {ref_answer}\n"
        #     f"[bold blue]start:[/]              {start}\n"
        #     f"[bold blue]end:[/]                {end}\n"
        #     f"[bold blue]final:[/]              {final}\n"
        # )

        return final, start_pos, end_pos