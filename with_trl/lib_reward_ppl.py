import logging
import os
import re
import typing

import general_utils
import more_itertools
import numpy as np
import peft
import rich
import rich.console
import torch
import torch.cuda.amp
import torch.distributed
import transformers

import lib_base_classes
import lib_bisect_tokens
import lib_utils

LOGGER = logging.getLogger(__name__)
RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "0"))


def global_do_checks(model):
    local_rank = LOCAL_RANK
    assert model.device.index == local_rank, f"{model.device.index = }, {local_rank = }"
    assert torch.cuda.current_device() == local_rank, (
        torch.cuda.current_device(),
        local_rank,
    )
    assert torch.distributed.get_backend() == "nccl", torch.distributed.get_backend()
    torch.distributed.barrier()


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


def _maybe_frozen_head(model, input_dict, use_frozen_head):
    if use_frozen_head:
        with torch.no_grad():
            return model(**input_dict)
    return model(**input_dict)


def _maybe_autocast(model_or_fn: typing.Callable, dtype: torch.dtype):
    if dtype is None:
        return model_or_fn
    return torch.autocast(device_type="cuda", dtype=dtype)(model_or_fn)  # type: ignore


def clone_hf_model(
    hf_model: transformers.PreTrainedModel,
) -> transformers.PreTrainedModel:
    assert isinstance(hf_model, transformers.PreTrainedModel), type(hf_model)
    copy = type(hf_model)(hf_model.config)
    copy.load_state_dict(hf_model.state_dict())

    return copy


class ScratchpadRewardFn(lib_base_classes.Reward):
    def __init__(
        self,
        *,
        ref_model_is_encoder_decoder: bool,
        ref_inference_fn: typing.Callable,
        inputs_device: torch.device,
        tokenizer: transformers.PreTrainedTokenizerBase,
        metric_fn: typing.Callable,
    ):
        super().__init__()
        raise NotImplementedError("Fix the extractor stuff")

        # ----------------------------------------------------------------
        # Set Attributes
        # ----------------------------------------------------------------
        self._ref_model_is_encoder_decoder = ref_model_is_encoder_decoder
        self._show_answer_replacement = False
        self._reward_tokenizer = tokenizer
        self._ref_inference_fn = ref_inference_fn
        self._no_answer_rate = lib_utils.MovingAverage(10000)
        self._inputs_device = inputs_device
        self._extractor = extractor
        self._tokenizer = tokenizer
        self._metric = metric_fn

    def __call__(
        self,
        *,
        queries: list[str],
        responses: list[str],
        ref_answers: list[str],
    ):
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
        output = np.mean(
            self._metric(
                queries=queries,
                responses=responses,
                ref_answers=ref_answers,
            )["em_accuracy"]
        )

        rich.print(f"[red bold]REWARD - INSTANT PER BATCH METRIC ACC: {output = :0.2%}")

        # Get the answers.
        question_tok = self._tokenizer(queries, padding=True, return_tensors="pt")

        timer_flags = dict(
            disable=True,
            cuda_sync=False,
            accelerate_sync=False,
            accelerator=None,
            log_level=logging.INFO,
            logger=LOGGER,
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
                more_itertools.zip_equal(responses, ref_answers)
            ):
                scratchpads.append(
                    self.replace_answer(
                        original_generation=self._extractor(output),
                        ref_answer=ref_answer,
                    )[0]
                )

        # Find the answer tokens.
        with timer("Extracting the answer tokens", **timer_flags):
            (
                tok_outputs,
                start_end_outputs,
                str_matches_outputs,
            ) = lib_bisect_tokens.extract_match_tokens(
                regexes=[re.escape(ref_answer) for ref_answer in ref_answers],
                tokenizer=self._reward_tokenizer,
                strings=scratchpads,
                tokenizer_kwargs=dict(
                    return_tensors="pt",
                    padding=True,
                ),
                verbose=False,
            )

        assert torch.cuda.current_device() == RANK, (torch.cuda.current_device(), RANK)
        assert len(start_end_outputs) == len(
            scratchpads
        ), f"{len(start_end_outputs) = }, {len(scratchpads) = }"

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
                        scratchpad_ids, skip_special_tokens=True
                    )
                    scratchpad_after = " ".join(
                        [
                            self._reward_tokenizer.decode(token)
                            if m == 0
                            else " <<REF_ANSWER>>"
                            for token, m in zip(scratchpad_ids, mask)
                        ]
                    ).replace("<pad>", "")

                    rich.print(
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
        if self._ref_model_is_encoder_decoder:
            # TODO: Maybe we don't have to recompute the logits.
            # We should get them for generation and for training.
            with timer("Moving things to GPU", **timer_flags):
                input_dict = dict(
                    input_ids=question_tok["input_ids"].to(self._inputs_device),
                    attention_mask=question_tok["attention_mask"].to(
                        self._inputs_device
                    ),
                    decoder_input_ids=tok_outputs["input_ids"].to(self._inputs_device),
                    decoder_attention_mask=tok_outputs["attention_mask"].to(
                        self._inputs_device
                    ),
                )

            with timer(
                "> Computing the logits with the ref model for the reward."
                + f"{question_tok['input_ids'].shape = }",
                **timer_flags,
            ):
                logits = self._ref_inference_fn(**input_dict).logits

        else:
            # TODO(@julesgm): Fix this
            assert False
            logits = self._ref_model(
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

            masks = torch.stack(masks).to(reward_model_outputs_scratchpad.device)
            probs = reward_model_outputs_scratchpad.clone()
            probs[masks == 0] = 1
            logp = probs.log()
            logp_per_seq = logp.sum(-1)
            final_output = logp_per_seq.detach()

        return list(final_output)

    def replace_answer(
        self, *, original_generation: str, ref_answer: str
    ) -> tuple[str, int, int]:
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
            start = original_generation[: answer.start()]
            end = original_generation[answer.end() :]
        elif mode == "remove_end":
            start = original_generation[: answer.start()]
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


class RewardForwardWrapper:
    """
    Meant to work with either a fixed trlAutoModelWithValueHead or a PeftModel
    """

    def __init__(
        self,
        *,
        ppo_trainer_model,
        ppo_trainer_ref_model,
        use_peft,
    ):
        self._ppo_model = ppo_trainer_model
        self._ppo_ref = ppo_trainer_ref_model
        self._use_peft = use_peft

    def __call__(self, *args, **kwargs):
        is_peft = isinstance(self._ppo_model.pretrained_model, peft.PeftModel)
        assert self._use_peft == is_peft, f"{self._use_peft} == {is_peft}"

        if is_peft:
            assert self._ppo_ref is None
        ref_mode = (not is_peft) and (self._ppo_ref is not None)

        assert is_peft ^ ref_mode, f"{is_peft} ^ {ref_mode}"
        rich.print(f"[red bold]{is_peft = } {ref_mode = }")

        if is_peft:
            with self._ppo_model.pretrained_model.disable_adapter():
                with torch.no_grad():
                    return self._ppo_model.pretrained_model(*args, **kwargs)

        elif ref_mode:
            self._ppo_ref.eval()
            with torch.no_grad():
                return self._ppo_ref(*args, **kwargs)

        raise ValueError("Should not be here")
