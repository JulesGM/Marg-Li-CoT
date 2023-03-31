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

import lib_bisect_tokens
import lib_data
import lib_metric

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


def clone_hf_model(
        hf_model: transformers.PreTrainedModel
) -> transformers.PreTrainedModel:
    
    assert isinstance(hf_model, transformers.PreTrainedModel), type(hf_model)
    copy = type(hf_model)(hf_model.config)
    copy.load_state_dict(hf_model.state_dict())
    return copy


class ScratchpadRewardFn(torch.nn.Module):
    def __init__(
        self, *, 
        reward_model_hf_name_or_path: str,
        get_extra_info_fn: typing.Callable[
            [typing.Union[str, typing.List[str]]], 
            typing.Union[typing.Dict, typing.List[typing.Dict]]
        ],
        reward_tokenizer: transformers.PreTrainedTokenizerBase, 
        do_single_proc: bool,
        metric_fn,
    ):
        super().__init__()

        #----------------------------------------------------------------
        # Build Models
        #----------------------------------------------------------------
        reward_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            reward_model_hf_name_or_path)
        
        if do_single_proc:
            non_distributed = [clone_hf_model(
                reward_model).eval().to(LOCAL_RANK)]
        else:
            non_distributed = None

        #----------------------------------------------------------------
        # Set Attributes
        #----------------------------------------------------------------
        self._non_distributed_reward_model = non_distributed
        self._show_answer_replacement      = False
        self._get_extra_info_fn            = get_extra_info_fn
        self._reward_tokenizer             = reward_tokenizer
        self._do_single_proc               = do_single_proc
        self._no_answer_rate               = lib_data.MovingAverage(10000)
        self._reward_model                 = reward_model.eval()
        self._conv_to_num                  = lib_data.ConvToNum()
        self._metric                       = metric_fn
        if os.environ["ACCELERATE_MIXED_PRECISION"] == "bf16":
            self._dtype = torch.bfloat16
        elif os.environ["ACCELERATE_MIXED_PRECISION"] == "fp16":
            self._dtype = torch.float16
        elif os.environ["ACCELERATE_MIXED_PRECISION"] == "no":
            self._dtype = torch.float32
        else:
            raise ValueError(os.environ["ACCELERATE_MIXED_PRECISION"])

        self._prep_models()


    def _prep_models(self):
        for p in self._reward_model.parameters():
            p.requires_grad = False

        if self._non_distributed_reward_model:
            for p in more_itertools.one(
                self._non_distributed_reward_model).parameters():
                p.requires_grad = False

        assert all(
            isinstance(x, torch.nn.parameter.Parameter) 
            for x in self.parameters()
        ), (collections.Counter(type(x) for x in self.parameters()))
        req_grad_self = [x.requires_grad for x in self.parameters()] 
        assert not any(req_grad_self), (
            f"{np.mean(req_grad_self) = :0.1%}"
        )
        reg_grad_distributed = [
            x.requires_grad for x 
            in self._reward_model.parameters()
        ]
        assert not any(reg_grad_distributed), (
            f"{np.mean(reg_grad_distributed) = :0.1%}"
        )

        if self._non_distributed_reward_model:
            req_grad_single = [
                x.requires_grad for x in 
                more_itertools.one(
                self._non_distributed_reward_model
                ).parameters()
            ]
            assert not any(req_grad_single), (
                f"{np.mean(req_grad_single) = :0.1%}"
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
        if not is_distributed:
            assert self._non_distributed_reward_model, (
                self._non_distributed_reward_model)

        assert prompts, is_distributed
        assert samples, is_distributed
        assert outputs, is_distributed

        #######################################################################
        # - Sanity checks 
        # - Varia prep
        #######################################################################
        
        output = np.mean(self._metric(prompts, samples, outputs)["em_accuracy"])
        LOGGER.info(f"[red bold]REWARD - INSTANT PER BATCH METRIC ACC: {output = :0.2%}")

        assert isinstance(prompts   , list), f"{type(prompts   ).mro() = }"
        assert isinstance(prompts[0], str ), f"{type(prompts[0]).mro() = }"
        assert isinstance(outputs   , list), f"{type(outputs   ).mro() = }"
        assert isinstance(outputs[0], str ), f"{type(outputs[0]).mro() = }"

        global_do_checks(self._reward_model)
        if self._non_distributed_reward_model:
            global_do_checks(more_itertools.one(self._non_distributed_reward_model))

        # Pick the correct model depending on whether we are distributed or not.
        model = (
            self._reward_model if is_distributed 
            else more_itertools.one(
                self._non_distributed_reward_model
            )
        )

        assert model is not None, f"model is None. {is_distributed = }"
        assert not model.training, f"{model.training = }"
        assert not any( x.requires_grad for x in model.parameters()), (
            f"{np.mean([x.requires_grad for x in model.parameters()]):0.1%}"
        )

        # Get the answers.
        assert (torch.cuda.current_device() == torch.distributed.get_rank()), (
            torch.cuda.current_device(), torch.distributed.get_rank())
        
        extra_info = self._get_extra_info_fn(
            sample_str=prompts, miss_ok=False)
        ref_answers = [x["answer"] for x in extra_info]
        question_tok = self._reward_tokenizer(
            prompts, padding=True, return_tensors="pt"
        )

        timer_flags = dict(
            accelerate_sync = False,
            accelerator     = None,
            log_level       = logging.INFO,
            cuda_sync       = True,
            disable         = True,
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
            f"{len(start_end_outputs) = }, {len(scratchpads) = }"
        )
        
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
            assert model.device.type == "cuda", (
                f"{model.device.type = }"
            )
            distributed_str = "DISTRIBUTED" if is_distributed else "NON-DISTRIBUTED"
            
            # TODO: Maybe we don't have to recompute the logits. 
            # We should get them for generation and for training. 
            with timer("Moving things to GPU", **timer_flags):
                input_dict = dict(
                    input_ids              = question_tok["input_ids"     ].to(model.device),
                    attention_mask         = question_tok["attention_mask"].to(model.device),
                    decoder_input_ids      = tok_outputs ["input_ids"     ].to(model.device),
                    decoder_attention_mask = tok_outputs ["attention_mask"].to(model.device),
                )

            with timer(
                f"[{distributed_str}]:\n> Computing the logits with the ref model for the reward." +
                f"{question_tok['input_ids'].shape = }"
                , **timer_flags):
                
                accel_state_mixed = os.environ["ACCELERATE_MIXED_PRECISION"]
                
                # -------------------------------------------------------------
                # We need to handle autocast in single proc mode, because 
                # the model is not passed tp accelerate
                # -------------------------------------------------------------
                with torch.no_grad():
                    assert not model.training, f"{model.training = }"                    
                    if not is_distributed and not accel_state_mixed == "no":
                        if accel_state_mixed == "bf16":
                            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                                logits = model(**input_dict).logits
                        elif accel_state_mixed == "fp16": 
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                logits = model(**input_dict).logits
                        else:
                            raise ValueError(f"{accel_state_mixed = }")
                    else:
                        logits = model(**input_dict).logits

                    if not is_distributed:
                        if accel_state_mixed == "bf16":
                            assert logits.dtype == torch.bfloat16, f"{logits.dtype = }"
                        elif accel_state_mixed == "fp16":
                            assert logits.dtype == torch.float16 , f"{logits.dtype = }"
                        elif accel_state_mixed == "no":
                            assert logits.dtype == torch.float   , f"{logits.dtype = }"
                        else:
                            raise ValueError(f"{accel_state_mixed = }")

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