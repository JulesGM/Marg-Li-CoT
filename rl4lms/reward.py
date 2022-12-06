import re
from typing import *

from beartype import beartype
from collections import abc
import datasets
import tqdm
import rich
import rich.console
import transformers
import torch
import numpy as np
import pandas as pd

import rl4lms.envs.text_generation.observation as rl4lms_observation
import rl4lms.envs.text_generation.reward as rl4lms_reward
import rl4lms.envs.text_generation.registry as rl4lms_registry


console = rich.console.Console(force_terminal=True)
ANSWER_REMOVAL_PAT = re.compile(r"(.*)Answer:(.*)")


# def flan_t5_answer_removal(input_text):

#     match_object = ANSWER_REMOVAL_PAT.match(input_text)

#     if match_object is not None:
#         matches = [x.strip() for x in match_object.groups()]
#         # rich.print(f"[bold blue]flan_t5_answer_removal[/]: \"{matches}\"")
#         assert len(matches) == 2, matches    

#     else:
#         matches = ["", input_text]

#     return matches


def flan_t5_answer_removal(input_text):

    input_text = input_text.strip().removesuffix(".")
    splits     = input_text.split(".")
    if len(splits) > 1:
        answer     = splits[-1].strip()
        scratchpad = ".".join(splits[:-1])
    else:
        answer     = input_text
        scratchpad = ""

    return scratchpad, answer


class ScratchpadAnswerReward(rl4lms_reward.RewardFunction):    
    @beartype
    def __init__(
        self,
        *,
        reward_model:            transformers.PreTrainedModel,
        reward_tokenizer:        transformers.PreTrainedTokenizerBase,
        answer_remover_fn:       abc.Callable[[str], str],
        reward_tokenizer_kwargs: dict[str, Any] = None,
    ) -> None:

        rich.print("[bright_magenta bold]#" * 80)
        rich.print("[bright_magenta bold]# [bright_yellow]REWARD ScratchpadAnswerReward")
        rich.print("[bright_magenta bold]#" * 80)
        
        super().__init__()
        assert torch.cuda.is_available(), "Reward model must be on GPU"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._metric_model            = reward_model.to(self._device)
        self._reward_tokenizer        = reward_tokenizer
        self._reward_tokenizer_kwargs = reward_tokenizer_kwargs
        self._answer_remover_fn       = answer_remover_fn

    def __call__(
        self, 
        current_observation: rl4lms_observation.Observation,
        action: int,
        next_observation: rl4lms_observation.Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:


        if done:
            with torch.inference_mode():
                ###############################################################
                # Remove the generated answer from the generated text
                ###############################################################
                generated_scratchpad, generated_answer = self._answer_remover_fn(
                    next_observation.context_text
                )

                ###############################################################
                # Prepare the sample:
                ###############################################################
                encoder_inputs = self._reward_tokenizer(
                    current_observation.prompt_or_input_text,
                    return_tensors="pt",
                    **self._reward_tokenizer_kwargs,
                ).to(self._device)

                assert "add_special_tokens" not in self._reward_tokenizer_kwargs

                encoded_generated_scratchpad = self._reward_tokenizer(
                    generated_scratchpad,
                    add_special_tokens=False,
                    **self._reward_tokenizer_kwargs,
                )["input_ids"]

                encoded_answ = self._reward_tokenizer(
                    next_observation.target_or_reference_texts,
                    add_special_tokens=False,
                    **self._reward_tokenizer_kwargs,
                )["input_ids"]

                assert len(encoded_answ) == 1, len(encoded_answ)
                encoded_answ = encoded_answ[0]
                assert isinstance(encoded_answ, list), type(encoded_answ)
                
                new_list = (
                    encoded_generated_scratchpad +
                    encoded_answ +
                    [self._reward_tokenizer.eos_token_id]
                )
                
                decoder_input_ids = torch.tensor(new_list).to(self._device)
                assert decoder_input_ids.ndim == 1, decoder_input_ids.shape

                console.print("[bold red]#[/]" * 80)
                console.print(
                    f"[bold blue]target                [/]: "
                    f"\"{self._reward_tokenizer.decode(encoded_answ)}\""
                )
                assert len(next_observation.target_or_reference_texts) == 1, (
                    next_observation.target_or_reference_texts)
                console.print(
                    f"[bold blue]reference text        [/]: "
                    f"[bold green]\"{next_observation.target_or_reference_texts[0]}\"[/]"
                )
                console.print(
                    f"[bold  blue]generated ANSWER            [/]: "
                    f"[red   bold]\"{generated_answer    }\"[/]"
                )


                console.print(
                    f"[bold blue]encoder_input_ids text[/]: "
                    f"\"{current_observation.prompt_or_input_text}\""
                )
                console.print(
                    f"[bold blue]decoder_input_ids text[/]: "
                    f"\"{self._reward_tokenizer.decode(decoder_input_ids)}\""
                )
                console.print(
                    f"[bold  blue]generated SCRATCHPAD        [/]: "
                    f"[green     ]\"{generated_scratchpad}\"[/]"
                )
                console.print(
                    f"[bold blue]raw generated text    [/]: "
                    f"[red     ]\"{next_observation.context_text}\"[/]"
                )
                
                assert next_observation.context_encoded_pt.shape[0] == 1, (
                    next_observation.context_encoded_pt.shape
                )

                console.print(
                    f"[bold  blue]raw generated len     [/]: "
                    f"{len(next_observation.context_encoded_pt.squeeze(0))}"
                )

                ###############################################################
                # Compute the reward:
                ###############################################################
                full_predictions = self._metric_model(
                    decoder_input_ids=decoder_input_ids.unsqueeze(0).to(self._device),
                    **{k: v.to(self._device) for k, v in encoder_inputs.items()},
                ).logits.log_softmax(dim=-1)
                assert full_predictions.ndim == 3, (full_predictions.ndim, 3,)
                
                ###############################################################
                # Extract the likelihood of the answer
                ###############################################################
                assert full_predictions.shape[0] == 1, full_predictions.shape
                full_predictions = full_predictions.squeeze(0)

                logp = full_predictions.gather(
                    dim=-1, 
                    index=decoder_input_ids.unsqueeze(0)
                ).squeeze(0)

                # console.print(f"[bold blue]full_predictions .shape[/]: {full_predictions .shape}")
                # console.print(f"[bold blue]decoder_input_ids.shape[/]: {decoder_input_ids.shape}")
                # console.print(f"[bold blue]logp             .shape[/]: {logp             .shape}")
                # console.print(f"[bold blue]decoder_input_ids.shape[/]: {decoder_input_ids.shape}")

                assert full_predictions.shape[0] == logp.shape[0], (full_predictions.shape[0], logp.shape[0],)
                assert logp.ndim        == 1, (logp.ndim    , 1,)

                score_mask = torch.tensor(
                    [0] * len(encoded_generated_scratchpad) + 
                    [1] * len(encoded_answ) +
                    [1]
                ).to(self._device)

                logp_answer = logp[score_mask == 1].sum(-1)

                return logp_answer.item()

        return 0.


class BatchedScratchpadAnswerReward(rl4lms_reward.BatchedRewardFunction):    
    @beartype
    def __init__(
        self, 
        reward_model:            transformers.PreTrainedModel,
        reward_tokenizer:        transformers.PreTrainedTokenizerBase,
        reward_tokenizer_kwargs: dict[str, Any] = None,
        answer_remover_fn:       abc.Callable[[str], str] = flan_t5_answer_removal,
    ) -> None:
        
        rich.print("[bright_magenta bold]#" * 80)
        rich.print("[bright_magenta bold]# [bright_yellow]REWARD ScratchpadAnswerReward")
        rich.print("[bright_magenta bold]#" * 80)

        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._reward_model            = reward_model.to(self._device)
        self._reward_tokenizer        = reward_tokenizer
        self._reward_tokenizer_kwargs = reward_tokenizer_kwargs
        self._answer_remover_fn       = answer_remover_fn

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:
        
        dones = np.array(dones)
        
        if not np.any(dones):
            return 0

        idx_dones = []
        done_prompt_texts = []
        done_gen_texts = []
        done_ref_texts = []

        for idx, is_done in enumerate(dones):
            if is_done:
                idx_dones.append(idx)
                done_prompt_texts.append(prompt_texts[idx])
                done_gen_texts.append(gen_texts[idx])
                done_ref_texts.append(ref_texts[idx])

        with torch.inference_mode():
            done_prompts_encoded_ids         = self._reward_tokenizer(done_prompt_texts, return_tensors="pt", padding=True,)
            generated_scratchpads, _         = zip(*[self._answer_remover_fn(gen_text) for gen_text in gen_texts])
            done_gen_scratchpads_encoded_ids = self._reward_tokenizer(generated_scratchpads)
            done_ref_encoded_ids             = self._reward_tokenizer(done_ref_texts)
            new_output_encoded_ids           = [sp + answ for sp, answ in zip(done_gen_scratchpads_encoded_ids, done_ref_encoded_ids)]
            output_reward_mask               = torch.nn.utils.rnn.pad_sequence([
                                                torch.tensor(len(sp) * [0] + len(answ) * [1]) for sp, answ in 
                                                zip(done_gen_scratchpads_encoded_ids, done_ref_encoded_ids)
                                                ], batch_first=True)
            new_output_encoded_ids           = self._reward_tokenizer.pad(
                                                dict(input_ids=new_output_encoded_ids), 
                                                return_tensors="pt"
                                                )
            

            ###############################################################
            # Compute the reward:
            ###############################################################
            full_predictions = self._metric_model(
                decoder_input_ids=new_output_encoded_ids["input_ids"].to(self._device),
                decoder_attention_mask=new_output_encoded_ids["attention_mask"].to(self._device),
                **done_prompts_encoded_ids,
            ).logits.log_softmax(dim=-1)
            assert full_predictions.ndim == 3, (full_predictions.ndim, 3,)
            

            ###############################################################
            # Extract the likelihood of the answer
            ###############################################################
            logp = full_predictions.gather(
                dim=-1, 
                index=new_output_encoded_ids["input_ids"]
            )

            logp[output_reward_mask == 0] = 0
            logp_answer = logp.sum(-1)

            return logp_answer.item()




rl4lms_registry.RewardFunctionRegistry.add(
    "scratchpad_answer_reward",
    ScratchpadAnswerReward,
)