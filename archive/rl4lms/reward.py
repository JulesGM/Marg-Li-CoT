import enum
import logging
import os
import random
import re
import textwrap
from collections import abc
from typing import *

import general_utils as utils
import more_itertools
import numpy as np
import rich
import rich.console
import text2digits
import torch
import transformers
from beartype import beartype

import rl4lms.envs.text_generation.observation as rl4lms_observation
import rl4lms.envs.text_generation.registry as rl4lms_registry
import rl4lms.envs.text_generation.reward as rl4lms_reward


class ParallelizeMode(str, enum.Enum):
    data_parallel = "data_parallel"
    parallelize = "parallelize"
    single_gpu = "single_gpu"
    nothing = "nothing"


console = rich.console.Console(force_terminal=True)
ANSWER_REMOVAL_PAT = re.compile(r"(.*)Answer:(.*)")
LOGGER = logging.getLogger(__name__)


# def flan_t5_answer_removal(input_text):

#     match_object = ANSWER_REMOVAL_PAT.match(input_text)

#     if match_object is not None:
#         matches = [x.strip() for x in match_object.groups()]
#         # rich.print(f"[bold blue]flan_t5_answer_removal[/]: \"{matches}\"")
#         assert len(matches) == 2, matches

#     else:
#         matches = ["", input_text]

#     return matches


class AnswerExtractionModes(str, enum.Enum):
    last_sentence = "last_sentence"
    last_number = "last_number"


def flan_t5_answer_removal(input_text, mode=AnswerExtractionModes.last_number):
    if mode == AnswerExtractionModes.last_sentence:
        input_text = input_text.strip().removesuffix(".")
        splits = input_text.split(".")
        if len(splits) > 1:
            answer = splits[-1].strip()
            scratchpad = ".".join(splits[:-1])
        else:
            answer = input_text
            scratchpad = ""
    elif mode == AnswerExtractionModes.last_number:
        match_obj = more_itertools.last(NUM_PAT.finditer(input_text), default=None)
        if match_obj is not None:
            answer = match_obj.group()
            scratchpad = input_text[: match_obj.start()]
        else:
            scratchpad = ""
            answer = input_text
    else:
        raise ValueError(
            f"Unknown mode: {mode}, should be one of {list(AnswerExtractionModes)}"
        )

    return scratchpad, answer


def flan_t5_answer_joiner(
    scratchpad, answer, tokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(scratchpad, list), type(scratchpad)
    if scratchpad:
        assert isinstance(scratchpad[0], int), type(scratchpad[0])
    assert isinstance(answer, list), type(answer)

    if answer:
        assert isinstance(answer[0], int), type(answer[0])

    assert isinstance(tokenizer, transformers.PreTrainedTokenizerBase), type(tokenizer)

    joiner_tokens = tokenizer.encode(". Answer: ", add_special_tokens=False,)

    output = scratchpad + joiner_tokens + answer + [tokenizer.eos_token_id]

    mask = (
        [False] * len(scratchpad)
        + [False] * len(joiner_tokens)
        + [True] * len(answer)
        + [True]
    )

    assert all([isinstance(x, int) for x in output]), output
    return torch.tensor(output, dtype=torch.long), torch.tensor(mask, dtype=bool)


def deal_with_words(text: str) -> Optional[float]:
    converted = text2digits.Text2Digits().convert(text)
    output = NUM_PAT.findall(converted)

    if not output:
        return None

    utils.debug_rank_0(LOGGER, "[bold blue]" + "#" * 80)
    utils.debug_rank_0(
        LOGGER,
        f"[bold blue]# text2digits[/]:\n"
        f" \t -> [green]source:[/]    {text}\n"
        f" \t -> [green]converted:[/] {converted}\n"
        f" \t -> [green]final:[/]     {output}",
    )
    utils.debug_rank_0(LOGGER, "[bold blue]" + "#" * 80)

    return output


NUM_PAT = re.compile(r"\d+(?:[\,\.]\d+)?")


class RunningAverageWindow:
    def __init__(self, window_size: int) -> None:
        self._window_size = window_size
        self._window = np.empty(self._window_size)
        self.reset()

    def __len__(self) -> int:
        return self._size

    def reset(self) -> None:
        self._window[:] = 0
        self._pointer = 0
        self._size = 0

    def update(self, value: float) -> None:
        self._window[self._pointer] = value
        self._pointer = (self._pointer + 1) % self._window_size
        self._size = min(self._size + 1, self._window_size)

    def mean(self) -> float:
        return np.mean(self._window[: self._size])

    def sum(self) -> float:
        return np.sum(self._window[: self._size])

    def std(self) -> float:
        return np.std(self._window[: self._size])


class ScratchpadAnswerReward(rl4lms_reward.RewardFunction, torch.nn.Module):
    @beartype
    def __init__(
        self,
        *,
        reward_model:            transformers.PreTrainedModel,
        reward_tokenizer:        transformers.PreTrainedTokenizerBase,
        generation_splitter_fn:  abc.Callable[[str], str],
        sp_answer_joiner_fn:     abc.Callable[[str, str], tuple[torch.Tensor, torch.Tensor]],
        reward_tokenizer_kwargs: dict[str, Any] = None,
    ) -> None:

        utils.info_rank_0(LOGGER, "[bright_magenta bold]#" * 80)
        utils.info_rank_0(
            LOGGER,
            "[bright_magenta bold]# [bright_yellow]REWARD ScratchpadAnswerReward",
        )
        utils.info_rank_0(LOGGER, "[bright_magenta bold]#" * 80)

        super().__init__()
        assert torch.cuda.is_available(), "Reward model must be on GPU"
        self._device = int(os.environ.get("LOCAL_RANK", "0"))
        self._reward_tokenizer = reward_tokenizer
        self._reward_tokenizer_kwargs = reward_tokenizer_kwargs
        self._split_generated = generation_splitter_fn
        self._join_sp_answer = sp_answer_joiner_fn

        state_dict = reward_model.state_dict()
        model = type(reward_model)(reward_model.config)
        model.load_state_dict(state_dict)
        self._model = model
        self._running_average_empty_sp = RunningAverageWindow(1000)
        self._running_average_length_sp = RunningAverageWindow(1000)
        self._running_average_length_no_zero_sp = RunningAverageWindow(1000)


    @classmethod
    def _wrap(cls, text, width=80):
        preprocessed_text = text.replace("\n", " ").strip()
        lines = textwrap.wrap(preprocessed_text, width)
        return "\n".join(["\t" + x.strip() for x in lines])

    @classmethod
    def _log(cls, title, text):
        LOGGER.warning(f"[bold blue]{title}:     \n[green]{cls._wrap(text)}")

    def __call__(
        self,
        current_observation: rl4lms_observation.Observation,
        action: int,
        next_observation: rl4lms_observation.Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        assert "add_special_tokens" not in self._reward_tokenizer_kwargs
        assert "return_tensors" not in self._reward_tokenizer_kwargs

        if done:
            with torch.inference_mode():
                ###############################################################
                # Remove the generated answer from the generated text
                ###############################################################
                generated_scratchpad, generated_answ = self._split_generated(
                    next_observation.context_text
                )

                ###############################################################
                # Prepare the sample:
                # -------------------------------------------------------------
                # Encoder input: prompt
                # Decoder input: encoded(scratchpad) + encoded(answer)
                ###############################################################
                # -------------------------------------------------------------
                # Tokenize the prompt
                # -------------------------------------------------------------
                # Needs special tokens and tensors because it's going to
                # the encoder directly
                encoder_inputs = self._reward_tokenizer(
                    current_observation.prompt_or_input_text,
                    return_tensors="pt",
                    **self._reward_tokenizer_kwargs,
                )

                # -------------------------------------------------------------
                # Tokenize the scratchpad
                # -------------------------------------------------------------
                # No special tokens or tensors because it's getting
                # joined with the answer
                encoded_generated_scratchpad = self._reward_tokenizer(
                    generated_scratchpad,
                    add_special_tokens=False,
                    **self._reward_tokenizer_kwargs,
                )["input_ids"]

                raenz = self._running_average_length_no_zero_sp
                rae = self._running_average_empty_sp
                if encoded_generated_scratchpad:
                    assert isinstance(encoded_generated_scratchpad[0], int), (
                        type(encoded_generated_scratchpad[0]),
                    )
                    rae.update(0)
                    raenz.update(len(encoded_generated_scratchpad))
                    self._log(
                        "Scratchpad length (no zero)",
                        f"{raenz.mean():.1f} +- {raenz.std():.1f}",
                    )
                else:
                    rae.update(1)
                    self._log(
                        "Got an empty scratchpad. ", f"{rae.mean():.1%} or {int(rae.sum())}/{len(rae)}"
                    )

                ral = self._running_average_length_sp
                ral.update(len(encoded_generated_scratchpad))
                self._log("Scratchpad length", f"{ral.mean():.1f} +- {ral.std():.1f}")


                # -------------------------------------------------------------
                # Tokenize the answer
                # -------------------------------------------------------------
                # No special tokens or tensors because it's getting
                # joined with the scratchpad
                assert len(next_observation.target_or_reference_texts) == 1
                encoded_ref_answ = self._reward_tokenizer(
                    next_observation.target_or_reference_texts[0],
                    add_special_tokens=False,
                    **self._reward_tokenizer_kwargs,
                )["input_ids"]
                assert isinstance(encoded_ref_answ, list), type(encoded_ref_answ)
                assert isinstance(encoded_ref_answ[0], int), type(encoded_ref_answ[0])

                # -------------------------------------------------------------
                # Join the scratchpad and the answer.
                # -------------------------------------------------------------
                decoder_input_ids, answer_mask = self._join_sp_answer(
                    scratchpad=encoded_generated_scratchpad,
                    answer=encoded_ref_answ,
                    tokenizer=self._reward_tokenizer,
                )
                decoder_input_ids = decoder_input_ids
                assert decoder_input_ids.ndim == 1, decoder_input_ids.shape
                LOGGER.info("[bold red]" + "#" * 80)
                assert (
                    len(next_observation.target_or_reference_texts) == 1
                ), next_observation.target_or_reference_texts
                assert (
                    next_observation.context_encoded_pt.shape[0] == 1
                ), next_observation.context_encoded_pt.shape

                ###############################################################
                # Compute the reward:
                # -------------------------------------------------------------
                # Just run the model on the decoder inputs
                ###############################################################
                
                full_predictions = self._model(
                    decoder_input_ids=decoder_input_ids.unsqueeze(0).to(self._model.device),
                    **{k: v.to(self._model.device) 
                    for k, v in encoder_inputs.items()},
                ).logits.log_softmax(dim=-1)
                assert full_predictions.ndim == 3, (
                    full_predictions.ndim,
                    3,
                )

                ###############################################################
                # Extract the likelihood of the answer
                # -------------------------------------------------------------
                # Gather the log-probabilities of the generated text,
                # and then sum up the log-probabilities of the answer.
                ###############################################################
                assert full_predictions.shape[0] == 1, full_predictions.shape
                full_predictions = full_predictions.squeeze(0)
                logp = full_predictions.gather(
                    index=decoder_input_ids.unsqueeze(0).to(self._model.device), dim=-1,
                ).squeeze(0)
                assert full_predictions.shape[0] == logp.shape[0], (
                    full_predictions.shape[0],
                    logp.shape[0],
                )
                assert logp.ndim == 1, (
                    logp.ndim,
                    1,
                )
                logp_answer = logp[answer_mask].sum(-1)

                ###############################################################
                # Logging
                ###############################################################
                # self._log("Prompt", current_observation.prompt_or_input_text)
                # self._log("Generated text", next_observation.context_text)
                # self._log("Generated scratchpad", generated_scratchpad)
                # self._log("Generated answer", generated_answ)
                # self._log("Target", self._reward_tokenizer.decode(encoded_ref_answ))
                # self._log(
                #     "Reference text", next_observation.target_or_reference_texts[0]
                # )
                # self._log(
                #     "decoder_input_ids text",
                #     self._reward_tokenizer.decode(decoder_input_ids),
                # )
                # self._log(
                #     "positively masked",
                #     self._reward_tokenizer.decode(decoder_input_ids[answer_mask]),
                # )

                return logp_answer.item()

            return 0.0


class BatchedScratchpadAnswerReward(rl4lms_reward.BatchedRewardFunction, torch.nn.Module):
    @beartype
    def __init__(
        self,
        *,
        reward_model: transformers.PreTrainedModel,
        reward_tokenizer: transformers.PreTrainedTokenizerBase,
        sp_answer_joiner_fn: abc.Callable[
            [list[int], list[int]], tuple[list[int], list[bool]]
        ],
        generation_splitter_fn: abc.Callable[[str], str] = flan_t5_answer_removal,
    ) -> None:

        utils.info_rank_0(LOGGER, "[bright_magenta bold]#" * 80)
        utils.info_rank_0(
            LOGGER,
            "[bright_magenta bold]# [bright_yellow]REWARD ScratchpadAnswerReward",
        )
        utils.info_rank_0(LOGGER, "[bright_magenta bold]#" * 80)

        super().__init__()
        
        self._reward_tokenizer = reward_tokenizer
        self._split_generated = generation_splitter_fn
        self._sp_answer_joiner = sp_answer_joiner_fn
        
        state_dict = reward_model.state_dict()
        model = type(reward_model)(reward_model.config)
        model.load_state_dict(state_dict)
        self._reward_model = reward_model

    @classmethod
    def _wrap(cls, text, width=80):
        preprocessed_text = text.replace("\n", " ").strip()
        lines = textwrap.wrap(preprocessed_text, width)
        return "\n".join(["\t" + x.strip() for x in lines])

    @classmethod
    def _log(cls, title, text):
        LOGGER.warning(f"[bold blue]{title}:     \n[green]{cls._wrap(text)}")

    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:

        assert len(prompt_texts) == len(gen_texts) == len(ref_texts) == len(dones), (
            f"{len(prompt_texts) = }\n",
            f"{len(gen_texts)    = }\n",
            f"{len(ref_texts)    = }\n",
            f"{len(dones)        = }\n",
        )

        # Can't do anything if they're not all done
        if not all(dones):
            return np.zeros(
                len(prompt_texts), 
            )

        assert not isinstance(self._reward_model, transformers.PreTrainedModel), (
            type(self._reward_model).mro()
        )

        with torch.inference_mode():
            # -------------------------------------------------------------
            # Tokenize the prompt
            # -------------------------------------------------------------
            # Needs special tokens and tensors because it's going to
            # the encoder directly
            done_prompts_encoded_ids = self._reward_tokenizer(
                prompt_texts, return_tensors="pt", padding=True, truncation=True,
            ).input_ids

            # Split the generated texts
            generated_scratchpads, generated_answ = zip(
                *[self._answer_remover_fn(gen_text) for gen_text in gen_texts]
            )

            # -------------------------------------------------------------
            # Tokenize the scratchpad and ref answers
            # -------------------------------------------------------------
            # No special tokens or tensors because they're getting joined
            gen_scratchpads_ids = self._reward_tokenizer(
                generated_scratchpads, add_special_tokens=False,
            )
            ref_ids = self._reward_tokenizer(ref_texts, add_special_tokens=False,)

            # -------------------------------------------------------------
            # Join the scratchpads and reference answers
            # -------------------------------------------------------------
            decoder_input_ids, reward_mask = zip(
                *[
                    self._sp_answer_joiner(
                        scratchpad=sp, answer=answ, tokenizer=self._reward_tokenizer,
                    )
                    for sp, answ in zip(gen_scratchpads_ids, ref_ids)
                ]
            )
            output_reward_mask = torch.nn.utils.rnn.pad_sequence(
                reward_mask, batch_first=True, padding_value=False,
            )
            decoder_input_ids = self._reward_tokenizer.pad(
                dict(input_ids=decoder_input_ids), return_tensors="pt", padding=True,
            )

            ###############################################################
            # Compute the reward:
            ###############################################################
            # TODO: Non Tested!
            full_predictions = self._reward_model(
                decoder_input_ids=decoder_input_ids["input_ids"],
                decoder_attention_mask=decoder_input_ids["attention_mask"],
                **{k: v for k, v in done_prompts_encoded_ids.items()},
            ).logits.log_softmax(dim=-1)
            assert full_predictions.ndim == 3, (
                full_predictions.ndim,
                3,
            )

            ###############################################################
            # Extract the likelihood of the answer
            ###############################################################
            # TODO: Non Tested!
            logp = full_predictions.gather(dim=-1, index=decoder_input_ids["input_ids"])
            logp_answer = (logp * output_reward_mask)
            print(f"{logp_answer.shape = }")
            assert logp_answer.shape == (len(gen_texts),)

            ###############################################################
            # Logging
            ###############################################################
            idx = random.randint(0, len(gen_texts) - 1)
            self._log("Prompt", prompt_texts[idx])
            self._log("Generated text", gen_texts[idx])
            self._log("Generated scratchpad", generated_scratchpads[idx])
            self._log("Generated answer", generated_answ[idx])
            self._log("Target", self._reward_tokenizer.decode(ref_ids[idx]))
            self._log("Reference text", ref_texts[idx])
            self._log(
                "decoder_input_ids text",
                self._reward_tokenizer.decode(decoder_input_ids[idx]),
            )
            self._log(
                "positively masked",
                self._reward_tokenizer.decode(decoder_input_ids[idx][reward_mask[idx]]),
            )

            return logp_answer.tolist()


class FlanT5ScratchpadAnswerReward(ScratchpadAnswerReward):
    @beartype
    def __init__(self, **kwargs,) -> None:
        super().__init__(
            sp_answer_joiner_fn=flan_t5_answer_joiner,
            generation_splitter_fn=flan_t5_answer_removal,
            **kwargs,
        )


class FlanT5BatchedScratchpadAnswerReward(BatchedScratchpadAnswerReward):
    def __init__(self, **kwargs,) -> None:
        super().__init__(
            sp_answer_joiner_fn=flan_t5_answer_joiner,
            generation_splitter_fn=flan_t5_answer_removal,
            **kwargs,
        )


rl4lms_registry.RewardFunctionRegistry.add(
    "scratchpad_answer_reward", ScratchpadAnswerReward,
)
rl4lms_registry.RewardFunctionRegistry.add(
    "batch_scratchpad_answer_reward", BatchedScratchpadAnswerReward,
)
rl4lms_registry.RewardFunctionRegistry.add(
    "flan_t5_scratchpad_answer_reward", FlanT5ScratchpadAnswerReward,
)
rl4lms_registry.RewardFunctionRegistry.add(
    "flan_t5_batch_scratchpad_answer_reward", FlanT5BatchedScratchpadAnswerReward,
)
