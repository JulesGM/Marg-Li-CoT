from __future__ import annotations
import abc
import bisect
import collections
import copy
import contextlib
import dataclasses
import enum
import itertools
import json
import math
import os
import pathlib
import sys
import typing
from typing import Optional, Union

import accelerate
import more_itertools
import numpy as np
import peft
# import peft_qlora
import rich
import rich.layout
import rich.markup
import rich.panel
import rich.table
import torch
import transformers
import transformers.tokenization_utils
import trl
import trl.core
import trl.models
import tqdm


SCRIPT_DIR = pathlib.Path(__file__).absolute().parent

sys.path.append(str(SCRIPT_DIR.parent))

from with_trl import lib_base_classes
from with_trl import lib_utils
from with_trl import lib_peft_utils

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))


class FixedPPOConfig(trl.PPOConfig):
    """Fixes the serialization of the config object.
    
    Out of the box, TRL tries to log it's config object as a json string, but
    doesn't do a good job at all of serializing some of the objects.
        
    This class fixes that.

    """
    def to_dict(self):
        new_self = copy.deepcopy(self)
        deepspeed_plugin_key = "deepspeed_plugin"
        kwargs_handlers_key = "kwargs_handlers"
        hf_ds_config = "hf_ds_config"
        accelerator_kwargs = new_self.accelerator_kwargs

        if deepspeed_plugin_key in accelerator_kwargs:
            accelerator_kwargs[deepspeed_plugin_key] = vars(
                accelerator_kwargs[deepspeed_plugin_key]) | {
                    "__class__": accelerate.utils.DeepSpeedPlugin.__name__
                }
            deepspeed_plugin = accelerator_kwargs[deepspeed_plugin_key]
            deepspeed_plugin[hf_ds_config] = vars(
                deepspeed_plugin[hf_ds_config]) | {
                    "__class__": deepspeed_plugin[hf_ds_config].__class__.__name__
                }
        
        kwargs_handlers = accelerator_kwargs[kwargs_handlers_key]
        for i, entry in enumerate(kwargs_handlers):
            if isinstance(entry, accelerate.utils.DistributedDataParallelKwargs):
                kwargs_handlers[i] = vars(entry) | {
                    "__class__": accelerate.utils.DistributedDataParallelKwargs.__name__
                }

        return super(FixedPPOConfig, new_self).to_dict()


# @dataclasses.dataclass
# class Args:

#     @dataclasses.dataclass
#     class Model:
#         batch_size:             int
#         inference_batch_size:   int
#         mini_batch_size:        int
#         model_name:             str

#     def __post_init__(self):
#         if self.curriculum_schedule:
#             self.curriculum_schedule = CurriculumSchedule(
#                 [CurriculumSchedule.CurriculumEntry(x) for x in self.curriculum_schedule]
#             )

#         self.answer_only_path    = pathlib.Path(self.answer_only_path)
#         self.dataset_name        = lib_data.DatasetChoices(self.dataset_name)
#         self.generation_kwargs   = self.generation_kwargs
#         self.model               = Args.Model(**self.model)
#         self.precision           = lib_utils.ValidPrecisions(self.precision)
#         self.peft_config         = peft.LoraConfig(**self.peft_config)
#         self.task_name           = lib_utils.Task(self.task_name)
#         self.wandb_dir           = pathlib.Path(self.wandb_dir)
#         self.arithmetic_dataset_root_folder_dir = pathlib.Path(self.arithmetic_dataset_root_folder_dir)
#         self.inference_generation_kwargs        = self.inference_generation_kwargs
    
#     answer_only_path:    pathlib.Path
#     curriculum_schedule: CurriculumSchedule | None
#     dataset_name:        str
#     generation_kwargs:   transformers.GenerationConfig
#     model:               Model
#     peft_config:         peft.LoraConfig
#     task_name:           lib_utils.Task
#     wandb_dir:           pathlib.Path
#     arithmetic_dataset_root_folder_dir: pathlib.Path
#     inference_generation_kwargs:        transformers.GenerationConfig

#     name:                        str
#     learning_rate:               float
#     answer_only:                 bool
#     answer_only_max_length:      int
#     input_max_length:            int
#     eval_every:                  int
#     eval_subset_size:            int
#     gradient_accumulation_steps: int
#     just_metrics:                bool
#     kl_penalty_mode:             str
#     peft_do_all_lin_layers:      bool
#     use_curriculum:              bool
#     use_peft:                    bool
#     use_few_shots:               bool
#     precision:                   torch.dtype | str
    
#     reward_type:                 Optional[str]
#     wandb_project:               str


class AccuracyMaintainer(abc.ABC):
    @abc.abstractmethod
    def __init__(self, limit_to_respect):
        assert isinstance(limit_to_respect, float), type(limit_to_respect)
        assert limit_to_respect >= 0, limit_to_respect
        assert limit_to_respect <= 1, limit_to_respect
        
    @abc.abstractmethod
    def get_stats(self) -> dict:
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        pass



class MaxLevelGlobalAvgAcc(AccuracyMaintainer):
    def __init__(self, limit_to_respect: float):
        super().__init__(limit_to_respect)
        
        self._max_level = None
        self._max_level_sum = None
        self._max_level_count = None
        self._limit_to_respect = limit_to_respect
        self._latest_distribution = None

    def _max_level_avg(self):
        return self._max_level_sum / self._max_level_count


    def get_stats(self) -> dict:
        return dict(
            max_level=self._max_level,
            max_level_acc=self._max_level_avg(),
            max_level_sum=self._max_level_sum,
            max_level_count=self._max_level_count,
            limit_to_respect=self._limit_to_respect,
            latest_distribution=self._latest_distribution,
        )
    
    def __call__(self, batch_difficulty_level, reward_output_values) -> bool:
        """
        
        For the hardest problems, only add them if we have enough good answers.
        
        We should probably do this for each level.

        """

        received_max_level = max(batch_difficulty_level)
        assert self._max_level is None or received_max_level >= self._max_level, (
            received_max_level, self._max_level)
        # The curriculum must have a strictly increasing max level

        indices_ok = []
        for idx, level in enumerate(batch_difficulty_level):
            
            if level == received_max_level:
                reward_output = reward_output_values[idx]
                assert reward_output == 0. or reward_output == 1., reward_output
                is_good = bool(reward_output)

                ok_to_add_bad = (
                    # Prevent divisions by zero in _max_level_avg by short-circuiting
                    (not self._max_level_count)
                    or
                    (self._max_level_avg() < self._limit_to_respect)
                )

                if is_good or ok_to_add_bad:
                    if self._max_level is None or received_max_level > self._max_level:
                        self._max_level = received_max_level
                        self._max_level_sum = 1
                        self._max_level_count = int(is_good)

                    elif received_max_level == self._max_level:
                        self._max_level_sum += 1
                        self._max_level_count += int(is_good)
                    
                    indices_ok.append(idx)
            else:
                indices_ok.append(idx)
        
        stats = collections.defaultdict(list)
        
        for idx in indices_ok:
            stats[level].append(reward_output_values[idx])
        
        self._latest_distribution = stats

        return indices_ok, stats
        

# class CurriculumPonderedGlobalAvgAcc(AccuracyMaintainer):
#     """
#     Heuristic of some kind. Feels a bit silly.
    
#     """
#     def __init__(self, limit_to_respect: float):
#         super().__init__(limit_to_respect)
#         assert False, "Not implemented"

#         self._sums = {}
#         self._counts = {}
#         self._limit_to_respect = limit_to_respect
#         self._latest_normalized_curriculum = None

#     def _curriculum_avg(self):
#         assert self._sums.keys() == self._counts.keys(), ()
#         weight_sum = sum(self._latest_normalized_curriculum.values())
#         assert math.isclose(weight_sum, 1.), weight_sum

#         return sum(
#             [self._sums[key] * self._latest_curriculum[key] / self._counts[key] for key in self._sums]
#         )
    

#     def set_curriculum(self, curriculum: dict[int, float]) -> None:
#         sum_curr = sum(curriculum.values())
#         self._latest_normalized_curriculum = {k: v / sum_curr for k, v in curriculum.items()}

    
#     def get_stats(self) -> dict:
#         return dict(
#             sums=self._sums,
#             counts=self._counts,
#             latest_normalized_curriculum=self._latest_normalized_curriculum,
#             limit_to_respect=self._limit_to_respect,
#             skips_in_a_row=self._skips_in_a_row,
#         )

#     def __call__(self, level, is_good) -> bool:
        
#         assert isinstance(is_good, bool), type(is_good)
#         assert isinstance(level, int), type(level)
#         assert self._latest_normalized_curriculum, self._latest_normalized_curriculum

#         if level not in self._sums:
#             self._sums  [level] = 0
#             self._counts[level] = 0

#         self._sums  [level] += int(is_good)
#         self._counts[level] += 1
        
#         do_skip = self._max_level_avg() < self._limit_to_respect
#         self._update_skips_in_a_row(do_skip)
#         return not do_skip


class Maintainers(str, enum.Enum):
    MAX_LEVEL_GLOBAL_AVG_ACC = "MaxLevelGlobalAvgAcc"
    # CURRICULUM_PONDERED_GLOBAL_AVG_ACC = "CurriculumPonderedGlobalAvgAcc"


MAINTAINER_NAME_TO_CLASS = {
    Maintainers.MAX_LEVEL_GLOBAL_AVG_ACC: MaxLevelGlobalAvgAcc,
    # Maintainers.CURRICULUM_PONDERED_GLOBAL_AVG_ACC: CurriculumPonderedGlobalAvgAcc,
}


def generate(
        *, 
        answer_extractor,
        batch,
        forward_tokenizer: transformers.PreTrainedTokenizerBase,
        generation_batch_size: int,
        generation_kwargs,
        post_process_gen_fewshots_fn,
        ppo_batch_size: int,
        ppo_trainer: trl.PPOTrainer,
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
        step_information,
        task_name: str,
        use_few_shots: int,
    ):

    print(f"[RANK: {RANK}] lib_trl_utils.batched_unroll: ({step_information = }) >>>")
    assert isinstance(batch["tok_ref_query"]   , (list, torch.Tensor)), type(batch["tok_ref_query"]   ).mro()
    assert isinstance(batch["tok_ref_query"][0], torch.Tensor        ), type(batch["tok_ref_query"][0])

    raw_gen_outputs, scores = batched_unroll(
        accelerator                  = ppo_trainer.accelerator,
        accelerated_model            = ppo_trainer.model,
        ref_text                     = batch["ref_qa_answer"],
        difficulty_levels            = batch["difficulty_level"],
        query_tensors                = batch["tok_ref_query"],
        generation_batch_size        = generation_batch_size,
        generation_kwargs            = generation_kwargs,
        post_process_gen_fewshots_fn = post_process_gen_fewshots_fn,
        prediction_tokenizer         = prediction_tokenizer,
        forward_tokenizer            = forward_tokenizer,
        task_name                    = task_name,
        use_few_shots                = use_few_shots,
    )

    lib_utils.named_barrier(f"{__file__}: {lib_utils.get_linenumber()}")
    print(f"{RANK} lib_trl_utils.batched_unroll <<<")
    if task_name == lib_utils.Task.MAIN:
        outputs = keep_good_one_generation(
            answer_extractor     = answer_extractor,
            difficulty_levels    = batch["difficulty_level"],
            generations          = raw_gen_outputs,
            ppo_batch_size= ppo_batch_size,
            num_return_seq       = generation_kwargs["num_return_sequences"],
            other_rewards        = dict(scores=scores),
            prediction_tokenizer = prediction_tokenizer,
            ref_answers          = batch["ref_qa_answer"],
        )

    else:
        # If we're on the sentiment task,
        # we return num sequences = 1.
        assert task_name == lib_utils.Task.SENTIMENT, task_name
        assert scores is None, scores
        assert (
            generation_kwargs["num_return_sequences"] == 1
        ), generation_kwargs["num_return_sequences"]

        outputs = raw_gen_outputs
        assert len(outputs.response_tensors) == batch_size, (
            len(outputs.response_tensors),
            batch_size,
        )

    lib_utils.named_barrier(f"{__file__}: {lib_utils.get_linenumber()}")

    outputs = lib_base_classes.BatchedUnrollReturn(
        response_tensors=unpad(
            responses=outputs.response_tensors,
            eos_token_id=prediction_tokenizer.eos_token_id,
            pad_token_id=prediction_tokenizer.pad_token_id,
        ),
        any_tokenizer=prediction_tokenizer,
    )

    return outputs


def rich_escape(value):
    return rich.markup.escape(str(value))


def check_qty_of_token_id(
    list_of_sequences: list[torch.Tensor], qty: int, token_id: int
):
    for sequence in list_of_sequences:
        qty_found = (sequence == token_id).long().sum().item()
        assert qty_found == qty, (qty_found, qty)


def check_max_qty_of_token_id(
    list_of_sequences: list[torch.Tensor], max_qty: int, token_id: int
):
    for sequence in list_of_sequences:
        qty_found = (sequence == token_id).sum().item()
        assert qty_found <= max_qty, (qty_found, max_qty)


def keep_good_one_generation(
    *,
    answer_extractor,
    difficulty_levels: list[int],
    ppo_batch_size: int,
    generations: lib_base_classes.BatchedUnrollReturn,
    num_return_seq: int,
    other_rewards: Optional[torch.Tensor],
    prediction_tokenizer: Optional[transformers.PreTrainedTokenizerBase],  # type: ignore
    ref_answers: Union[list[list[str]], torch.Tensor],
) -> lib_base_classes.BatchedUnrollReturn:
    """

    Return the index of the generation with the:
        - Max reward of the generations with good answers if there is one
        - Max reward of the generations with bad  answers otherwise

    """
    assert isinstance(generations, lib_base_classes.BatchedUnrollReturn
        ), type(generations)

    array_response_text = np.array(
        generations.response_text,
        dtype=object,
    ).reshape((ppo_batch_size, num_return_seq))

    device = generations.response_tensors[0].device
    response_tensors = prediction_tokenizer.pad(
            dict(input_ids=generations.response_tensors), 
            return_tensors="pt", 
            padding=True,
    )["input_ids"].reshape(ppo_batch_size, num_return_seq, -1)
    assert isinstance(ref_answers[0][0], str), type(ref_answers[0][0])
    selections = []

    table = rich.table.Table(
        "Difficulty Level",
        "Ref Comparable", 
        "Generated answers", 
        "Ratio Good",
        title=f"(RANK={RANK}) keep_good_one_generation", 
        show_lines=True,
        expand=True,
    )
    ratios_good = []


    breakpoint()
    for b_idx in range(ppo_batch_size):
        # Extract the reference answer
        ref_comparable = answer_extractor(ref_answers[b_idx])
        assert ref_comparable is not None, ref_answers[b_idx]

        # Extract the generated answers
        gen_answ_beams = [answer_extractor(gen) for gen in array_response_text[b_idx]]
        is_good_beams = [gen == ref_comparable for gen in gen_answ_beams]
        ratio_good_beams = np.mean(np.array(is_good_beams, dtype=np.float32))
        values_to_counts = collections.Counter(x for x in gen_answ_beams)

        # Inner table with the different extracted generations        
        inner_table = rich.table.Table(
            expand        = True,
            show_edge     = False,
            show_footer   = False,
            title_justify = "center",
        )
        inner_table.add_column("Extracted Gen.", justify="left")
        inner_table.add_column(         "Count", justify="left")

        for val, count in values_to_counts.most_common():
            color = "[green]" if val == ref_comparable else ""
            inner_table.add_row(f"{color}`{val}`", f"{color}{count}")
        
        table.add_row(
            rich_escape(difficulty_levels[b_idx]),
            f"`{rich_escape(ref_comparable)}`",
            inner_table,
            f"[green]{ratio_good_beams:0.2%}[/] or " + 
            f"[green]{np.sum(is_good_beams)}[/]/[green]{len(is_good_beams)}[/]",
        )

        if any(is_good_beams):
            # Return the good with the max other reward
            good_idx = [i for i, g in enumerate(is_good_beams) if g]
            
            if other_rewards:
                other_reward_tensor = torch.tensor(other_rewards["scores"][b_idx])
                good_rewards = other_reward_tensor[torch.tensor(good_idx)]                

                assert good_rewards.ndim == 1, (
                    f"{good_rewards.ndim = }, "
                    f"{good_rewards = }, "
                    f"{good_idx = }"
                )
                
                selection = good_idx[np.argmax(good_rewards)]
                assert other_reward_tensor[selection] == torch.max(good_rewards)

            else:
                assert False
                good_idx_id = torch.randint(0, len(good_idx), (1,))
                selection = good_idx[good_idx_id]
        else:
            # Sample
            if other_rewards:
                selection = torch.distributions.Categorical(
                    torch.tensor(other_rewards["scores"][b_idx]).softmax(-1)
                ).sample()
            else:
                assert False
                selection = np.random.randint(0, num_return_seq, (1,))[0]
                
        selections .append(       selection)
        ratios_good.append(ratio_good_beams)

    ratio_avg = np.mean(ratios_good)
    at_least_one_good = np.mean([ratio > 0 for ratio in ratios_good])
    table.caption = (
        f"Ratio Average: {ratio_avg:0.2%}, " +
        f"At least one good ratio: {at_least_one_good:0.2%}"
    )

    # if RANK == 0:
    #     # rich.print(table)

    selections = torch.tensor(selections)
    assert selections.shape == (ppo_batch_size,), f"{selections.shape = } {ppo_batch_size = }"
    output_generations_tensors = []

    for idx in range(ppo_batch_size):
        output_generations_tensors.append(
            response_tensors[idx][selections[idx]].detach().clone().to(device)
        )

    return lib_base_classes.BatchedUnrollReturn(
        any_tokenizer=prediction_tokenizer,
        response_tensors=output_generations_tensors,
    )

def print_table(
    *,
    call_source: str,
    extra_columns: Optional[dict[str, list]] = None,
    difficulty_levels: list[int],
    generation_kwargs,
    log_header: str,
    name: str,
    qty: int,
    queries: list[str],
    responses: list[str],
    rewards: list[float],
):
    if extra_columns is None:
        extra_columns = {}

    assert (
        len(rewards) == len(responses) == len(queries)
    ), f"{len(rewards) = } {len(responses) = } {len(queries) = }"

    for k, v in extra_columns.items():
        assert len(v) == len(rewards), f"{k = } {len(v) = } {len(rewards) = }"

    table = rich.table.Table(
        "Difficulty Level",
        "Training Query",
        "Training Response",
        "Reward",
        *extra_columns.keys(),
        show_lines=True,
        title=(
            f"{RANK = } - " + 
            f"{rich_escape(name)} - " +
            f"{rich_escape(log_header)} - " +
            f"<{call_source}>.lib_trl_utils.print_table(...): Samples:"
        ),
    )

    for difficulty_level, query, response, reward, *extra in itertools.islice(
        more_itertools.zip_equal(
            difficulty_levels,
            queries,
            responses,
            rewards,
            *extra_columns.values(),
        ),
        qty,
    ):
        # Escape brackets for rich
        rindex = query.rfind("Q:")
        if rindex == -1:
            rindex = 0
        
        table.add_row(
            rich_escape(difficulty_level),
            f"[black on white]{rich_escape(query[rindex:])}",
            f"[black on white]{rich_escape(response)}",
            f"{reward:0.3}",
            *map(rich_escape, extra),
        )

    if RANK == 0:
        kwargs_table = rich.table.Table("Key", "Value", title="Gen Kwargs")
        for k, v in generation_kwargs.items():
            kwargs_table.add_row(rich.markup.escape(str(k)), rich.markup.escape(str(v)))
        rich.print(kwargs_table)
        
        rich.print(table)


def check_all_start_with_token_id(tensors, token_id: int):
    """
    Description: Makes sure all the tensors in the list start with the same token id.

    Intent: All the decoder inputs of an encoder-decoder model should start with the same,
            fixed token id. This is a sanity check to make surethat they do.

    """
    starts = [r[0].item() for r in tensors]
    if not all(s == token_id for s in starts):
        raise ValueError(f"{token_id = }\n" f"{collections.Counter(starts) = }\n")


def check_all_end_with_token_id(tensors, token_id: int):
    """
    Description: Makes sure all the tensors in the list end with the same token id.

    Intent: All the decoder inputs of an encoder-decoder model should end with the same,
            fixed token id. This is a sanity check to make surethat they do.

    """
    ends = [r[-1].item() for r in tensors]
    if not all(s == token_id for s in ends):
        raise ValueError(f"{token_id = }\n" f"{collections.Counter(ends) = }\n")


def print_trainable_parameters(
    model: torch.nn.Module,
    do_print: bool = True,
) -> int:
    """

    Description: Prints the number of trainable parameters in the model, returns the number.

    Intent: Mostly of use with peft & LoRA, to see how many parameters are actually trainable.

    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if do_print:
        rich.print(
            f"[bold blue]({RANK}/{WORLD_SIZE}):[/] "
            f"trainable params: {rich_escape(trainable_params)} || "
            f"all params: {rich_escape(all_param)} || "
            f"trainable%: {100 * trainable_params / all_param}"
        )

    return trainable_params


def load_then_peft_ize_model(
    *,
    adapter_name: str,
    forward_tokenizer: transformers.PreTrainedTokenizer,
    just_device_map: bool,
    model_name: str,
    peft_config,
    peft_do_all_lin_layers: bool,
    precision,
    prediction_tokenizer: transformers.PreTrainedTokenizer,
    trust_remote_code: bool,
    use_peft: bool,
    we_pretrained_it: bool,
    adapter_path: Optional[str],
):
    
    if use_peft:
        lora_config = peft_config

    if just_device_map:
        assert not use_peft, "not written with that in mind"

    ###########################################################################
    # Model Class Specific Options
    ###########################################################################
    if use_peft:
        assert lora_config.task_type == peft.TaskType.CAUSAL_LM, lora_config.task_type
        
    transformers_cls = transformers.AutoModelForCausalLM  # type: ignore

    ###########################################################################
    # Init the Pre-Trained Model
    # -> Precision specific
    ###########################################################################
    assert isinstance(precision, torch.dtype) or precision == "auto", (
        f"{type(precision) = }")
    
    extra_args = {}
    if model_name.startswith("google/gemma-2-2b"):
        # extra_args["attn_implementation"] = "eager"
        extra_args["attn_implementation"] = "flash_attention_2"
    else:
        # sanity check
        assert not "gemma" in model_name, model_name
    
    pretrained_model = transformers_cls.from_pretrained(
        model_name,
        device_map="auto" if just_device_map else None,
        torch_dtype=precision,
        pad_token_id=prediction_tokenizer.pad_token_id,
        eos_token_id=prediction_tokenizer.eos_token_id,
        trust_remote_code=trust_remote_code,
        **extra_args,
    )

    if we_pretrained_it:
        pretrained_model = peft.PeftModel.from_pretrained(
            pretrained_model,
            model_id=adapter_path,
        )
        pretrained_model.merge_and_unload()

    assert not just_device_map or all(
        x.device.type == "cuda" 
        for x in pretrained_model.parameters()
    )

    ###########################################################################
    # Fix tokenizers for causal models
    ###########################################################################
    # Fix pretrained model to handle the new pad token
    assert len(forward_tokenizer) == len(prediction_tokenizer), (
        f"{len(forward_tokenizer) = }\n" 
        f"{len(prediction_tokenizer) = }"
    )
    assert forward_tokenizer.pad_token_id == prediction_tokenizer.pad_token_id, (
        f"{forward_tokenizer.pad_token_id = }\n"
        f"{prediction_tokenizer.pad_token_id = }"
    )
    assert forward_tokenizer.pad_token == prediction_tokenizer.pad_token, (
        f"{forward_tokenizer.pad_token = }\n"
        f"{prediction_tokenizer.pad_token = }"
    )
    assert forward_tokenizer.eos_token_id == prediction_tokenizer.eos_token_id, (
        forward_tokenizer.eos_token_id, prediction_tokenizer.eos_token_id
    )
    

    ###########################################################################
    # Peft-ize the model
    ###########################################################################
    if use_peft:
        if peft_do_all_lin_layers:
            assert False
            lora_config.target_modules = lib_peft_utils.find_all_linear_names(
                precision, 
                pretrained_model,
            )
        
        # if "microsoft/phi-2" == model_name:
        #     lora_config.target_modules = [
        #         "Wqkv",
        #         "out_proj",
        # ]  
        
        if model_name.startswith("google/gemma-2-2b"):
            lora_config.target_modules = [
                "q_proj", 
                "o_proj", 
                "k_proj", 
                "v_proj", 
                "gate_proj", 
                "up_proj", 
                "down_proj",
            ]

        pretrained_model = peft.get_peft_model(
            pretrained_model,
            lora_config,
            adapter_name=adapter_name,
        )

    return pretrained_model

def load_tokenizers(model_name):
    
    if pathlib.Path(model_name).exists() and pathlib.Path(model_name).is_dir():
        with open(pathlib.Path(model_name) / "adapter_config.json") as f:
            model_name = json.load(f)["base_model_name_or_path"]

    prediction_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, padding_side="left")
    forward_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, padding_side="right")

    for tokenizer in (forward_tokenizer, prediction_tokenizer):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    assert forward_tokenizer.eos_token_id == prediction_tokenizer.eos_token_id
    assert forward_tokenizer.pad_token_id == prediction_tokenizer.pad_token_id
    
    return dict(
        forward_tokenizer   =forward_tokenizer, 
        prediction_tokenizer=prediction_tokenizer,
    )


class MultiAdapterWrapper(torch.nn.Module):
    """
    The idea is to switch adapters whenever we use the model.
    """
    def __init__(
            self, 
            *, 
            trl_model_with_peft: trl.AutoModelForCausalLMWithValueHead,
            adapter_name: str, 
            peft_config: peft.PeftConfig,
            add_adapter: bool,
        ):
        super().__init__()
        
        self._adapter_name = adapter_name
        self._is_peft_model = True

        self._peft_config: peft.PeftConfig = peft_config
        self._peft_model: peft.PeftModel = self._trl_model.pretrained_model
        self._trl_model = trl_model_with_peft

        if add_adapter:
            self._peft_model.add_adapter(adapter_name, peft_config)


    def _activate(self):
        self._peft_model.set_adapter(self._adapter_name)
        return self

    @contextlib.contextmanager
    def disable_adapter(self):
        with self._peft_model.disable_adapter() as ctx:
            yield ctx

    def forward(self, *args, **kwargs):
        self._activate()
        return self._trl_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self._activate()
        return self._trl_model.generate(*args, **kwargs)
    
    @property
    def adapter_name(self):
        return self._name

    @property
    def active_adapter(self):
        return self._peft_model.active_adapter
    
    @property
    def config(self):
        return self._peft_model.config
    
    @property
    def is_peft_model(self):
        return self._is_peft_model

    @property
    def peft_config(self):
        return self._peft_config

    @property
    def pretrained_model(self):
        return self._trl_model.pretrained_model

    @property
    def v_head(self):
        return self._trl_model.v_head


def init_model(
    *,
    model_name: str,
    peft_do_all_lin_layers: bool,
    peft_config: Optional[dict[str, typing.Any]],
    precision,
    trust_remote_code,
    use_peft: bool,
    adapter_path: typing.Optional[str | pathlib.Path],
    we_pretrained_it: bool,
) -> tuple[
    typing.Union[
        trl.models.AutoModelForCausalLMWithValueHead,
        trl.models.AutoModelForSeq2SeqLMWithValueHead,
    ],
    transformers.PreTrainedTokenizerBase,  # type: ignore
    transformers.PreTrainedTokenizerBase,  # type: ignore
]:
    """

    Description: Initializes the model, tokenizer, and LoRA config, & does the precision stuff.

    Intent: The LoRA stuff & the precision stuff is repetitive.

    Currently:
        1. Init the pretrained model
            -> If int8, prepare for int8 training w/ peft.
        2. Peft-ize the pretrained model
        3. Initialize the trl ValueHead model from the peft-model

    .. That's what they use in the example:
    https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py#L194

    The question is, why does peft + value-model work with gpt and not t5?

    It further doesn't work with "prepare_for_int8" training, but that's not the question right now.

    """


    assert precision, precision

    ###########################################################################
    # Tokenizer stuff
    ###########################################################################
    tmp_tokenizers = load_tokenizers(model_name)
    forward_tokenizer    = tmp_tokenizers["forward_tokenizer"]
    prediction_tokenizer = tmp_tokenizers["prediction_tokenizer"]
    del tmp_tokenizers

    ###########################################################################
    # HF Raw Model Stuff
    ###########################################################################
    pretrained_model = load_then_peft_ize_model(
        adapter_name           = "default",
        forward_tokenizer      = forward_tokenizer,
        just_device_map        = False,
        model_name             = model_name,
        peft_config            = peft_config,
        peft_do_all_lin_layers = peft_do_all_lin_layers,
        precision              = precision,
        prediction_tokenizer   = prediction_tokenizer,
        trust_remote_code      = trust_remote_code,
        use_peft               = use_peft,
        we_pretrained_it       = we_pretrained_it,
        adapter_path           = adapter_path,
    )

    model = trl.AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model,

    )
    model.gradient_checkpointing_disable = (
        model.pretrained_model.gradient_checkpointing_disable)
    model.gradient_checkpointing_enable = (
        model.pretrained_model.gradient_checkpointing_enable)

    if precision in (
        torch.float16,
        torch.bfloat16,
    ):
        dtype = (
            torch.float16
            if precision == torch.float16
            else torch.bfloat16)
        
        model.v_head.to(dtype=dtype)
    
    output = print_trainable_parameters(model, True)
    assert output > 0

    return dict(
        model=model, 
        forward_tokenizer=forward_tokenizer, 
        prediction_tokenizer=prediction_tokenizer,
    )


class CurriculumSchedule:
    @dataclasses.dataclass
    class CurriculumEntry:
        """
        Only steps are comparable
        """

        step: int
        enabled_difficulties: set[int]

        def __post_init__(self):
            # Make sure the proportions sum to approx 1
            self.enabled_difficulties = set(self.enabled_difficulties)
            self.check()

        def check(self) -> None:
            # Check types
            assert isinstance(self.enabled_difficulties, set), (
                type(self.enabled_difficulties)
            )
            assert isinstance(self.step, int), type(self.step)
            assert all(isinstance(x, int) for x in self.enabled_difficulties), (
                self.enabled_difficulties
            )

    CE = CurriculumEntry

    def __init__(self, entries: list[CE]=None, literals=None):
        # Make sure the indices are strictly increasing
        # and start at 0
        assert entries or literals, (entries, literals)
        assert not (entries and literals), (entries, literals)
        if not entries:
            assert isinstance(literals, list), literals
            assert all(isinstance(x, (list, tuple)) for x in literals), literals
            assert all(len(x) == 2 for x in literals), literals
            entries = [
                self.CE(step=step, enabled_difficulties=enabled_difficulties) 
                for step, enabled_difficulties in literals
            ]
            del literals

        self._entries = entries
        self.check()
    
    def __call__(self, step: int) -> CurriculumEntry:
        assert isinstance(step, int), type(step)

        # Index of first <= step.
        index = bisect.bisect_right(
            [x.step for x in self._entries],
            step
        ) - 1
        entry = self[index]

        assert entry.step <= step, (entry.step, step)
        not_last = index < len(self._entries) - 1

        if not_last:
            assert step < self[index + 1].step

        return entry

    def __len__(self) -> int:
        return len(self._entries)
    
    def __getitem__(self, index) -> CurriculumEntry:
        return self._entries[index]

    def __bool__(self) -> bool:
        return bool(self._entries)
    
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._entries})>"

    def __str__(self) -> str:
        return self.__repr__()

    def check(self) -> None:
        assert all(
            isinstance(x, self.CE) 
            for x in self._entries), self._entries
        assert all(
            x.step < y.step 
            for x, y in zip(self._entries, self._entries[1:])
        ), self._entries
        assert self._entries[0].step == 0, self._entries[0].step
        
    def check(self):
        for i, entry in enumerate(self._entries):
            assert entry.step == i, (entry.step, i)
            assert isinstance(entry.enabled_difficulties, set), type(entry.enabled_difficulties)
            assert all(
                isinstance(x, int) 
                for x in entry.enabled_difficulties
            ), type(entry.enabled_difficulties)


def _display_table_return_sequences(
    *,
    ref_text_answers,
    batch_size,
    difficulty_levels,
    prediction_tokenizer,
    query_tensors,
    responses,
    sequences_scores,
):
    # Prep info for the table
    assert isinstance(query_tensors, (list, tuple, np.ndarray, torch.Tensor)), type(query_tensors).mro()
    
    # Create the table
    for batch_idx in range(batch_size):
        table = rich.table.Table(
            title       = f"{RANK} - Batch Unroll", 
            show_lines  = True,
            show_header = False,
            show_edge   = True,
            highlight   = True,
        )

        # Sort by likelihood
        sorted_seq_idx = sorted(
            range(len(sequences_scores[batch_idx])),
            key     = lambda seq_idx: sequences_scores[batch_idx][seq_idx],
            reverse = True,
        )

        # Decode the generated text
        output_text   = prediction_tokenizer.batch_decode(
            responses[batch_idx], 
            skip_special_tokens=False,
        )

        if max(sorted_seq_idx) > len(output_text) - 1:
            raise ValueError(f"{sorted_seq_idx}, {len(output_text) - 1}")
        
        sorted_texts  = [output_text                [seq_idx] for seq_idx in sorted_seq_idx]
        sorted_scores = [sequences_scores[batch_idx][seq_idx] for seq_idx in sorted_seq_idx]

        for (seq_idx, val), score in more_itertools.zip_equal(
                enumerate(sorted_texts), sorted_scores
            ):
            val == ref_text_answers[batch_idx]
            table.add_row(
                ("[bold green]Selected[/]\n" if seq_idx == 0 else "") +
                f"{difficulty_levels[batch_idx] = } \n" + 
                f"{ref_text_answers [batch_idx] = }",
                f"{batch_idx                    = } \n" +
                f"{seq_idx                      = } \n" +
                f"{score.item()                 = :0.3f}",
                str(val).strip(),
            )
    
        # if RANK == 0:
        #     rich.print(table)


def fix_sequence_scores(*, model_outputs, prediction_tokenizer):
    ###################################################################
    # Obtain a "sequences_scores" from a "scores" tensor.
    #
    # For when scores exists and sequences_scores doesn't. "scores" is then removed.
    # 
    # Ignores all eos == pad tokens except a single one at the end of the sequence,
    # if there is one.
    ###################################################################
    if "sequences_scores" not in model_outputs:

        assert prediction_tokenizer.pad_token_id == prediction_tokenizer.eos_token_id, (
            "This code assumes that the pad token is the eos token.")
        
        lm_logits = torch.stack(model_outputs["scores"], dim=1).contiguous()
        
        # Compute the score masking all of the tokens after the first eos token.
        # We need to be able to modify the labels, so we clone them.
        labels = model_outputs["sequences"].clone()[:, -lm_logits.shape[1]:].contiguous()
        pad_mask = (labels == prediction_tokenizer.pad_token_id).long()
        for i in range(labels.shape[0]):
            non_zero = pad_mask[i].nonzero()
            if len(non_zero):
                bound = non_zero[0].item()
                labels[i, bound:] = -100

        assert labels.shape[:2] == lm_logits.shape[:2], (labels.shape, lm_logits.shape)

        loss_fct = torch.nn.CrossEntropyLoss()
        losses = []
        for i in range(labels.shape[0]):
            losses.append(loss_fct(lm_logits[i], labels[i],))
        model_outputs["sequences_scores"] = torch.tensor(losses).to(lm_logits.device)    
    
    del model_outputs["scores"]
    assert "scores" not in model_outputs, model_outputs.keys()
    assert "sequences_scores" in model_outputs, model_outputs.keys()

    return model_outputs


def validate_responses(*, responses, prediction_tokenizer):
    ###########################################################################
    # Make sure that the responses are valid:
    # - They should not contain any pad tokens, except if it's also an 
    #   eos token.
    # - They should not contain more than one eos token, and if they do, 
    #   it should be at the end.
    # This code should not break with SoS tokens, as we removed the prompt.
    ###########################################################################
    for response in responses:
        if not isinstance(response, torch.Tensor):
            assert isinstance(response, list), type(response)
            assert isinstance(response[0], int), type(response[0])
            

        qty_pad_token_id = (
            response == prediction_tokenizer.pad_token_type_id
        ).long().sum().item()
        assert qty_pad_token_id <= 1, qty_pad_token_id
        
        if qty_pad_token_id:
            # There should only be a pad token id if it's also the eos token,
            # and it should be at the very end.
            assert prediction_tokenizer.eos_token_id == prediction_tokenizer.pad_token_id, (
                prediction_tokenizer.eos_token_id, prediction_tokenizer.pad_token_id,)
            assert response[-1] == prediction_tokenizer.eos_token_id, response
        
        qty_eos_tokens = (response == prediction_tokenizer.eos_token_id).long().sum().item()
        assert qty_eos_tokens <= 1, qty_eos_tokens
        if qty_eos_tokens:
            # If there is an eos token, it should be at the end.
            assert response[-1] == prediction_tokenizer.eos_token_id, response

    return responses


def remove_prompt_and_padding(
        *,
        tokenized,
        model_outputs,
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
        forward_tokenizer: transformers.PreTrainedTokenizerBase,
        fn_fix_few_shots: Optional[typing.Callable],
    ) -> dict[str, list]:

    outputs = collections.defaultdict(list)

    model_outputs = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in model_outputs.items() 
        if not isinstance(v, tuple)
    }
    
    ###########################################################################
    # Remove the prompt.
    ###########################################################################
    # They all have the same prompt, ,so we can just cut the first 
    # `query_len` tokens.
    query_len = tokenized["input_ids"].shape[1]
    raw_sequences = model_outputs["sequences"]
    model_outputs["sequences"] = model_outputs["sequences"][:, query_len:]

    ###########################################################################
    # Fix the few shots.
    ###########################################################################
    if fn_fix_few_shots:
        model_outputs["sequences"] = fn_fix_few_shots(
            input_ids=tokenized["input_ids"],
            raw_gen_outputs=model_outputs["sequences"],
            forward_tokenizer=forward_tokenizer,
        )

    ###########################################################################
    # Remove the padding
    ###########################################################################
    assert len(model_outputs["sequences"]), model_outputs.keys()
    masks = model_outputs["sequences"] == prediction_tokenizer.pad_token_id
    
    for id_in_sub_batch in range(len(masks)):
        mask = masks[id_in_sub_batch]
        seq = model_outputs["sequences"][id_in_sub_batch]
        
        # Keep one eos/pad token if there are any at the end.
        if prediction_tokenizer.eos_token_id in seq:
            end = torch.nonzero(mask, as_tuple=False)[0, 0].item() + 1
        else:
            end = None

        new_seq = seq[:end]

        assert (
            prediction_tokenizer.eos_token_id == 
            prediction_tokenizer.pad_token_id
        ), (
            prediction_tokenizer.eos_token_id, 
            prediction_tokenizer.pad_token_id,
        )
        
        # CHECKS
        # We want to make sure that the outputs aren't somehow left padded
        first_token_is_eos_or_pad = new_seq[0] == prediction_tokenizer.eos_token_id

        # We want to make sure that there is only one eos token at the end.
        assert not (first_token_is_eos_or_pad and len(new_seq) > 1), (new_seq, )

        if not len(new_seq) > 1:
            decoded_new_seq = prediction_tokenizer.decode(new_seq)
            decoded_raw_sequences = prediction_tokenizer.decode(raw_sequences[id_in_sub_batch])

        if len(new_seq) > 1:
            assert new_seq[-2] != prediction_tokenizer.pad_token_id, (
                new_seq[-2], prediction_tokenizer.pad_token_id,)
    
        # There sould be a total of 1 eos/pad tokens at most.
        qty_eos_tokens = (
            new_seq == prediction_tokenizer.eos_token_id
        ).long().sum().item()

        assert qty_eos_tokens <= 1, (new_seq, prediction_tokenizer.eos_token_id,)

        # Shorten everything by how much we cut.
        for key, value in model_outputs.items():
            assert key in {"sequences", "sequences_scores", "beam_indices"}, key
            sample = value[id_in_sub_batch]

            if   isinstance(sample, torch.Tensor) and sample.ndim == 1:
                # Cut the padding
                assert key in {"sequences", "beam_indices"}, key
                outputs[key].append(sample[:end])

            elif isinstance(sample, torch.Tensor) and sample.ndim == 0:
                assert key == "sequences_scores", key
                # Not changed in any way
                outputs[key].append(sample)
                
            else:
                raise ValueError(
                    (
                        key, 
                        (sample.shape, sample.ndim) 
                        if isinstance(sample, torch.Tensor) else None, 
                        type(sample).mro(),
                    )
                )

    return dict(outputs)


def batched_unroll(
    *,
    accelerator,
    accelerated_model,
    forward_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    difficulty_levels,
    generation_kwargs: dict[str, typing.Any],
    generation_batch_size: int,
    post_process_gen_fewshots_fn,
    prediction_tokenizer: transformers.PreTrainedTokenizerBase,  # type: ignore
    query_tensors: list[torch.Tensor],
    ref_text: list[str],
    task_name: lib_utils.Task,
    use_few_shots: bool,
) -> lib_base_classes.BatchedUnrollReturn:
    
    """
    Batched generation.
        - Validate and prepare the inputs.
        - Unwrap the model.
        - Split batches to sub-batches of size generation_batch_size.
        - Iterate over the sub-batches, and generate the sequences.

    Arguments:
        - accelerator: The Huggingface Accelerate Accelerator.
        - accelerated_model: The model wrapped by Huggingface Accelerate.
        - ref_text: The reference text. Only used for display or logging.
        - difficulty_levels: The difficulty levels of the queries.
        - generation_kwargs: The Huggingface generation kwargs.
        - 
        
    """

    batch_size = len(query_tensors)
    num_seqs = generation_kwargs.get("num_return_sequences", 1)
    assert isinstance(query_tensors, list) or (isinstance(query_tensors, torch.Tensor) and query_tensors.ndim == 2), type(query_tensors)
    assert all(isinstance(x, torch.Tensor) for x in query_tensors), type(query_tensors)

    ###########################################################################
    # Validate the inputs, cleanup gen-kwargs
    ###########################################################################
    assert prediction_tokenizer.pad_token_id == prediction_tokenizer.eos_token_id, (
        prediction_tokenizer.pad_token_id, prediction_tokenizer.eos_token_id
    )

    # Make sure the generation batch size is not too large.
    if generation_batch_size is None or generation_batch_size > len(query_tensors):
        generation_batch_size = len(query_tensors)

    # We are generating, so we need to set the padding side to the left.
    assert prediction_tokenizer.padding_side == "left", (
        prediction_tokenizer.padding_side)
    
    # We copy the generation kwargs to avoid modifying the original.
    generation_kwargs = generation_kwargs.copy()
    
    # The following are set in the call. 
    if "output_scores" in generation_kwargs:
        del generation_kwargs["output_scores"]

    if "return_dict_in_generate" in generation_kwargs:
        del generation_kwargs["return_dict_in_generate"]
    
    # Generate. We have a sub-batch for generation, to allow for larger PPO batch sizes.
    model = accelerator.unwrap_model(accelerated_model)

    ###########################################################################
    # Iterate on sub-batches
    ###########################################################################
    outputs = collections.defaultdict(list)
    for i in tqdm.trange(
        0, 
        len(query_tensors), 
        generation_batch_size, 
        desc="Batched Unroll",
    ):        
    
        #######################################################################
        # Generate the sequences
        #######################################################################
        tokenized = prediction_tokenizer.pad(
            dict(input_ids=query_tensors[i : i + generation_batch_size]),
            return_tensors="pt",
            padding=True,
            
        ).to(accelerator.device)

        is_training_pre_gen = model.training

        assert tokenized["input_ids"].shape[0] == min(
            generation_batch_size, len(query_tensors) - i), (
            tokenized["input_ids"].shape[0], 
            generation_batch_size, 
            len(query_tensors) - i,
        )
        

        model_outputs = model.eval().generate(
            tokenized["input_ids"],
            return_dict_in_generate = True, 
            output_scores           = True,
            pad_token_id            = prediction_tokenizer.pad_token_id,
            **generation_kwargs,
        )
        
        expected_num_sequences = (
            len(tokenized["input_ids"]) * 
            generation_kwargs["num_return_sequences"]
        )

        assert model_outputs["sequences"].shape[0] == expected_num_sequences, (
            model_outputs["sequences"].shape, 
            len(tokenized["input_ids"]), 
            generation_kwargs["num_return_sequences"],
        )
        
        model.train(is_training_pre_gen)
        model_outputs = dict(**model_outputs)
        model_outputs = fix_sequence_scores(
            model_outputs        = model_outputs, 
            prediction_tokenizer = prediction_tokenizer
        )

        #######################################################################
        # Remove the prompt & padding (keep an EOS token).
        #######################################################################
        # Acts in place
        partial_outputs = remove_prompt_and_padding(
            tokenized            = tokenized,
            model_outputs        = model_outputs,
            prediction_tokenizer = prediction_tokenizer,
            forward_tokenizer    = forward_tokenizer,
            fn_fix_few_shots     = post_process_gen_fewshots_fn if use_few_shots else None,
        )

        for key, value in partial_outputs.items():
            assert len(value) == expected_num_sequences, (key, len(value), expected_num_sequences)
            outputs[key].extend(value)

    ###########################################################################
    # Place things in an array with the shape:
    #   (batch_size, num_seqs, VARIABLE_LENGTH_SEQUENCES)
    #
    # They have the object type to allow for variable length sequences.
    # We use this instead of nested / jagged tensors, potentially only out of
    # ignorance.
    ###########################################################################
    
    # temp_sequences = np.ndarray(dtype=object, shape=(batch_size, num_seqs,))
    # for i in range(batch_size):
    #     for j in range(num_seqs):
    #         temp_sequences[i, j] = outputs["sequences"][i * num_seqs + j]

    # outputs["sequences"] = temp_sequences
    # del temp_sequences
    # outputs["sequences_scores"] = torch.tensor(
    #     outputs["sequences_scores"]).reshape(batch_size, num_seqs,)
    # responses = outputs["sequences"]

    ###########################################################################
    #
    # Remove the repeating outputs.
    # 
    # Few-shot generation causes the model to create other fake questions
    # that it answers to as well. We only want the real questions, and
    # the associated outputs.
    #
    # We leave it to the datasets to provide their own way of removing
    # everything after the answer to the real question.
    #
    ###########################################################################

    # if task_name == lib_utils.Task.MAIN:
    #     if use_few_shots:
    #         responses = post_process_gen_fewshots_fn(
    #             raw_gen_outputs = responses, 
    #             any_tokenizer   = prediction_tokenizer,
    #         )

    #         validate_responses(
    #             prediction_tokenizer = prediction_tokenizer,
    #             responses            = responses, 
    #         )

    # else:
    #     assert post_process_gen_fewshots_fn is None
    
    responses = outputs["sequences"]
    return_obj = lib_base_classes.BatchedUnrollReturn(
        response_tensors = responses,
        any_tokenizer    = prediction_tokenizer,
    )

    sequences_scores = None
    if task_name != lib_utils.Task.SENTIMENT:
        assert "sequences_scores" in outputs, outputs.keys()

        responses = np.ndarray(shape=(batch_size, num_seqs,), dtype=object)
        for i in range(batch_size):
            for j in range(num_seqs):
                responses[i, j] = outputs["sequences"][i * num_seqs + j]

        sequences_scores = np.array(
            outputs["sequences_scores"],
            dtype=float
        ).reshape(batch_size, num_seqs)

        #######################################################################
        # We have multiple returned equences: Create the table
        #######################################################################
        _display_table_return_sequences(
            ref_text_answers     = ref_text,
            batch_size           = batch_size,
            difficulty_levels    = difficulty_levels,
            prediction_tokenizer = prediction_tokenizer,
            query_tensors        = query_tensors,
            responses            = responses,
            sequences_scores     = sequences_scores,
        )

    return return_obj, sequences_scores


def unpad(responses, pad_token_id, eos_token_id):
    # Remove the padding
    final_responses = []
    for response in responses:
        init_len = len(response)
        # Remove the padding:
        response = response[response != pad_token_id]
        modified = init_len != len(response)

        # If we have modified the len, then we are guaranteed that the
        # sentence ends, and so there needs to be an eos token.
        # We might have cut it off if pad_token_id == eos_token_Id,
        # so we need to add it if it's not there.
        if modified and (
            not response.shape[0] or 
            response[-1] != eos_token_id
        ):
            response = torch.cat([
                response, 
                torch.tensor([eos_token_id]).to(response.device),
            ])
            
        final_responses.append(response)
    return final_responses
