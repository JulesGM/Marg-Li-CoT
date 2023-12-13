#!/usr/bin/env python
from __future__ import annotations
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
# os.environ["DATASETS_VERBOSITY"] = "warning"
# os.environ["WANDB_SILENT"] = "true"
# os.environ["NCCL_DEBUG"] = "WARN"

DETERMINISTIC = False
if DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import collections
import enum
import getpass
import itertools
import logging
import pathlib
import random
import typing
from typing import Any, Optional, Union

import accelerate
import datasets
import fire
import more_itertools as mit
import numpy as np
import peft
import rich
import rich.console
import rich.highlighter
import rich.logging
import rich.markup
import rich.status
import rich.table
import rich.traceback 
import torch
import torch.backends
import torch.backends.cuda
import torch.backends.cudnn
import torch.utils
import torch.utils.data
import torch.utils.data.sampler
import transformers
import trl
import trl_fork
import wandb
import accelerate.utils
from tqdm import tqdm

import lib_base_classes
import lib_data
import lib_eval
import lib_trl_utils
import lib_utils

datasets.disable_caching()
rich.traceback.install(
    console=rich.console.Console(
        force_terminal=True
))

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "0"))
RANK = int(os.environ.get("RANK", "0"))
LOGGER = logging.getLogger(__name__)

np.random.seed(0)
random.seed(1)
torch.manual_seed(2)
torch.cuda.manual_seed_all(3)
trl.set_seed(4)
trl_fork.set_seed(4)

torch.backends.cuda.matmul.allow_tf32 = not DETERMINISTIC
torch.backends.cudnn.allow_tf32 = not DETERMINISTIC
torch.use_deterministic_algorithms(DETERMINISTIC)

DEFAULT_LOG_STATS_VERBOSE = True
DEFAULT_REWARD_VERBOSE = False

##############################################################################
##############################################################################
DEFAULT_USE_FEW_SHOTS = True
DEFAULT_GEN_KWARGS = dict(
    min_new_tokens       = 5,
    num_return_sequences = 14,
    num_beams            = 14,

    ##########################################
    synced_gpus        = os.getenv("ACCELERATE_DEEPSPEED_ZERO_STAGE", "") == "3",
    temperature        = 1.,
    # top_k              = 0.0,
    # top_p              = 1.0,
    use_cache          = True,
    repetition_penalty = 1,
    ##########################################
)

DEFAULT_KL_PENALTY_MODE = "kl"

CS = lib_trl_utils.CurriculumSchedule
CE = lib_trl_utils.CurriculumSchedule.CE


DEFAULT_CURRICULUM_SCHEDULE = [
    (0,   {
        1: 1.,
    }),
    # (100, {0: 0.66, 1: 0.34}),
    # (200, {0: 0.50, 1: 0.50}),
]

DEFAULT_USE_CURRICULUM = True
DEFAULT_TRL_LIBRARY_MODE = lib_utils.TrlLibraryMode.TRL
DEFAULT_TASK_NAME: str = lib_utils.Task.MAIN
DEFAULT_EVAL_EVERY: int = 0
DEFAULT_VALUE_MODEL_PRETRAIN_AMOUNT: int = 1

class TrainingState(enum.Enum):
        REGULAR_TRAINING = "regular_training"
        VALUE_MODEL_PRETRAINING = "value_model_pretraining"


if DEFAULT_TASK_NAME == lib_utils.Task.MAIN:
    DEFAULT_REWARD_TYPE = lib_utils.RewardChoices.EXACT_MATCH
    DEFAULT_DATASET_NAME = lib_data.DatasetChoices.ARITHMETIC
    DEFAULT_WANDB_PROJECT: str = f"rl_{DEFAULT_DATASET_NAME.value}"

    # -------------------------------------------------------
    #########################################################
    DEFAULT_GEN_KWARGS["do_sample"] = False
    # DEFAULT_MODEL_NAME = "EleutherAI/gpt-j-6B"; DEFAULT_BATCH_SIZE: int = 2; DEFAULT_MINI_BATCH_SIZE: int = 1
    DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"; DEFAULT_BATCH_SIZE: int = 2; DEFAULT_MINI_BATCH_SIZE: int = 1
    # DEFAULT_MODEL_NAME = "EleutherAI/pythia-70m-deduped"; DEFAULT_BATCH_SIZE: int = 2; DEFAULT_MINI_BATCH_SIZE: int = 2
    
    DEFAULT_ANSWER_ONLY = True 
    DEFAULT_GEN_KWARGS["max_new_tokens"] = 10 if DEFAULT_ANSWER_ONLY else 200

    DEFAULT_INFERENCE_BATCH_SIZE: int = DEFAULT_BATCH_SIZE
    DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 1
    DEFAULT_CAUSAL_QUESTION_PREFIX: str = None
    DEFAULT_CAUSAL_QUESTION_SUFFIX: str = None
    DEFAULT_PEFT_QLORA_MODE = False
    DEFAULT_PRECISION = lib_utils.ValidPrecisions.bfloat16

    DEFAULT_ANSWER_ONLY_PATH = (
        # "/network/scratch/g/gagnonju/saved_scratchpad_gen_outputs/chatgpt-3.5-commonsenseqa-scratchpads/cond-on-answers/commonsenseqa.chatgpt"
        "/network/scratch/g/gagnonju/saved_scratchpad_gen_outputs/chatgpt-3.5-commonsenseqa-scratchpads/not-cond-on-answers/commonsenseqa.chatgpt"
    )
    #########################################################
    # -------------------------------------------------------


    DEFAULT_PEFT_DO_ALL_LIN_LAYERS = True
    assert DEFAULT_EVAL_EVERY == 0 or DEFAULT_EVAL_EVERY >= (
        DEFAULT_GRADIENT_ACCUMULATION_STEPS // WORLD_SIZE
    ), (
        f"DEFAULT_EVAL_EVERY ({DEFAULT_EVAL_EVERY}) must be >= "
        f"DEFAULT_GRADIENT_ACCUMULATION_STEPS "
        f"({DEFAULT_GRADIENT_ACCUMULATION_STEPS})"
    )

    DEFAULT_INFERENCE_GEN_KWARGS = DEFAULT_GEN_KWARGS.copy()
    DEFAULT_INFERENCE_GEN_KWARGS["num_beams"] = 1
    DEFAULT_INFERENCE_GEN_KWARGS["num_return_sequences"] = 1
    DEFAULT_INFERENCE_GEN_KWARGS["do_sample"] = False
    # We could use a custom batch size too.


elif DEFAULT_TASK_NAME == lib_utils.Task.SENTIMENT:
    assert False
    DEFAULT_WANDB_PROJECT: str = "sentiment"
    DEFAULT_GEN_KWARGS["max_new_tokens"] = 20
    DEFAULT_GEN_KWARGS["min_new_tokens"] = 4
    DEFAULT_GEN_KWARGS["num_return_sequences"] = 1
    DEFAULT_GEN_KWARGS["do_sample"] = True
    DEFAULT_EVAL_BATCH_SIZE: int = 24
    DEFAULT_MINI_BATCH_SIZE: int = 24
    DEFAULT_BATCH_SIZE: int = 24
    DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 1

    DEFAULT_PEFT_QLORA_MODE = True
    DEFAULT_REWARD_TYPE: typing.Optional[str] = None

    DEFAULT_PRECISION = lib_utils.ValidPrecisions.bfloat16
    # DEFAULT_MODEL_NAME: str                   = "huggyllama/llama-65b"
    DEFAULT_MODEL_NAME: str = "tiiuae/falcon-40b-instruct"
    DEFAULT_CAUSAL_QUESTION_PREFIX: str = ""
    DEFAULT_CAUSAL_QUESTION_SUFFIX: str = ""

    DEFAULT_DATASET_NAME = lib_data.DatasetChoices.SENTIMENT
    DEFAULT_INFERENCE_BATCH_SIZE: int = 96
    DEFAULT_INFERENCE_GEN_KWARGS = DEFAULT_GEN_KWARGS.copy()

else:
    raise ValueError(f"Unknown task name: {DEFAULT_TASK_NAME}")


##############################################################################
##############################################################################
DEFAULT_EVAL_QTY: int = 1200
DEFAULT_NUM_EPOCHS: int = 10
DEFAULT_USE_PEFT: bool = True

DEFAULT_LEARNING_RATE: float = 1.41e-4

DEFAULT_PEFT_CONFIG = dict(
    bias           = "none",
    inference_mode = False,
    lora_dropout   = 0.,
    lora_alpha     = 16,
    r              = 16,
)

DEFAULT_INPUT_MAX_LENGTH = 115
# get username with os

if RANK == 0:
    DEFAULT_WANDB_DIR = pathlib.Path("/tmp") / f"{getpass.getuser()}_{os.getpid()}"
else:
    DEFAULT_WANDB_DIR = None
DEFAULT_ARITHMETIC_DATASET_ROOT_FOLDER_DIR = "/home/mila/g/gagnonju/Marg-Li-CoT/with_trl/libs_data/arithmetic/"


def main(
    name: Optional[str],
    *,
    answer_only:                                 bool = DEFAULT_ANSWER_ONLY,
    answer_only_path:                             str = DEFAULT_ANSWER_ONLY_PATH,
    arithmetic_dataset_root_folder_dir                = DEFAULT_ARITHMETIC_DATASET_ROOT_FOLDER_DIR,
    batch_size:                                   int = DEFAULT_BATCH_SIZE,
    causal_question_prefix:                       str = DEFAULT_CAUSAL_QUESTION_PREFIX,
    causal_question_suffix:                       str = DEFAULT_CAUSAL_QUESTION_SUFFIX,
    curriculum_schedule                               = DEFAULT_CURRICULUM_SCHEDULE,
    dataset_name:                                 str = DEFAULT_DATASET_NAME,
    extr_arith_ignore_one_line:                  bool = True,
    eval_every:                                   int = DEFAULT_EVAL_EVERY,
    eval_subset_size:                             int = DEFAULT_EVAL_QTY,
    generation_kwargs:                 dict[str, Any] = DEFAULT_GEN_KWARGS,
    gradient_accumulation_steps:                  int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    input_max_length:                             int = DEFAULT_INPUT_MAX_LENGTH,
    inference_gen_kwargs:              dict[str, Any] = DEFAULT_INFERENCE_GEN_KWARGS,
    inference_batch_size:                         int = DEFAULT_INFERENCE_BATCH_SIZE,
    just_metrics:                                bool = False,
    kl_penalty_mode                                   = DEFAULT_KL_PENALTY_MODE,
    learning_rate:                              float = DEFAULT_LEARNING_RATE,
    mini_batch_size:                              int = DEFAULT_MINI_BATCH_SIZE,
    model_name:                                   str = DEFAULT_MODEL_NAME,
    peft_config_dict:        Optional[dict[str, Any]] = DEFAULT_PEFT_CONFIG,
    peft_do_all_lin_layers:                      bool = DEFAULT_PEFT_DO_ALL_LIN_LAYERS,
    precision                                         = DEFAULT_PRECISION,
    reward_type: None | str | lib_utils.RewardChoices = DEFAULT_REWARD_TYPE,
    task_name:                                    str = DEFAULT_TASK_NAME,
    trl_library_mode:        lib_utils.TrlLibraryMode = DEFAULT_TRL_LIBRARY_MODE,
    use_curriculum:                              bool = DEFAULT_USE_CURRICULUM,
    use_peft:                                    bool = DEFAULT_USE_PEFT,
    use_few_shots:                                int = DEFAULT_USE_FEW_SHOTS,
    wandb_dir:                                    str = DEFAULT_WANDB_DIR,
    wandb_project:                                str = DEFAULT_WANDB_PROJECT,
):
    args = locals().copy()
    precision = lib_utils.ValidPrecisions(precision)  # type: ignore

    # Display command line args
    if RANK == 0:
        lib_utils.readable(args, "Command line args")

    # Check some command line args
    assert kl_penalty_mode in {"kl", "abs", "mse", "full"}, kl_penalty_mode
    task_name = lib_utils.Task(task_name)
    dataset_name = lib_data.DatasetChoices(dataset_name)
    trl_library_mode = lib_utils.TrlLibraryMode(trl_library_mode)
    trl_library = lib_utils.TRL_LIBRARIES[trl_library_mode]
    
    if use_curriculum:
        assert curriculum_schedule
        if not isinstance(curriculum_schedule, CS):
            curriculum_schedule = CS(literals=curriculum_schedule)
        else:
            # Only in else because CS.__init__ already calls check
            curriculum_schedule.check()

        # Just test a few
        curriculum_schedule(0)
        curriculum_schedule(1)
        all_levels = sorted(set(itertools.chain.from_iterable(
            entry.proportions.keys() for entry in curriculum_schedule
        )))

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[rich.logging.RichHandler(markup=True)],
        format=f"[{RANK}/{WORLD_SIZE}] %(funcName)s:%(lineno)d - %(message)s",
    )
    logging.getLogger("transformers").setLevel(logging.ERROR)

    if RANK != 0:
        logging.getLogger("peft_lora").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)

    ###########################################################################
    # Find the type of model we are using
    ###########################################################################
    hf_config = transformers.AutoConfig.from_pretrained(  # type: ignore
        model_name, 
    )
    
    if hf_config.is_encoder_decoder:
        assert False
        assert causal_question_prefix == "", (
            causal_question_prefix,
            causal_question_suffix,
        )
        assert causal_question_suffix == "", (
            causal_question_suffix,
            causal_question_suffix,
        )

    assert peft_config_dict is not None
    assert "task_type" not in peft_config_dict, peft_config_dict

    if not hf_config.is_encoder_decoder:
        peft_config_dict["task_type"] = peft.TaskType.CAUSAL_LM
    elif hf_config.is_encoder_decoder:
        peft_config_dict["task_type"] = peft.TaskType.SEQ_2_SEQ_LM
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    if not peft_do_all_lin_layers:
        assert "target_modules" not in peft_config_dict, peft_config_dict
        peft_config_dict[
            "target_modules"
        ] = peft.utils.other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
            hf_config.model_type
        ]

    accelerator_kwargs = dict(
        kwargs_handlers=[accelerate.utils.DistributedDataParallelKwargs(
                find_unused_parameters=False
        )])

    ppo_config_dict = dict(
        accelerator_kwargs          = accelerator_kwargs,
        batch_size                  = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        learning_rate               = learning_rate,
        log_with                    = "wandb",
        mini_batch_size             = mini_batch_size,
        model_name                  = model_name,
        kl_penalty                  = kl_penalty_mode,   
    )

    trl_config: trl.PPOConfig = trl_library.PPOConfig(**ppo_config_dict)

    if RANK == 0:
        lib_utils.readable(vars(trl_config), title="ppo_config")

    if task_name == lib_utils.Task.MAIN:
        reward_type = lib_utils.RewardChoices(reward_type)

    if RANK == 0:
        wandb_dir = pathlib.Path(wandb_dir)
        if not wandb_dir.exists():
            wandb_dir.mkdir(parents=True)
            assert wandb_dir.exists(), wandb_dir
            
        wandb.init(
            save_code=True,
            project=wandb_project,
            entity="julesgm",
            name=name,
            config=dict(
                generation_kwargs = generation_kwargs,
                peft_config_dict  = peft_config_dict,
                ppo_config_args   = ppo_config_dict,
                script_args       = args,
            ),
            dir=wandb_dir,
        )

    assert isinstance(trl_config.model_name, str), type(trl_config.model_name)

    ###########################################################################
    # Load Model
    ###########################################################################
    with lib_utils.maybe_context_manager(
        lambda: rich.status.Status(
            f"[bold green]({RANK}/{WORLD_SIZE})Loading model: "
            f"[white]{rich.markup.escape(str(trl_config.model_name))} [green]...",
            spinner="weather",
        ),
        disable=RANK != 0,
    ):
        output = lib_trl_utils.init_model(
            model_name             = trl_config.model_name,
            peft_config_dict       = peft_config_dict,
            # peft_qlora_mode       = peft_qlora_mode,
            peft_do_all_lin_layers = peft_do_all_lin_layers,
            precision              = precision,
            trl_library_mode       = trl_library_mode,
            use_peft               = use_peft,
        )

        # Deal with fork vs non-fork
        forward_tokenizer    = output["forward_tokenizer"]
        prediction_tokenizer = output["prediction_tokenizer"]
        trainer_kwargs       = {}

        if trl_library_mode == lib_utils.TrlLibraryMode.TRL:
            trainer_kwargs[       "model"] = output[       "model"]
        elif trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
            trainer_kwargs["policy_model"] = output["policy_model"]
            trainer_kwargs[ "value_model"] = output[ "value_model"]
        else:
            raise ValueError(f"Unknown trl_library_mode: {trl_library_mode}")

        eos_token_id = forward_tokenizer.eos_token_id
        pad_token_id = forward_tokenizer.pad_token_id
        assert eos_token_id == prediction_tokenizer.eos_token_id
        assert pad_token_id == prediction_tokenizer.pad_token_id

    
    ###########################################################################
    # Extract "\n"
    ###########################################################################
    # assert not stop_at_line_return
    # if stop_at_line_return:
    #     if dataset_name == lib_data.DatasetChoices.ARITHMETIC:
    #         assert False
    #     line_return_tok = lib_utils.line_return_token(any_tokenizer=prediction_tokenizer)
    #     assert "eos_token_id" not in generation_kwargs, generation_kwargs
    #     assert "eos_token_id" not in inference_gen_kwargs, inference_gen_kwargs
    #     generation_kwargs   ["eos_token_id"] = line_return_tok
    #     inference_gen_kwargs["eos_token_id"] = line_return_tok
        
    # elif True:

    # This is a way to stop decoding.
    line_return_tok = prediction_tokenizer.encode("\n!")[-1]
    decoding_test_output = prediction_tokenizer.decode([line_return_tok])
    assert decoding_test_output == "!", (
        f"\"{decoding_test_output}\" != \"!\""
    )

    assert "eos_token_id" not in generation_kwargs, generation_kwargs
    assert "eos_token_id" not in inference_gen_kwargs, inference_gen_kwargs
    generation_kwargs   ["eos_token_id"] = line_return_tok
    inference_gen_kwargs["eos_token_id"] = line_return_tok

    # else:
    #     generation_kwargs["eos_token_id"] = eos_token_id
    #     inference_gen_kwargs["eos_token_id"] = eos_token_id

    ###########################################################################
    # Load Datasets
    ###########################################################################
    dataset = lib_data.prep_dataset_rl(
        any_tokenizer=forward_tokenizer,
        answer_only=answer_only,
        answer_only_path=answer_only_path,
        dataset_name=dataset_name,
        input_max_length=input_max_length,
        question_prefix=causal_question_prefix,
        question_suffix=causal_question_suffix,
        split=lib_utils.CVSets.TRAIN,
        use_few_shots=use_few_shots,
        arithmetic_dataset_root_folder_dir=arithmetic_dataset_root_folder_dir,
        extr_arith_ignore_one_line=extr_arith_ignore_one_line,
        use_curriculum=use_curriculum,
    )

    eval_dataset = lib_data.prep_dataset_rl(
        any_tokenizer=forward_tokenizer,
        answer_only=answer_only,
        answer_only_path=answer_only_path,
        dataset_name=dataset_name,
        input_max_length=input_max_length,
        question_prefix=causal_question_prefix,
        question_suffix=causal_question_suffix,
        split=lib_utils.CVSets.VALID,
        use_few_shots=use_few_shots,
        arithmetic_dataset_root_folder_dir=arithmetic_dataset_root_folder_dir,
        extr_arith_ignore_one_line=extr_arith_ignore_one_line,
        use_curriculum=False,
    )
    
    ###########################################################################
    # Set model name specific flags
    ###########################################################################
    if not hf_config.is_encoder_decoder:
        if peft_config_dict:
            assert peft_config_dict["task_type"] == peft.TaskType.CAUSAL_LM, (
                peft_config_dict["task_type"])

    ###########################################################################
    # Prep Training
    ###########################################################################
    data_collator = (
        lib_utils.collator if task_name == lib_utils.Task.SENTIMENT 
        else lib_data.data_item_collator
    )

    if use_curriculum:
        data_collator_arg = None
        dataset_arg       = None
    else:
        data_collator_arg = data_collator   
        dataset_arg       = dataset
        
    ppo_trainer: trl.PPOTrainer = trl_library.PPOTrainer(
        config        = trl_config,
        data_collator = data_collator_arg,
        dataset       = dataset_arg,
        ref_model     = None,
        tokenizer     = forward_tokenizer,
        **trainer_kwargs,
    )
    
    metrics, reward_fn = lib_eval.make_metric_and_reward_fn(
        accelerator_device        = ppo_trainer.accelerator.device,
        accelerator_num_processes = ppo_trainer.accelerator.num_processes,
        dataset                   = dataset,
        dataset_name              = dataset_name,
        extractor                 = dataset.get_extractor(),
        pad_token                 = forward_tokenizer.pad_token,
        reward_type               = reward_type,
        task_name                 = task_name,
        use_peft                  = use_peft,
    )
    
    policy_model = (
        ppo_trainer.model 
        if trl_library_mode == lib_utils.TrlLibraryMode.TRL 
        else ppo_trainer.policy_model
    )

    train_eval = lib_eval.EvalLoop(
        accelerated_model    = policy_model,
        accelerator          = ppo_trainer.accelerator,
        batch_size           = inference_batch_size,
        dataset              = dataset,
        dataset_type         = dataset_name,
        eval_subset_size     = eval_subset_size,
        forward_tokenizer    = forward_tokenizer,
        inference_gen_kwargs = inference_gen_kwargs,
        metrics              = metrics,
        prediction_tokenizer = prediction_tokenizer,
        reward_fn            = reward_fn,
        split                = lib_utils.CVSets.TRAIN,
        task_name            = task_name,
        use_few_shots        = use_few_shots,
    )

    eval_eval = lib_eval.EvalLoop(
        accelerated_model    = policy_model,
        accelerator          = ppo_trainer.accelerator,
        batch_size           = inference_batch_size,
        dataset              = eval_dataset,
        dataset_type         = dataset_name,
        forward_tokenizer    = forward_tokenizer,
        eval_subset_size     = eval_subset_size,
        inference_gen_kwargs = inference_gen_kwargs,
        metrics              = metrics,
        prediction_tokenizer = prediction_tokenizer,
        reward_fn            = reward_fn,
        split                = lib_utils.CVSets.VALID,
        task_name            = task_name,
        use_few_shots        = use_few_shots,
    )

    if just_metrics:
        train_eval(0)
        eval_eval(0)
        return

    ###########################################################################
    # Training Loop
    ###########################################################################
    epoch_count = -1
    if use_curriculum:
        dataloader = torch.utils.data.DataLoader(
            batch_size = batch_size,
            dataset    = dataset, 
            collate_fn = data_collator,
        )
        dataloader.dataset.set_proportion_difficulties(
            curriculum_schedule(ppo_trainer.current_step).proportions
        )
    else:
        dataloader = ppo_trainer.dataloader

    current_step = 0
    for epoch in itertools.count():
        epoch_count += 1
        for batch_idx, batch in enumerate(
            lib_utils.progress(
                description=(
                    f"Epoch {epoch_count}, "
                    f"Global Epoch: {epoch}"
                    ),
                disable=True,
                seq=dataloader,
            )
        ):
            if use_curriculum:
                dataloader.dataset.set_proportion_difficulties(
                    curriculum_schedule(ppo_trainer.current_step).proportions
                )
            
            ############################################################
            # Keys of batch:
            #   - "query"
            #   - "input_ids"
            #   - "ref_answer" if in GSM8K
            #   - "ref_scratchpad"
            ############################################################

            if eval_every and batch_idx % eval_every == 0:
                if RANK == 0: rich.print("[red bold]DOING EVAL: [white]TRAIN SET")
                train_eval(ppo_trainer.current_step)
                if RANK == 0: rich.print("[red bold]DOING EVAL: [white]EVAL SET")
                eval_eval(ppo_trainer.current_step)
                if RANK == 0: rich.print("[red bold]DONE WITH EVAL")

            outputs = lib_trl_utils.generate(
                accelerator                  = ppo_trainer.accelerator, 
                answer_extractor             = dataset.get_extractor(),
                batch                        = batch, 
                batch_size                   = batch_size,
                dataset_name                 = dataset_name,
                generation_kwargs            = generation_kwargs, 
                policy_model                 = policy_model, 
                post_process_gen_fewshots_fn = dataset.post_process_gen_fewshots,
                prediction_tokenizer         = prediction_tokenizer,
                task_name                    = task_name,
                use_few_shots                = use_few_shots,
                step_information             = dict(
                    trainer_step = ppo_trainer.current_step,
                    batch_idx    = batch_idx,
                    epoch_idx    = epoch, 
                ),
            )

            reward_output = reward_fn(
                batch=batch,
                responses=outputs.response_text,
            )

            ###########################################################################
            # Checks & Step
            #################
            # - For encoder decoders, the answers should start with the pad token
            # - For all models, the answers should not have any pad tokens in them
            # - For all models, the answers should have one or fewer eos token in them
            ###########################################################################
            if ppo_trainer.is_encoder_decoder:
                assert isinstance(pad_token_id, int), type(pad_token_id)
                lib_trl_utils.check_all_start_with_token_id(
                    outputs.response_tensors, pad_token_id)

            # There should be no pad_token_ids, but the pad
            # token id might be the eos token id, so we can't
            # just blindly check for the pad token id
            if pad_token_id != eos_token_id:
                lib_trl_utils.check_qty_of_token_id(
                    list_of_sequences=outputs.response_tensors,
                    qty=0,
                    token_id=pad_token_id,
                )

            if RANK == 0: print(f"{RANK} ppo_trainer.step >>>")

            step_kwargs = {}
            if trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
                step_kwargs["answers"] = batch.tok_ref_answer                    


            # Log scores per difficulty level
            if use_curriculum:
                per_level = {level: [] for level in all_levels}
                
                for score in mit.zip_equal(batch.difficulty_level, reward_output.values):
                    per_level[score[0]].append(score[1])

                # Accumulate with accelerate
                per_level = {
                    k: torch.tensor(v, device=ppo_trainer.accelerator.device) 
                    for k, v in per_level.items()
                }
                per_level = ppo_trainer.accelerator.gather(per_level)

                if RANK == 0:
                    for k, v in per_level.items():
                        if v is not None and len(v) > 0:
                            wandb.log(
                                {
                                    f"reward_per_level/{k}": v.mean().item(),
                                    f"sample_count_per_level/{k}": 
                                    len(v) / (batch_size * ppo_trainer.accelerator.num_processes),
                                }, 
                                step=ppo_trainer.current_step,
                            )

            stats = ppo_trainer.step(
                queries   = batch.tok_ref_query,
                responses = outputs.response_tensors,
                scores    = reward_output.values,
                **step_kwargs,
            )
            

            if RANK == 0: 
                print(f"{RANK} ppo_trainer.step done <<<")

            # Log stats
            assert isinstance(reward_output.values, list), type(reward_output.values)
            assert isinstance(stats, dict), type(stats)

            batch_stats = dict(
                response         = prediction_tokenizer.batch_decode(
                    outputs.response_tensors),
                query            = batch.detok_ref_query,
                ref_answer       = batch.detok_ref_answer,
                ref_scratchpad   = batch.detok_ref_scratchpad,
                difficulty_level = batch.difficulty_level,
            )

            assert current_step == ppo_trainer.current_step, (
                current_step, ppo_trainer.current_step)
            
            ppo_trainer.log_stats(
                batch   = batch_stats,
                rewards = [x.to(torch.float32) for x in reward_output.values],
                stats   = stats,
                columns_to_log = [
                    "query", 
                    "response", 
                    "ref_answer", 
                    "ref_scratchpad", 
                    "difficulty_level",
                ],
            )
            
            current_step += 1
            ppo_trainer.current_step = current_step
            assert current_step == ppo_trainer.current_step, (
                current_step, ppo_trainer.current_step)
            

            lib_trl_utils.print_table(
                call_source       = "main_loop",
                extra_columns     = reward_output.logging_columns,
                generation_kwargs = generation_kwargs,
                log_header        = f"(e{epoch}-b{batch_idx}) ",
                name              = str(name),
                qty               = 5,
                queries           = batch.detok_ref_query,
                responses         = outputs.response_text,
                rewards           = reward_output.values,
                difficulty_levels = batch.difficulty_level,
            )



if __name__ == "__main__":
    lib_utils.print_accelerate_envs()
    fire.Fire(main)
