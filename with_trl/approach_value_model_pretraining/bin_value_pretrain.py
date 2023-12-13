print("Starting bin_value_pretrain.py")
import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
# os.environ["DATASETS_VERBOSITY"] = "warning"
# os.environ["WANDB_SILENT"] = "true"
# os.environ["NCCL_DEBUG"] = "WARN"

DETERMINISTIC = False
if DETERMINISTIC:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import contextlib
import datetime
import enum
import itertools as it
import json
import logging
import pathlib
import random
import shutil
import sys
from typing import Any, Optional, Union

import accelerate
import datasets
import fire
import more_itertools as mit
import numpy as np
import peft
import rich
import rich.console
import rich.markup
import rich.status
import rich.table
import rich.traceback
import torch
import transformers
import trl
import wandb

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent
sys.path.append(str(SCRIPT_DIR.parent))

import lib_constant
import lib_data
import lib_eval
import lib_trl_utils
import lib_utils


datasets.disable_caching()
RANK = int(os.getenv("RANK", 0))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

# datasets.logging.set_verbosity_warning()
# transformers.logging.set_verbosity_warning()  # type: ignore
# logging.getLogger("datasets").setLevel(logging.WARNING)
# logging.getLogger("transformers").setLevel(logging.WARNING)
# logging.getLogger("deepspeed").setLevel(logging.WARNING)

np.random.seed(0)
random.seed(1)
torch.manual_seed(2)
torch.cuda.manual_seed_all(3)
trl.set_seed(4)

torch.backends.cuda.matmul.allow_tf32 = not DETERMINISTIC
torch.backends.cudnn.allow_tf32 = not DETERMINISTIC
torch.use_deterministic_algorithms(DETERMINISTIC)


###############################################################################
# Will change depending on the model.
###############################################################################
DEFAULT_MODEL_NAME = "EleutherAI/gpt-j-6B"
# DEFAULT_MODEL_NAME = "EleutherAI/pythia-70m-deduped"

DEFAULT_BATCH_SIZE = 4
DEFAULT_EVAL_BATCH_SIZE = DEFAULT_BATCH_SIZE
DEFAULT_TRAIN_NUM_EPOCHS = 1

DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_WANDB_DIR = lib_utils.get_tmp_dir() / "bin_value_pretrain"
DEFAULT_WANDB_ENTITY = "julesgm"
DEFAULT_WANDB_PROJECT = "value-pretraining"

DEFAULT_CKPT_ROOT = pathlib.Path(
    os.getenv("SCRATCH", "/network/scratch/g/gagnonju")
) / "MargLiCotCkpts" / "ValueFnPretraining"


DEFAULT_LEARNING_RATE: float = 1.41e-5
DEFAULT_GEN_KWARGS = dict(
    min_new_tokens       = 1,
    num_return_sequences = 2,
    early_stopping       = True,
    do_sample            = True,
    
    synced_gpus          = False,
    repetition_penalty   = 1,
    temperature          = 1.,
    top_k                = 0.0,
    top_p                = 1.0,
    use_cache            = True,
    max_new_tokens       = 100,
)

DEFAULT_PEFT_CONFIG_DICT = dict(
    inference_mode=False,
    lora_dropout=0.,
    lora_alpha=16,
    bias="none",
    r=16,
)


###############################################################################
# Will Never change.
###############################################################################
DEFAULT_DO_ALL_LIN_LAYERS = True
DEFAULT_USE_FEW_SHOTS = True
DEFAULT_PEFT_QLORA_MODE = False
DEFAULT_CAUSAL_QUESTION_PREFIX = None
DEFAULT_CAUSAL_QUESTION_SUFFIX = None
DEFAULT_USE_PEFT = True
DEFAULT_INPUT_MAX_LENGTH = 115

DEFAULT_NUM_TRAIN_BATCHES_BETWEEN_EVAL = 16
DEFAULT_NUM_SAMPLES_EVAL = 128
DEFAULT_PRECISION = lib_utils.ValidPrecisions.bfloat16
DEFAULT_TASK_NAME = lib_utils.Task.MAIN
DEFAULT_REWARD_TYPE = lib_utils.RewardChoices.EXACT_MATCH
DEFAULT_TRL_LIBRARY_MODE = lib_utils.TrlLibraryMode.TRL


class CVSets(enum.Enum):
    TRAIN = "train"
    EVAL = "eval"


def prepare_model_inputs(
    *, 
    is_encoder_decoder: bool,
    queries: torch.Tensor,
    responses: torch.Tensor,
    data_collator,
    current_device: int,
) -> dict[str, torch.Tensor]:
    if is_encoder_decoder:
        assert False
        input_data = data_collator(
            [{"input_ids": q, "attention_mask": torch.ones_like(q)} for q in queries]
        ).to(current_device)

        decoder_inputs = data_collator(
            [{"input_ids": r, "attention_mask": torch.ones_like(r)} for r in responses]
        ).to(current_device)

        input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
        input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]

    else:
        assert (
            isinstance(queries[0], torch.Tensor) and 
            isinstance(responses[0], torch.Tensor)
        ), (type(queries).mro(), type(responses).mro())

        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
        
        input_data = data_collator(
            [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} 
             for ids in input_ids]
        ).to(current_device)

    input_data.pop("labels", None)  # we don't want to compute LM losses

    return input_data


def forward_pass(
    *,
    accelerator:       accelerate.Accelerator,
    data_collator,
    model:             trl.PreTrainedModelWrapper,
    queries:           torch.Tensor,
    responses:         torch.Tensor,
    forward_tokenizer: transformers.PreTrainedTokenizerBase,
):

    bs = len(queries)
    is_encoder_decoder = accelerator.unwrap_model(
        model).pretrained_model.config.is_encoder_decoder
    
    assert not is_encoder_decoder

    ############################################################################
    # Prepare model inputs
    ############################################################################
    model_inputs = prepare_model_inputs(
        is_encoder_decoder=is_encoder_decoder, 
        queries=queries,
        responses=responses,
        data_collator=data_collator,
        current_device=accelerator.device,
    )

    pad_first = forward_tokenizer.padding_side == "left"
    model_inputs["input_ids"] = accelerator.pad_across_processes(
        model_inputs["input_ids"], 
        dim=1, 
        pad_index=forward_tokenizer.pad_token_id, 
        pad_first=pad_first,
    )
    model_inputs["attention_mask"] = accelerator.pad_across_processes(
        model_inputs["attention_mask"], 
        dim=1, 
        pad_index=0, 
        pad_first=pad_first,
    )

    if is_encoder_decoder:
        assert False
        model_inputs["decoder_input_ids"] = accelerator.pad_across_processes(
            model_inputs["decoder_input_ids"],
            dim=1,
            pad_index=forward_tokenizer.pad_token_id,
            pad_first=pad_first,
        )
        model_inputs["decoder_attention_mask"] = accelerator.pad_across_processes(
            model_inputs["decoder_attention_mask"], 
            dim=1, 
            pad_index=0, 
            pad_first=pad_first
        )

    # Make sure all batch sizes match
    assert len(responses) == bs, (len(responses), bs)
    _, _, values = model(**model_inputs)


    ############################################################################
    # Prepare masks
    ############################################################################
    if is_encoder_decoder:
        assert False
        input_ids = model_inputs["decoder_input_ids"]
        attention_mask = model_inputs["decoder_attention_mask"]
    else:
        attention_mask = model_inputs["attention_mask"]

    masks = torch.zeros_like(attention_mask)
    masks[:, :-1] = attention_mask[:, 1:]

    for j in range(bs):
        if is_encoder_decoder:
            # Decoder sentence starts always in the index 
            # 1 after padding in the Enc-Dec Models
            start = 1
            end = attention_mask[j, :].sum() - 1
        else:
            start = len(queries[j]) - 1
            
            # Offset left padding
            if attention_mask[j, 0] == 0:  
                start += attention_mask[j, :].nonzero()[0]
            end = start + len(responses[j])

        masks[j, :start] = 0
        masks[j, end:] = 0

    return values[:, :-1], masks[:, :-1]


def compute_rewards(
    *,
    scores: torch.FloatTensor,
    masks: torch.LongTensor,
):
    """
    Applies the mask, then puts the reward on the last non zero element of the sequence.
    """
    
    rewards = []

    assert masks.dtype == torch.long, masks.dtype

    for score, mask in zip(scores, masks):
        # reward is preference model score + KL penalty
        reward = torch.zeros_like(mask, dtype=torch.bfloat16)
        last_non_masked_index = mask.nonzero()[-1]
        reward[last_non_masked_index] += score
        rewards.append(reward)

    return torch.stack(rewards)


class ValueModelPretrainer:
    def __init__(
        self,
        *,
        accelerator: accelerate.Accelerator,
        answer_extractor,
        console,
        dataset_name: str,
        forward_tokenizer: transformers.PreTrainedTokenizerBase,
        gamma: float,
        lam: float,
        metrics,
        model: trl.AutoModelForCausalLMWithValueHead,
        optimizer: torch.optim.Optimizer,
        post_process_gen_fewshots_fn,
        prediction_tokenizer: transformers.PreTrainedTokenizerBase,
        reward_fn,
        task_name: str,
        use_few_shots: bool,
    ):
        self._accelerator                  = accelerator
        self._answer_extractor             = answer_extractor
        self._console                      = console
        self._dataset_name                 = dataset_name
        self._forward_tokenizer            = forward_tokenizer
        self._gamma                        = gamma
        self._lam                          = lam
        self._metrics                      = metrics
        self._model                        = model
        self._optimizer                    = optimizer
        self._post_process_gen_fewshots_fn = post_process_gen_fewshots_fn
        self._prediction_tokenizer         = prediction_tokenizer
        self._reward_fn                    = reward_fn
        self._task_name                    = task_name
        self._use_few_shots                = use_few_shots
        self._wandb_table                  = {}


    @property
    def model(self):
        return self._model

    @property
    def accelerator(self):
        return self._accelerator


    def _v_loss(
        self, 
        *, 
        masks: torch.Tensor,
        scores: torch.Tensor,
        values: torch.Tensor,
        
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        lastgaelam = 0
        advantages_reversed = []
        assert len(scores.shape), 2

        values = values * masks

        rewards = compute_rewards(masks=masks, scores=scores)
        rewards = rewards.to(masks.device) * masks
        gen_len = rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta      = rewards[:, t] + self._gamma * nextvalues - values[:, t]
            lastgaelam = delta + self._gamma * self._lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = trl.trainer.ppo_trainer.masked_whiten(advantages, masks)
        advantages = advantages.detach()

        vf_losses = (values - returns) ** 2
        vf_loss = 0.5 * trl.trainer.ppo_trainer.masked_mean(vf_losses, masks)

        return (
            vf_loss, 
            advantages.detach().cpu(),
            returns   .detach().cpu(),
            rewards   .detach().cpu(),
            vf_losses .detach().cpu(),
        )

    def _show_table(
            self, *, 
            tokens, 
            values, 
            advantages, 
            returns, 
            rewards, 
            losses, 
            masks, 
            cv_set, 
            global_step,
        ):
        
        title = f"({cv_set.value}) Value Table:"
        
        rich_table = rich.table.Table(
            title         = title, 
            show_header   = False,
            show_lines    = True,
            title_justify = "left",
        )
        
        max_len = 100
        wandb_padding_amount = max_len - len(values) - 1    
        old_wandb_table = self._wandb_table.get(cv_set, None)
    

        if old_wandb_table is not None:
            self._wandb_table[cv_set] = wandb.Table(
                columns=[str(i) for i in range(max_len)], 
                data=old_wandb_table.data,
            )
        else:
            self._wandb_table[cv_set] = wandb.Table(
                columns=[str(i) for i in range(max_len)],
            )

        assert len(tokens) == len(values    ), (len(tokens), len(values    ))
        assert len(tokens) == len(advantages), (len(tokens), len(advantages))
        assert len(tokens) == len(returns   ), (len(tokens), len(returns   ))
        assert len(tokens) == len(rewards   ), (len(tokens), len(rewards   ))
        assert len(tokens) == len(losses    ), (len(tokens), len (losses    ))
        
        rows = {}
        rows["tokens"] = [
            rich.markup.escape(self._forward_tokenizer.decode(x)) 
            for x in tokens
        ]
        rows["values"    ] = [f"{x:0.2}" for x in values]
        rows["advantages"] = [f"{x:0.2}" for x in advantages]
        rows["returns"   ] = [f"{x:0.2}" for x in returns]
        rows["rewards"   ] = [f"{x:0.2}" for x in rewards]
        rows["losses"    ] = [f"{x:0.2}" for x in losses]

        for row_name, data in rows.items():
            rich_table.add_row(f"[bold blue]{row_name}", *data)
            self._wandb_table[cv_set].add_data(row_name, *data, *["" for _ in range(wandb_padding_amount)])
        
        # A separator of sorts
        self._wandb_table[cv_set].add_data(*["###" for _ in range(max_len)])
        self._console.print(rich_table)
        wandb.log(
            {f"{lib_constant.WANDB_NAMESPACE}/{cv_set.value}/table": self._wandb_table[cv_set]}, 
            step=global_step,
        )

    def step(
        self,
        *,
        batch,
        batch_idx: int,
        cv_set: CVSets,
        do_log: bool,
        epoch: int,
        forward_data_collator,
        policy_outputs,
        global_step: int,
    ):

        if cv_set == CVSets.TRAIN:
            self._model.train()
        else:
            self._model.eval()

        with (
            self._accelerator.accumulate(self._model) 
            if cv_set == CVSets.TRAIN 
            else torch.no_grad()
        ):

            if cv_set == CVSets.TRAIN:
                self._optimizer.zero_grad()
            

            responses = [
                torch.tensor(x, device="cpu")
                if not isinstance(x, torch.Tensor) else x.cpu()
                for x in policy_outputs.response_tensors
            ]
            queries = [
                torch.tensor(x, device="cpu")
                if not isinstance(x, torch.Tensor) else x.cpu()
                for x in batch.tok_ref_query
            ]

            values, masks = forward_pass(
                accelerator       = self._accelerator,
                data_collator     = forward_data_collator,
                forward_tokenizer = self._forward_tokenizer,
                model             = self._model,
                queries           = queries,
                responses         = responses,
            )

            reward_output = self._reward_fn(
                batch=batch,
                responses=policy_outputs.response_text,
            )

            # Compute metrics
            metrics_outputs = {}
            for metric_name, metric in self._metrics.items():
                local = metric(
                    batch     = batch,
                    responses = policy_outputs.response_text,
                )
                metrics_outputs[metric_name] = self._accelerator.gather_for_metrics(
                    torch.tensor(local.values, device=self._accelerator.device)).mean()


            loss, advantages, returns, rewards, vf_losses = self._v_loss(
                scores = torch.stack(reward_output.values).to(self._accelerator.device),
                values = values,
                masks  = masks,
            )

            if RANK == 0 and global_step % 10 == 0:
                rand_idx = random.randint(0, len(advantages) - 1)
                b_masks  = masks[rand_idx].bool().cpu()
                
                self._show_table(
                    advantages  = advantages[rand_idx].detach().masked_select(b_masks),
                    cv_set      = cv_set,
                    global_step = global_step,
                    losses      = vf_losses [rand_idx].detach().masked_select(b_masks),
                    masks       = masks.cpu(),
                    returns     = returns   [rand_idx].detach().masked_select(b_masks),
                    rewards     = rewards   [rand_idx].detach().masked_select(b_masks),
                    tokens      = responses [rand_idx],
                    values      = values    [rand_idx].detach().cpu().masked_select(b_masks),
                )

            if RANK == 0 and do_log:

                wandb.log({
                    f"{lib_constant.WANDB_NAMESPACE}/{cv_set.value}/value_loss": loss, 
                }, step=global_step)

                for k, v in metrics_outputs.items():
                    wandb.log(
                        {f"{lib_constant.WANDB_NAMESPACE}/{cv_set.value}/{k}": v}, 
                        step=global_step,
                    )

            ###########################################################################
            # Print Rewards
            ###########################################################################

            if cv_set == CVSets.TRAIN:
                self._accelerator.backward(loss)
                self._optimizer.step()

            device = self._accelerator.device
            return (
                self._accelerator.gather(loss      .detach().to(device).contiguous()).cpu(),
                self._accelerator.gather(advantages.detach().to(device).contiguous()).cpu(),
                self._accelerator.gather(returns   .detach().to(device).contiguous()).cpu(), 
                metrics_outputs,
            )



def main(
    wandb_run_name                 : str,
    batch_size                     : int   = DEFAULT_BATCH_SIZE,
    causal_question_prefix         : str   = DEFAULT_CAUSAL_QUESTION_PREFIX,
    causal_question_suffix         : str   = DEFAULT_CAUSAL_QUESTION_SUFFIX,
    ckpt_root                      : str   = DEFAULT_CKPT_ROOT,
    dataset_name                   : lib_data.DatasetChoices   = lib_data.DatasetChoices.COMMONSENSEQA_MC,
    eval_batch_size                : int   = DEFAULT_EVAL_BATCH_SIZE,
    generation_kwargs              : dict  = DEFAULT_GEN_KWARGS,
    gradient_accumulation_steps    : int   = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    input_max_length               : int   = DEFAULT_INPUT_MAX_LENGTH,
    model_name                     : str   = DEFAULT_MODEL_NAME,
    learning_rate                  : float = DEFAULT_LEARNING_RATE,
    num_samples_eval               : int   = DEFAULT_NUM_SAMPLES_EVAL,
    num_train_batches_between_eval : int   = DEFAULT_NUM_TRAIN_BATCHES_BETWEEN_EVAL,
    peft_config_dict               : dict  = DEFAULT_PEFT_CONFIG_DICT,
    peft_do_all_lin_layers         : dict  = DEFAULT_DO_ALL_LIN_LAYERS,
    precision                      : lib_utils.ValidPrecisions = DEFAULT_PRECISION,
    reward_type                    : lib_utils.RewardChoices   = DEFAULT_REWARD_TYPE,
    stop_at_line_return            : bool  = True,
    task_name                      : str   = DEFAULT_TASK_NAME,
    train_num_epochs               : int   = DEFAULT_TRAIN_NUM_EPOCHS,
    trl_library_mode               : lib_utils.TrlLibraryMode  = DEFAULT_TRL_LIBRARY_MODE,
    use_few_shots                  : bool  = DEFAULT_USE_FEW_SHOTS,
    use_peft                       : bool  = DEFAULT_USE_PEFT,
    wandb_dir                      : str   = DEFAULT_WANDB_DIR,
    wandb_entity                   : str   = DEFAULT_WANDB_ENTITY,
    wandb_project                  : str   = DEFAULT_WANDB_PROJECT,
):
    
    precision = lib_utils.ValidPrecisions(precision)  # type: ignore
    args = locals().copy()
    
    rich_console = rich.console.Console(force_terminal=True, width=240)
    rich.traceback.install(console=rich_console)

    ckpt_root = pathlib.Path(ckpt_root)
    assert ckpt_root and ckpt_root.exists() and ckpt_root.is_dir(), (
        ckpt_root, ckpt_root.exists(), ckpt_root.is_dir())

    # Display command line args
    if RANK == 0:
        table = rich.table.Table(
            "Key", "Value", 
            title="Command Line Arguments", 
            show_lines=True,
        )
        for key, value in sorted(args.items(), key=lambda x: x[0]):
            table.add_row(
                "[bold]" + 
                rich.markup.escape(str(key)), 
                rich.markup.escape(str(value))
            )
        rich.print(table)
    
    # Check some command line args
    task_name = lib_utils.Task(task_name)
    dataset_name = lib_data.DatasetChoices(dataset_name)
    trl_library_mode = lib_utils.TrlLibraryMode(trl_library_mode)
    # trl_library = lib_utils.TRL_LIBRARIES[trl_library_mode]

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[rich.logging.RichHandler(markup=True)],
        format=f"[{RANK}/{WORLD_SIZE}] %(funcName)s:%(lineno)d - %(message)s",
    )
    logging.getLogger("transformers").setLevel(logging.ERROR)

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.utils.DistributedDataParallelKwargs(
                find_unused_parameters=True,
            )
        ]
    )

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
    assert "task_type" not in peft_config_dict

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

    if task_name == lib_utils.Task.MAIN:
        reward_type = lib_utils.RewardChoices(reward_type)

    if RANK == 0:
        wandb.init(
            save_code=True,
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=dict(
                generation_kwargs=generation_kwargs,
                peft_config_dict=peft_config_dict,
                script_args=args,
            ),
            dir=wandb_dir,
        )

    assert isinstance(model_name, str), type(model_name)

    ###########################################################################
    # Load Model
    ###########################################################################
    with lib_utils.maybe_context_manager(
        lambda: rich.status.Status(
            f"[bold green]({RANK}/{WORLD_SIZE})Loading model: "
            f"[white]{rich.markup.escape(str(model_name))} [green]...",
            spinner="weather",
        ),
        disable=RANK != 0,
    ):
        output = lib_trl_utils.init_model(
            model_name       = model_name,
            peft_config_dict = peft_config_dict,
            precision        = precision,
            trl_library_mode = trl_library_mode,
            use_peft         = use_peft,
            peft_do_all_lin_layers = peft_do_all_lin_layers,
        )

        # Deal with fork vs non-fork
        forward_tokenizer = output["forward_tokenizer"]
        prediction_tokenizer = output["prediction_tokenizer"]
        if trl_library_mode == lib_utils.TrlLibraryMode.TRL:
            model = output["model"]
        elif trl_library_mode == lib_utils.TrlLibraryMode.TRL_FORK:
            model = output["value_model"]
        else:
            raise ValueError(f"Unknown trl_library_mode: {trl_library_mode}")

        eos_token_id = forward_tokenizer.eos_token_id
        assert eos_token_id == prediction_tokenizer.eos_token_id
        pad_token_id = forward_tokenizer.pad_token_id
        assert pad_token_id == prediction_tokenizer.pad_token_id

        if stop_at_line_return:
            assert (
                "eos_token_id" not in generation_kwargs
            ), generation_kwargs["eos_token_id"]

            generation_kwargs["eos_token_id"] = (
                lib_utils.line_return_token(forward_tokenizer)
            )

    ###########################################################################
    # Load Datasets
    ###########################################################################
    dataset = lib_data.prep_dataset_rl(
        answer_only      = False,
        answer_only_path = None,
        input_max_length = input_max_length,
        question_prefix  = causal_question_prefix,
        question_suffix  = causal_question_suffix,
        any_tokenizer    = forward_tokenizer,
        use_few_shots    = use_few_shots,
        dataset_name     = dataset_name,
        split            = lib_utils.CVSets.TRAIN,
    )

    eval_dataset = lib_data.prep_dataset_rl(
        answer_only      = False,
        answer_only_path = None,
        input_max_length = input_max_length,
        question_prefix  = causal_question_prefix,
        question_suffix  = causal_question_suffix,
        any_tokenizer    = forward_tokenizer,
        use_few_shots    = use_few_shots,
        dataset_name     = dataset_name,
        split            = lib_utils.CVSets.VALID,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn = lib_data.data_item_collator, 
        batch_size = batch_size,
        shuffle    = True,
    )

    eval_dataloader_small = torch.utils.data.DataLoader(
        torch.utils.data.Subset(eval_dataset, range(num_samples_eval)), 
        collate_fn = lib_data.data_item_collator, 
        batch_size = eval_batch_size,
        shuffle    = False,
    )
    
    eval_dataloader_all = torch.utils.data.DataLoader(
        eval_dataset, 
        collate_fn = lib_data.data_item_collator, 
        batch_size = eval_batch_size,
        shuffle    = False,
    )

    metrics, reward_fn = lib_eval.make_metric_and_reward_fn(
        accelerator_device        = accelerator.device,
        accelerator_num_processes = accelerator.num_processes,
        reward_type               = reward_type,
        task_name                 = task_name,
        extractor                 = dataset.get_extractor(),
        use_peft                  = use_peft,
    )

    ###########################################################################
    # Set model name specific flags
    ###########################################################################
    if not hf_config.is_encoder_decoder:
        if peft_config_dict:
            assert peft_config_dict["task_type"] == peft.TaskType.CAUSAL_LM
    
    optimizer = torch.optim.Adam(params=[
            x for x in accelerator.unwrap_model(model).v_head.parameters() 
            if x.requires_grad
        ],
        lr=learning_rate,
    )

    ###########################################################################
    # TRAINING LOOP
    ###########################################################################
    (
        model, optimizer, dataloader, eval_dataloader_small, eval_dataloader_all,
    ) = accelerator.prepare(
        model, optimizer, dataloader, eval_dataloader_small, eval_dataloader_all,
    )

    collator = transformers.DataCollatorForLanguageModeling(
        forward_tokenizer, mlm=False)

    value_trainer = ValueModelPretrainer(
        accelerator                  = accelerator,
        answer_extractor             = dataset.get_extractor(),
        console                      = rich_console,
        dataset_name                 = dataset_name,
        forward_tokenizer            = forward_tokenizer,
        gamma                        = 1.,
        lam                          = .95,
        metrics                      = metrics,
        model                        = model,
        optimizer                    = optimizer,
        post_process_gen_fewshots_fn = dataset.post_process_gen_fewshots,
        prediction_tokenizer         = prediction_tokenizer,
        reward_fn                    = reward_fn,
        task_name                    = task_name,
        use_few_shots                = use_few_shots,
    )

    global_step = 0
    for epoch in range(train_num_epochs):
        while True:
            train_data_loader = enumerate(lib_utils.progress(
                dataloader,
                description=f"Epoch {epoch}",
                disable=RANK!=0,
            ))
            
            # If we can't do a single batch, then we are done with the epoch
            at_least_one = False
            
            for train_batch_idx, batch in it.islice(
                train_data_loader,
                num_train_batches_between_eval,
            ):
                global_step += len(batch.detok_ref_scratchpad) * WORLD_SIZE
                at_least_one = True

                with value_trainer.accelerator.unwrap_model(
                    value_trainer.model
                ).pretrained_model.disable_adapter():

                    outputs = lib_trl_utils.generate(
                        accelerator                  = value_trainer.accelerator, 
                        answer_extractor             = dataset.get_extractor(),
                        batch                        = batch, 
                        batch_size                   = batch_size,
                        dataset_name                 = dataset_name,
                        generation_kwargs            = generation_kwargs, 
                        policy_model                 = value_trainer.model, 
                        post_process_gen_fewshots_fn = dataset.post_process_gen_fewshots,
                        prediction_tokenizer         = prediction_tokenizer,
                        task_name                    = task_name,
                        use_few_shots                = use_few_shots,
                    )

                    value_trainer.step(
                        batch                 = batch,
                        batch_idx             = train_batch_idx,
                        cv_set                = CVSets.TRAIN,
                        epoch                 = epoch,
                        forward_data_collator = collator, 
                        do_log                = True,
                        policy_outputs        = outputs,
                        global_step           = global_step,
                    )

            if not at_least_one:
                break

            for eval_batch_idx, batch in enumerate(lib_utils.progress(
                description=f"Validation {epoch}",
                disable=RANK!=0,
                seq=eval_dataloader_small,
            )):
                with torch.no_grad():
                    value_trainer.step(
                        batch                 = batch,
                        batch_idx             = eval_batch_idx,
                        cv_set                = CVSets.EVAL,
                        epoch                 = epoch,
                        forward_data_collator = collator,
                        do_log                = True,
                        policy_outputs        = outputs,
                        global_step           = global_step,
                    )



if __name__ == "__main__":
    fire.Fire(main)