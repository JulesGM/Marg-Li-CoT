import copy
import logging
import os
import sys
from typing import *

import rich
import torch
import transformers

sys.path.append("/home/mila/g/gagnonju/RL4LMs")
import rl4lms.envs.text_generation.registry as rl4lms_registry
import rl4lms.envs.text_generation.policy.seq2seq_policy as rl4lms_seq2seq_policy
from rl4lms.envs.text_generation import hf_generation_utils 
import transformers
import transformers.deepspeed
import deepspeed

from our_scratchpad.bin_deepspeed_experim import OptimizerMerger

import general_utils as utils

LOGGER = logging.getLogger(__name__)


CONSOLE = rich.console.Console(
    width=80,
    force_terminal=True,
)

def msg(text, color, inner_color=None):
    if inner_color is None:
        inner_color = color

    utils.info_rank_0(LOGGER, f"[{color} bold]#" * 80)
    utils.info_rank_0(LOGGER, f"[{color} bold]#" * 80)
    utils.info_rank_0(LOGGER, f"[{color} bold]# [bold {inner_color}]{text}")
    utils.info_rank_0(LOGGER, f"[{color} bold]#" * 80)


class PrecisionControlSeq2SeqLMActorCriticPolicy(
    rl4lms_seq2seq_policy.Seq2SeqLMActorCriticPolicy):
    def __init__(
        self,
        *,
        from_pretrained_kwargs,
        head_kwargs,
        same_model_for_value,
        ref_model=None,
        **kwargs,
    ):
        
        utils.info_rank_0(LOGGER, "[bright_magenta bold]#" * 80)
        utils.info_rank_0(LOGGER, "[bright_magenta bold]# [bright_cyan]POLICY PrecisionControlSeq2SeqLMActorCriticPolicy.__call__")
        utils.info_rank_0(LOGGER, "[bright_magenta bold]#" * 80)

        self._reward_model_container = [ref_model] if ref_model else None
        self._head_kwargs            = head_kwargs
        self._same_model_for_value   = same_model_for_value
        self._from_pretrained_kwargs = from_pretrained_kwargs

        super().__init__(**kwargs)
        

    def _build_model_heads(self, model_name: str):
        """
        - 1. We reuse the reward model for the ref model.
        - 2. We reuse the policy model for the value model.
        """

        #######################################################################
        # Policy Model
        #######################################################################
        msg("Loading Policy Model", "bright_cyan")
        self._policy_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name, **self._from_pretrained_kwargs)
        
        self._policy_model.__class__ = hf_generation_utils.override_generation_routines(
            type(self._policy_model)
        )

        #######################################################################
        # Create Value Model
        #######################################################################
        if self._same_model_for_value:            
            msg("Reusing the policy model as the value model", "bright_cyan")
            self._value_model = self._policy_model
        else:
            msg("Loading Value Model", "bright_cyan")
            self._value_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                model_name, **self._from_pretrained_kwargs
            )
        
        #######################################################################
        # Reference Model
        #######################################################################
        if self._reward_model_container:
            msg("Moving ref ref model to attribute", "bright_cyan")
            self._ref_model = self._reward_model_container[0]
            for p in self._ref_model.parameters():
                p.requires_grad = False
            del self._reward_model_container
        else:
            self._ref_model = None

        if not self._ref_model:
            msg("Copying the policy model as the ref model", "bright_cyan")
            self._ref_model = copy.deepcopy(self._policy_model).eval()
            for p in self._ref_model.parameters():
                p.requires_grad = False

        #######################################################################
        # Create value head
        #######################################################################
        self._value_head = torch.nn.Linear(
            self._value_model.config.hidden_size, 
            1,
            bias=False, 
            **self._head_kwargs,
        )
        rich.print("[red bold]Done Loading Value Model[/]")

        #######################################################################
        # Parallelization
        #######################################################################
        assert torch.cuda.is_available(), "requires CUDA"
        if torch.cuda.is_available():
            if self._apply_model_parallel and self._policy_model.is_parallelizable:
    
                self._policy_model.parallelize()
                self._ref_model.parallelize()

                if not self._value_model is self._policy_model:
                    self._value_model.parallelize()

                self._value_head = self._value_head.to(self.device)

            else:  # else defaults to data parallel
                assert False
                self._policy_model = torch.nn.DataParallel(self._policy_model.to(self.device))
                self._ref_model    = torch.nn.DataParallel(self._ref_model   .to(self.device))
                self._value_model  = torch.nn.DataParallel(self._value_model .to(self.device))
                self._value_head   = torch.nn.DataParallel(self._value_head  .to(self.device))

        else:
            assert False    
        


class DeepSpeedExperimentationPolicy(rl4lms_seq2seq_policy.Seq2SeqLMActorCriticPolicy):
    def __init__(self, *args, ds_configs, **kwargs):
    
        kwargs["apply_model_parallel"] is False, kwargs["apply_model_parallel"]
        super().__init__(*args, **kwargs)
        assert self._apply_model_parallel is False, self._apply_model_parallel
                
        self._ds_train_config, self._ds_inference_config = ds_configs
        self._dschf = transformers.deepspeed.HfDeepSpeedConfig(self._ds_train_config)

        deepspeed.init_distributed()
        self._policy_model = deepspeed.initialize(
            model=self._policy_model, 
            dist_init_required=False,
            config_params=self._ds_train_config,
        )[0]
        self._value_model = deepspeed.initialize(
            model=self._value_model, 
            dist_init_required=False,
            config_params=self._ds_train_config,
        )[0]
        self._value_head = deepspeed.initialize(
            model=self._value_head,
            dist_init_required=False,
            config_params=self._ds_train_config,
        )[0]

        self._ref_model = deepspeed.init_inference(
            model=self._ref_model,
            mp_size=os.environ["WORLD_SIZE"],
            config=self._ds_inference_config,
        )
        
        self.optimizer = OptimizerMerger(
            [self._policy_model, self._value_model, self._value_head]
        )


    def _build_model_heads(self, model_name: str):
        self._policy_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name)
        self._policy_model.__class__ = hf_generation_utils.override_generation_routines(
            type(self._policy_model))

        self._value_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name)
        self._ref_model = copy.deepcopy(self._policy_model).eval()

        self._value_head = torch.nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False)

rl4lms_registry.PolicyRegistry.add(
    "deepspeed_experimentation_policy",
    DeepSpeedExperimentationPolicy,
)


rl4lms_registry.PolicyRegistry.add(
    "precision_control_seq2seq_lm_actor_critic_policy",
    PrecisionControlSeq2SeqLMActorCriticPolicy,
)
