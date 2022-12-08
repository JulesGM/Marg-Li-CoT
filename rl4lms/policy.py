import copy
from typing import *
import sys

import rich
import torch
import transformers

sys.path.append("/home/mila/g/gagnonju/RL4LMs")
import rl4lms.envs.text_generation.registry as rl4lms_registry
import rl4lms.envs.text_generation.policy.seq2seq_policy as rl4lms_seq2seq_policy
from rl4lms.envs.text_generation import hf_generation_utils 

CONSOLE = rich.console.Console(
    width=80,
    force_terminal=True,
)

def msg(text, color, inner_color=None):
    if inner_color is None:
        inner_color = color

    CONSOLE.print(f"[{color} bold]#" * 80)
    CONSOLE.print(f"[{color} bold]# [bold {inner_color}]{text}")
    CONSOLE.print(f"[{color} bold]#" * 80)


class PrecisionControlSeq2SeqLMActorCriticPolicy(
    rl4lms_seq2seq_policy.Seq2SeqLMActorCriticPolicy):
    def __init__(
        self,
        *args,
        from_pretrained_kwargs,
        head_kwargs,
        same_model_for_value,
        ref_model=None,
        **kwargs,
    ):
        
        CONSOLE.print("[bright_magenta bold]#" * 80)
        CONSOLE.print("[bright_magenta bold]# [bright_cyan]POLICY PrecisionControlSeq2SeqLMActorCriticPolicy.__call__")
        CONSOLE.print("[bright_magenta bold]#" * 80)

        self._reward_model_container = [ref_model] if ref_model else None
        self._head_kwargs            = head_kwargs
        self._same_model_for_value   = same_model_for_value
        self._from_pretrained_kwargs = from_pretrained_kwargs

        super().__init__(*args, **kwargs)
        
        #######################################################################
        # Check the devices of each module in this module.
        # It is strange because the original code doesn't have to do this.
        #######################################################################

        # self.to("cuda")
        # assert self.device == "cuda", f"Got `{self.device}`"
        # for k, v in vars(self).items():
        #     if k == "features_extractor_class":
        #         continue
            
        #     if hasattr(v, "device"):
        #         assert v.device == self.device, f"{k}:{v.device} != {self.device}"
        #         print("{k} is ok - device")

        #     if hasattr(v, "parameters"):
        #         try:
        #             for p in v.parameters():
        #                 assert p.device == self.device, f"{k}: {p.device} != {self.device}"
        #         except TypeError:
        #             print(f"{k}")
        #             raise
        #         print("{k} is ok - parameters")

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
        

rl4lms_registry.PolicyRegistry.add(
    "precision_control_seq2seq_lm_actor_critic_policy",
    PrecisionControlSeq2SeqLMActorCriticPolicy,
)
