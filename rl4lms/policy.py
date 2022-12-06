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


class PrecisionControlSeq2SeqLMActorCriticPolicy(
    rl4lms_seq2seq_policy.Seq2SeqLMActorCriticPolicy):
    def __init__(
        self,
        *args,
        from_pretrained_kwargs,
        head_kwargs,
        **kwargs,
    ):
        
        rich.print("[bright_magenta bold]#" * 80)
        rich.print("[bright_magenta bold]# [bright_cyan]POLICY PrecisionControlSeq2SeqLMActorCriticPolicy.__call__")
        rich.print("[bright_magenta bold]#" * 80)

        self._from_pretrained_kwargs = from_pretrained_kwargs
        self._head_kwargs = head_kwargs
        super().__init__(*args, **kwargs)
        
        #######################################################################
        # Check the devices
        #######################################################################
        self.to("cuda")
        assert self.device == "cuda", f"Got `{self.device}`"
        for k, v in vars(self).items():
            if k == "features_extractor_class":
                continue
            
            if hasattr(v, "device"):
                assert v.device == self.device, f"{k}:{v.device} != {self.device}"
                print("{k} is ok - device")

            if hasattr(v, "parameters"):
                try:
                    for p in v.parameters():
                        assert p.device == self.device, f"{k}: {p.device} != {self.device}"
                except TypeError:
                    print(f"{k}")
                    raise
                print("{k} is ok - parameters")

    def _build_model_heads(self, model_name: str):
        rich.print("[red bold]Loading Policy Model[/]")
        self._policy_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name, **self._from_pretrained_kwargs)
        self._policy_model.__class__ = hf_generation_utils.override_generation_routines(
            type(self._policy_model)
        )

        rich.print("[red bold]Loading Value Model[/]")
        self._value_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name, **self._from_pretrained_kwargs)

        rich.print("[red bold]Copying to ref[/]")
        self._ref_model = copy.deepcopy(self._policy_model).eval()
        self._value_head = torch.nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False, **self._head_kwargs,
        )
        rich.print("[red bold]Done Loading Value Model[/]")

        # apply model parallel
        assert torch.cuda.is_available(), "requires CUDA"

        if torch.cuda.is_available():
            self._do_parallelize = False

            if self._do_parallelize:
                assert False
                if self._apply_model_parallel and self._policy_model.is_parallelizable:
                    self._policy_model.parallelize()
                    self._ref_model.parallelize()
                    self._value_model.parallelize()
                    self._value_head = self._value_head.to(self.device)
                else:  # else defaults to data parallel
                    self._policy_model = torch.nn.DataParallel(self._policy_model.to(self.device))
                    self._ref_model    = torch.nn.DataParallel(self._ref_model   .to(self.device))
                    self._value_model  = torch.nn.DataParallel(self._value_model .to(self.device))
                    self._value_head   = torch.nn.DataParallel(self._value_head  .to(self.device))
            else:
                self._apply_model_parallel = False

                self._policy_model = self._policy_model .to(self.device)
                self._ref_model    = self._ref_model    .to(self.device)
                self._value_model  = self._value_model  .to(self.device)
                self._value_head   = self._value_head   .to(self.device)

        assert self._policy_model.device == self.device, f"{self._policy_model.device} != {self.device}"
        assert self._ref_model.device    == self.device, f"{self._ref_model   .device} != {self.device}"
        assert self._value_model.device  == self.device, f"{self._value_model .device} != {self.device}"
        # assert self._value_head.device   == self.device, f"{self._value_head  .device} != {self.device}"
        for v in self._value_head.parameters():
            assert v.device == self.device, f"{v.device} != {self.device}"
        

rl4lms_registry.PolicyRegistry.add(
    "precision_control_seq2seq_lm_actor_critic_policy",
    PrecisionControlSeq2SeqLMActorCriticPolicy,
)
