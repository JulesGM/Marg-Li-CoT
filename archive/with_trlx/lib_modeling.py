
from typing import Union

import peft
import torch
import transformers
from transformers.modeling_outputs import ModelOutput

from trlx.models.modeling_base import PreTrainedModelWrapper
from trlx.utils.modeling import (
    hf_get_hidden_size,
    make_head,
)
from trlx.models.modeling_ppo import (
    Seq2SeqLMOutputWithValue,
    PreTrainedModelWrapper
)


class AutoModelForSeq2SeqLMWithHydraValueHead(PreTrainedModelWrapper):
    _supported_modules = ["v_head", "frozen_head"]
    _supported_args    = ["num_layers_unfrozen"]

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        peft_config,
        device
    ):
        super().__init__(base_model)
        self.v_head = make_head(hf_get_hidden_size(self.base_model.config), 1)
        
        self.peft_config = peft_config
        
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.frozen_head = base_model

        self.base_model = peft.get_peft_model(base_model, peft_config).bfloat16()
        type(self.base_model).print_trainable_parameters(self.base_model)
        
        self.v_head.to(self.base_model.dtype).to(device)
    
    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.base_model.generate(*args, **kwargs)

    def forward_hydra(self, **forward_kwargs):

        return_dict = forward_kwargs.get("return_dict", True)
        
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"]          = True

        forward_kwargs["output_attentions"]    = False
        forward_kwargs["use_cache"]            = False

        hydra_outputs = self.frozen_head(**forward_kwargs)
        if not return_dict:
            return hydra_outputs.logits
        return hydra_outputs
    

    def forward(self, **forward_kwargs):
        forward_kwargs["output_hidden_states"] = True
        forward_kwargs["return_dict"         ] = True

        outputs           = self.base_model(**forward_kwargs)
        last_hidden_state = outputs.decoder_hidden_states[-1]
        value             = self.v_head(last_hidden_state).squeeze(-1)

        return Seq2SeqLMOutputWithValue(**outputs, value=value)