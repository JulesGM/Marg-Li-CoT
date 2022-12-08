"""

File intended to contain the code adapted from OpenAI's original PPO for NLP code.

Sources: 
- https://github.com/CarperAI/trlx/blob/master/trlx/utils/modeling.py
- https://github.com/CarperAI/trlx/blob/master/trlx/model/accelerate_ppo_model.py#L76

"""
import time
from typing import *

from beartype import beartype
import torch
import transformers
import wandb

import general_utils as utils

class GPT2ForTokenClassificationWithActivationOnTop(transformers.GPT2ForTokenClassification):
    @beartype
    def __init__(self, activation_fn: Callable[[torch.Tensor], torch.Tensor], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._activation_fn = activation_fn

    def forward(self, *args, **kwargs):
        logits = super().forward(*args, **kwargs).logits
        if self._activation_fn:
            return self._activation_fn(logits)
        return logits


def whiten(values, shift_mean=True):
    """
    Whiten values.
    """
    mean = torch.mean(values)
    var = torch.var(values)

    whitened = (values - mean) * torch.rsqrt(var + 1e-8)

    if not shift_mean:
        whitened += mean

    return whitened


def clip_by_value(x, tensor_min, tensor_max)-> torch.Tensor:
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """

    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    
    return clipped


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """

    log_p = logits.log_softmax(dim=2)
    log_py = torch.gather(log_p, 2, labels.unsqueeze(2)).squeeze(-1)

    return log_py




def make_experience(
    *,
    all_tokens_input_ids:      torch.Tensor,
    all_tokens_attention_mask: torch.Tensor,
    query_tensors:             torch.Tensor,
    response_tensors:          torch.Tensor,
    scores:                    torch.Tensor,
    logprobs_generated:        torch.Tensor,
    logprobs_generated_fixed:  torch.Tensor,
    value_model:               transformers.GPT2PreTrainedModel,
    init_kl_coef:              float,

    num_scratchpads:           int,
    batch_size:                int,


    # Arguments for logging
    batch_idx:            int,
    logger,    
    logger_kwargs:        dict[str, Any], 
):
    """
        Computes the KL reward and the value of the response.
    """
    
    all_tokens_input_ids      = all_tokens_input_ids     .reshape(num_scratchpads * batch_size, -1)
    all_tokens_attention_mask = all_tokens_attention_mask.reshape(num_scratchpads * batch_size, -1)
    scores                    = scores                   .reshape(num_scratchpads * batch_size, -1)
    logprobs_generated        = logprobs_generated       .reshape(num_scratchpads * batch_size, -1)
    logprobs_generated_fixed  = logprobs_generated_fixed .reshape(num_scratchpads * batch_size, -1)
    query_tensors             = query_tensors            .reshape(num_scratchpads * batch_size, -1)
    response_tensors          = response_tensors         .reshape(num_scratchpads * batch_size, -1)


    #######################################################################
    # Compute the logits and the values with the trainable and fixed models
    #######################################################################
    # Precompute logprobs, values
    with torch.no_grad():
        v = value_model(
            input_ids=all_tokens_input_ids, 
            attention_mask=all_tokens_attention_mask,
        ).logits
        utils.check_equal(v.shape[-1], 1)
        v = v.squeeze(-1)

    start       = query_tensors.shape[1] - 1
    end         = query_tensors.shape[1] + response_tensors.shape[1] - 1
    all_values  = v[:, start - 1:end - 1]

    #######################################################################
    # Compute rewards
    #######################################################################
    kls                 = logprobs_generated - logprobs_generated_fixed
    non_score_rewards   = - init_kl_coef * kls
    all_rewards         = non_score_rewards.clone()
    utils.check_equal(scores.shape, (all_rewards.shape[0], 1))
    scores = scores.squeeze(-1)
    all_rewards[:, -1] += scores
    
    return dict(
        response_tensors = response_tensors,
        query_tensors    = query_tensors,
        logprobs         = logprobs_generated,
        values           = all_values,
        rewards          = all_rewards,
    )
    
    

def loss(
    all_logprobs:     torch.Tensor,
    all_rewards:      torch.Tensor,
    all_values:       torch.Tensor,
    query_tensors:    torch.Tensor,
    response_tensors: torch.Tensor,

    model:            transformers.GPT2LMHeadModel,
    value_model:      transformers.GPT2ForTokenClassification,
    gamma:            float,           
    lambda_:          float,
    cliprange_value:  float,
    cliprange:        float,
    vf_coef:          float,
    num_scratchpads:  int,
    batch_size:       int,
):
    """
    Parameters:
        - all_logprobs: logprob for question + answer
        - all_rewards: afaik this is the KL reward with the final reward at the final step
        - all_values: values predicted by the value funciton
        - config:
            - method:
                - gamma
                - lam
                - cliprange_value
                - cliprange
                - vf_coef
        - model:
        - query_tensors: [question]
        - response_tensors: [answer]
    """

    all_logprobs     = all_logprobs    .reshape(num_scratchpads * batch_size, all_rewards.shape[-1]).detach()
    all_values       = all_values      .reshape(num_scratchpads * batch_size, -1).detach()
    all_rewards      = all_rewards     .reshape(num_scratchpads * batch_size, -1).detach()
    query_tensors    = query_tensors   .reshape(num_scratchpads * batch_size, -1).detach()
    response_tensors = response_tensors.reshape(num_scratchpads * batch_size, -1).detach()

    utils.check_equal(all_values.shape[-1], response_tensors.shape[-1])
    utils.check_equal(all_rewards.shape, all_logprobs.shape[:2])

    ###########################################################################
    # Compute the advantages
    ###########################################################################
    last_gae_lam = 0.
    advantages_reversed = []
    gen_len = response_tensors.shape[1]

    for t in reversed(range(gen_len)):
        nextvalues = all_values [:, t + 1] if t < gen_len - 1 else 0.0
        delta      = all_rewards[:, t] + gamma * nextvalues - all_values[:, t]
        last_gae_lam = delta + gamma * lambda_ * last_gae_lam
        advantages_reversed.append(last_gae_lam)
    
    advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

    returns = (advantages + all_values)
    advantages = whiten(advantages)
    advantages = advantages.detach()

    ###########################################################################
    # Compute the prediction
    ###########################################################################
    all_tokens = torch.cat((query_tensors, response_tensors), dim=1)
    logits = model(all_tokens).logits
    vpred  = value_model(all_tokens).logits
    utils.check_equal(vpred.shape[-1], 1)
    vpred = vpred.squeeze(-1)

    ###########################################################################
    # Compute the loss of the value function
    ###########################################################################
    # Only the generation part of the values / logprobs is needed
    vpred = vpred[:, - gen_len - 1: - 1]

    vpredclipped = clip_by_value(
        vpred,
        all_values - cliprange_value,
        all_values + cliprange_value,
    )

    vf_losses1 = (vpred        - returns.detach()) ** 2
    vf_losses2 = (vpredclipped - returns.detach()) ** 2
    vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))

    ###########################################################################
    # Compute the loss of the policy (policy gradient loss)
    ###########################################################################
    # Only the generation part of the values / logprobs is needed
    logprob = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
    logprob = logprob[:, - gen_len:]
    ratio = torch.exp(logprob - all_logprobs[:, -gen_len:].detach())
    

    pg_losses  = - advantages * ratio
    pg_losses2 = - advantages * torch.clamp(
        ratio,
        1.0 - cliprange,
        1.0 + cliprange,
    )
    pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))

    ###########################################################################
    # Final loss
    ###########################################################################
    model_loss = pg_loss + vf_coef * vf_loss

    return model_loss
