"""

File intended to contain the code adapted from OpenAI's original PPO for NLP code.

Sources: 
- https://github.com/CarperAI/trlx/blob/master/trlx/utils/modeling.py
- https://github.com/CarperAI/trlx/blob/master/trlx/model/accelerate_ppo_model.py#L76

"""

import torch
import torch.nn.functional as F


def whiten(values, shift_mean=True):
    """
    Whiten values.
    """
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def loss(
    all_logprobs,  # floats, with_grad, batch_size x seq_len x vocab_size
    all_rewards,  # floats, with_grad, batch_size x seq_len 
    all_values,  # floats, with_grad, batch_size x seq_len
    model,  # ?
    query_tensors,  # long, (no grad), [batch_size, seq_len]
    response_tensors,  # long, (no_grad), [batch_size, max_seq_len]
    config,  # namespace
):
    """
    Parameters:
        - all_logprobs: logprob for question + [scratchpad] + answer
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

    ###########################################################################
    # Compute the advantages
    ###########################################################################
    lastgaelam = 0
    advantages_reversed = []
    gen_len = response_tensors.shape[1]  # [batch * num_scratchpads?] x seq_len

    for t in reversed(range(gen_len)):
        nextvalues = all_values[:, t + 1] if t < gen_len - 1 else 0.0  # float, with_grad?, batch_size
        delta = all_rewards[:, t] + config.method.gamma * nextvalues - all_values[:, t]  # float, with_grad?, batch_size
        lastgaelam = delta + config.method.gamma * config.method.lam * lastgaelam   # generalized advantage estimator,  float, with_grad?, batch_size
        advantages_reversed.append(lastgaelam)
    
    advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)  # float, with grad ?, batch_size x seq_len

    returns = advantages + all_values
    advantages = whiten(advantages)
    advantages = advantages.detach()


    ###########################################################################
    # Compute the prediction
    ###########################################################################
    all_tokens = torch.cat((query_tensors, response_tensors), dim=1)
    logits, _, vpred = model(all_tokens)  # The logits are only used in the PG loss



    ###########################################################################
    # Compute the loss of the value function
    ###########################################################################
    # Only the generation part of the values / logprobs is needed
    vpred   = vpred  [:, - gen_len - 1: - 1] 

    vpredclipped = clip_by_value(
        vpred,
        all_values - config.method.cliprange_value,
        all_values + config.method.cliprange_value,
    )

    vf_losses1 = (vpred - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))


    ###########################################################################
    # Compute the loss of the policy (policy gradient loss)
    ###########################################################################
    # Only the generation part of the values / logprobs is needed
    logprob = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
    logprob = logprob[:, - gen_len:]
    ratio = torch.exp(logprob - all_logprobs)

    pg_losses = - advantages * ratio
    pg_losses2 = - advantages * torch.clamp(
        ratio,
        1.0 - config.method.cliprange,
        1.0 + config.method.cliprange,
    )
    pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))


    ###########################################################################
    # Final loss
    ###########################################################################
    model_loss = pg_loss + config.method.vf_coef * vf_loss
    

    return model_loss