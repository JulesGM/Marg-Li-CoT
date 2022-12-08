import torch
import numpy as np
import general_utils as utils       

class AC:
    def __init__(self, policy, value, device, lambda_=None):
        self.policy  = policy
        self.value   = value
        self.device  = device

        if lambda_ is None:
            self.lambda_ = .99
        else:
            self.lambda_ = lambda_


    def __call__(self, observation):
        assert False, "Do this directly with the policy"
        observation = torch.tensor(observation, device=self.device).float()
        action_logits = self.policy(observation)
        action = torch.distributions.Categorical(logits=action_logits).sample()
        # action = torch.argmax(action_logits, dim=-1)
        return action.item(), action_logits
    
    def update(self, rollouts):
        policy_loss = 0
        value_loss = 0

        for rollout in rollouts:
            observations = torch.tensor([
                x["observation"] for x in rollout
            ], device=self.device).float()

            values = self.value(observations).squeeze()
            
            for t, step in reversed(list(enumerate(rollout))):
                
                next_observation = step["next_observation"]
                observation      = step["observation"]
                action           = step["action"]
                reward           = step["reward"]
                logits           = step["logits"]
                done             = step["done"]

                next_observation = torch.tensor(next_observation, device=self.device).detach()
                observation      = torch.tensor(observation,      device=self.device).detach()
                action           = torch.tensor(action,           device=self.device, dtype=int).detach()
                reward           = torch.tensor(reward,           device=self.device).detach()
                done             = torch.tensor(done,             device=self.device).detach()

                if not done:
                    next_value = values[t + 1]
                else:
                    next_value = torch.tensor(0)
                
                assert values[t].requires_grad
                td_error = reward + self.lambda_ * next_value.detach() - values[t]
                
                policy_loss += - torch.log_softmax(
                    logits, dim=-1
                )[action] * td_error.detach()

                value_loss += td_error ** 2
            
        return policy_loss, value_loss
