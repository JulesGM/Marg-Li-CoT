import copy
import enum
from typing import *

import numpy as np
import rich
import torch

import utils

class ReplayBuffer:
    def __init__(
        self, *, 
        size: int, 
        output_device, 
        state_shape: Tuple[int, ...], 
        action_shape: Tuple[int, ...],
        qty_memory_samples: int,
    ):
        self.output_device = output_device
        self.action_shape  = action_shape
        self.state_shape   = state_shape
        self.size          = size
        self.qty_memory_samples = qty_memory_samples

        self.observations      = np.zeros((size,  *state_shape), dtype=np.float32)
        self.next_observations = np.zeros((size,  *state_shape), dtype=np.float32)
        self.actions           = np.zeros((size,             1), dtype=np.float32)
        self.rewards           = np.zeros((size,             1), dtype=np.float32)
        self.dones             = np.zeros((size,             1), dtype=np.float32)

        self.full = False
        self.ptr = 0
    
    def store(
        self, *, 
        observation:      np.ndarray, 
        next_observation: np.ndarray, 
        action:           np.ndarray, 
        reward:           float, 
        done:             bool,
    ):
        self.observations     [self.ptr] = observation
        self.next_observations[self.ptr] = next_observation
        self.actions          [self.ptr] = action
        self.rewards          [self.ptr] = reward
        self.dones            [self.ptr] = done

        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0
    
    def sample(self) -> Dict[str, torch.Tensor]:
        
        indices = np.random.randint(0, self.size if self.full else self.ptr, size=self.qty_memory_samples)

        output = dict(
            observation      = torch.as_tensor(self.observations     [indices], device=self.output_device),
            next_observation = torch.as_tensor(self.next_observations[indices], device=self.output_device),
            action           = torch.as_tensor(self.actions          [indices], device=self.output_device, dtype=int),
            reward           = torch.as_tensor(self.rewards          [indices], device=self.output_device),
            done             = torch.as_tensor(self.dones            [indices], device=self.output_device, dtype=int),
        )

        for k, v in output.items():
            assert v.shape[0] == self.qty_memory_samples, (k, v.shape[0], self.qty_memory_samples)

        return output


class AC_Mixin:
    def update(self, rollouts):
        all_observations = []
        policy_loss      = []
        value_loss       = []

        ###############################################################################
        # Prepare the observations
        ###############################################################################
        for rollout in rollouts:
            rollout_observation = []
            for observations in rollout:
                rollout_observation.append(
                    observations["observation"])
            all_observations.append(
                rollout_observation)

        for rid, rollout in enumerate(rollouts):
            values = self.value(torch.tensor(
                all_observations[rid], 
                device=self.device,
            )).squeeze(-1)

            rollout_policy_loss, rollout_value_loss = self.loss(rollout, values)

            policy_loss.append(rollout_policy_loss)
            value_loss.append(rollout_value_loss)

        return torch.mean(torch.stack(policy_loss)), torch.mean(torch.stack(value_loss))

class SARSA:
    class SarsaTypes(str, enum.Enum):
        REGULAR = "regular"
        EXPECTATION = "expectation"

    def __init__(
        self,
        *,
        q:             torch.nn.Module,
        device:        str,
        gamma:         float,
        tau:           float,
        sarsa_type:    str,
        action_space_size:  int,
    ):
        self.q             = q
        self.tau           = tau
        self.gamma         = gamma
        self.device        = device
        self.sarsa_type    = sarsa_type
        self.action_space_size = action_space_size

    def __call__(self, observation: np.ndarray) -> Tuple[int, float]:
        
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        distribution = (self.q(observation) / self.tau).softmax(dim=-1)
        action = torch.distributions.Categorical(distribution).sample().item()

        return action, 0.

    def update(self, rollouts):
        all_observations = []
        q_loss           = []

        ###############################################################################
        # Prepare the observations
        ###############################################################################
        for rollout in rollouts:
            rollout_observation = []

            for observations in rollout:
                rollout_observation.append(
                    observations["observation"])

            all_observations.append(
                rollout_observation)

        for rid, rollout in enumerate(rollouts):
            values = self.q(torch.tensor(all_observations[rid], device=self.device,)).squeeze(-1)
            rollout_q_loss = self.loss(rollout, values)
            assert rollout_q_loss.shape == ()
            q_loss.append(rollout_q_loss)

        return torch.mean(torch.stack(q_loss))

    def loss(self, rollout, values):
        rollout_q_loss = []
        for t, step in reversed(list(enumerate(rollout))):                
            action      = step["action"]
            reward      = step["reward"]
            done        = step["done"]
            next_action = step["next_action"]

            action = torch.tensor(action, device=self.device)
            reward = torch.tensor(reward, device=self.device)
            
            assert done == (len(rollout) - 1 == t),(done, len(rollout) - 1, t)

            if done:
                next_value = torch.tensor(0., device=self.device)

            else:
                if self.sarsa_type == self.SarsaTypes.REGULAR:
                    if next_action is None:
                        next_value = 0
                    else:
                        assert next_action.shape == (), next_action.shape
                        next_value = values[t + 1][next_action]
                        assert next_value.shape == (), next_value.shape

                elif self.sarsa_type == self.SarsaTypes.EXPECTATION:
                    if done:
                        next_value = torch.tensor(0, device=self.device)
                    else:
                        next_value = (values[t + 1] * (values[t + 1] / self.tau).softmax(dim=-1)).sum()

            next_value = next_value.detach()
            
            value = values[t]

            assert value.requires_grad
            
            assert next_value.shape == (), next_value.shape
            assert action    .shape == (), action.shape
            assert value     .shape == (self.action_space_size,), value.shape
            value = value[action]
            assert value     .shape == (), value.shape
            assert reward    .shape == (), reward.shape

            target = reward + self.gamma * next_value
            assert target    .shape == (), target.shape
            q_loss = torch.nn.functional.smooth_l1_loss(value, target)

            rollout_q_loss.append(q_loss)

        return torch.mean(torch.stack(rollout_q_loss))


class DQN:
    def __init__(
        self,
        *,
        q:             torch.nn.Module,
        device:        str,
        gamma:         float,
        tau:           float,
        replay_buffer: ReplayBuffer,
        eps_min:       float,
        eps_max:       float,
        eps_decay:     float,
        action_space_size:  int,
        observation_space_shape: int,
        double_dqn:    bool,
        double_dqn_target_update_period: int,
        double_dqn_soft_update_ratio: float,
    ):
        self.q                               = q
        self.q_target                        = None
        self.tau                             = tau
        self.gamma                           = gamma
        self.device                          = device
        self.replay_buffer                   = replay_buffer
        self.action_space_size               = action_space_size
        self.observation_space_shape         = observation_space_shape
        self.eps_min                         = eps_min
        self.eps_max                         = eps_max
        self.eps_decay                       = eps_decay
        self.eps                             = eps_max
        self.double_dqn                      = double_dqn
        self.double_dqn_target_update_period = double_dqn_target_update_period
        self.double_dqn_soft_update_ratio    = double_dqn_soft_update_ratio
        self.double_dqn_update_count         = 0

        if self.double_dqn:
            self.update_target_q()


    def __call__(self, observation: np.ndarray) -> Tuple[int, float]:
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)

        if self.eps > np.random.rand():
            action = np.random.randint(0, self.action_space_size)
        else:
            action = self.q(observation).argmax().item()

        return action, 0.

    def update_target_q(self):
        rich.print("[red bold]Updating target Q network[/red bold]")
        assert self.double_dqn
        if self.q_target is None:
            self.q_target = copy.deepcopy(self.q)
            self.q_target.eval()
        else:
            for target_param, local_param in zip(self.q_target.parameters(), self.q.parameters()):
                target_param.data.copy_(self.double_dqn_soft_update_ratio * local_param.data + (1.0 - self.double_dqn_soft_update_ratio) * target_param.data)

        for layer in self.q_target.parameters():
                layer.requires_grad = False

    def update(self, rollouts):
        for rollout in rollouts:
            for step in rollout:
                self.replay_buffer.store(
                    observation      = step["observation"],
                    next_observation = step["next_observation"],
                    action           = step["action"], 
                    reward           = step["reward"],
                    done             = step["done"],
                )

        # rich.print(f"[green bold]Sampling from replay buffer: {len(rollout) = }")
        samples = self.replay_buffer.sample()

        ###############################################################################
        # Prepare the observations
        ###############################################################################

        assert samples["observation"].shape == (self.replay_buffer.qty_memory_samples, *self.observation_space_shape), (
            samples["observation"].shape, (self.replay_buffer.qty_memory_samples, *self.observation_space_shape))

        assert samples["next_observation"].shape == (self.replay_buffer.qty_memory_samples, *self.observation_space_shape), (
            samples["next_observation"].shape, (self.replay_buffer.qty_memory_samples, *self.observation_space_shape))

        values      = self.q(samples["observation"]     ).squeeze(-1)

        if self.double_dqn:
            next_values = self.q_target(samples["next_observation"]).squeeze(-1)
            self.double_dqn_update_count += 1
            if self.double_dqn_update_count % self.double_dqn_target_update_period == 0:
                self.update_target_q()
                self.double_dqn_update_count = 0
        else:
            next_values = self.q(samples["next_observation"]).squeeze(-1)
        
        assert values     .shape == (self.replay_buffer.qty_memory_samples, self.action_space_size), (values.shape)
        assert next_values.shape == (self.replay_buffer.qty_memory_samples, self.action_space_size), next_values.shape

        self.eps = max(self.eps_min, self.eps_decay * self.eps)

        rich.print(f"[green bold]Epsilon:[/] {self.eps = }, {self.eps_min = }, {self.eps_decay = }")
    
        return self.loss(samples, values, next_values.detach()).mean()

    def loss(self, samples, values, next_values):
        """

        Off policy loss. Samples are taken at random from the replay buffer.

        Could be vetorized.

        """
        
        # print(f"{samples['reward']     .shape = }")
        # print(f"{samples['done']       .shape = }")
        # print(f"{next_values           .shape = }")
        # print(f"{next_values.argmax(-1).shape = }")
        # print(f"{values                .shape = }")

        dones   = samples["done"].squeeze(-1)
        actions = samples["action"]
        rewards = samples["reward"].squeeze(-1)

        next_values = next_values.max(-1).values * (1 - dones)

        # print(f"{type(actions) = }")
        # print(f"{actions.dtype = }")
        # print(f"{actions.shape = }")
        # print(f"{values.shape  = }")

        values = values.gather(index=actions, dim=-1).squeeze(-1)
        
        # print(f"Post: {values.shape  = }")

        assert dones      .ndim == 1, dones      .shape
        assert next_values.ndim == 1, next_values.shape
        assert values     .ndim == 1, values     .shape
        
        assert dones            .shape ==  next_values      .shape     , (dones      .shape, next_values.shape)
        assert next_values      .shape == (next_values      .shape[0],),  next_values.shape
        assert values           .shape == (values           .shape[0],),  values     .shape
        assert rewards          .shape == (rewards          .shape[0],),  rewards    .shape
        assert next_values      .shape ==  rewards          .shape,       next_values.shape
        assert values           .shape ==  rewards          .shape,       values     .shape

        assert values.requires_grad
        q_loss = torch.nn.functional.smooth_l1_loss(values, rewards + self.gamma * next_values.detach())
        
        assert q_loss.ndim == 0, q_loss.shape

        return q_loss

class AC(AC_Mixin):
    def __init__(self, policy, value, device, lambda_, gamma):
        self.policy  = policy
        self.value   = value
        self.device  = device
        self.lambda_ = lambda_
        self.gamma   = gamma

    def __call__(self, observation):
        observation = torch.tensor(observation, device=self.device).float()
        action_logits = self.policy(observation)
        action = torch.distributions.Categorical(logits=action_logits).sample()
        # action = torch.argmax(action_logits, dim=-1)
        return action.item(), action_logits


    def loss(self, rollout, values):
        rollout_value_loss = []
        rollout_policy_loss = []

        for t, step in reversed(list(enumerate(rollout))):                
            action = step["action"]
            reward = step["reward"]
            logits = step["logits"]
            done   = step["done"]

            action = torch.tensor(action, device=self.device, dtype=int).detach()
            reward = torch.tensor(reward, device=self.device).detach()
            done   = torch.tensor(done,   device=self.device).detach()

            ###############################################################################
            # Compute the Value Loss.
            # This is the JUICE.
            ###############################################################################
            if not done:
                next_value = values[t + 1]
            else:
                next_value = torch.tensor(0., device=self.device)

            td_error = reward + self.lambda_ * next_value.detach() - values[t]

            rollout_policy_loss.append(- torch.log_softmax(
                logits, dim=-1)[action] * td_error.detach()
            )
            
            rollout_value_loss .append(td_error ** 2)

            assert logits    .requires_grad
            assert td_error  .requires_grad
            assert values[t] .requires_grad
        
        rollout_policy_loss = torch.stack(rollout_policy_loss).sum()
        rollout_value_loss  = torch.stack(rollout_value_loss ).sum()

        return rollout_policy_loss, rollout_value_loss

    def ppo_loss(self, rollout, values):
        rollout_value_loss = []
        rollout_policy_loss = []
        advantages_reversed = []
        lastgaelam = 0

        for t, step in reversed(list(enumerate(rollout))):                
            action = step["action"]
            reward = step["reward"]
            logits = step["logits"]
            done   = step["done"]

            action = torch.tensor(action, device=self.device, dtype=int        ).detach()
            reward = torch.tensor(reward, device=self.device, dtype=torch.float).detach()
            done   = torch.tensor(done,   device=self.device, dtype=torch.float).detach()

            ###############################################################################
            # Compute the Value Loss.
            # This is the JUICE.
            ###############################################################################
            if not done:
                next_value = values[t + 1]
            else:
                next_value = torch.tensor(0., device=self.device)

            td_error   = reward   + self.lambda_ * next_value.detach() - values[t]
            lastgaelam = td_error + self.lambda_ * self.gamma * lastgaelam
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1]).detach()
        returns = advantages + values


        assert not torch.isnan(advantages).any(), "before - Advantages contain NaNs."
        assert not torch.isinf(advantages).any(), "before - Advantages contain Infs."
        advantages = (advantages - advantages.mean().detach()) / (advantages.std().detach() + 1e-8)
        assert not torch.isnan(advantages).any(), "Advantages contain NaNs."
        assert not torch.isinf(advantages).any(), "Advantages contain Infs."


        all_logits = torch.stack([step["logits"][step["action"]] for step in rollout])

        rollout_policy_loss.append(- torch.log_softmax(
            all_logits, dim=-1)[action] * advantages.detach()
        )
        rollout_value_loss .append(advantages ** 2)

        assert logits    .requires_grad
        assert td_error  .requires_grad
        assert values[t] .requires_grad
        
        rollout_policy_loss = torch.stack(rollout_policy_loss).sum()
        rollout_value_loss = torch.stack(rollout_value_loss ).sum()

        return rollout_policy_loss, rollout_value_loss





