import copy
import enum
from typing import *

import numpy as np
import scipy.signal
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
        store_logits: bool = False,
    ):
        self.output_device = output_device
        self.action_shape  = action_shape
        self.state_shape   = state_shape
        self.size          = size
        self.qty_memory_samples = qty_memory_samples

        self.observations      = np.zeros((size,   *state_shape), dtype=np.float32)
        self.next_observations = np.zeros((size,   *state_shape), dtype=np.float32)
        self.actions           = np.zeros((size,              1), dtype=np.float32)
        self.rewards           = np.zeros((size,              1), dtype=np.float32)
        self.dones             = np.zeros((size,              1), dtype=np.float32)
        if store_logits:
            self.logits        = np.zeros((size,  *action_shape), dtype=np.float32)

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


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
        x1 + discount * x2,
        x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer:
    """
    From https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, *, obs_dim, size, gamma, lam, dtype, device):

        self._obs_buf:  Final[np.ndarray] = np.zeros((size, obs_dim,))
        self._act_buf:  Final[np.ndarray] = np.zeros((size, ))
        self._adv_buf:  Final[np.ndarray] = np.zeros((size,))
        self._rew_buf:  Final[np.ndarray] = np.zeros((size,))
        self._ret_buf:  Final[np.ndarray] = np.zeros((size,))
        self._val_buf:  Final[np.ndarray] = np.zeros((size,))
        self._logp_buf: Final[np.ndarray] = np.zeros((size,))
        
        self._lam:      Final[float] = lam
        self._gamma:    Final[float] = gamma
        self._max_size: Final[float] = size

        self._dtype:    Final[float] = dtype
        self._device:   Final[float] = device

        # Things that can change
        self._ptr: int = 0 
        self._path_start_idx: int = 0
        self._total_steps_seen: int = 0

    def store(self, *, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self._ptr < self._max_size     # buffer has to have room so you can store

        self._obs_buf [self._ptr] = obs
        self._act_buf [self._ptr] = act
        self._rew_buf [self._ptr] = rew
        self._val_buf [self._ptr] = val
        self._logp_buf[self._ptr] = logp
        
        self._ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.

        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self._path_start_idx, self._ptr)
        rews = np.append(self._rew_buf[path_slice], last_val)
        vals = np.append(self._val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self._gamma * vals[1:] - vals[:-1]
        self._adv_buf[path_slice] = discount_cumsum(deltas, self._gamma * self._lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self._ret_buf[path_slice] = discount_cumsum(rews, self._gamma)[:-1]
        
        self._path_start_idx = self._ptr
        

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self._ptr == self._max_size    # buffer has to be full before you can get

        
        # the next two lines implement the advantage normalization trick
        adv_mean = self.distr_mean(self._adv_buf)
        adv_std = self.distr_std(self._adv_buf)

        self._adv_buf = (self._adv_buf - adv_mean) / (adv_std + 1e-8)

        data = dict(
            obs =torch.tensor(self._obs_buf [:self._ptr], device=self._device, dtype=self._dtype), 
            act =torch.tensor(self._act_buf [:self._ptr], device=self._device, dtype=self._dtype),
            ret =torch.tensor(self._ret_buf [:self._ptr], device=self._device, dtype=self._dtype),
            adv =torch.tensor(self._adv_buf [:self._ptr], device=self._device, dtype=self._dtype),
            logp=torch.tensor(self._logp_buf[:self._ptr], device=self._device, dtype=self._dtype),
        )

        self._total_steps_seen += self._ptr
        rich.print(f"\n[bold]ppo_buff | Steps seen: {self._total_steps_seen = }")
        self._ptr = 0
        self._path_start_idx = 0

        return {k: torch.as_tensor(v, dtype=self._dtype, device=self._device) for k, v in data.items()}

    def distr_sum(self, arr):
        """
        Sums an array across all MPI processes, and returns the sum.
        """
        return np.sum(arr)

    def distr_mean(self, arr):
        """
        Averages an array across all MPI processes, and returns the average.
        """
        return np.mean(arr)

    def distr_std(self, arr):
        """
        Computes the standard deviation of an array across all MPI processes, and returns the result.
        """
        return np.std(arr)

    def reset(self,):
        self._ptr: int = 0 
        self._path_start_idx: int = 0
    

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


class PPO:
    def __init__(self, *, 
        policy, 
        value, 
        buffer_max_size, 
        act_dim, 
        obs_dim, 
        dtype, 
        device: str, 
        lambda_: float, 
        gamma: float,
    ):
        self.policy  = policy
        self.value   = value
        self.device  = device
        self.lambda_ = lambda_
        self.gamma   = gamma

        self._ppo_buff = PPOBuffer(
            obs_dim=obs_dim, 
            size=buffer_max_size, 
            gamma=gamma, 
            lam=lambda_, 
            dtype=dtype, 
            device=device,
        )

    def __call__(self, observation):
        observation = torch.tensor(observation, device=self.device).float()
        action_logits = self.policy(observation)
        action = torch.distributions.Categorical(logits=action_logits).sample()
        # action = torch.argmax(action_logits, dim=-1)
        return action.item(), action_logits

    def add(self, *, observation, action, reward, done, logp):
        # rich.print(f"[bold red]Adding.")
        self._ppo_buff.store(obs=observation, act=action, rew=reward, val=done, logp=logp)

    def buffer_finish_path(self):
        self._ppo_buff.finish_path()

    def get_minibatches(self, batch_size):
        data = self._ppo_buff.get()
        # Check that all sizes are the same
        assert all(data["obs"].shape[0] == x.shape[0] for x in data.values()), {
            k: data["obs"].shape[0] == v.shape[0] for k, v in data.items()}

        for i in range(0, len(data["obs"]), batch_size):
            yield {k: v[i:i + batch_size] for k, v in data.items()}

    def loss(self, rollouts, new_logprobs, new_values):
        clip_range = 0.2

        assert new_values.requires_grad
        assert new_logprobs.requires_grad

        ratio = torch.exp(rollouts["logp"] - new_logprobs)        
        policy_loss_1 = rollouts["adv"] * ratio
        policy_loss_2 = rollouts["adv"] * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

        rollout_value_loss = ((new_values - rollouts["ret"]) ** 2).mean()

        return policy_loss, rollout_value_loss





