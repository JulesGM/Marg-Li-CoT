import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.utils import explained_variance
from rl4lms.envs.text_generation.policy.base_policy import EvaluateActionsOutput
from rl4lms import conv_bfloat16

import rl4lms.algorithms.ppo.ppo as rl4lms_ppo
import rl4lms.envs.text_generation.registry as rl4lms_registry

from our_scratchpad.bin_deepspeed_experim import OptimizerMerger
import policy

class DeepSpeedPPO(rl4lms_ppo.PPO):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert isinstance(self.policy, policy.DeepSpeedExperimentationPolicy), type(self.policy)
        assert isinstance(self.policy.optimizer, OptimizerMerger), type(self.policy.optimizer)


    def train(self) -> None:
        """

        Update policy using the currently gathered rollout buffer.

        There is only one change: we do self.optimizer.backward(loss) instead of loss.backward().
        This is required

        """
        # Switch to train mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for batch_ix, rollout_data in enumerate(list(self.rollout_buffer.get(self.batch_size))):
                # self.verify_rollout_data(rollout_data)
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                evaluation_output: EvaluateActionsOutput = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values, log_prob, entropy = evaluation_output.values, evaluation_output.log_prob, evaluation_output.entropy
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()
                                  ) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                # if batch_ix == 0 and epoch == 0:
                #     assert th.allclose(th.mean(ratio), th.tensor(
                #         1.0), atol=1e-3), "Cannot reconstruct probability distribution. Please check your policy network implementation"

                #     assert th.allclose(values, rollout_data.old_values, atol=1e-3), "Cannot reconstruct values. Please check your value network implementation"

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * \
                    th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = conv_bfloat16(th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio).cpu()).numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()

                ##################################################
                self.policy.optimizer.backward(loss) # ONLY CHANGE
                ##################################################

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates",
                           self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        train_info = {
            "ppo/entropy_loss":  np.mean(entropy_losses).item(),
            "ppo/policy_gradient_loss": np.mean(pg_losses).item(),
            "ppo/value_loss": np.mean(value_losses).item(),
            "ppo/approx_kl": np.mean(approx_kl_divs).item(),
        }

        self._tracker.log_training_infos(train_info)


rl4lms_registry.AlgorithmRegistry.add(
    "deepspeed_ppo",
    DeepSpeedPPO,
)

rl4lms_registry.WrapperRegistry.add(
    "deepspeed_ppo",
    rl4lms_registry.wrap_onpolicy_alg,
)
