# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticCNN
from rsl_rl.storage import RolloutStorage

class DyrosPPO(PPO):
    """Custom PPO implementation for Dyros robot.
    
    This class inherits from the standard RSL-RL PPO and can be customized
    to override default behaviors.
    """
    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Bound loss parameters
        bound_loss_cfg: dict | None = None,
        # LCP loss parameters
        lcp_loss_cfg: dict | None = None,
    ) -> None:
        super().__init__(policy, storage, num_learning_epochs, num_mini_batches, clip_param, gamma, lam, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, use_clipped_value_loss, schedule, desired_kl, normalize_advantage_per_mini_batch, device, rnd_cfg, symmetry_cfg, multi_gpu_cfg)
        print("[INFO] Using custom DyrosPPO Algorithm")

        if bound_loss_cfg is not None:
            self.use_bound_loss = bound_loss_cfg
        else:
            self.use_bound_loss = None

        if lcp_loss_cfg is not None:
            self.use_lcp_loss = lcp_loss_cfg
        else:
            self.use_lcp_loss = None

    def update(self) -> dict[str, float]:
            mean_value_loss = 0
            mean_surrogate_loss = 0
            mean_entropy = 0
            # RND loss
            mean_rnd_loss = 0 if self.rnd else None
            # Symmetry loss
            mean_symmetry_loss = 0 if self.symmetry else None
            # Bound loss
            mean_bound_loss = 0 if self.use_bound_loss else None
            # LCP loss
            mean_lcp_loss = 0 if self.use_lcp_loss else None
            
            # Get mini batch generator
            if self.policy.is_recurrent:
                generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            else:
                generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

            # Iterate over batches
            for (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch,
                masks_batch,
            ) in generator:
                num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
                original_batch_size = obs_batch.batch_size[0]

                # Check if we should normalize advantages per mini batch
                if self.normalize_advantage_per_mini_batch:
                    with torch.no_grad():
                        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)


                # Perform symmetric augmentation
                if self.symmetry and self.symmetry["use_data_augmentation"]:
                    # Augmentation using symmetry
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    # Returned shape: [batch_size * num_aug, ...]
                    obs_batch, actions_batch = data_augmentation_func(
                        obs=obs_batch,
                        actions=actions_batch,
                        env=self.symmetry["_env"],
                    )
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                    # Repeat the rest of the batch
                    old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                    target_values_batch = target_values_batch.repeat(num_aug, 1)
                    advantages_batch = advantages_batch.repeat(num_aug, 1)
                    returns_batch = returns_batch.repeat(num_aug, 1)

                if self.use_lcp_loss:
                    obs_actor_batch = self.policy.get_actor_obs(obs_batch)
                    obs_actor_batch = self.policy.actor_obs_normalizer(obs_actor_batch)
                    obs_est_batch = obs_actor_batch.clone()
                    obs_est_batch.requires_grad_(True)
                    self.policy._update_distribution(obs_est_batch)
                    self.policy.distribution.sample()
                else:
                    self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])

                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
                # Note: We only keep the entropy of the first augmentation (the original one)
                mu_batch = self.policy.action_mean[:original_batch_size]
                sigma_batch = self.policy.action_std[:original_batch_size]
                entropy_batch = self.policy.entropy[:original_batch_size]

                if self.use_lcp_loss:
                    lcp_loss = self._calc_gradient_penalty(obs_est_batch, actions_log_prob_batch, self.use_lcp_loss["is_lcp"])

                # Compute KL divergence and adapt the learning rate
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)

                        # Reduce the KL divergence across all GPUs
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.gpu_world_size

                        # Update the learning rate only on the main process
                        # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                        #       then the learning rate should be the same across all GPUs.
                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        # Update the learning rate for all GPUs
                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()

                        # Update the learning rate for all parameter groups
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Symmetry loss
                if self.symmetry:
                    # Obtain the symmetric actions
                    # Note: If we did augmentation before then we don't need to augment again
                    if not self.symmetry["use_data_augmentation"]:
                        data_augmentation_func = self.symmetry["data_augmentation_func"]
                        obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                        # Compute number of augmentations per sample
                        num_aug = int(obs_batch.shape[0] / original_batch_size)

                    # Actions predicted by the actor for symmetrically-augmented observations
                    mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                    # Compute the symmetrically augmented actions
                    # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                    # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                    # using the mean of the distribution.
                    action_mean_orig = mean_actions_batch[:original_batch_size]
                    _, actions_mean_symm_batch = data_augmentation_func(
                        obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                    )

                    # Compute the loss
                    mse_loss = torch.nn.MSELoss()
                    symmetry_loss = mse_loss(
                        mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                    )
                    # Add the loss to the total loss
                    if self.symmetry["use_mirror_loss"]:
                        loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                    else:
                        symmetry_loss = symmetry_loss.detach()

                # RND loss
                # TODO: Move this processing to inside RND module.
                if self.rnd:
                    # Extract the rnd_state
                    # TODO: Check if we still need torch no grad. It is just an affine transformation.
                    with torch.no_grad():
                        rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                        rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                    # Predict the embedding and the target
                    predicted_embedding = self.rnd.predictor(rnd_state_batch)
                    target_embedding = self.rnd.target(rnd_state_batch).detach()
                    # Compute the loss as the mean squared error
                    mseloss = torch.nn.MSELoss()
                    rnd_loss = mseloss(predicted_embedding, target_embedding)

                if self.use_bound_loss:
                    bound_loss_coef = self.use_bound_loss["bound_loss_coef"]
                    soft_bound = self.use_bound_loss["bound_range"]
                    bound_loss = self.bound_loss(mu_batch, soft_bound)
                    loss += bound_loss_coef * bound_loss 

                if self.use_lcp_loss:
                    lcp_loss_coef = self.use_lcp_loss["gradient_penalty_coef"]
                    if self.use_lcp_loss["is_lcp"]:
                        loss += lcp_loss_coef * lcp_loss

                # Compute the gradients for PPO
                self.optimizer.zero_grad()
                loss.backward()
                # Compute the gradients for RND
                if self.rnd:
                    self.rnd_optimizer.zero_grad()
                    rnd_loss.backward()

                # Collect gradients from all GPUs
                if self.is_multi_gpu:
                    self.reduce_parameters()

                # Apply the gradients for PPO
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                # Apply the gradients for RND
                if self.rnd_optimizer:
                    self.rnd_optimizer.step()

                # Store the losses
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy_batch.mean().item()
                # RND loss
                if mean_rnd_loss is not None:
                    mean_rnd_loss += rnd_loss.item()
                # Symmetry loss
                if mean_symmetry_loss is not None:
                    mean_symmetry_loss += symmetry_loss.item()
                # -- Bound loss
                if mean_bound_loss is not None:
                    mean_bound_loss += bound_loss.item()\
                # LCP loss
                if mean_lcp_loss is not None:
                    mean_lcp_loss += lcp_loss.item()

            # Divide the losses by the number of updates
            num_updates = self.num_learning_epochs * self.num_mini_batches
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_entropy /= num_updates
            if mean_rnd_loss is not None:
                mean_rnd_loss /= num_updates
            if mean_symmetry_loss is not None:
                mean_symmetry_loss /= num_updates
            # -- For Bound loss
            if mean_bound_loss is not None:
                mean_bound_loss /= num_updates
            # -- For LCP loss
            if mean_lcp_loss is not None:
                mean_lcp_loss /= num_updates

            # Clear the storage
            self.storage.clear()

            # Construct the loss dictionary
            loss_dict = {
                "value": mean_value_loss,
                "surrogate": mean_surrogate_loss,
                "entropy": mean_entropy,
            }
            if self.rnd:
                loss_dict["rnd"] = mean_rnd_loss
            if self.symmetry:
                loss_dict["symmetry"] = mean_symmetry_loss
            if self.use_bound_loss:
                loss_dict["bound"] = mean_bound_loss
            if self.use_lcp_loss:
                loss_dict["lcp"] = mean_lcp_loss
                
            return loss_dict

    def bound_loss(self, mu, soft_bound):
        mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
        mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
        b_loss = (mu_loss_low + mu_loss_high).sum()
        return b_loss

    def _update_distribution_from_normalized_actor_obs(self, obs_batch, masks=None, hidden_state=None) -> torch.Tensor:
        """Update policy distribution using a *leaf* normalized actor-observation tensor.

        Returns:
            The leaf tensor (normalized actor obs) with requires_grad=True, which should be used as the input
            to torch.autograd.grad for LCP/gradient-penalty calculations.
        """
        actor_obs = self.policy.get_actor_obs(obs_batch)

        # CNN policies return (mlp_obs, cnn_obs_dict)
        if isinstance(actor_obs, tuple):
            mlp_obs, cnn_obs = actor_obs
            mlp_obs = self.policy.actor_obs_normalizer(mlp_obs)
            mlp_obs = mlp_obs.detach()
            mlp_obs.requires_grad_(True)
            # ActorCriticCNN expects both inputs for distribution update
            self.policy._update_distribution(mlp_obs, cnn_obs)  # type: ignore[attr-defined]
            return mlp_obs

        # Recurrent policies need memory before distribution update
        if getattr(self.policy, "is_recurrent", False):
            actor_obs = self.policy.actor_obs_normalizer(actor_obs)
            actor_obs = actor_obs.detach()
            actor_obs.requires_grad_(True)
            out_mem = self.policy.memory_a(actor_obs, masks, hidden_state).squeeze(0)  # type: ignore[attr-defined]
            self.policy._update_distribution(out_mem)  # type: ignore[attr-defined]
            return actor_obs

        # Standard MLP policy
        actor_obs = self.policy.actor_obs_normalizer(actor_obs)
        actor_obs = actor_obs.detach()
        actor_obs.requires_grad_(True)
        self.policy._update_distribution(actor_obs)  # type: ignore[attr-defined]
        return actor_obs

    def _calc_gradient_penalty(self, obs: torch.Tensor, actions_log_prob_batch, is_loss=True):
        grad_log_prob = torch.autograd.grad(
            actions_log_prob_batch.sum(),
            obs,
            create_graph=is_loss,
            retain_graph=True,
        )[0]
        gradient_penalty_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
        return gradient_penalty_loss