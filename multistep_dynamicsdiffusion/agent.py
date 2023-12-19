# Copyright (c) Facebook, Inc. and its affiliates.

import abc
import copy

import torch
import torch.nn.functional as F
from accelerate.utils.dataclasses import DynamoBackend

from . import logger, utils


class Agent(object):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    @abc.abstractmethod
    def train(self, training=True):
        """Sets the agent in either training or evaluation mode."""

    @abc.abstractmethod
    def update(self, replay_buffer, step):
        """Main function of the agent that performs learning."""

    @abc.abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""


class DynamicsDiffusionAgent(Agent):
    """DynamicsDiffusion agent."""

    def __init__(
        self,
        env_name,
        obs_dim,
        action_dim,
        action_range,
        dx,
        num_train_steps,
        temp,
        actor,
        actor_lr,
        actor_betas,
        actor_update_freq,
        actor_mve,
        actor_detach_rho,
        actor_dx_threshold,
        critic,
        critic_lr,
        critic_tau,
        critic_target_update_freq,
        critic_target_mve,
        full_target_mve,
        discount,
        seq_batch_size,
        seq_train_length,
        step_batch_size,
        update_freq,
        model_update_freq,
        rew_hidden_dim,
        rew_hidden_depth,
        rew_lr,
        done_hidden_dim,
        done_hidden_depth,
        done_lr,
        done_ctrl_accum,
        model_update_repeat,
        model_free_update_repeat,
        horizon,
        warmup_steps,
        det_suffix,
        traj_optimizer,
        num_particles,
    ):
        super().__init__()
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.num_train_steps = num_train_steps
        self.det_suffix = det_suffix
        self.num_particles = num_particles

        self.accelerator = utils.AcceleratorManager.get_accelerator()

        self.dx = dx
        self.actor = actor

        self.discount = discount
        self.discount_horizon = torch.tensor(
            [discount**i for i in range(horizon)]
        ).to(self.accelerator.device)
        self.seq_batch_size = seq_batch_size

        self.seq_train_length = seq_train_length

        self.step_batch_size = step_batch_size
        self.update_freq = update_freq
        self.model_update_repeat = model_update_repeat
        self.model_update_freq = model_update_freq
        self.model_free_update_repeat = model_free_update_repeat

        self.horizon = horizon

        self.warmup_steps = warmup_steps

        self.temp = temp

        self.rew = utils.mlp(obs_dim + action_dim, rew_hidden_dim, 1, rew_hidden_depth)
        self.rew_opt = torch.optim.Adam(self.rew.parameters(), lr=rew_lr)
        self.rew, self.rew_opt = self.accelerator.prepare(self.rew, self.rew_opt)

        self.done = utils.mlp(
            obs_dim + action_dim, done_hidden_dim, 1, done_hidden_depth
        )
        self.done_ctrl_accum = done_ctrl_accum
        self.done_opt = torch.optim.Adam(self.done.parameters(), lr=done_lr)
        self.done, self.done_opt = self.accelerator.prepare(self.done, self.done_opt)

        self.critic = critic
        if critic is not None:
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_target.train()
        if self.critic is not None:
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
            self.critic_tau = critic_tau
            self.critic_target_update_freq = critic_target_update_freq

        self.critic_target_mve = critic_target_mve
        self.full_target_mve = full_target_mve
        if critic_target_mve or full_target_mve:
            assert self.critic is not None

        if full_target_mve:
            assert ~critic_target_mve

        self.critic, self.critic_opt = self.accelerator.prepare(
            self.critic, self.critic_opt
        )
        self.critic_target = self.accelerator.prepare(self.critic_target)

        mods = [self.actor]
        params = utils.get_params(mods)
        self.actor_opt = torch.optim.Adam(params, lr=actor_lr, betas=actor_betas)
        self.actor_update_freq = actor_update_freq
        self.actor_mve = actor_mve
        self.actor_detach_rho = actor_detach_rho
        self.actor_dx_threshold = actor_dx_threshold

        self.actor, self.actor_opt = self.accelerator.prepare(
            self.actor, self.actor_opt
        )

        self.train()
        self.last_step = 0
        self.rolling_dx_loss = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["dx"].model = self.accelerator.unwrap_model(d["dx"].model)
        d["actor"] = self.accelerator.unwrap_model(d["actor"])
        d["critic"] = self.accelerator.unwrap_model(d["critic"])
        d["critic_target"] = self.accelerator.unwrap_model(d["critic_target"])
        d["done"] = self.accelerator.unwrap_model(d["done"])
        d["rew"] = self.accelerator.unwrap_model(d["rew"])
        if self.accelerator.state.dynamo_plugin.backend != DynamoBackend.NO:
            d["dx"].model = d["dx"].model._orig_mod
            d["actor"] = d["actor"]._orig_mod
            d["critic"] = d["critic"]._orig_mod
            d["critic_target"] = d["critic_target"]._orig_mod
            d["done"] = d["done"]._orig_mod
            d["rew"] = d["rew"]._orig_mod
        return d

    def __setstate__(self, d):
        self.__dict__ = d

        self.actor, self.actor_opt = self.accelerator.prepare(
            self.actor, self.actor_opt
        )
        self.critic_target = self.accelerator.prepare(self.critic_target)
        self.critic, self.critic_opt = self.accelerator.prepare(
            self.critic, self.critic_opt
        )
        self.done, self.done_opt = self.accelerator.prepare(self.done, self.done_opt)
        self.rew, self.rew_opt = self.accelerator.prepare(self.rew, self.rew_opt)
        self.dx.model = self.accelerator.prepare(self.dx.model)

        if "full_target_mve" not in d:
            self.full_target_mve = False

        if "actor_dx_threshold" not in d:
            self.actor_dx_threshold = None
            self.rolling_dx_loss = None

    def train(self, training=True):
        self.training = training
        self.dx.train(training)
        self.rew.train(training)
        self.done.train(training)
        self.actor.train(training)
        if self.critic is not None:
            self.critic.train(training)

    def reset(self):
        pass

    def act(self, obs, sample=False):
        obs = torch.tensor(
            obs,
            device=self.accelerator.device,
            dtype=torch.float16
            if self.accelerator.mixed_precision == "fp16"
            else torch.float32,
        )
        obs = obs.unsqueeze(dim=0)

        if not sample:
            actions, _, _ = self.actor(obs)
        else:
            with torch.no_grad():
                _, actions, _ = self.actor(obs)

        actions = actions.clamp(*self.action_range)
        assert actions.ndim == 3 and actions.shape[0] == 1
        return utils.to_np(actions[0])

    def expand_Q(self, xs, critic, sample=True, discount=False):
        assert xs.dim() == 2
        n_batch = xs.size(0)
        _, us, log_p_us = self.actor(xs)
        pred_obs = self.dx.unroll(xs, us[:, :-1], detach_xt=self.actor_detach_rho)

        all_obs = torch.cat((xs.unsqueeze(1), pred_obs), dim=1)
        xu = torch.cat((all_obs, us), dim=2)
        dones = self.done(xu).sigmoid().squeeze(dim=2)
        not_dones = 1.0 - dones
        not_dones = utils.accum_prod(not_dones)
        rewards = not_dones * self.rew(xu).squeeze(2)
        rewards -= self.temp.alpha.detach() * log_p_us.squeeze()
        if discount:
            rewards *= self.discount_horizon.unsqueeze(0)
        total_rewards = rewards.sum(dim=1).unsqueeze(1)

        if critic is not None:
            with utils.eval_mode(critic):
                _, us_, log_prob_ = self.actor(all_obs[:, -1])
                q1, q2 = critic(all_obs[:, -1], us_)
            q = torch.min(q1, q2)
            last_not_dones = not_dones[:, -1]
            terminal_reward = q - self.temp.alpha.detach() * log_prob_
            if discount:
                terminal_reward *= self.discount**self.horizon
            total_terminal_rewards = terminal_reward.sum(dim=1)
            total_rewards += last_not_dones.view(-1, 1) * total_terminal_rewards

        first_log_p = log_p_us[0]
        total_log_p_us = log_p_us.sum(dim=1)
        return total_rewards, first_log_p, total_log_p_us

    def update_actor_and_alpha(self, xs, step):
        assert xs.ndimension() == 2
        n_batch, _ = xs.size()

        do_model_free_update = (
            step < self.warmup_steps
            or self.horizon == 0
            or not self.actor_mve
            or (
                self.actor_dx_threshold is not None
                and self.rolling_dx_loss is not None
                and self.rolling_dx_loss > self.actor_dx_threshold
            )
        )

        if do_model_free_update:
            # Do vanilla SAC updates while the model warms up.
            # i.e., fit to just the Q function
            _, pi, first_log_p = self.actor(xs)
            actor_Q1, actor_Q2 = self.critic(xs, pi)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.temp.alpha.detach() * first_log_p - actor_Q).mean()
        else:
            # Switch to the model-based updates.
            # i.e., fit to the controller's sequence cost
            rewards, first_log_p, total_log_p_us = self.expand_Q(
                xs, self.critic, sample=True, discount=True
            )
            assert total_log_p_us.size() == rewards.size()
            actor_loss = -(rewards / self.horizon).mean()

        logger.logkv_mean("train_actor/loss", actor_loss, step)
        logger.logkv_mean("train_actor/entropy", -first_log_p.mean(), step)

        self.actor_opt.zero_grad()
        self.accelerator.backward(actor_loss)
        self.actor_opt.step()
        self.temp.update(first_log_p, step)

        logger.logkv_mean("train_alpha/value", self.temp.alpha, step)

    def update_critic(self, xs, xps, us, rs, not_done, step):
        assert xs.ndimension() == 2
        n_batch, _ = xs.size()
        rs = rs.squeeze().permute(1, 0)
        us = us.permute(1, 0, 2)
        not_done = not_done.squeeze().permute(1, 0)

        with torch.no_grad():
            if not self.critic_target_mve or step < self.warmup_steps:
                mu, target_us, log_pi = self.actor(xps)
                log_pi = log_pi.squeeze(1)

                target_Q1, target_Q2 = [
                    Q.squeeze(1) for Q in self.critic_target(xps, target_us)
                ]
                target_Q = (
                    torch.min(target_Q1, target_Q2) - self.temp.alpha.detach() * log_pi
                ).squeeze()
                assert target_Q.size() == rs.size()
                assert target_Q.ndimension() == 2
                target_Q = rs + not_done * self.discount * target_Q
                target_Q = target_Q.detach()
            else:
                target_Q, first_log_p, total_log_p_us = self.expand_Q(
                    xps, self.critic_target, sample=True, discount=True
                )
                target_Q = target_Q - self.temp.alpha.detach() * first_log_p
                target_Q = rs + not_done * self.discount * target_Q
                target_Q = target_Q.detach()

        current_Q1, current_Q2 = [Q.squeeze(2) for Q in self.critic(xs, us)]
        assert current_Q1.size() == target_Q.size()
        assert current_Q2.size() == target_Q.size()
        Q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        logger.logkv_mean("train_critic/Q_loss", Q_loss, step)
        current_Q = torch.min(current_Q1, current_Q2)
        logger.logkv_mean("train_critic/value", current_Q.mean(), step)

        self.critic_opt.zero_grad()
        self.accelerator.backward(Q_loss)
        logger.logkv_mean("train_critic/Q_loss", Q_loss, step)
        self.critic_opt.step()

    def update_critic_mve(
        self, first_xs, first_us, first_rs, next_xs, first_not_dones, step
    ):
        """MVE critic loss from Feinberg et al (2015)"""
        assert first_xs.dim() == 2
        assert first_us.dim() == 2
        assert first_rs.dim() == 2
        assert next_xs.dim() == 2
        assert first_not_dones.dim() == 2
        n_batch = next_xs.size(0)

        # unroll policy, concatenate obs and actions
        pred_us, log_p_us, pred_xs = self.dx.unroll_policy(
            next_xs, self.actor, sample=True, detach_xt=self.actor_detach_rho
        )
        all_obs = torch.cat((first_xs.unsqueeze(0), next_xs.unsqueeze(0), pred_xs))
        all_us = torch.cat([first_us.unsqueeze(0), pred_us])
        xu = torch.cat([all_obs, all_us], dim=2)
        horizon_len = all_obs.size(0) - 1  # H

        # get immediate rewards
        pred_rs = self.rew(xu[1:-1])  # t from 0 to H - 1
        rewards = torch.cat([first_rs.unsqueeze(0), pred_rs]).squeeze(2)
        rewards = rewards.unsqueeze(1).expand(-1, horizon_len, -1)
        log_p_us = log_p_us.unsqueeze(1).expand(-1, horizon_len, -1)

        # get not dones factor matrix, rows --> t, cols --> k
        first_not_dones = first_not_dones.unsqueeze(0)
        init_not_dones = torch.ones_like(
            first_not_dones
        )  # we know the first states are not terminal
        pred_not_dones = 1.0 - self.done(xu[2:]).sigmoid()  # t from 1 to H
        not_dones = torch.cat(
            [init_not_dones, first_not_dones, pred_not_dones]
        ).squeeze(2)
        not_dones = not_dones.unsqueeze(1).repeat(1, horizon_len, 1)
        triu_rows, triu_cols = torch.triu_indices(
            row=horizon_len + 1, col=horizon_len, offset=1, device=not_dones.device
        )
        not_dones[triu_rows, triu_cols, :] = 1.0
        not_dones = not_dones.cumprod(dim=0).detach()

        # get lower-triangular reward discount factor matrix
        discount = torch.tensor(self.discount, device=rewards.device)
        discount_exps = torch.stack(
            [torch.arange(-i, -i + horizon_len) for i in range(horizon_len)], dim=1
        )
        r_discounts = discount ** discount_exps.to(rewards.device)
        r_discounts = r_discounts.tril().unsqueeze(-1)

        # get discounted sums of soft rewards (t from -1 to H - 1 (k from t to H - 1))
        alpha = self.temp.alpha.detach()
        soft_rewards = (not_dones[:-1] * rewards) - (
            discount * alpha * not_dones[1:] * log_p_us
        )
        soft_rewards = (r_discounts * soft_rewards).sum(0)

        # get target q-values, final critic targets
        target_q1, target_q2 = self.critic_target(all_obs[-1], all_us[-1])
        target_qs = torch.min(target_q1, target_q2).squeeze(-1).expand(horizon_len, -1)
        q_discounts = discount ** torch.arange(horizon_len, 0, step=-1).to(
            target_qs.device
        )
        target_qs = target_qs * (not_dones[-1] * q_discounts.unsqueeze(-1))
        critic_targets = (soft_rewards + target_qs).detach()

        # get predicted q-values
        with utils.eval_mode(self.critic):
            q1, q2 = self.critic(
                all_obs[:-1].flatten(end_dim=-2), all_us[:-1].flatten(end_dim=-2)
            )
            q1, q2 = q1.reshape(horizon_len, n_batch), q2.reshape(horizon_len, n_batch)
        assert q1.size() == critic_targets.size()
        assert q2.size() == critic_targets.size()

        # update critics
        q1_loss = (not_dones[:-1, 0] * (q1 - critic_targets).pow(2)).mean()
        q2_loss = (not_dones[:-1, 0] * (q2 - critic_targets).pow(2)).mean()
        Q_loss = q1_loss + q2_loss

        logger.logkv_mean("train_critic/Q_loss", Q_loss, step)
        current_Q = torch.min(q1, q2)
        logger.logkv_mean("train_critic/value", current_Q.mean(), step)

        self.critic_opt.zero_grad()
        self.accelerator.backward(Q_loss)
        logger.logkv_mean("train_critic/Q_loss", Q_loss, step)
        self.critic_opt.step()
        if hasattr(self.critic, "module"):
            self.critic.module.log(step)
        else:
            self.critic.log(step)

    def update(self, replay_buffer, step):
        self.last_step = step
        if step % self.update_freq != 0:
            return

        if (
            (self.horizon > 1 or not self.critic)
            and (step % self.model_update_freq == 0)
            and (self.actor_mve or self.critic_target_mve)
        ):
            for i in range(self.model_update_repeat):
                obses, actions, rewards = replay_buffer.sample_multistep(
                    self.seq_batch_size, self.seq_train_length
                )
                assert obses.ndimension() == 3
                dx_loss = self.dx.update_step(obses, actions, rewards, step)
                if self.actor_dx_threshold is not None:
                    if self.rolling_dx_loss is None:
                        self.rolling_dx_loss = dx_loss
                    else:
                        factor = 0.9
                        self.rolling_dx_loss = (
                            factor * self.rolling_dx_loss + (1.0 - factor) * dx_loss
                        )

        n_updates = 1 if step < self.warmup_steps else self.model_free_update_repeat
        for i in range(n_updates):
            (
                obs,
                actions,
                rewards,
                next_obs,
                not_done,
                not_done_no_max,
            ) = replay_buffer.sample_planning(self.step_batch_size, self.horizon)

            if self.critic is not None:
                if self.full_target_mve:
                    self.update_critic_mve(
                        obs, actions, rewards, next_obs, not_done_no_max, step
                    )
                else:
                    self.update_critic(
                        obs, next_obs, actions, rewards, not_done_no_max, step
                    )

            if step % self.actor_update_freq == 0:
                self.update_actor_and_alpha(obs, step)

            if self.rew_opt is not None:
                self.update_rew_step(obs, actions[0], rewards[0], step)

            self.update_done_step(obs, actions[0], not_done_no_max[0], step)

            if self.critic is not None and step % self.critic_target_update_freq == 0:
                utils.soft_update_params(
                    self.critic, self.critic_target, self.critic_tau
                )

    def update_rew_step(self, obs, action, reward, step):
        assert obs.dim() == 2
        reward = reward.unsqueeze(1)

        xu = torch.cat((obs, action), dim=1)
        pred_reward = self.rew(xu)
        assert pred_reward.size() == reward.size()
        reward_loss = F.mse_loss(pred_reward, reward, reduction="mean")

        self.rew_opt.zero_grad()
        self.accelerator.backward(reward_loss)
        self.rew_opt.step()

        logger.logkv_mean("train_model/reward_loss", reward_loss, step)

    def update_done_step(self, obs, action, not_done, step):
        assert obs.dim() == 2
        batch_size, _ = obs.shape

        done = 1.0 - not_done

        xu = torch.cat((obs, action), dim=1)

        pred_logits = self.done(xu)
        n_done = torch.sum(done)
        if n_done > 0.0:
            pos_weight = (batch_size - n_done) / n_done
        else:
            pos_weight = torch.tensor(1.0)
        done_loss = F.binary_cross_entropy_with_logits(
            pred_logits, done, pos_weight=pos_weight, reduction="mean"
        )

        self.done_opt.zero_grad()
        self.accelerator.backward(done_loss)
        self.done_opt.step()

        logger.logkv_mean("train_model/done_loss", done_loss, step)

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: torch.Tensor,
        num_particles: int,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (torch.Tensor): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        with torch.no_grad():
            batch_size, population_size, horizon, _ = action_sequences.shape
            initial_obs_batch = initial_state.unsqueeze(1).repeat(
                1, num_particles * population_size, 1
            )
            initial_obs_batch = initial_obs_batch.flatten(0, 1)
            action_sequences = action_sequences.repeat_interleave(num_particles, dim=1)
            action_sequences = action_sequences.flatten(0, 1)

            pred_next_states = self.dx.unroll(initial_obs_batch, action_sequences)

            pred_next_states[:, 0] = initial_obs_batch
            pred_next_states = pred_next_states.permute(1, 0, 2)
            action_sequences = action_sequences.permute(1, 0, 2)
            xu = torch.cat((pred_next_states, action_sequences), dim=2)
            dones = self.done(xu).sigmoid().squeeze(dim=2)
            not_dones = 1.0 - dones
            not_dones = utils.accum_prod(not_dones)
            rewards = not_dones * self.rew(xu).squeeze(2)

            total_rewards = rewards.sum(dim=0)

            total_rewards = total_rewards.unflatten(
                0, (batch_size, population_size, num_particles)
            )
            return total_rewards.mean(dim=-1)

    def trajectory_eval_fn(self, initial_state, action_sequences):
        return self.evaluate_action_sequences(
            action_sequences,
            initial_state=initial_state,
            num_particles=self.num_particles,
        )
