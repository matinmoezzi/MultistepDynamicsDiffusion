from . import logger, utils
import torch
import torch.nn.functional as F

from diffusers import CMStochasticIterativeScheduler

from .ema import ExponentialMovingAverage


class DiffusionDx:
    def __init__(
        self,
        env_name,
        obs_dim,
        action_dim,
        detach_xt,
        clip_grad_norm,
        lr,
        model,
        scheduler,
        num_inference_steps,
        ema_rate,
    ):
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.detach_xt = detach_xt
        self.clip_grad_norm = clip_grad_norm
        self.lr = lr
        self.model = model
        self.scheduler = scheduler

        self.accelerator = utils.AcceleratorManager.get_accelerator()

        if hasattr(self.model, "configure_optimizers"):
            self.optimizer = self.model.configure_optimizers()
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )
        self.num_inference_steps = num_inference_steps
        self.dtype = (
            torch.float16
            if self.accelerator.mixed_precision == "fp16"
            else torch.float32
        )

        self.ema = ExponentialMovingAverage(self.model.parameters(), ema_rate)

        # Manually freeze the goal locations
        if env_name == "gym_petsReacher":
            self.freeze_dims = torch.LongTensor([7, 8, 9])
        elif env_name == "gym_petsPusher":
            self.freeze_dims = torch.LongTensor([20, 21, 22])
        else:
            self.freeze_dims = None

    def update_step(self, obs, action, reward, step):
        obs = obs.to(self.accelerator.device).to(self.dtype)
        action = action.to(self.accelerator.device).to(self.dtype)

        assert obs.dim() == 3
        T, batch_size, _ = obs.shape

        # Convert from (T, B, obs_dim) to (B, T, obs_dim)
        obs = obs.permute(1, 0, 2)
        action = action.permute(1, 0, 2)

        x0 = obs[:, 1:, :]

        noise = torch.randn(x0.shape, device=x0.device)

        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

        timesteps_idx = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            dtype=torch.int64,
        )
        if isinstance(self.scheduler, CMStochasticIterativeScheduler):
            timesteps = torch.take(self.scheduler.timesteps, timesteps_idx).to(
                self.accelerator.device
            )
        else:
            timesteps = timesteps_idx.to(self.accelerator.device)

        noisy_obs = self.scheduler.add_noise(x0, noise, timesteps)

        noise_pred = self.model(
            noisy_obs,
            timesteps.to(self.dtype),
            initial_state=obs[:, 0, :],
            actions=action[:, :-1, :],
        )
        loss = F.mse_loss(noise_pred, noise)
        self.accelerator.backward(loss)

        if self.clip_grad_norm:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm
            )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.ema.update(self.model.parameters())

        logger.logkv_mean("train_diffusion/loss", loss, step)

    def unroll(self, x, us, detach_xt=False):
        x = x.to(self.accelerator.device).to(self.dtype)
        us = us.to(self.accelerator.device).to(self.dtype)

        assert x.dim() == 2
        assert us.dim() == 3
        n_batch = x.size(0)

        if self.freeze_dims is not None:
            obs_frozen = x[:, self.freeze_dims]

        if detach_xt:
            x = x.detach()

        next_states = torch.randn(
            size=(n_batch, us.size(1), x.size(1)), device=x.device, dtype=x.dtype
        )

        self.scheduler.set_timesteps(self.num_inference_steps)

        for t in self.scheduler.timesteps:
            scaled_next_states = self.scheduler.scale_model_input(next_states, t)

            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
            model_output = self.model(
                scaled_next_states,
                t.to(x.device).to(x.dtype),
                initial_state=x,
                actions=us,
            )
            self.ema.restore(self.model.parameters())

            next_states = self.scheduler.step(model_output, t, next_states).prev_sample

        if self.freeze_dims is not None:
            next_states[..., self.freeze_dims] = obs_frozen

        return next_states

    def train(*args):
        pass
