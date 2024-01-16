import datetime
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import torch
import wandb

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = True

import copy
import os
import sys
import shutil
import time

from setproctitle import setproctitle
from tqdm import trange
from accelerate import PartialState


from multistep_dynamicsdiffusion import logger, utils
from multistep_dynamicsdiffusion.logger import DynamicsDiffusionLogger
from multistep_dynamicsdiffusion.utils import (
    AcceleratorManager,
    get_dx_model,
    get_runtime_choice,
    if_resolver,
    karras_distillation,
    steps_to_human_readable,
)

setproctitle("multistep_dynamicsdiffusion")

import hydra


from multistep_dynamicsdiffusion.video import VideoRecorder
from multistep_dynamicsdiffusion.replay_buffer import ReplayBuffer
from hydra.core.hydra_config import HydraConfig


class Workspace(object):
    def __init__(self, cfg, work_dir):
        self.cfg = cfg

        self.work_dir = work_dir

        utils.set_seed_everywhere(cfg.seed)
        self.device = AcceleratorManager.get_accelerator().device
        self.env = utils.make_norm_env(cfg)
        self.episode = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.done = False

        cfg.obs_dim = int(self.env.observation_space.shape[0])
        cfg.action_dim = self.env.action_space.shape[0]
        cfg.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]

        self.agent = hydra.utils.instantiate(cfg.agent)

        if isinstance(cfg.replay_buffer_capacity, str):
            cfg.replay_buffer_capacity = int(eval(cfg.replay_buffer_capacity))

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
            normalize_obs=cfg.normalize_obs,
        )
        self.replay_dir = os.path.join(self.work_dir, "replay")

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)

        self.step = 0
        self.steps_since_eval = 0
        self.steps_since_save = 0
        self.best_eval_rew = None

    def evaluate(self):
        distributed_state = PartialState()
        if distributed_state.local_process_index != 0:
            return
        episode_rewards = []
        for episode in range(self.cfg.num_eval_episodes):
            if self.cfg.fixed_eval:
                self.env.set_seed(episode)
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    if self.cfg.normalize_obs:
                        mu, sigma = self.replay_buffer.get_obs_stats()
                        obs_norm = (obs - mu) / sigma
                        action = self.agent.act(obs_norm, sample=False)
                    else:
                        action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(
                    action[0]
                )  # step the first action of the sequence to the environment
                self.video_recorder.record(self.env)
                episode_reward += reward
            episode_rewards.append(episode_reward)

            self.video_recorder.save(f"{self.step}.mp4")
            logger.logkv_mean("eval/episode_reward", episode_reward, self.step)
        if self.cfg.fixed_eval:
            self.env.set_seed(None)
        logger.dump(self.step)
        return np.mean(episode_rewards)

    # @profile
    def run(self):
        assert not self.done
        assert self.episode_reward == 0.0
        assert self.episode_step == 0
        self.agent.reset()
        obs = self.env.reset()

        start_time = time.time()
        for _ in trange(
            self.step,
            int(self.cfg.num_train_steps),
            initial=self.step,
            total=int(self.cfg.num_train_steps),
            desc="Training",
            file=sys.stdout,
        ):
            if self.done:
                if self.step > 0:
                    logger.logkv_mean(
                        "train/episode_reward", self.episode_reward, self.step
                    )
                    logger.logkv_mean(
                        "train/duration", time.time() - start_time, self.step
                    )
                    logger.logkv_mean("train/episode", self.episode, self.step)
                    start_time = time.time()
                    if self.step > self.cfg.num_seed_steps:
                        logger.dump(self.step)

                if self.steps_since_eval >= self.cfg.eval_freq:
                    logger.logkv_mean("eval/episode", self.episode, self.step)
                    eval_rew = self.evaluate()
                    self.steps_since_eval = 0

                    if self.best_eval_rew is None or eval_rew > self.best_eval_rew:
                        self.save(tag="best")
                        self.best_eval_rew = eval_rew

                    self.replay_buffer.save_data(self.replay_dir)
                    self.save(tag="eval_latest")

                if (
                    self.step > 0
                    and self.cfg.save_freq
                    and self.steps_since_save >= self.cfg.save_freq
                ):
                    tag = str(self.step).zfill(self.cfg.save_zfill)
                    self.save(tag="latest")
                    self.steps_since_save = 0

                if self.cfg.num_initial_states is not None:
                    self.env.set_seed(self.episode % self.cfg.num_initial_states)
                obs = self.env.reset()
                self.agent.reset()
                self.done = False
                self.episode_reward = 0
                self.episode_step = 0
                self.episode += 1

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    if self.cfg.normalize_obs:
                        mu, sigma = self.replay_buffer.get_obs_stats()
                        obs_norm = (obs - mu) / sigma
                        actions = self.agent.act(obs_norm, sample=True)
                    else:
                        actions = self.agent.act(obs, sample=True)
                action = actions[0]

            # run training update
            if self.step >= self.cfg.num_seed_steps - 1:
                if (
                    self.step
                    <= self.cfg.num_seed_steps + self.cfg.model_pretrain_seed_steps
                ):
                    self.agent._update_dx(
                        self.replay_buffer,
                        self.step,
                        self.cfg.model_pretrain_update_repeat,
                    )
                else:
                    self.agent.update(self.replay_buffer, self.step)

            next_obs, reward, self.done, _ = self.env.step(action)

            # allow infinite bootstrap
            done_float = float(self.done)
            done_no_max = (
                done_float
                if self.episode_step + 1 < self.env._max_episode_steps
                else 0.0
            )
            self.episode_reward += reward

            self.replay_buffer.add(
                obs, action, reward, next_obs, done_float, done_no_max
            )

            obs = next_obs
            self.episode_step += 1
            self.step += 1
            self.steps_since_eval += 1
            self.steps_since_save += 1

        if self.steps_since_eval > 1:
            logger.logkv_mean("eval/episode", self.episode, self.step)
            self.evaluate()

        if self.cfg.delete_replay_at_end:
            if os.path.exists(self.replay_dir):
                shutil.rmtree(self.replay_dir)

    def save(self, tag="latest"):
        if PartialState().local_process_index == 0:
            path = os.path.join(self.work_dir, f"{tag}.pt")
            torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path, map_location=PartialState().device)

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d["env"]
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        # override work_dir
        self.env = utils.make_norm_env(self.cfg)
        if "max_episode_steps" in self.cfg and self.cfg.max_episode_steps is not None:
            self.env._max_episode_steps = self.cfg.max_episode_steps
        self.episode_step = 0
        self.episode_reward = 0
        self.done = False

        if os.path.exists(self.replay_dir):
            self.replay_buffer.load_data(self.replay_dir)

        self.temp = hydra.utils.instantiate(self.cfg.agent.temp)


# Registering the custom resolver with the name 'if_else'
OmegaConf.register_new_resolver("if_else", if_resolver, replace=True)

OmegaConf.register_new_resolver("get_dx_model", get_dx_model, replace=True)

OmegaConf.register_new_resolver(
    "karras_distillation", karras_distillation, replace=True
)
OmegaConf.register_new_resolver("get_runtime_choice", get_runtime_choice, replace=True)
OmegaConf.register_new_resolver(
    "human_readable_steps", steps_to_human_readable, replace=True
)


@hydra.main(version_base=None, config_path="../config/", config_name="train")
def main(cfg):
    from train import Workspace as W

    # this needs to be done for successful pickle
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    work_dir = Path(HydraConfig.get().run.dir, "train").resolve()

    accelerator = AcceleratorManager.get_accelerator(
        cpu=True if cfg.device == "cpu" else False
    )

    log_suffix = f"[{accelerator.device}:{accelerator.local_process_index}]"
    DynamicsDiffusionLogger.configure(
        str(work_dir),
        log_frequency=cfg.log_freq,
        log_suffix=log_suffix,
        format_strs=cfg.format_strs,
    )

    wandb.init(
        project="dynamicsdiffusion",
        sync_tensorboard=True,
        config=OmegaConf.to_container(cfg),
        name=f"{cfg.agent.env_name}_{cfg.dx.scheduler._target_.split('.')[-1]}_{cfg.dx.model._target_.split('.')[-1]}_{steps_to_human_readable(cfg.agent.num_train_steps)}_{datetime.datetime.now():%Y%m%d-%H:%M:%S}",
    )

    # Choosing seed based on global rank
    cfg.seed += accelerator.process_index

    fname = cfg.checkpoint_path
    if os.path.exists(fname):
        print(f"Resuming from {fname}")
        try:
            with open(fname, "rb") as f:
                workspace = W.load(fname)
                workspace.work_dir = work_dir
                workspace.replay_dir = os.path.join(work_dir, "replay")
                workspace.cfg = cfg
        except Exception as e:
            print("Failed to load checkpoint. Starting from scratch.", e)
            workspace = W(cfg, work_dir=work_dir)
    else:
        workspace = W(cfg, work_dir=work_dir)

    workspace.run()


if __name__ == "__main__":
    main()
