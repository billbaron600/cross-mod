from utils.RLBenchFunctions.train_policy import train_policy, run_train_policy
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils.RLBenchFunctions.custom_envs import instantiate_environment
import pickle
import torch
import os
from stable_baselines3 import SAC, PPO
from utils.RLBenchFunctions.train_policy import train_policy,RLBenchLoggingCallback,EntCoefScheduleCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from utils.RLBenchFunctions.fill_buffer import preload_from_config
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer,DictReplayBuffer
from utils.Classes.custom_replay_buffer import ScaledRewardBuffer, RunningMeanStd
from utils.RLBenchFunctions.relabel_reward_buffer import relabel_buffer_rewards
import numpy as np

from gymnasium import spaces
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import matplotlib.pyplot as plt
from utils.RLBenchFunctions.custom_envs import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


def plot_replay_buffer_rewards(model, *, figsize=(10, 5), show=True):
    """
    Visualise every episode stored in `model.replay_buffer`.

    • Figure 1  – per-step reward curves  
    • Figure 2  – cumulative-reward curves

    Parameters
    ----------
    model    : Stable-Baselines3-style model that owns a `replay_buffer`
    figsize  : tuple, default (10, 5) – size passed to `plt.figure`
    show     : bool,  default True    – if False, caller can overlay / save later
    """
    rb = model.replay_buffer

    # ------------------------------------------------------------------
    # 1) slice the valid portion of the buffer
    # ------------------------------------------------------------------
    dones   = rb.dones[: rb.pos]                         # (N,) bool
    rewards = rb.rewards[: rb.pos]                       # (N,) or (N,1)

    # squeeze possible singleton second dimension
    if rewards.ndim == 2 and rewards.shape[1] == 1:
        rewards = rewards.squeeze(1)

    # ------------------------------------------------------------------
    # 2) locate episode boundaries (done == True)
    # ------------------------------------------------------------------
    episodes = []
    start = 0
    for i, d in enumerate(dones):
        if d:
            episodes.append((start, i))
            start = i + 1
    if start < len(rewards):                             # unfinished final ep
        episodes.append((start, len(rewards) - 1))

    # ------------------------------------------------------------------
    # 3) Figure 1 – per-step rewards
    # ------------------------------------------------------------------
    plt.figure(figsize=figsize)
    for ep_idx, (s, e) in enumerate(episodes):
        ts = np.arange(e - s + 1)                        # timestep within ep
        plt.plot(ts, rewards[s:e + 1], label=f"ep {ep_idx}")
    plt.title("Replay-buffer per-step rewards")
    plt.xlabel("Timestep in episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    #plt.legend()
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 4) Figure 2 – cumulative rewards
    # ------------------------------------------------------------------
    plt.figure(figsize=figsize)
    for ep_idx, (s, e) in enumerate(episodes):
        r = rewards[s:e + 1]
        plt.plot(np.cumsum(r), label=f"ep {ep_idx}")
    plt.title("Replay-buffer cumulative rewards")
    plt.xlabel("Timestep in episode")
    plt.ylabel("Cumulative reward")
    plt.grid(alpha=0.3)
    #plt.legend()
    plt.tight_layout()

    if show:
        plt.show()



class RewardStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # `infos` is a list of dicts, one per parallel env
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                if "reward_mean" in info:
                    # record scalars under TB “train/…”
                    self.logger.record("train/reward_mean", info["reward_mean"])
                    self.logger.record("train/reward_std",  info["reward_std"])
                    break  # only log once per call
        return True

class ClipGradientsCallback(BaseCallback):
    def __init__(self, max_norm: float = 1.0, verbose: int = 0):
        super().__init__(verbose)
        self.max_norm = max_norm

    def _on_rollout_end(self) -> None:
        # clip both actor and critic after each rollout
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(),  self.max_norm)
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_norm)

    def _on_step(self) -> bool:
        # this must exist, but we don’t need to do anything each step
        return True
class AdaptiveCriticLrCallback(BaseCallback):
    """
    After each rollout, read the latest train/critic_loss from the logger
    and feed it into a ReduceLROnPlateau scheduler attached to the critic optimizer.
    """
    def __init__(
        self,
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.factor   = factor
        self.patience = patience
        self.min_lr   = min_lr
        # will hold our scheduler once training starts
        self.scheduler = None

    def _on_training_start(self) -> None:
        # attach scheduler to the critic optimizer
        critic_opt = self.model.critic.optimizer
        self.scheduler = ReduceLROnPlateau(
            critic_opt,
            mode='min',
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
            verbose=bool(self.verbose),
        )

    def _on_rollout_end(self) -> None:
        # fetch the most recent critic_loss from SB3’s logger
        loss = self.model.logger.name_to_value.get("train/critic_loss")
        if loss is not None:
            # feed it to the scheduler
            self.scheduler.step(loss)

    def _on_step(self) -> bool:
        # no per‑step logic needed
        return True

class BufferRewardStatsCallback(BaseCallback):
    """
    Every `log_freq` calls, compute statistics over the current replay buffer rewards:
    - mean
    - standard deviation
    - interquartile range (IQR)
    - percent of entries equal to the minimum
    - percent of entries equal to the maximum
    and log them to TensorBoard.
    """
    def __init__(self, log_freq: int = 10000, verbose: int = 0, clamp_range: int = 2):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.clamp_range = clamp_range

    def _on_step(self) -> bool:
        # Only log every `log_freq` calls
        if self.n_calls % self.log_freq == 0:
            buf = self.model.replay_buffer
            # Get only the valid portion of the buffer
            rewards = buf.rewards[: buf.pos]
            if rewards.size > 0:
                arr = rewards.reshape(-1)
                count = arr.size

                mean_val = float(arr.mean())
                std_val  = float(arr.std(ddof=0))
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                iqr_val = float(q3 - q1)

                min_val = arr.min()
                max_val = arr.max()
                count_min = np.sum(arr == min_val)
                count_max = np.sum(arr == max_val)
                pct_min = count_min / count
                pct_max = count_max / count

                std_div_range = std_val #/self.clamp_range

                # Record to TensorBoard
                self.logger.record("buffer/mean_combined_reward", mean_val)
                self.logger.record("buffer/std_combined_reward", std_val)
                self.logger.record("buffer/iqr_reward", iqr_val)
                self.logger.record("buffer/pct_min_reward", pct_min)
                self.logger.record("buffer/pct_max_reward", pct_max)
                self.logger.record("buffer/std_div_range", std_div_range)
        return True

class CriticQLogger(BaseCallback):
    """
    Log Q1/Q2 statistics from a fresh replay-buffer batch every `freq` steps.
    Works with MultiInputPolicy and buffers that store numpy *or* torch tensors.
    """

    def __init__(self, batch_size: int = 512, freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.batch_size = batch_size
        self.freq       = freq

    # --- util: tensor → cpu numpy, handle dicts -------------------------
    @staticmethod
    def _to_numpy(obj):
        if isinstance(obj, dict):
            return {k: CriticQLogger._to_numpy(v) for k, v in obj.items()}
        if torch.is_tensor(obj):
            return obj.detach().cpu().numpy()
        return obj

    # --------------------------------------------------------------------
    def _on_step(self) -> bool:
        if self.n_calls % self.freq != 0:
            return True

        rb = self.model.replay_buffer
        if rb.size() < self.batch_size:
            return True  # not enough data yet

        # -------- sample batch ------------------------------------------
        batch  = rb.sample(self.batch_size)      # no env arg → raw obs
        device = self.model.device

        # convert observations
        obs_np = self._to_numpy(batch.observations)
        obs_t, _ = self.model.policy.obs_to_tensor(obs_np)  # -> torch on device

        actions_t = torch.as_tensor(batch.actions, device=device)

        # -------- critic forward ----------------------------------------
        with torch.no_grad():
            q1, q2 = self.model.policy.critic(obs_t, actions_t)

        # -------- tensorboard scalars -----------------------------------
        self.logger.record("diag/q1_mean", q1.mean().item())
        self.logger.record("diag/q2_mean", q2.mean().item())
        self.logger.record("diag/q1_std",  q1.std(unbiased=False).item())
        self.logger.record("diag/q2_std",  q2.std(unbiased=False).item())
        self.logger.dump(self.num_timesteps)

        return True