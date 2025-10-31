from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils.RLBenchFunctions.custom_envs import instantiate_environment
import pickle
import torch
import os
from stable_baselines3 import SAC, PPO
#from utils.RLBenchFunctions.train_policy import RLBenchLoggingCallback,EntCoefScheduleCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from utils.RLBenchFunctions.fill_buffer import preload_from_config
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer,DictReplayBuffer
from utils.RLBenchFunctions.combine_and_flatten_obs_and_action import combine_observation_and_action

import numpy as np
from typing import Union

def expand_replay_buffer(model, new_size: int, fill_value: Union[int, float] = 0) -> None:
    """
    In-place expansion of a Stable-Baselines3 DictReplayBuffer (or subclass).

    Parameters
    ----------
    model      : SB3 algorithm instance that already owns a replay buffer.
    new_size   : int, desired capacity (must exceed current size).
    fill_value : number used for the padding slots (default 0; set to np.nan if preferred).

    Raises
    ------
    ValueError if `new_size` <= current capacity.
    """
    buf = model.replay_buffer
    old_size = buf.buffer_size
    if new_size <= old_size:
        raise ValueError(f"new_size ({new_size}) must be larger than old_size ({old_size})")

    def _pad(arr: np.ndarray) -> np.ndarray:
        pad_shape = (new_size - old_size, *arr.shape[1:])
        pad = np.full(pad_shape, fill_value, dtype=arr.dtype)
        return np.concatenate((arr, pad), axis=0)

    # ------------------------------------------------------------------
    # 1) Dict fields: observations & next_observations
    # ------------------------------------------------------------------
    for dict_name in ("observations", "next_observations"):
        if hasattr(buf, dict_name):
            d = getattr(buf, dict_name)
            for k, v in d.items():
                if isinstance(v, np.ndarray) and v.shape[0] == old_size:
                    d[k] = _pad(v)

    # ------------------------------------------------------------------
    # 2) Explicit critical arrays
    # ------------------------------------------------------------------
    for attr_name in (
        "actions",
        "rewards",
        "dones",
        "raw_rewards",
        "raw_rewards_unclipped",
        "is_demo"
    ):
        if hasattr(buf, attr_name):
            v = getattr(buf, attr_name)
            if isinstance(v, np.ndarray) and v.shape[0] == old_size:
                setattr(buf, attr_name, _pad(v))

    # ------------------------------------------------------------------
    # 3) Catch-all for any remaining numpy attributes
    # ------------------------------------------------------------------
    for attr_name in dir(buf):
        if attr_name.startswith("__") or attr_name in ("observations", "next_observations"):
            continue
        v = getattr(buf, attr_name)
        if isinstance(v, np.ndarray) and v.shape[0] == old_size:
            setattr(buf, attr_name, _pad(v))

    # ------------------------------------------------------------------
    # 4) Update meta-fields
    # ------------------------------------------------------------------
    buf.buffer_size = new_size     # inside the buffer
    model.buffer_size = new_size   # on the algorithm

    # Recompute running stats if your custom buffer supports it
    if hasattr(buf, "recompute_stats") and callable(buf.recompute_stats):
        buf.recompute_stats()

    #print(f"Replay buffer expanded from {old_size} → {new_size}")

    return model



class RunningMeanStd:
    """Track running mean and variance (Welford)."""
    def __init__(self, eps=1e-8):
        self.count = eps
        self.mean  = 0.0
        self.var   = 1.0

    def update(self, x: np.ndarray):
        self.mean = x.mean()
        self.var = x.var()



    def update(self, x: np.ndarray):
        batch_mean = x.mean()
        batch_var  = x.var()
        batch_count = x.size

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count
    




class ScaledRewardBuffer(DictReplayBuffer):
    """
    1. Normalise & clip rewards         (same as before)
    2. Keep a fixed demo-to-agent ratio (sample override)
    3. 'Warm-up → online' switch so we don't mis-label demos
    """
    def __init__(self, *args, demo_fraction=None, min_reward_threshold = None,reward_shift: float = 0.0,std_scale_factor: float = 1.0,adjust_mean_every = 5e6,success_bonus = 5.0,restrict_recent: bool = True,recent_window: int = 50_000, distribute_steps = 1, clip_limit = (-20.0,20.0),raw_clip_limit = (-20.0,20.0),**kwargs):
        super().__init__(*args, **kwargs)
        self.rms = RunningMeanStd()
        self.raw_rewards = np.zeros((self.buffer_size,1),dtype=np.float32)
        self.raw_rewards_unclipped = np.zeros((self.buffer_size,1),dtype=np.float32)
        self.std_scale_factor = std_scale_factor

        # ─── demo bookkeeping ───────────────────────────────────
        self.demo_fraction   = demo_fraction
        self.is_demo         = np.zeros(self.buffer_size, dtype=bool)
        #self.is_demo         = np.zeros(self.buffer_size, dtype=bool)
        self.infos = []
        self._online_phase   = False        # becomes True after you tag demos
        self.success_bonus = success_bonus
        self.distribute_steps = distribute_steps
        self.clip_limit = clip_limit
        self.raw_clip_limit = raw_clip_limit
        self.restrict_recent = restrict_recent
        self.recent_window = recent_window
        self.adjust_mean_every = adjust_mean_every
        self.reward_shift = reward_shift
        self.n_envs = 1

        self.min_reward_threshold = min_reward_threshold

    
    # ────────────────────────────────────────────────────────────────────
    def set_reward_threshold(self, value =  None):
        """
        If `value` is not None, every transition whose *raw* reward
        (before clipping or normalisation) is < value will be ignored:
            • it will not be counted in mean/std statistics
            • it will never be sampled in minibatches
        Pass `None` to disable the filter.
        """
        self.min_reward_threshold = value

    def _valid_mask(self, size: int) -> np.ndarray:
        """
        Boolean mask (shape = (size,)) indicating transitions that satisfy
        the reward-threshold (or everything if the threshold is off).
        """
        if self.min_reward_threshold is None:
            return np.ones(size, dtype=bool)   # keep all
        return self.raw_rewards_unclipped[:size, 0] >= self.min_reward_threshold


    # ---------- writing ----------
    def add(self, obs, next_obs, action, reward, done, infos):
        
        #SCRATCH START
        
        info_dict = infos[0]
        if info_dict.get("success") is not None:
            if info_dict.get("success") is True:
                #print("Rewards Before:")
                #print(self.rewards[self.pos-self.distribute_steps:self.pos])
                bonus = self.success_bonus
                reward = 1.0
                
            #else:
            #    r = 0.0
        #else:
        
        #    r=0.0
        self.raw_rewards_unclipped[self.pos] = reward
        self.raw_rewards[self.pos] = reward
        #SRATCH END
        
        # ─── NEW: don't write on top of demos ───────────────────────
        if self.full and self.is_demo[self.pos]:
            start = self.pos
            while self.is_demo[self.pos]:
                self.pos = (self.pos + 1) % self.buffer_size
                if self.pos == start:          # all-demo edge-case
                    raise RuntimeError("ReplayBuffer is completely filled with demos.")

        # OLD starts
        """
        r = np.asarray(reward, dtype=np.float32).reshape(-1, 1)
        info_dict = infos[0]
        

        self.raw_rewards_unclipped[self.pos] = r        
        r = np.clip(r,self.raw_clip_limit[0],self.raw_clip_limit[1])
        info_dict = infos[0]
        if info_dict.get("success") is not None:
            if info_dict.get("success") is True:
                #print("Rewards Before:")
                #print(self.rewards[self.pos-self.distribute_steps:self.pos])
                bonus = self.success_bonus
                r+=bonus

        self.raw_rewards[self.pos] = r        # keep raw
        #self.rms.update(r)                    # streaming update (optional)

        
        if self.pos % self.adjust_mean_every == 0:
            self.recompute_stats()

        
        
        #scaled = np.clip((r - self.rms.mean) /
        #                 np.sqrt(self.rms.var + 1e-8), self.clip_limit[0], self.clip_limit[1])
        
        scaled = np.clip((r - self.min) /
                         (self.max-self.min), 0, 1)
        
        scaled += self.reward_shift
        """
        scaled = reward

        super().add(obs, next_obs, action, scaled, done, infos)
        
        #Append infos to this list
        self.infos.append(infos)

        # After demos are tagged, every *new* transition is agent-generated.
        if self._online_phase:
            self.is_demo[self.pos - 1] = False

        if reward>0.0:
            #print("Added to success buffer")
            tempPos = self.pos-1
            self.is_demo[tempPos]=True
            indices_added = [tempPos]
            for i in range(300):
                tempPos-=1
                indices_added.append(tempPos)
                is_done = self.dones[tempPos]
                if is_done==True or tempPos<=0:
                    #print(indices_added)
                    break
                else:
                    self.is_demo[tempPos] = True
            
            # Compute percentage of True between indices 0 and self.pos
            percent_true = np.mean(self.is_demo[:self.pos]) * 100

            # Print a friendly summary
            n_true = self.is_demo[:self.pos].sum()
            #print(f"Demo indices [0:{self.pos}]: {n_true}/{self.pos} are True → {percent_true:.2f}%")
            



    # ------------------------------------------------------------------
    # Helper: slice the i-th element along the first dimension,
    #         for dicts OR ndarrays.
    # ------------------------------------------------------------------
    @staticmethod
    def _slice_first_dim(x, i):
        if isinstance(x, dict):
            return {k: v[i] for k, v in x.items()}
        else:
            return x[i]



    
    def manage_success_bonus(self,info_dict):
        if info_dict.get("success") is not None:
            if info_dict.get("success") is True:
                #print("Rewards Before:")
                #print(self.rewards[self.pos-self.distribute_steps:self.pos])
                bonus = self.success_bonus
                steps_to_distribute = self.distribute_steps
                current_pos = self.pos    
                self.raw_rewards[current_pos] = self.raw_rewards[current_pos] + bonus
                    
                #Recompute mean and std and scale after adding success bonus    
                self.recompute_stats()
                

    def update_clipped_rewards(self) -> None:
        """
        Clip `self.raw_rewards_unclipped` into [`low`, `high`] and store the
        result in `self.raw_rewards`.

        Assumes:
            • self.raw_clip_limit  → (low, high) tuple
            • self.pos (optional) → how many slots are actually filled
            • arrays are shape (buffer_size, 1) and dtype float32
        """
        #Apply clip to raw rewards
        raw_low, raw_high = self.raw_clip_limit
        end = getattr(self, "pos", self.raw_rewards_unclipped.shape[0])

        np.clip(
            self.raw_rewards_unclipped[:end],
            raw_low,
            raw_high,
            out=self.raw_rewards[:end],
        )

        # Add success bonus (simple for-loop, no masks, no NumPy tricks)
        if hasattr(self, "infos") and self.success_bonus != 0:
            for idx in range(end):                     # end is self.pos or buffer_size
                info_entry = self.infos[idx]
                info_dict = info_entry[0]             # already a dict

                if info_dict.get("success") is not None:
                    # Add bonus if "success" key exists and is True
                    if info_dict.get("success") is True:
                        self.raw_rewards[idx, 0] += self.success_bonus

        

        #Apply clip to standardized rewards
        #clip_low, clip_high = self.clip_limit
        #end = getattr(self, "pos", self.raw_rewards.shape[0])

        #np.clip(
        #    self.raw_rewards[:end],
        #    clip_low,
        #    clip_high,
        #    out=self.raw_rewards[:end],
        #)




    def recompute_stats(self,new_raw_clip_limit = None,new_clip_limit=None,print_len_valid=False):
        """
        Re-estimate reward mean/variance on
            {all demos}  ∪  {recent_window agents}
        but weight the two subsets according to
            demo_fraction : (1-demo_fraction)
        so the statistics reflect the mixture the
        policy actually trains on.
        """
        for i in range(0,self.pos):
            info_dict = self.infos[i][0]
            if info_dict.get("success") is not None:
                if info_dict.get("success") is True:
                    #print("Rewards Before:")
                    #print(self.rewards[self.pos-self.distribute_steps:self.pos])
                    bonus = self.success_bonus
                    self.rewards[i] = 1.0
                else:
                    self.rewards[i] = 0.0
            else:
                self.rewards[i] = 0.0    

        """
        normalzing
        size = self.buffer_size if self.full else self.pos

        if new_clip_limit is not None:
            self.clip_limit = new_clip_limit
        
        if new_raw_clip_limit is not None:
            self.raw_clip_limit = new_raw_clip_limit
            self.update_clipped_rewards()


        #if self.min_reward_threshold is not None:
        valid = self._valid_mask(size)

        if print_len_valid is True:
            print(f"Valid transitions (above threshold): {valid.sum()}")

        # ─── 1. index sets ──────────────────────────────────────────
        demo_idx  = np.flatnonzero(self.is_demo[:size] & valid)
        agent_idx = np.flatnonzero(~self.is_demo[:size] & valid)


        if self.restrict_recent:
            recent_start = max(0, size - self.recent_window)
            agent_idx    = agent_idx[agent_idx >= recent_start]

        # ── print final tally (threshold  +  recent-window filter) ──────────────
        if print_len_valid:
            total_final = demo_idx.size + agent_idx.size
            print(f"Valid transitions after threshold + recent window:")
            print(f"  demos : {demo_idx.size}")
            print(f"  agent : {agent_idx.size}")
            print(f"  total : {total_final}")

        if demo_idx.size == 0 and agent_idx.size == 0:
            return                                           # nothing to do(
        
        self.rewards[:size] = np.clip(
            (self.raw_rewards[:size] - self.min) /
            (self.max-self.min),
            0,1
        ) - self.reward_shift
        """

        """
        ORIGINAL STANDARDIZING
        # ─── 2. sample-set means / variances ───────────────────────
        demo_mean = self.raw_rewards[demo_idx].mean()  if demo_idx.size  else 0.0
        demo_var  = self.raw_rewards[demo_idx].var()   if demo_idx.size  else 0.0
        agnt_mean = self.raw_rewards[agent_idx].mean() if agent_idx.size else 0.0
        agnt_var  = self.raw_rewards[agent_idx].var()  if agent_idx.size else 0.0

        #Mjulitply variance here
        demo_var = demo_var * self.std_scale_factor
        agnt_var = agnt_var * self.std_scale_factor

        # ─── 3. weights that mirror mini-batch composition ─────────
        w_demo  = self.demo_fraction  if demo_idx.size  else 0.0
        w_agent = 1.0 - w_demo        if agent_idx.size else 0.0
        norm    = w_demo + w_agent
        if norm == 0.0:          # (defensive: should not happen)
            return
        w_demo  /= norm
        w_agent /= norm

        # ─── 4. combine by law-of-total-variance ───────────────────
        grand_mean = w_demo * demo_mean + w_agent * agnt_mean
        grand_var  = (w_demo  * (demo_var  + (demo_mean - grand_mean) ** 2) +
                    w_agent * (agnt_var  + (agnt_mean - grand_mean) ** 2))

        # ─── 5. store into running-stats object & rescale buffer ───
        self.rms = RunningMeanStd()          # fresh container
        self.rms.mean = grand_mean
        self.rms.var  = grand_var
        self.rms.count = 1.0                 # dummy – not used elsewhere

        self.rewards[:size] = np.clip(
            (self.raw_rewards[:size] - self.rms.mean) /
            np.sqrt(self.rms.var + 1e-8),
            self.clip_limit[0], self.clip_limit[1]
        ) - self.reward_shift
        """

    def sample(self, batch_size: int, env=None):
        if self.demo_fraction is None:
            return super().sample(batch_size=batch_size)
        
        size = self.buffer_size if self.full else self.pos
        valid = self._valid_mask(size)

        demo_idx  = np.flatnonzero(self.is_demo[:size] & valid)
        agent_idx = np.flatnonzero(~self.is_demo[:size] & valid)

        k_demo  = min(int(batch_size * self.demo_fraction), demo_idx.size)
        k_agent = batch_size - k_demo

        # --- pick demos ---------------------------------------------------------
        demo_part = np.random.choice(
            demo_idx, k_demo, replace=k_demo > demo_idx.size
        )

        # --- pick agents --------------------------------------------------------
        if self.restrict_recent:                          # default = True
            recent_start   = max(0, size - self.recent_window)
            recent_mask    = agent_idx >= recent_start
            recent_agents  = agent_idx[recent_mask]
        else:
            recent_agents  = agent_idx                   # entire set

        # if we have enough recent data, great; otherwise fall back to full pool
        if recent_agents.size >= k_agent:
            agent_part = np.random.choice(
                recent_agents, k_agent, replace=k_agent > recent_agents.size
            )
        else:
            # fill what we can with recent, grab the remainder from all agents
            need        = k_agent - recent_agents.size
            older_part  = np.random.choice(
                agent_idx, need, replace=need > agent_idx.size
            )
            agent_part  = np.concatenate([recent_agents, older_part])

        # --- compose final minibatch -------------------------------------------
        idx = np.concatenate([demo_part, agent_part])
        np.random.shuffle(idx)
        return self._get_samples(idx, env=env)
    

    # ---------- one-liner you call after pre-loading ----------
    def tag_first_as_demo(self,pos=None):
        """
        Mark everything currently in the buffer as demonstration data
        and switch the buffer to ‘online’ mode so future additions are
        treated as agent-generated.
        """
        if pos is None:
            upto = self.pos if not self.full else self.buffer_size
        else:
            upto = pos
        self.is_demo[:upto] = True
        self._online_phase  = True

    def relabel_rewards(
        self,
        reward_model,                 # your ensemble model
        *,
        feat_stats=None,
        safety_factor: float = 0.0,
        batch_size: int = 2048,
        device: str = "cuda"):
        """
        Re-evaluate every transition with `reward_model` and update the buffer.

        Parameters
        ----------
        reward_model   : nn.Module   (expects (B, 2, D) input, returns list[n_heads])
        safety_factor  : float       (default 0 → use mean reward only)
        feat_stats     : whatever you pass to combine_observation_and_action
        batch_size     : mini-batch size for relabelling loop
        device         : "cuda" | "cpu" | None (auto-select)
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        reward_model = reward_model.to(device).eval()

        n_filled = getattr(self, "pos", self.actions.shape[0])
        low_raw, high_raw = self.raw_clip_limit

        for start in range(0, n_filled, batch_size):
            end = min(start + batch_size, n_filled)
            B = end - start

            # -------- slice observations / next-obs / actions ----------
            obs_batch   = [{k: v[i] for k, v in self.observations.items()}
                        for i in range(start, end)]
            nobs_batch  = [{k: v[i] for k, v in self.next_observations.items()}
                        for i in range(start, end)]
            act_batch   = self.actions[start:end]

            # -------- build (B, 2, D) feature tensor ----------
            feats_now   = [combine_observation_and_action(o,  a, feat_stats)
                        for o, a in zip(obs_batch,  act_batch)]
            feats_next  = [combine_observation_and_action(no, a, feat_stats)
                        for no, a in zip(nobs_batch, act_batch)]
            seq = torch.tensor(
                    np.stack([feats_now, feats_next], axis=1),  # (B, 2, D)
                    dtype=torch.float32,
                    device=device
                )

            # -------- forward pass through ensemble ----------
            with torch.no_grad():
                outs = reward_model(seq, update_stats=False)      # list[n_heads]
                # (n_heads, B, 2)
                stacked = torch.stack(outs, dim=0).squeeze(-1)    # squeeze if last dim=1
                last_step = stacked[..., -1]                      # (n_heads, B)

                mean = last_step.mean(dim=0)                     # (B,)
                std  = last_step.std(dim=0, unbiased=False)       # (B,)
                r    = mean - safety_factor * std                # (B,)

            # -------- write back to buffer ----------
            r_np = r.cpu().numpy()
            self.raw_rewards_unclipped[start:end, 0] = r_np
            np.clip(r_np, low_raw, high_raw,
                    out=self.raw_rewards[start:end, 0])

        # refresh running statistics / normalisers
        if hasattr(self, "recompute_stats") and callable(self.recompute_stats):
            self.recompute_stats()

        print(f"Relabelled {n_filled} transitions "
            f"(safety_factor={safety_factor}).")