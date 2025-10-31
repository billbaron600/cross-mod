import copy
import numpy as np
import torch
import os
from utils.RLBenchFunctions.fill_buffer import preload_from_config
from utils.Classes.custom_replay_buffer import ScaledRewardBuffer, RunningMeanStd
from utils.RLBenchFunctions.combine_and_flatten_obs_and_action import combine_observation_and_action
import pickle

def batched_forward(x, model, chunk=8192):
    """
    Run `model` (an ensemble) in slices so we never hand the transformer
    more than `chunk` tokens at once.

    Returns the same structure as a single forward call:
        list[ Tensor(B, 1, 1) ]   # len == ensemble size
    """
    outputs_per_member = None           # will become list[list[Tensor]]

    with torch.no_grad():
        for i in range(0, x.size(0), chunk):
            y_list = model(x[i:i + chunk], update_stats=False)   # list[length=M]

            if outputs_per_member is None:                       # initialise
                outputs_per_member = [[] for _ in range(len(y_list))]

            for m, y in enumerate(y_list):
                outputs_per_member[m].append(y)                  # append chunk

    # concatenate chunks for each member
    return [torch.cat(member_chunks, dim=0) for member_chunks in outputs_per_member]

def relabel_buffer_rewards(model, config,
                           safety_factor=1.0, success_bonus=0.0,clip_limit = (-7.0,7.0),
                           device="cuda",hard_sync=True,db_index=None):
    """
    Build a *fresh* replay buffer containing:
      • all expert / non-expert demos (via preload_from_config)
      • all existing agent-generated transitions *re-scored*
        with the current reward ensemble

    Returns the new buffer and also patches model.replay_buffer in-place.
    """
    old_buf = model.replay_buffer
    if old_buf is None or old_buf.pos == 0:
        raise ValueError("Nothing to relabel - the old replay buffer is empty")

    # ── 1. create an empty buffer with the same spec ─────────────────────
    new_buf = ScaledRewardBuffer(
        buffer_size      = old_buf.buffer_size,
        observation_space= old_buf.observation_space,
        action_space     = old_buf.action_space,
        clip_limit       = clip_limit,
        device           = old_buf.device,
        optimize_memory_usage = old_buf.optimize_memory_usage,
        # keep any custom kwargs you use (demo_fraction, etc.)
    )

    # temporarily swap so preload_from_config pushes into *new* buffer
    model.replay_buffer = new_buf
    preload_from_config(model,
                        config,
                        success_bonus=success_bonus,
                        device=device,
                        safety_factor=safety_factor,
                        db_index=db_index)
    new_buf.tag_first_as_demo() #maek those as the new demo

    # ── 2. prepare reward ensemble once ──────────────────────────────────
    reward_path = os.path.join(config.iteration_working_dir,
                               "reward_model_ensemble.pt")
    reward_ensemble = torch.load(reward_path, map_location=device)
    reward_ensemble = reward_ensemble.to(device).eval()

    # number of transitions already written (all demos)
    demo_cutoff = new_buf.pos if not new_buf.full else new_buf.buffer_size

    # ── 3. collect agent transitions and build one big feature tensor ────────
    size_old   = old_buf.buffer_size if old_buf.full else old_buf.pos
    agent_idx  = np.arange(demo_cutoff, size_old)

    transitions = []            # (obs, next_obs, action, done, info, feat_np)
    feats_list  = []

    #oad the updated preference database
    db_path = os.path.join(config.iteration_working_dir, "preference_database_"+str(db_index)+".pkl")
    with open(db_path, "rb") as f:
        db = pickle.load(f)

    for idx in agent_idx:
        obs      = {k: v[idx] for k, v in old_buf.observations.items()}
        next_obs = {k: v[idx] for k, v in old_buf.next_observations.items()}
        action   = old_buf.actions[idx]
        done     = old_buf.dones[idx]
        info     = old_buf.infos[idx]

        feat_np = combine_observation_and_action(
            obs, action, include_action=False, feat_stats=db.feat_stats
        )
        feats_list.append(feat_np)
        transitions.append((obs, next_obs, action, done, info))

    feat_batch = (torch.tensor(np.stack(feats_list, axis=0),  # (B, D)
        dtype=torch.float32,
        device=device).unsqueeze(1))
    
    # ── 4. single reward-ensemble forward pass ───────────────────────────────
    with torch.no_grad():
        print(feat_batch.shape)
        #outs    = reward_ensemble(feat_batch, update_stats=False)
        outs = batched_forward(feat_batch, reward_ensemble)
        stacked = torch.stack(outs, dim=0).squeeze(-1)       # (heads, B)
        mean_v  = stacked.mean(0)                            # (B,)
        std_v   = stacked.std(0, unbiased=False)             # (B,)
        rewards = mean_v - safety_factor * std_v             # (B,)

    rewards = rewards.cpu().numpy()                          # back to NumPy

    # ── 5. write transitions into the *new* buffer via .add ──────────────────
    for i, (obs, next_obs, action, done, info) in enumerate(transitions):
        
        #print(i)
        r = rewards[i]
        if isinstance(info, dict) and info.get("success"):
            r += success_bonus

        #r = np.clip(r, new_buf.clip_limit[0], new_buf.clip_limit[1]).astype(np.float32)
        new_buf.add(obs, next_obs, action, r, done, info)

    # ── 4. renormalise rewards once with a full pass ─────────────────────
    new_buf.rms = RunningMeanStd()               # reset
    upto = new_buf.buffer_size if new_buf.full else new_buf.pos
    new_buf.rms.update(new_buf.raw_rewards[:upto].flatten())
    new_buf.recompute_stats()                    # z-score + clip in place

    # ── 5. activate online phase and swap buffer back into the model ────
    model.replay_buffer = new_buf

    
    return model