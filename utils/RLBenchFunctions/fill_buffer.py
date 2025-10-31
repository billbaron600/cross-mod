import os
import pickle
from itertools import chain

import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import SAC
import os, pickle, torch
from itertools import chain
from stable_baselines3.common.buffers import DictReplayBuffer
from utils.RLBenchFunctions.combine_and_flatten_obs_and_action import combine_observation_and_action
import numpy as np
from utils.Classes.policy_trajectory_class import PolicyTrajectory
from typing import List
#from utils.Classes.custom_replay_buffer import ScaledRewardBuffer
from typing import List, Tuple

import matplotlib.pyplot as plt
from statistics import median, mean
import random
from typing import List, Optional

def plot_seed_success_stats(
    config,
    *,
    filename: str = "full_ivk_trajectories.pkl",
    show: bool = True,
):
    """
    For each seed in `config.seeds`, load the pickled trajectory file and plot:
      1.  The number of successful trajectories.
      2.  The average length (in observations) of those successful trajectories.

    Parameters
    ----------
    config : object
        Must expose
            • iteration_working_dir : str
            • seeds                : Iterable (e.g. list[int] or list[str])
    filename : str, optional
        Name of the pickle file to load in every seed directory.
    show : bool, optional
        If True (default) call plt.show() after plotting.

    Returns
    -------
    successes_per_seed : List[int]
        Number of successful demos for each seed (in the order of config.seeds).
    avg_len_per_seed   : List[float]
        Average length of *successful* demos for each seed.
    """
    successes_per_seed: List[int] = []
    avg_len_per_seed: List[float] = []
    seed_labels: List[str] = []

    for seed in config.seeds:
        seed_path = os.path.join(config.iteration_working_dir, str(seed))
        pkl_path = os.path.join(seed_path, filename)

        if not os.path.isfile(pkl_path):
            print(f"[WARNING] Missing file: {pkl_path} – skipping this seed.")
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        successful = data.get("successful", [])
        # The trajectories list may be stored under either key
        trajs = data.get("trajectories", data.get("full_trajectories", []))

        success_indices = [i for i, ok in enumerate(successful) if ok]
        n_success = len(success_indices)
        lengths = [
            len(trajs[i].observations) for i in success_indices
        ] if success_indices else []

        avg_len = float(sum(lengths) / len(lengths)) if lengths else 0.0

        successes_per_seed.append(n_success)
        avg_len_per_seed.append(avg_len)
        seed_labels.append(str(seed))

    # ── Plot 1: number of successful trajectories ───────────────────────
    plt.figure(figsize=(8, 4))
    plt.bar(seed_labels, successes_per_seed)
    plt.xlabel("Seed")
    plt.ylabel("# Successful Trajectories")
    plt.title("Successful Trajectories per Seed")
    plt.tight_layout()

    # ── Plot 2: average length of successful trajectories ───────────────
    plt.figure(figsize=(8, 4))
    plt.bar(seed_labels, avg_len_per_seed)
    plt.xlabel("Seed")
    plt.ylabel("Average Length (observations)")
    plt.title("Average Length of Successful Trajectories per Seed")
    plt.tight_layout()

    if show:
        plt.show()

    #return successes_per_seed, avg_len_per_seed


class ExpertDatabase_DEPRECATED:
    """
    Splits IVK trajectories into expert / non-expert buckets and
    patches the final `info` dict of each expert rollout to flag success.

    Parameters
    ----------
    config : Any
        Has an `iteration_working_dir` attribute containing *full_ivk_trajectories.pkl*.
    min_length : int, optional
        Trajectories shorter than this (len(traj.observations)) are discarded.
    """

    def __init__(self, config, *, min_length: int = 30) -> None:
        self.expert_demos: List = []
        self.non_expert_demos: List = []

        # 1. Load file --------------------------------------------------------
        pkl_path = os.path.join(config.iteration_working_dir,
                                "full_ivk_trajectories.pkl")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        trajectories = data["full_trajectories"]
        successful   = data["successful"]

        # 2. Filter / bucket --------------------------------------------------
        for traj, is_successful in zip(trajectories, successful):
            if len(traj.observations) < min_length:
                continue

            if is_successful or traj.rewards[-1]>0.0:
                # -- ensure success flag is present in final info ------------
                if not traj.info:                       # safety: empty list
                    traj.info = [{}]
                last_info = traj.info[-1]
                if last_info.get("success") is not True:
                    last_info["success"] = True
                self.expert_demos.append(traj)
            else:
                self.non_expert_demos.append(traj)

    # Convenience helpers -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.expert_demos) + len(self.non_expert_demos)

    def __repr__(self) -> str:
        return (f"<ExpertDatabase | experts={len(self.expert_demos)} | "
                f"non-experts={len(self.non_expert_demos)}>")

    def all(self) -> List:
        """Concatenate expert and non-expert trajectories."""
        return self.expert_demos + self.non_expert_demos


class ExpertDatabase:
    """
    Splits IVK trajectories into expert / non-expert buckets and (optionally)
    balances the number of expert demos drawn from each seed directory.

    Parameters
    ----------
    config         : object with `iteration_working_dir` attribute.
    results_path   : str | None               (see docstring above)
    sample_evenly  : bool                     (default False)
    min_length     : int                      (default 30)
    """

    def __init__(self, config,
                 *, results_path=None,
                 n_trajs_to_keep = None,
                 n_percent_shorts=None,
                 sample_evenly: bool = False,
                 min_length: int = 30,
                 use_mean_instead_of_median: bool=False):

        self.expert_demos: List = []
        self.non_expert_demos: List = []
        self.seed_expert_points = {}       # 🔹 new: seed → total obs count
        self.use_mean_instead_of_median = use_mean_instead_of_median

        # ------------------------------------------------------------------
        # 1. Load trajectories either from a combined file or per-seed dirs
        # ------------------------------------------------------------------
        if results_path is None:
            pkl_path = os.path.join(config.iteration_working_dir,
                                    "full_ivk_trajectories.pkl")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            sources = [("combined", data)]
        else:
            results_path = os.path.abspath(results_path)
            seed_dirs = sorted([d for d in os.listdir(results_path) if d.isdigit()],
                               key=int)
            if not seed_dirs:
                raise FileNotFoundError(f"No numeric sub-folders in {results_path}")

            sources = []
            for sd in seed_dirs:
                pkl_path = os.path.join(results_path, sd, "full_ivk_trajectories.pkl")
                with open(pkl_path, "rb") as f:
                    sources.append((sd, pickle.load(f)))

        # ------------------------------------------------------------------
        # 2. Split into expert / non-expert per seed
        # ------------------------------------------------------------------
        rng = random.Random(42)
        per_seed_success = {}
        per_seed_fail    = {}

        for seed, data in sources:
            seed_success, seed_fail = [], []

            for traj, is_successful in zip(data["full_trajectories"],
                                           data["successful"]):
                if len(traj.observations) < min_length:
                    continue

                if is_successful or getattr(traj, "rewards", [-1])[-1] > 0.0:
                    if not traj.info:                      # ensure success flag
                        traj.info = [{}]
                    traj.info[-1]["success"] = True
                    seed_success.append(traj)
                else:
                    seed_fail.append(traj)

            per_seed_success[seed] = seed_success
            per_seed_fail[seed]    = seed_fail
            # 🔹 total expert points for this seed
            self.seed_expert_points[seed] = sum(len(t.observations) for t in seed_success)

        # ------------------------------------------------------------------
        # 3. Optional balanced sampling
        # ------------------------------------------------------------------
        if n_percent_shorts is not None:
            if not (0.0 < n_percent_shorts <= 1.0):
                raise ValueError("n_percent_shorts must be in (0, 1].")

            print("\n--- Fractional sampling summary --------------------------")
            counts = [len(v) for v in per_seed_success.values()]
            print("Successful counts per seed :", counts)
            print(f"Requested fraction         : {n_percent_shorts:.2f}")

            for seed, trajs in per_seed_success.items():
                k = max(1, int(round(len(trajs) * n_percent_shorts)))
                sampled = rng.sample(trajs, k) if k < len(trajs) else trajs
                self.expert_demos.extend(sampled)
                print(f"Seed {seed}: taking {k}/{len(trajs)} expert demos "
                      f"({self.seed_expert_points[seed]} obs)")
        else:
            if n_trajs_to_keep is not None:
                counts = [len(v) for v in per_seed_success.values()]
                print("\n--- Fixed-per-seed sampling summary -----------------------")
                print("Successful counts per seed :", counts)
                print(f"Requested keep per seed    : {n_trajs_to_keep}")
                print("Expert points per seed      :",
                    [self.seed_expert_points[s] for s in per_seed_success])

                M = n_trajs_to_keep
                print("\n--- Even sampling summary --------------------------------")
                print("Successful counts per seed :", counts)
                print("Number of Successful Counts set Explictly:", M)
                # 🔹 print expert-point counts
                print("Expert points per seed      :", [self.seed_expert_points[s] for s in per_seed_success])

                for seed, trajs in per_seed_success.items():
                    k = min(M, len(trajs))
                    sampled = rng.sample(trajs, k) if k < len(trajs) else trajs
                    self.expert_demos.extend(sampled)
                    print(f"Seed {seed}: taking {k}/{len(trajs)} expert demos "
                        f"({self.seed_expert_points[seed]} obs)")
            elif results_path and sample_evenly:
                counts = [len(v) for v in per_seed_success.values()]
                if not counts:
                    raise ValueError("No successful trajectories found in any seed!")
                
                if self.use_mean_instead_of_median is True:
                    M = int(mean(counts))
                    print("\n--- Even sampling summary --------------------------------")
                    print("Successful counts per seed :", counts)
                    print("Mean successful count       :", M)
                    # 🔹 print expert-point counts
                    print("Expert points per seed      :", [self.seed_expert_points[s] 
                                                        for s in per_seed_success])
                else:
                    M = int(median(counts))
                    print("\n--- Even sampling summary --------------------------------")
                    print("Successful counts per seed :", counts)
                    print("Median successful count     :", M)
                    # 🔹 print expert-point counts
                    print("Expert points per seed      :", [self.seed_expert_points[s] for s in per_seed_success])

                for seed, trajs in per_seed_success.items():
                    k = min(M, len(trajs))
                    sampled = rng.sample(trajs, k) if k < len(trajs) else trajs
                    self.expert_demos.extend(sampled)
                    print(f"Seed {seed}: taking {k}/{len(trajs)} expert demos "
                        f"({self.seed_expert_points[seed]} obs)")
            else:
                for trajs in per_seed_success.values():
                    self.expert_demos.extend(trajs)

            for trajs in per_seed_fail.values():
                self.non_expert_demos.extend(trajs)

    # ----------------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.expert_demos) + len(self.non_expert_demos)

    def __repr__(self) -> str:
        return (f"<ExpertDatabase | experts={len(self.expert_demos)} | "
                f"non-experts={len(self.non_expert_demos)}>")

    def all(self) -> List:
        """Concatenate expert and non-expert trajectories."""
        return self.expert_demos + self.non_expert_demos


def preload_from_config(model: SAC, config, n_trajs_to_keep=None,n_percent_shorts=None,use_mean_instead_of_median=False,results_path=None,sample_evenly=False,plot_success_from_demos=False,db_index=None,normalize_actions=True,skip_idx=(0,-1),min_z_gripper=None,window_size=0,max_length = 50000,include_non_expert=False,success_bonus=0.0,device: str = "cpu",safety_factor = 0.0,clip_limit = (-10.0,10.0)):
    """
    Load expert & non-expert trajectories from the preference database,
    score them with the trained reward ensemble, and seed them into
    model.replay_buffer.
    """

    """
    
    if plot_success_from_demos is True:
        plot_seed_success_stats(config)

    db = ExpertDatabase(config,n_percent_shorts=n_percent_shorts,results_path=results_path,n_trajs_to_keep=n_trajs_to_keep,sample_evenly=sample_evenly,use_mean_instead_of_median=use_mean_instead_of_median)
    

    # 3) Flatten any nested lists in expert / non-expert
    def flatten(traj_list):
        return list(chain.from_iterable(
            t if isinstance(t, (list, tuple)) else [t]
            for t in traj_list
        ))

    expert_trajs    = flatten(db.expert_demos)
    expert_trajs = [traj for traj in flatten(db.expert_demos) if len(traj.observations) <= max_length]

    if include_non_expert is True:
        nonexpert_trajs = flatten(db.non_expert_demos)
    else:
        nonexpert_trajs = []
    all_trajs       = expert_trajs + nonexpert_trajs
    n_expert        = len(expert_trajs)
    n_nonexpert     = len(nonexpert_trajs)
    """

    pkl_path = os.path.join(config.iteration_working_dir,"full_ivk_trajectories.pkl")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)


    original_pos = None
    if include_non_expert == False:
        all_trajs = []
        for i in range(len(data["successful"])):
            if data["successful"][i]==True:
                all_trajs.append(data["full_trajectories"][i])
                
                print("Seed Success: " + str(i))
                #for quick in data["full_trajectories"][i].actions:
                    #print(quick[2])

    else:
        all_trajs = data["full_trajectories"]

    print("Number of Trajs: " + str(len(all_trajs)))





    # 5) Score each trajectory in batch and push transitions
    total_transitions = 0

    with torch.no_grad():
        for traj in all_trajs:
            # ---------- sanity check ----------
            if len(traj.observations) == 0 or len(traj.observations) != len(traj.actions):
                print(f"Skipping empty/malformed trajectory, "
                    f"obs={len(traj.observations)}, actions={len(traj.actions)}")
                continue

            # 4) Apply terminal success bonus if needed (successful trajectories have "dones==True for the last entry")
                #raw_rewards[-1] += success_bonus
            #Make sure the last point is always true for dones
            traj.dones[-1]=True


            # Overwrite trajectory rewards and push each transition
            #traj.rewards = step_rewards

            #skip_idx = (1,1e6)
            
            for obs, action, next_obs, reward, done, info in zip(
                traj.observations[skip_idx[0]:],        # skip first `skip_idx` steps
                traj.actions[skip_idx[0]:],
                traj.next_observations[skip_idx[0]:],
                traj.rewards[skip_idx[0]:],
                traj.dones[skip_idx[0]:],
                traj.info[skip_idx[0]:],
            ):
                if hasattr(model.replay_buffer, "is_demo"):
                    model.replay_buffer.is_demo[model.replay_buffer.pos] = True
                if normalize_actions is True:
                    action_use = model.policy.scale_action(action)
                else:
                    action_use = action
                model.replay_buffer.add(obs, next_obs, action_use, reward, done, [info])
                total_transitions += 1

    #print(f"Added {n_expert} expert trajectories, "
    #      f"{n_nonexpert} non-expert trajectories, "
    #      f"for a total of {total_transitions} transitions.")
    print("Added " + str(len(all_trajs))+" trajectories to the buffer")
    #Mark those points as demo points
    #model.replay_buffer.tag_first_as_demo(pos = model.replay_buffer.pos)
    return model




def preload_from_config_DEPRECATED(model: SAC, config, db_index=None,skip_idx=(0,-1),min_z_gripper=None,window_size=0,max_length = 500,include_non_expert=False,success_bonus=0.0,device: str = "cpu",safety_factor = 0.0,clip_limit = (-10.0,10.0)):
    """
    Load expert & non-expert trajectories from the preference database,
    score them with the trained reward ensemble, and seed them into
    model.replay_buffer.
    """
    # 1) Load the updated preference database
    db_path = os.path.join(config.iteration_working_dir, "preference_database_"+str(db_index)+".pkl")
    with open(db_path, "rb") as f:
        db = pickle.load(f)

    # 2) Load the trained reward-ensemble
    reward_path = os.path.join(config.iteration_working_dir, "reward_model_ensemble.pt")
    reward_ensemble = torch.load(reward_path, map_location=device)
    reward_ensemble = reward_ensemble.to(device)
    reward_ensemble.eval()

    # 3) Flatten any nested lists in expert / non-expert
    def flatten(traj_list):
        return list(chain.from_iterable(
            t if isinstance(t, (list, tuple)) else [t]
            for t in traj_list
        ))

    expert_trajs    = flatten(db.expert_demos)
    expert_trajs = [traj for traj in flatten(db.expert_demos) if len(traj.observations) <= max_length]

    if include_non_expert is True:
        nonexpert_trajs = flatten(db.non_expert_demos)
    else:
        nonexpert_trajs = []
    all_trajs       = expert_trajs + nonexpert_trajs
    n_expert        = len(expert_trajs)
    n_nonexpert     = len(nonexpert_trajs)

    # 4) Ensure the replay buffer exists
    #if model.replay_buffer is None:
    #    model.replay_buffer = ReplayBuffer(
    #        buffer_size=model.buffer_size,
    #        observation_space=model.observation_space,
    #        action_space=model.action_space,
    #        device=model.device,
    #        optimize_memory_usage=model.optimize_memory_usage,
    #        clip_limit = clip_limit
    #    )
        # Alternatively, you can do:
        # model.learn(total_timesteps=0, reset_num_timesteps=True)

    if model.replay_buffer.pos >0:
        original_pos = model.replay_buffer.pos
        print("original_pos")
        model.replay_buffer.pos = 0
    else:
        original_pos = None
    # 5) Score each trajectory in batch and push transitions
    total_transitions = 0

    with torch.no_grad():
        for traj in all_trajs:
            # ---------- sanity check ----------
            if len(traj.observations) == 0 or len(traj.observations) != len(traj.actions):
                print(f"Skipping empty/malformed trajectory, "
                    f"obs={len(traj.observations)}, actions={len(traj.actions)}")
                continue

            if min_z_gripper is not None:
                if traj.observations[1]["gripper_pose"][2]<min_z_gripper:
                    continue

            # ---------- build per-step feature tensors ----------
            obs_tensor = traj.generate_tensor_from_trajectory(          # (N, 16)
                                feat_stats=db.feat_stats,
                                source="observations"        # <-- add this kwarg if your helper accepts it
                        ).to(device)

            next_obs_tensor = traj.generate_tensor_from_trajectory(     # (N, 16)
                                    feat_stats=db.feat_stats,
                                    source="next_observations"  # <-- same helper, but on next-obs
                            ).to(device)

            N = obs_tensor.size(0)          # same length for both

            # ---------- per-step, two-row forward passes ----------
            outs_per_head = None            # list[M] of lists
            for t in range(N):
                # stack obs_t and next_obs_t  -> (2, 16)
                #step_input = torch.stack([obs_tensor[t], next_obs_tensor[t]], dim=0)
                step_input = torch.stack([obs_tensor[t],next_obs_tensor[t]], dim=0)

                per_head = reward_ensemble(step_input, update_stats=False)
                # list[M] of shape (L,1) where L == step_input.size(0)

                if outs_per_head is None:
                    M = len(per_head)                           # ensemble size
                    outs_per_head = [[] for _ in range(M)]

                for h in range(M):
                    # keep only the FIRST row as the reward for time-index t
                    outs_per_head[h].append(per_head[h][0:1])   # (1,1), preserves dim

            # ---------- stack to (M, N) ----------
            outs    = [torch.cat(outs_per_head[h], dim=0) for h in range(M)]  # [(N,1)]
            stacked = torch.stack(outs, dim=0).squeeze(-1)                   # (M, N)

            # ---------- per-step mean/std + safety factor ----------
            means = stacked.mean(dim=0)                       # (N,)
            stds  = stacked.std(dim=0, unbiased=False)        # (N,)


            raw_rewards = means - safety_factor * stds        # (N,)

            # 4) Apply terminal success bonus if needed (successful trajectories have "dones==True for the last entry")
            if traj.dones[-1]:
                #raw_rewards[-1] += success_bonus
                traj.info[-1]["success"]=True
            else:
                traj.dones[-1]=True
                traj.info[-1]["success"]=False
            # 5) Normalize to [0, 1]
            #min_r = -1.0
            #max_r = 1.0
            #range_r = max_r - min_r
            #normalized = (raw_rewards - min_r) / range_r
            #normalized = raw_rewards + 0.5


            #clamp to be between 0 and 1
            clamped_rewards = raw_rewards #torch.clamp(raw_rewards, min=-7.0, max=7.0)
            # 6) Move to CPU + list
            step_rewards = clamped_rewards.cpu().tolist()

            # ---- Apply terminal success_bonus ----
            # If this traj ended in done=True, add bonus to its last step reward
            #if success_bonus != 0.0 and traj.dones[-1]:
            #    step_rewards[-1] += success_bonus

            # Overwrite rewards and push into the buffer
            traj.rewards = step_rewards

            # Overwrite trajectory rewards and push each transition
            traj.rewards = step_rewards
            for obs, action, next_obs, reward, done, info in zip(
                traj.observations[skip_idx[0]:],        # skip first `skip_idx` steps
                traj.actions[skip_idx[0]:],
                traj.next_observations[skip_idx[0]:],
                traj.rewards[skip_idx[0]:],
                traj.dones[skip_idx[0]:],
                traj.info[skip_idx[0]:],
            ):
                if normalize_actions is True:
                    action_use = model.policy.scale_action(action)
                else:
                    action_use = action
                model.replay_buffer.add(obs, next_obs, action_use, reward, done, [info])
                total_transitions += 1

    print(f"Added {n_expert} expert trajectories, "
          f"{n_nonexpert} non-expert trajectories, "
          f"for a total of {total_transitions} transitions.")
    
    #Mark those points as demo points
    model.replay_buffer.tag_first_as_demo(pos = model.replay_buffer.pos)
    return model