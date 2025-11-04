# refactored module (auto-generated)


def _load_ray_results(seed_path):
    """Load ray_tracing_results.pkl and return (mean_tor, var_tor, kept_rows)."""
    import os, pickle
    p = os.path.join(seed_path, "ray_tracing_results.pkl")
    with open(p, "rb") as f:
        data = pickle.load(f)
    return data.get("mean_tor"), data.get("var_tor"), data.get("kept_rows")

def _apply_cartesian_shift(xyz, shift_vec, indices=None):
    """Apply a fixed shift vector to selected indices (or all)."""
    import numpy as np
    if xyz is None or shift_vec is None: return xyz
    xyz = np.array(xyz, dtype=float).copy()
    shift_vec = np.array(shift_vec, dtype=float).reshape(1,3)
    if not indices:
        xyz += shift_vec
    else:
        idx = np.array(indices, dtype=int)
        xyz[idx] += shift_vec
    return xyz

def _iter_seeds(results_root, seeds):
    base = os.path.abspath(results_root)
    if seeds:
        folders = [str(i) for i in seeds]
    else:
        folders = sorted([d for d in os.listdir(base) if d.isdigit()], key=int)
    if not folders:
        raise FileNotFoundError(f"No numeric sub-folders in {base}")
    for s in folders:
        yield s, os.path.join(base, s)

# ---- explicit cross-mod imports ----
from cross_mod.sim.rollout import execute_trajectory, execute_trajectory_from_file, execute_trajectory_original
from cross_mod.planning.smoothing import interpolate_points, build_interpolated_trajectory, build_smooth_trajectory, interpolate_trajectory
from cross_mod.utils.geom import quaternion_distance, quaternion_slerp, euler_to_quaternions_use
from cross_mod.io.viz import plot_trajectory_grid, add_trajectory_line, set_cylinder_between_points

# ---- imports from original modules ----
from  utils.RLBenchFunctions.template_sensor_views import compute_camera_pose

from matplotlib.colors import ListedColormap     # ← add this

from pathlib import Path

from pyrep.objects.dummy import Dummy

from pyrep.objects.shape import Shape, PrimitiveShape

from pyrep.objects.vision_sensor import VisionSensor

from rlbench.action_modes.action_mode import MoveArmThenGripper

from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK,EndEffectorPoseViaPlanning, JointPosition

from rlbench.action_modes.gripper_action_modes import Discrete,GripperJointPosition

from rlbench.gym import RLBenchEnv

from rlbench.tasks import SlideBlockToTarget

from scipy.interpolate import interp1d

from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation as R,RotationSpline

from scipy.spatial.transform import Slerp

from transforms3d.euler import quat2euler, euler2quat

from types import MethodType

from utils.Classes.policy_trajectory_class import PolicyTrajectory

from utils.RLBenchFunctions.custom_action_modes import EndEffectorPoseViaPlanning_Record,IVKPlanningBounds,IVKPlanningBounds_NonDiscrete

from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds,MoveArmThenGripperWithBoundsDelta_IVK, EndEffectorPoseViaPlanning_Custom, MoveArmThenGripperWithBoundsDelta

from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import save_sample_trajectories

from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import visualize_trajectories

import concurrent.futures

import copy

import cv2

import gc

import matplotlib.pyplot as plt

import numpy as np

import os

import os, pickle, math

import pickle

import random

import re

import torch



def generate_interpolated_trajectories(
        config,
        init_orientation,                 # [roll,pitch,yaw]  (deg, XYZ)
        target_orientations,              # list  [[roll,pitch,yaw], …]
        target_indices,                   # list  [idx0, idx1, …]  same length
        seed_idx=0):
    """
    Linear Euler-angle interpolation (yaw-pitch-roll, ZYX).

    * Segment 0 :   init_orientation  →  target_orientations[0]
                    over indices   0 … target_indices[0]
    * Segment k :   target_orientations[k-1]  →  target_orientations[k]
                    over   target_indices[k-1]+1 … target_indices[k]

    XYZ positions are copied untouched.
    Saves all trajectories + one overlay plot.
    """

    # ── safety checks ──────────────────────────────────────────────────────
    if len(target_orientations) != len(target_indices):
        raise ValueError("target_orientations and target_indices must match")
    if sorted(target_indices) != list(target_indices):
        raise ValueError("target_indices must be monotonically increasing")

    # ── load xyz positions ────────────────────────────────────────────────
    folder = os.path.join(config.iteration_working_dir, str(seed_idx))
    with open(os.path.join(folder, "generated_trajectories.pkl"), "rb") as f:
        traj_xyz = pickle.load(f)                       # (M,N,3) torch
    traj_xyz = traj_xyz.clone()
    M, N, _ = traj_xyz.shape

    init_orientation   = np.asarray(init_orientation,  dtype=float)
    tgt_oris           = [np.asarray(o, dtype=float) for o in target_orientations]
    tgt_idxs           = target_indices

    # ── container for output ──────────────────────────────────────────────
    traj_out = torch.full(
        (M, N, 6), float("nan"),
        dtype=traj_xyz.dtype,
        device=traj_xyz.device,
    )
    traj_out[:, :, :3] = traj_xyz                       # copy xyz

    all_eulers = []                                     # for overlay plot

    # ── build interpolation segments once (index mapping) ────────────────
    segments = []
    prev_idx  = -1
    prev_ori  = init_orientation
    for idx, tgt_idx in enumerate(tgt_idxs):
        start = prev_idx + 1
        end   = tgt_idx
        if start > end:
            raise ValueError(f"segment {idx} has invalid indices")
        segments.append((start, end, prev_ori, tgt_oris[idx]))
        prev_idx, prev_ori = end, tgt_oris[idx]
    # handle any remaining tail (hold last orientation)
    if prev_idx + 1 < N:
        segments.append((prev_idx + 1, N - 1, prev_ori, prev_ori))

    # ── fill each trajectory ──────────────────────────────────────────────
    for m in range(M):
        xyz   = traj_xyz[m]
        valid = ~torch.isnan(xyz[:, 0])
        if valid.sum() < 2:
            all_eulers.append(None)
            continue

        eul_full = np.full((N, 3), np.nan, float)

        for (s, e, ori_a, ori_b) in segments:
            n  = e - s + 1
            alpha = np.linspace(0.0, 1.0, n)[:, None]
            eul_seg = ori_a + alpha * (ori_b - ori_a)   # (n,3)
            eul_full[s : e + 1] = eul_seg

        traj_out[m, :, 3:] = torch.tensor(
            eul_full, dtype=traj_out.dtype, device=traj_out.device
        )
        all_eulers.append(eul_full)

    # ── save trajectories ────────────────────────────────────────────────
    save_sample_trajectories(
        os.path.join(folder, "trajectories_with_orientations"), traj_out
    )

    # ── overlay plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    labels = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    x = np.arange(N)

    for eul in all_eulers:
        if eul is None:
            continue
        for i, ax in enumerate(axes):
            ax.plot(x, eul[:, i], linewidth=1, alpha=0.8)

    for ax, lab in zip(axes, labels):
        ax.set_ylabel(lab)
        ax.grid(True)
    axes[-1].set_xlabel('Index')
    fig.suptitle('All Trajectories: Euler Angles vs Index', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = os.path.join(folder, "trajectories_with_orientations",
                             "euler_angles_overlay.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"saved  {plot_path}")

    return traj_out

def apply_shifts_to_trajs(
    config,
    seed_idx_demo = None,
    cartesian_shift_amount = (0.0, 0.0, 0.0),
    indices_shift= (0,),
):
    

    path = os.path.join(
        config.iteration_working_dir, str(seed_idx_demo), "trajectories_with_orientations","sample_trajectories"
    )
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    shift_vec = np.asarray(cartesian_shift_amount, dtype=np.float32).ravel()
    if shift_vec.size != 3:
        print("⚠️  cartesian_shift_amount must have exactly 3 numbers; nothing done.")
        return

    base_re = re.compile(r"^(?P<name>.+?)(?:_shifted_(?P<idx>\d+))?\.npy$")

    all_files  = [f for f in os.listdir(path) if f.endswith(".npy")]
    base_files = [f for f in all_files if "_shifted_" not in f]

    for fname in base_files:
        base, _ = os.path.splitext(fname)          # 'foo' from 'foo.npy'

        # -------- find the next backup index ---------------------------
        existing = [
            int(m.group("idx"))
            for f in all_files
            if (m := base_re.match(f))
            and m.group("name") == base
            and m.group("idx") is not None
        ]
        next_idx = max(existing, default=-1) + 1

        full_path = os.path.join(path, fname)
        traj = np.load(full_path)                  # load BEFORE renaming

        # -------- rename original to backup -----------------------------
        backup_name = f"{base}_shifted_{next_idx}.npy"
        os.rename(full_path, os.path.join(path, backup_name))
        print(f"original renamed → {backup_name}")

        # -------- apply the shift --------------------------------------
        for row in indices_shift:
            if row >= traj.shape[0]:
                print(f"⚠️  index {row} out of bounds for {fname}; skipping.")
                continue
            traj[row, :3] += shift_vec            # shift cols 0-2

        # -------- save shifted array back to original filename ----------
        np.save(full_path, traj)
        print(f"saved shifted array → {fname}")

def generate_IVK_trajectories(config,discrete_gripper=False,gripper_pause=False,use_gripper_action_file=False,min_vals=None,max_vals=None,shift_mean=None,use_mean=False,use_gripper_orientation_file=False,gripper_provided=False,task=SlideBlockToTarget,limit_to_correction_indices=None,trajectory_indices = None,min_z=None):
    """Load 3D mean path per seed, apply optional shifts, and run execute_trajectory."""
    import os, numpy as np
    results_root = config.iteration_working_dir
    seeds = kwargs.get("limit_to_correction_indices", None)
    shift = kwargs.get("cartesian_shift_amount", None)
    indices_shift = kwargs.get("indices_shift", None)
    record_video = kwargs.get("record_video", False)
    out = []

    for seed_str, seed_path in _iter_seeds(results_root, seeds):
        mean_tor, var_tor, kept_rows = _load_ray_results(seed_path)
        if mean_tor is None:
            continue
        traj_xyz = np.asarray(mean_tor, dtype=float)
        if shift is not None:
            traj_xyz = _apply_cartesian_shift(traj_xyz, shift, indices_shift)

        res = execute_trajectory(
            task=None,
            trajectory=traj_xyz,
            seed_dir=seed_path,
            kept_rows=kept_rows,
            record_video=record_video,
            use_gripper_orientation_file=kwargs.get("use_gripper_orientation_file", True),
            use_gripper_action_file=kwargs.get("use_gripper_action_file", True),
        )
        out.append((seed_str, res))
    return out


def generate_IVK_trajectories_from_file(config,gripper_pause=False,task=SlideBlockToTarget,cam_pose=None,limit_to_correction_indices=None,record_video=True,discrete_gripper=True,working_dir=None,video_filename="trajectory_execution.mp4"):
    
    # Get the working directories we will be loading data in from
    working_dirs = config.working_dirs

    if limit_to_correction_indices is None:
        limit_to_correction_indices = config.seeds

    #for seed_idx,working_dir in enumerate(working_dirs):
    for idx in limit_to_correction_indices:
        seed_idx = config.seeds[idx]
        working_dir = config.working_dirs[seed_idx]
        seed_of_working_dir = config.seeds[seed_idx]    


        full_trajectories_seed = {"full_trajectories": [],
                            "successful":[]}

        with open(working_dir+'full_ivk_trajectories.pkl', "rb") as f:
            full_trajs = pickle.load(f)
        
        trajectory = full_trajs["full_trajectories"][0].actions

        full_trajectory, done,success = execute_trajectory_from_file(trajectory,gripper_pause=gripper_pause,cam_pose=cam_pose,record_video=record_video,seed=seed_of_working_dir,task=task,discrete_gripper=discrete_gripper,working_dir=working_dir,video_filename=video_filename)
        print(f"Trajectory {str(idx)} executed successfully: {success}")

        #append the key information to the full_trajectories dictionary
        full_trajectories_seed['full_trajectories'].append(full_trajs["full_trajectories"][0])
        full_trajectories_seed['successful'].append(success)
        
        with open(working_dir+'full_ivk_trajectories.pkl', 'wb') as f:
            pickle.dump(full_trajectories_seed, f)
            print("Saved to: " + working_dir+'full_ivk_trajectories.pkl')

def run_ivk_for_corrections(config,task=SlideBlockToTarget,limit_to_correction_indices=None):
    #This function generates the full IVK trajectoris for all the corrections made in this iteration
    from utils.Classes.preference_database import Correction
    working_dirs = config.working_dirs

    
    with open(config.path_to_current_trajectories,'rb') as file:
        current_policy_trajectories = pickle.load(file)

    if limit_to_correction_indices is None:
        #current_policy_trajectories = [current_policy_trajectories[i] for i in limit_to_correction_indices]
        #working_dirs = 
        limit_to_correction_indices = list(range(len(working_dirs)))

    
    execute_trajectory_kwargs = config.execute_trajectory_kwargs
    all_corrections = [] #list to hold all corrections

    #quaternion offsets
    offset_by = execute_trajectory_kwargs.pop('offset_by')

    for traj_index in limit_to_correction_indices:
        working_dir = working_dirs[traj_index]
        with open(working_dir+"generated_trajectories.pkl", 'rb') as file:
            trajectories = pickle.load(file)
        
        starting_env_state = current_policy_trajectories[traj_index].environment_states[config.frame_correction_indices[traj_index]]
        execute_trajectory_kwargs["starting_env_state"] = starting_env_state
        execute_trajectory_kwargs["seed"] = current_policy_trajectories[traj_index].env_seed
        count = 0
        
        #Create the creaction object for this correction
        correction_object = Correction(current_policy_trajectory=current_policy_trajectories[traj_index])

        #Set the offset_by parameter in the execture trajectory kwargs


        for idx, trajectory in enumerate(trajectories):
            #print("x")
            #execute_trajectory_kwargs["trajectory"]=trajectory.cpu().numpy()
            execute_trajectory_kwargs["video_filename"] = "correction"+str(traj_index)+"_trajectory"+str(count)+".mp4"
            execute_trajectory_kwargs["trajectory"] = trajectory.detach().cpu().numpy()
            correction_working_dir = os.path.join(working_dir, f"corrected_trajectory_{idx}")
            os.makedirs(correction_working_dir, exist_ok=True)
            execute_trajectory_kwargs["working_dir"]=correction_working_dir
            trajectory_data,done,success = execute_trajectory(task=task,**execute_trajectory_kwargs,offset_by=offset_by[traj_index])
            
            #Apppend to our correction object
            correction_object.append_corrections_list(trajectory_data,success_status=success)
            count+=1
            # Full file path to save the trajectory to (with the correct name)
            file_path = os.path.join(correction_working_dir, 'trajectory_data.pkl')
            #print(trajectory_data.video_path)

            # Pickle the variable
            with open(file_path, 'wb') as f:
                pickle.dump(trajectory_data, f)
        #Append the correcion object to the list
        all_corrections.append(correction_object)
        #Save the corrections object to the working_dir path
        file_path = os.path.join(working_dir,"original_trajectory_and_corrections.pkl")
        with open(file_path,'wb') as f:
            pickle.dump(correction_object,f)
    
    #Save all corrections
    save_dir = config.iteration_working_dir
    file_path = os.path.join(save_dir,"all_corrections.pkl")
    with open(file_path,'wb') as f:
        pickle.dump(all_corrections,f)

def merge_full_trajectories(
        config,
        results_root: str,                       # e.g. "run_results/reach_target/demos"
        fname: str = "full_ivk_trajectories.pkl",
        save: bool = True,
        limit_to_correction_indices = None,
        limit_to_first=True
    ):
    
    # ---------------------------------------------------------------------
    # 1) determine which sub-folders to load and where they live
    # ---------------------------------------------------------------------
    if limit_to_correction_indices is not None:
        # Use exactly the indices the caller specified, in order
        base_dir = getattr(config, "iteration_working_dir")
        folders = [str(i) for i in limit_to_correction_indices]
    else:
        # Fall back to scanning results_root for numeric sub-folders
        base_dir = os.path.abspath(results_root)
        folders = sorted(
            [d for d in os.listdir(base_dir) if d.isdigit()],
            key=int
        )
        if not folders:
            raise FileNotFoundError(f"No numeric sub-folders in {base_dir}")

    # ---------------------------------------------------------------------
    # 2) load, concatenate
    # ---------------------------------------------------------------------
    merged = {"full_trajectories": [], "successful": []}

    for d in folders:
        pkl_path = os.path.join(base_dir, d, fname)
        if not os.path.isfile(pkl_path):
            continue
            #raise FileNotFoundError(f"Expected file not found: {pkl_path}")

        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        if len(data["successful"])>0 and limit_to_first is True:
            data["full_trajectories"] = [data["full_trajectories"][0]]
            data["successful"] = [data["successful"][0]]

        merged["full_trajectories"].extend(data["full_trajectories"])
        merged["successful"].extend(data["successful"])

    # ---------------------------------------------------------------------
    # 3) optionally save the merged dict in iteration_working_dir
    # ---------------------------------------------------------------------
    if save:
        out_dir = getattr(config, "iteration_working_dir")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "wb") as fh:
            pickle.dump(merged, fh)
        print(f"merged dict saved → {out_path}")

    total = len(merged["successful"])
    successes = sum(merged["successful"])
    pct_success = 100.0 * successes / total if total > 0 else 0.0

    print(f"merged {len(merged['full_trajectories'])} trajectories "
          f"from {len(folders)} folders")

    print(f"{successes}/{total} successful "
          f"({pct_success:.2f}%)")

    return merged

def merge_full_trajectories_DEPRECATED(config,
                            results_root,                 # e.g. "run_results/reach_target/demos"
                            fname="full_ivk_trajectories.pkl",
                            save=True):
    
    results_root = os.path.abspath(results_root)

    # 1) collect numeric sub-folders in ascending order
    folders = sorted([d for d in os.listdir(results_root) if d.isdigit()],
                     key=int)
    if not folders:
        raise FileNotFoundError(f"No numeric sub-folders in {results_root}")

    merged = {"full_trajectories": [], "successful": []}

    # 2) load and extend
    for d in folders:
        pkl_path = os.path.join(results_root, d, fname)
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        merged["full_trajectories"].extend(data["full_trajectories"])
        merged["successful"].extend(data["successful"])

    # 3) optionally save into the config’s working dir
    if save:
        out_dir = getattr(config, "iteration_working_dir")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "wb") as fh:
            pickle.dump(merged, fh)
        print(f"merged dict saved → {out_path}")

    print(f"merged {len(merged['full_trajectories'])} trajectories "
          f"from {len(folders)} folders")
    return merged