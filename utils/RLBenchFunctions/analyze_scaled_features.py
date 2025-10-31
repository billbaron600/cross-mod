import os
from utils.Classes.preference_database import Correction,PreferenceDatabase
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

def analyze_scaled_features(config, database_index=None, device="cuda"):
    import os, pickle, torch

    # Load in the preference database
    quickpath = f"preference_database_{database_index}.pkl"
    db_path = os.path.join(config.iteration_working_dir, quickpath)
    with open(db_path, 'rb') as f:
        pref_db = pickle.load(f)

    # Get the pairwise comparisons
    comparisons = pref_db.pairwise_comparisons
    trajectories = []
    for trajA, trajB, _ in comparisons:
        a = trajA.generate_tensor_from_trajectory(feat_stats=pref_db.feat_stats).to(device)
        b = trajB.generate_tensor_from_trajectory(feat_stats=pref_db.feat_stats).to(device)
        trajectories.extend([a, b])

    # concatenate into one big [ΣN_i, D] tensor
    stacked = torch.cat(trajectories, dim=0)

    # column-wise stats
    col_min  = stacked.min(dim=0).values
    col_max  = stacked.max(dim=0).values
    col_mean = stacked.mean(dim=0)
    col_std  = stacked.std(dim=0, unbiased=False)

    pref_db.feat_stats["gripper_open"] = {
    "mean": np.array([1.0]),
    "std":  np.array([0.5]),
    "max":  np.array([1.0]),
    "min":  np.array([0.0])}

    # Reconstruct the per-column feature names in the same order
    # as flatten_observation would have concatenated them:
    field_order = [
        "joint_velocities",
        # choose "joint_positions_sincos" if present, else raw
        #"joint_positions",
        "joint_positions_sincos",
        "joint_forces",
        "gripper_open",
        "gripper_pose",
        "gripper_joint_positions",
        "gripper_touch_forces",
        "task_low_dim_state",
        "task_low_dim_state_delta"
    ]

    column_names = []
    for field in field_order:
        stats = pref_db.feat_stats[field]["mean"]
        dim = stats.shape[0]
        column_names += [field] * dim

    # pretty‐print
    header = f"{'Idx':>3} | {'Feature':>25} | {'Min':>12} | {'Max':>12} | {'Mean':>12} | {'Std':>12}"
    print(header)
    print("-" * len(header))
    for i, (mn, mx, mu, sd) in enumerate(zip(col_min, col_max, col_mean, col_std)):
        print(f"{i:3d} | {column_names[i]:>25} | {mn.item():12.6f} | {mx.item():12.6f} | {mu.item():12.6f} | {sd.item():12.6f}")

def analyze_scaled_features_DEPRECATED(config, database_index=None, device="cuda"):
    
    # Load in the preference database
    quickpath = "preference_database_" + str(database_index) + ".pkl"
    db_path = os.path.join(config.iteration_working_dir, quickpath)

    with open(db_path, 'rb') as f:
        pref_db = pickle.load(f)

    # Get the pairwise comparisons
    pairwise_comparisons = pref_db.pairwise_comparisons
    trajectories = []
    for idx, (trajA, trajB, label) in enumerate(pairwise_comparisons):
        trajA_vec = trajA.generate_tensor_from_trajectory(feat_stats=pref_db.feat_stats).to(device)  # [60×37]
        trajB_vec = trajB.generate_tensor_from_trajectory(feat_stats=pref_db.feat_stats).to(device)  # [60×37]
        trajectories.append(trajA_vec)
        trajectories.append(trajB_vec)

    # 1) concatenate into one big [ΣN_i, 37] tensor
    stacked = torch.cat(trajectories, dim=0)

    # 2) column‐wise extrema, mean, std
    col_min = stacked.min(dim=0).values
    col_max = stacked.max(dim=0).values
    col_mean = stacked.mean(dim=0)
    col_std  = stacked.std(dim=0, unbiased=False)  # population std

    # 3) pretty print
    print(f"{'Col':>3} | {'Min':>12} | {'Max':>12} | {'Mean':>12} | {'Std':>12}")
    print("-" * 66)
    for i, (mn, mx, mu, sd) in enumerate(zip(col_min, col_max, col_mean, col_std)):
        print(f"{i:3d} | {mn.item():12.6f} | {mx.item():12.6f} | {mu.item():12.6f} | {sd.item():12.6f}")



