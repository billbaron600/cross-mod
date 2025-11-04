# refactored module (auto-generated)


# ---- explicit cross-mod imports ----
from cross_mod.perception.density.stack import combine_density_views
from cross_mod.perception.multiview import trajectory_model
from cross_mod.planning.smoothing import interpolate_segment, rbf_kernel

# ---- imports from original modules ----
from rendering_helpers import *

from torch import nn

from utils.RLBenchFunctions.combine_density_views import combine_density_views

from utils.RLBenchFunctions.plottingFunctions.plot_traced_points import plot_and_save

import numpy as np

import os, shutil

import pickle

import torch



def get_true_first_position(config):
    #Get the actual first positoinn of the gripper from teh scene, which will be added too the mean at the first idne
    with open(config.path_to_current_trajectories,'rb') as file:
        current_policy_trajectories = pickle.load(file)
    
    first_positions = []
    for traj_index in range(config.number_corrections):
        starting_env_state = current_policy_trajectories[traj_index].environment_states[config.frame_correction_indices[traj_index]]
        gripper_pose = starting_env_state["tip_pose"]
        first_positions.append(gripper_pose[:3])
    return first_positions

def fit_NeRF_model(config,use_mean=False,segment_idx=0,iterations = 40000,lr_mean=5e-3,lr_std=1e-3,mean_shift_indices=[],use_intrinsics=False,policy_correction=False,limit_to_correction_indices=None,shift_mean=None,negative_z=True):
    #Get the key variables form teh config
    ray_tracing_params = config.ray_tracing_params
    working_dirs = config.working_dirs

    if limit_to_correction_indices is None:
        limit_to_correction_indices = list(range(len(working_dirs)))

    #Get the actual first positions
    if policy_correction is True:
        gripper_ground_truth_positions = get_true_first_position(config)
    else:
        gripper_ground_truth_positions = None

    for idx in limit_to_correction_indices:
        working_folder = working_dirs[idx]

        # Define the path to the pickle file
        pickle_path = os.path.join(working_folder, "grand_traj_tor_r.pkl")
        with open(pickle_path, "rb") as f:
            grand_traj_tor_r = pickle.load(f)

        config.ray_tracing_params["n_times"] = grand_traj_tor_r[0][0].shape[1]

        #Instatiate NeRF model
        trajectory_dist=trajectory_model(**ray_tracing_params)

    
        #Organize the views
        im_list_tor_all = combine_density_views(working_dir=working_folder,n_times=ray_tracing_params['n_times'],n_views=ray_tracing_params['n_views'])

        #Load in the sensor extrinsics
        poses_all=torch.load(working_folder+"poses_mobile.tar")
        poses = poses_all['extrinsics']
        poses = poses[ray_tracing_params['n_views'],:,:]
        if use_intrinsics is True:
            intrinsics = poses_all["intrinsics"]
            intrinsics_mats = intrinsics[ray_tracing_params['n_views'],:,:]
        else:
            intrinsics_mats = None

        #Fit the NeRF model
        mean_tor, var_tor, kept_rows=trajectory_dist.extract_mean_std_from_images(im_list_tor_all, poses,print_outputs=False,intrinsic_mats=intrinsics_mats,negative_z=negative_z)

        if shift_mean is not None:
            shift_mean = torch.from_numpy(shift_mean)
            shift_mean = shift_mean.to(mean_tor.device)
            mean_tor = mean_tor + shift_mean
        

        save_results_path = working_folder + 'ray_tracing_results.pkl'
        
        results = {
            'mean_tor': mean_tor.detach().cpu().numpy(),
            'var_tor':  var_tor.detach().cpu().numpy(),
            'kept_rows':np.array(kept_rows)
        }
        with open(save_results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved ray tracing results to: {save_results_path}")

        
        #var_tor = torch.cat((zero_row, var_tor), dim=0)

        #plot the interpolated mean for each coordinate
        plot_and_save(mean_tor,var_tor,working_folder,show_plots=False)


        #fitthe continuouse distribution to the mdoel
        #iterations = 40000 #was iniitally 40000 and 20000
        
        if use_mean==False:
            trajectory_dist.fit_continuous_function(mean_tor, var_tor,kept_rows=kept_rows,n_display=10000,n_iter=iterations,lr_mean=lr_mean,lr_std=lr_std)
            trajectory_dist.fit_continuous_function(mean_tor, var_tor,kept_rows=kept_rows,n_display=10000,n_iter=iterations,lr_mean=lr_mean/10,lr_std=lr_std/10)

        #save the model
        # Save
        save_to_path = working_folder+'trajectory_dist.pkl'
        with open(save_to_path, 'wb') as f:
            pickle.dump(trajectory_dist, f)

        print("Saved to: " + save_to_path)

        # ─────────────────────────── NEW BLOCK ────────────────────────────
        # Duplicate both pickle files into segment_<segment_idx>/ sub‑folder
        # Duplicate both pickle files into segment_<segment_idx>/ sub‑folder
        segment_dir = os.path.join(working_folder, f"segment_{segment_idx}")
        os.makedirs(segment_dir, exist_ok=True)

        plot_and_save(mean_tor, var_tor, segment_dir,show_plots=False)
        # use the correct variable names here
        for src in [save_results_path, save_to_path]:
            if os.path.exists(src):
                shutil.copy(src, segment_dir)

        print(f"Duplicated pickles to: {segment_dir}")