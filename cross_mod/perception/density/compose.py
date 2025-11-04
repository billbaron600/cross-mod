# refactored module (auto-generated)


# ---- imports from original modules ----
from PIL import Image

from matplotlib.image import imread

from torchvision.utils import save_image

from typing import List

from typing import List, Tuple, Dict

from utils.RLBenchFunctions.add_time_stamps_to_drawings import add_time_stamps_to_drawings

from utils.RLBenchFunctions.trajectory_generator import rbf_kernel,trajectory_model

import FrEIA.framework as Ff

import FrEIA.modules as Fm

import copy

import cv2

import cv2                                   # ← switched to OpenCV

import glob

import json

import matplotlib.pyplot as plt

import numpy as np

import os

import os, re, ast

import os, re, ast, copy

import os, re, ast, cv2, numpy as np

import os, re, glob

import os, shutil   # ←‑‑ NEW

import pickle

import re

import torch

import torch.nn as nn



def generate_densities(
    grand_traj_tor_r, segment_idx=0,noise_added=0.001,n_views=None, img_len=100, n_images=2,time_length=50, hdim=512,N_DIM=3,working_dir=None,training_iterations=5000):
    #N_DIM = 3

    if isinstance(img_len, (tuple, list)):
        #img_w, img_h = img_len       # e.g. (640, 480)
        #if isinstance(img_len, list):
        img_w = img_len[0]
        img_h = img_len[1]
        print("Breakpoint")
    else:                            # backwards-compatible
        img_w = img_h = img_len      # e.g. 250 × 250

    # create grid to query
    if n_views==None:
        pass
        #make list from 2 to n_images

    """
    xy_g = torch.linspace(0, 1, img_len)
    grid_xy_vals = torch.meshgrid(xy_g, xy_g)
    grid_xy = torch.cat(
        [grid_xy_vals[0][:, :, None], grid_xy_vals[1][:, :, None]], dim=-1
    ).reshape((-1, 2))

    # number of times steps of images produced
    time_range = torch.linspace(0, 1, time_length)
    """
    # create grid to query  ---------------------------------------------
    x_g = torch.linspace(0, 1, img_w)            # ➋ CHANGED (was xy_g)
    y_g = torch.linspace(0, 1, img_h)            # ➊ NEW
    grid_xy_vals = torch.meshgrid(x_g, y_g, indexing='ij')  # ➋ CHANGED
    grid_xy = torch.cat(
        [grid_xy_vals[0][..., None], grid_xy_vals[1][..., None]], dim=-1
    ).reshape(-1, 2)
    # -------------------------------------------------------------------

    time_range = torch.linspace(0, 1, time_length)

    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, hdim), nn.ReLU(), nn.Linear(hdim, dims_out)
        )

    for view in range(n_images):
        t_view_wt = grand_traj_tor_r[view]
        # time_len = t_view.shape[1]
        # times_tor = (torch.arange(0, time_len) / time_len)[None, :, None]
        # times_tor_tile = times_tor.repeat((t_view.shape[0], 1, 1))
        # t_view_wt = torch.cat([times_tor_tile, t_view], dim=-1)
        data = t_view_wt.reshape((-1, 3)).to(device)
        inn = Ff.SequenceINN(N_DIM)
        for k in range(8):
            inn.append(
                Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True
            )
        inn = inn.to(device)
        optimizer = torch.optim.Adam(inn.parameters(), lr=0.0003, weight_decay=0.00005)

        for i in range(training_iterations):
            optimizer.zero_grad()
            noise_in_space = (torch.randn_like(data) * noise_added).to(device)
            x = (data + noise_in_space).to(device)
            # pass to INN and get transformed variable z and log Jacobian determinant
            z, log_jac_det = inn(x)
            # calculate the negative log-likelihood of the model with a standard normal prior
            loss = 0.5 * torch.sum(z**2, 1) - log_jac_det
            loss = loss.mean() / N_DIM
            # backpropagate and update the weights
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print("{}: {}".format(i, loss))

        # create densities along time and save to a folder called "n_imgs"
        if working_dir==None:
            img_dir = None
        else:
            img_dir = os.path.join(working_dir, "traj_imgs")
            img_dir = img_dir + "/"
            os.makedirs(img_dir, exist_ok=True)

        """
        print("creating densities for view {}".format(view))
        for times_ind in range(len(time_range)):
            query_txy = torch.cat(
                [time_range[times_ind] * torch.ones(len(grid_xy), 1), grid_xy], dim=-1
            ).to(device)
            z, log_jac_det = inn(query_txy)
            loss = torch.exp(-0.5 * torch.sum(z**2, 1) - log_jac_det)
            loss_r = loss / loss.max()
            
            grid_xy_im = loss_r.reshape((img_len, img_len)).swapaxes(0, 1).cpu()
            # convert to img
            img = torch.zeros((img_len, img_len))
            for i in range(grid_xy_im.shape[0]):
                for j in range(grid_xy_im.shape[1]):
                    img[img_len - 1 - i, j] = grid_xy_im[i, j]
            #save_image(img.detach(), working_dir+"traj_imgs/img_{}_{}.png".format(view, times_ind))
            save_image(img.detach(), img_dir+"img_{}_{}.png".format(view, times_ind))
        inn = inn.cpu()
        """
        print("creating densities for view {}".format(view))
        for times_ind in range(len(time_range)):
            query_txy = torch.cat(
                [time_range[times_ind] * torch.ones(len(grid_xy), 1), grid_xy], dim=-1
            ).to(device)
            z, log_jac_det = inn(query_txy)
            loss   = torch.exp(-0.5 * torch.sum(z**2, 1) - log_jac_det)
            loss_r = loss / loss.max()

            # ----------- ONLY THREE LINES CHANGE ------------------------------
            # 1. reshape with (img_w, img_h) instead of (img_len, img_len)
            grid_xy_im = loss_r.reshape((img_w, img_h)).swapaxes(0, 1).cpu()

            # 2. allocate the empty image as (img_h, img_w)
            img = torch.zeros((img_h, img_w))

            # 3. use img_h when flipping rows
            for i in range(grid_xy_im.shape[0]):
                for j in range(grid_xy_im.shape[1]):
                    img[img_h - 1 - i, j] = grid_xy_im[i, j]
            # ------------------------------------------------------------------

            save_image(img.detach(), img_dir + f"img_{view}_{times_ind}.png")
        inn = inn.cpu()

    if working_dir is not None:
        src_dir = os.path.join(working_dir, "traj_imgs")
        if os.path.isdir(src_dir):
            segment_dir = os.path.join(working_dir, f"segment_{segment_idx}")
            dst_dir     = os.path.join(segment_dir, "traj_imgs")

            os.makedirs(segment_dir, exist_ok=True)
            # Remove any existing dst_dir to avoid copytree errors
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            shutil.copytree(src_dir, dst_dir)
            print(f"Duplicated traj_imgs to: {dst_dir}")

def combine_segments(config,segment_opens=None, limit_to_correction_indices=None, n_segments=1):
    """
    For each seed, concatenate all segment_* ray‑tracing results and
    **create a brand‑new `trajectory_model`** whose n_times matches the
    stacked mean/var length.

        <seed>/ray_tracing_results.pkl   (concatenated mean/var)
        <seed>/trajectory_dist.pkl       (fresh model, then fitted)
    """
    if limit_to_correction_indices is None:
        limit_to_correction_indices = config.seeds

    if segment_opens is not None:
        gripper_commands = []

    for seed in limit_to_correction_indices:
        seed_dir = os.path.join(config.iteration_working_dir, str(seed))

        mean_list, var_list = [], []
        base_td = None          # we’ll grab hyper‑params from segment_0

        # ── 1.  gather mean/var from each segment ──────────────────────
        for seg_idx in range(n_segments):
            seg_dir = os.path.join(seed_dir, f"segment_{seg_idx}")

            # ---- ray_tracing_results.pkl --------------------------------
            rt_pkl = os.path.join(seg_dir, "ray_tracing_results.pkl")
            if os.path.isfile(rt_pkl):
                with open(rt_pkl, "rb") as f:
                    rt = pickle.load(f)
                mean_list.append(rt["mean_tor"])
                var_list.append(rt["var_tor"])
            else:
                print(f"[Seed {seed}]  seg{seg_idx}: missing ray_tracing_results")

            # ---- trajectory_dist.pkl (only need 1st for params) ---------
            if base_td is None:
                td_pkl = os.path.join(seg_dir, "trajectory_dist.pkl")
                if os.path.isfile(td_pkl):
                    with open(td_pkl, "rb") as f:
                        base_td = pickle.load(f)

            if segment_opens is not None:
                isOpen = segment_opens[seg_idx]
                
                if isOpen == True:
                    grip_action = 0.04
                else:
                    grip_action = 0.0
                
                num_points = len(rt["mean_tor"])
                grip_actions = np.full(num_points, grip_action)
                gripper_commands.append(grip_actions)

        if segment_opens is not None:
            # paths
            original_path = os.path.join(seed_dir, "gripper_actions.npy")
            backup_path   = os.path.join(seed_dir, "gripper_actions_original.npy")
            # rename existing file if it’s there
            if os.path.exists(original_path):
                os.rename(original_path, backup_path)

            # now concatenate and save the new one
            gripper_commands_numpy = np.concatenate(gripper_commands)
            np.save(original_path, gripper_commands_numpy)
            print("Used Actions For Segments")

        if not mean_list:
            print(f"[Seed {seed}]  nothing found – skipped.")
            continue
        if base_td is None:
            print(f"[Seed {seed}]  no trajectory_dist.pkl in any segment – skipped.")
            continue

        # ── 2.  stack mean / var  ───────────────────────────────────────
        mean_all = np.concatenate(mean_list, axis=0)      # (T_all,3)
        var_all  = np.concatenate(var_list,  axis=0)
        with open(os.path.join(seed_dir, "ray_tracing_results.pkl"), "wb") as f:
            pickle.dump({"mean_tor": mean_all, "var_tor": var_all}, f)

        # ── 3.  build a *new* trajectory_model matching T_all ───────────
        T_all = mean_all.shape[0]                         # 50 × segs
        ray_params = dict(
            focal             = base_td.focal,
            height            = base_td.height,
            width             = base_td.width,
            near              = base_td.near,
            far               = base_td.far,
            n_weights         = base_td.n_weights,        # 30
            n_views           = base_td.n_views,
            n_times           = T_all,                    # e.g. 100
            n_samples         = base_td.n_samples,
            perturb           = base_td.perturb,
            gamma             = base_td.gamma,
            ray_dist_threshold= base_td.ray_dist_threshold,
            density_threshold = base_td.density_threshold,
            inducing_points   = [0, 1],                   # same default
            device            = base_td.device,
        )
        new_td = trajectory_model(**ray_params)

        # ── 4.  fit it to the concatenated mean / var ───────────────────
        new_td.fit_continuous_function(
            torch.from_numpy(mean_all).to("cuda"),
            torch.from_numpy(var_all).to("cuda"),
            n_iter   = 40_000,
            n_display= 5_000,
            lr_mean  = 5e-4,      #  (original 5e‑3 ÷ 10)
            lr_std   = 1e-4,      #  (original 1e‑3 ÷ 10)
        )

        # ── 5.  save the fresh, fitted model ────────────────────────────
        with open(os.path.join(seed_dir, "trajectory_dist.pkl"), "wb") as f:
            pickle.dump(new_td, f)

        print(f"[Seed {seed}]  merged T={T_all}, new model fitted and saved.")

def combine_segments_DEPRECATED(config, limit_to_correction_indices=None, n_segments: int = 1):
    """
    Concatenate the per‑segment ray‑tracing results into a single dictionary.

    Output path:
        <config.iteration_working_dir>/<seed>/ray_tracing_results.pkl
    """
    if limit_to_correction_indices is None:
        limit_to_correction_indices = config.seeds

    for corr_idx in limit_to_correction_indices:
        seed_dir = os.path.join(config.iteration_working_dir, str(corr_idx))

        mean_list, var_list = [], []

        for segment_idx in range(n_segments):
            seg_dir   = os.path.join(seed_dir, f"segment_{segment_idx}")
            seg_pkl   = os.path.join(seg_dir, "ray_tracing_results.pkl")

            if not os.path.isfile(seg_pkl):
                print(f"[Seed {corr_idx}] segment_{segment_idx} missing ‑ skipped")
                continue

            with open(seg_pkl, "rb") as f:
                seg_data = pickle.load(f)

            mean_list.append(seg_data["mean_tor"])
            var_list.append(seg_data["var_tor"])

        if not mean_list:
            print(f"[Seed {corr_idx}] No segment files found ‑ nothing combined.")
            continue

        # concatenate along the first (trajectory‑index) axis
        combined_mean = np.concatenate(mean_list, axis=0)
        combined_var  = np.concatenate(var_list,  axis=0)

        combined_dict = {
            "mean_tor": combined_mean,
            "var_tor":  combined_var,
        }

        #load in the trajectoory dist


        out_path = os.path.join(seed_dir, "ray_tracing_results.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(combined_dict, f)

        
        #Load in teh traj object and refit the density
        load_path = seed_dir + "/" + 'trajectory_dist.pkl'

        with open(load_path, 'rb') as f:
            trajectory_dist = pickle.load(f)

        
        #trajectory_dist.fit_continuous_function(mean_tor, var_tor,kept_rows=kept_rows,n_display=10000,n_iter=iterations,lr_mean=lr_mean,lr_std=lr_std)
        iterations = 40000
        lr_mean=5e-3
        lr_std=1e-3
        combined_mean = torch.from_numpy(combined_mean).to("cuda")
        combined_var  = torch.from_numpy(combined_var).to("cuda")
        trajectory_dist.fit_continuous_function(combined_mean, combined_var,n_display=5000,n_iter=iterations,lr_mean=lr_mean/10,lr_std=lr_std/10)

        print(f"[Seed {corr_idx}] combined ray_tracing_results.pkl written to {out_path}")
