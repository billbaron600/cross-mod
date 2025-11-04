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



def run_generate_densities_for_all_images(config,training_iterations=5000,segment_idx=0,limit_to_correction_indices=None):
    working_dirs = config.working_dirs
    density_generation_params = config.density_generation_params

    if limit_to_correction_indices is None:
        limit_to_correction_indices = list(range(len(working_dirs)))

    for idx in limit_to_correction_indices:
        working_folder = working_dirs[idx]

        # Define the path to the pickle file
        pickle_path = os.path.join(working_folder, "grand_traj_tor_r.pkl")
        with open(pickle_path, "rb") as f:
            grand_traj_tor_r = pickle.load(f)
        

        # walk [cam_idx][traj_idx] and convert any NumPy arrays to tensors
        for cam_i in range(len(grand_traj_tor_r)):
            for traj_i in range(len(grand_traj_tor_r[cam_i])):
                arr = grand_traj_tor_r[cam_i][traj_i]
                if not isinstance(arr, torch.Tensor):
                    grand_traj_tor_r[cam_i][traj_i] = torch.as_tensor(arr, dtype=torch.float32)
        
        
        config.density_generation_params["time_length"] = int(grand_traj_tor_r[0][0].shape[1])
        
        #add time points to the coordinates drawn on images
        all_view_list = add_time_stamps_to_drawings(grand_traj_tor_r)

        #add the workign dir paramter here
        density_generation_params["working_dir"] = working_folder

        
        n_views = config.ray_tracing_params["n_views"]

        

        #generate densitties using the drawn images and parameters defiend above
        generate_densities(all_view_list,segment_idx=segment_idx,n_views=n_views,**density_generation_params,training_iterations=training_iterations)
