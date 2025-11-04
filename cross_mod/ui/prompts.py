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



def query_user(config,limit_to_correction_indices=None):
    #Get the owrkign directories where our images lie we wish to draw on
    working_dirs = config.working_dirs
    draw_traj_config = config.draw_traj_config

    if limit_to_correction_indices is None:
        limit_to_correction_indices = list(range(len(working_dirs)))


    #iterate through the seeds, and have the user draw trajectories for each
    for idx in limit_to_correction_indices:
        working_folder = working_dirs[idx]
        #working_folder = "run_results/slide_block_to_target/0/"
        view_str= working_folder + "camera_angle_{}.png"

        #Query the user to drwa the images
        grand_traj_tor_r=draw_traj(view_str, **draw_traj_config)

        #Pickle the output to the working_folder directory
        pickle_path = os.path.join(working_folder, "grand_traj_tor_r.pkl")
        # Save grand_traj_tor_r to the pickle file
        with open(pickle_path, "wb") as f:
            pickle.dump(grand_traj_tor_r, f)

        print(f"Data saved to {pickle_path}")
