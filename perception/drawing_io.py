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



def load_grand_traj_from_file(
    config,
    working_dir: str,
    *,
    segment_idx: int,                 # ←–– NEW: keep only “…_segment_<segment_idx>”
    step_px: float = 10.0,
    render_trajectories: bool = False,
    dot_radius: int = 2,
    dot_color: Tuple[int, int, int] = (0, 0, 255),
    create_continuous_path: bool = True,
) -> List[List[torch.Tensor]]:
    """
    grand_traj_l[view_idx][traj_idx] → tensor (1, N, 2)

    • Loads trajectory lists whose *name* in trajectory_points.txt matches
      'camera_angle_{view}_traj_{traj}_segment_{segment_idx}' exactly.
    • Uses config["ray_tracing_params"]["n_views"] for the view indices.
    • Densifies, flips‑Y, and normalises exactly like draw_traj.
    • If render_trajectories=True, writes camera_angle_<v>_rendered.png.
    """
    # ----------------------------------------------------------------
    # (1) Load base resolution from view 0 (any view works).
    # ----------------------------------------------------------------
    sample_im = cv2.imread(os.path.join(working_dir, "camera_angle_0.png"))
    if sample_im is None:
        raise FileNotFoundError("camera_angle_0.png missing in working_dir")
    h_px, w_px = sample_im.shape[:2]

    # ----------------------------------------------------------------
    # (2) Parse trajectory_points.txt  →  dict[view][traj] = [(x,y),…]
    # ----------------------------------------------------------------
    txt_path = os.path.join(working_dir, "trajectory_points.txt")
    with open(txt_path, "r") as f:
        txt = f.read()

    # match:  camera_angle_<view>_traj_<traj>_segment_<seg> = [...]
    patt = re.compile(
        r"camera_angle_(\d+)_traj_(\d+)_segment_(\d+)\s*=\s*\[(.*?)\]", re.S
    )
    raw_views: Dict[int, Dict[int, List[Tuple[float, float]]]] = {}
    for vid, tid, sid, rhs in patt.findall(txt):
        if int(sid) != segment_idx:
            continue                                  # skip other segments
        raw_views.setdefault(int(vid), {})[int(tid)] = ast.literal_eval(
            "[" + rhs + "]"
        )

    # ----------------------------------------------------------------
    # (3) Iterate over *all* requested views
    # ----------------------------------------------------------------
    view_ids = config.ray_tracing_params["n_views"]
    grand_traj_l: List[List[torch.Tensor]] = []

    for v in view_ids:
        view_img_path = os.path.join(working_dir, f"camera_angle_{v}.png")
        view_img = cv2.imread(view_img_path)
        if view_img is None:
            raise FileNotFoundError(f"{view_img_path} not found")

        traj_list: List[torch.Tensor] = []
        trajs_for_view = raw_views.get(v, {})          # may be empty

        for t in sorted(trajs_for_view.keys()):
            if create_continuous_path:
                dense_px = densify_waypoints(trajs_for_view[t], step_px=step_px)
            else:
                dense_px = trajs_for_view[t]

            # ––––– render (unnormalised, original pixel coords)
            if render_trajectories:
                for x, y in dense_px:
                    cv2.circle(
                        view_img,
                        (int(round(x)), int(round(y))),
                        dot_radius,
                        dot_color,
                        -1,
                    )

            # ––––– flip‑Y + normalise for return tensor
            xy = torch.tensor(dense_px, dtype=torch.float32)  # (N, 2)
            xy[:, 1] = h_px - xy[:, 1]   # flip Y
            xy[:, 0] /= w_px             # normalise X
            xy[:, 1] /= h_px             # normalise Y
            traj_list.append(xy[None])   # (1, N, 2)

        grand_traj_l.append(copy.deepcopy(traj_list))

        # ––––– save rendered image (drawn or original)
        if render_trajectories:
            out_path = os.path.join(working_dir, f"camera_angle_{v}_rendered.png")
            cv2.imwrite(out_path, view_img)

    
    # ======================  NEW BLOCK  ======================
    # After all views processed, copy outputs to segment_<idx>/
    segment_dir = os.path.join(working_dir, f"segment_{segment_idx}")
    os.makedirs(segment_dir, exist_ok=True)

    # copy rendered images
    if render_trajectories:
        for v in view_ids:
            src = os.path.join(working_dir, f"camera_angle_{v}_rendered.png")
            if os.path.exists(src):
                shutil.copy(src, segment_dir)

    # also copy the trajectory_points.txt for traceability
    txt_src = os.path.join(working_dir, "trajectory_points.txt")
    if os.path.exists(txt_src):
        shutil.copy(txt_src, segment_dir)
    
    return grand_traj_l

def create_combined_object(config,segment_opens=None,segment_idx=0,limit_to_correction_indices=None,render_trajectories=False,create_continuous_path=True):
    working_dirs = config.working_dirs
    density_generation_params = config.density_generation_params

    if limit_to_correction_indices is None:
        limit_to_correction_indices = list(range(len(working_dirs)))


        

    for idx in limit_to_correction_indices:
        working_folder = working_dirs[idx]


        grand_traj_l = load_grand_traj_from_file(config,working_folder,segment_idx=segment_idx,render_trajectories=render_trajectories,create_continuous_path=create_continuous_path)
        #Pickle the output to the working_folder directory
        pickle_path = os.path.join(working_folder, "grand_traj_tor_r.pkl")
        # Save grand_traj_tor_r to the pickle file
        with open(pickle_path, "wb") as f:
            pickle.dump(grand_traj_l, f)

        grand_traj_np = [
            [traj.squeeze(0).detach().cpu().numpy()     # (N,2) ndarray
            for traj in view]
            for view in grand_traj_l
        ]

        # ------------------------------------------------------
        # Pickle the NumPy version
        # ------------------------------------------------------
        numpy_pickle_path = os.path.join(working_folder, "grand_traj_tor_r_numpy.pkl")
        with open(numpy_pickle_path, "wb") as f:
            pickle.dump(grand_traj_np, f)

        print(f"Saved NumPy version to: {numpy_pickle_path}")

        #Craete and save to the segment folder
        segment_dir = os.path.join(working_folder, f"segment_{segment_idx}")
        os.makedirs(segment_dir, exist_ok=True)

        seg_pickle_path = os.path.join(segment_dir, "grand_traj_tor_r.pkl")
        seg_numpy_pickle_path = os.path.join(
            segment_dir, "grand_traj_tor_r_numpy.pkl"
        )

        with open(seg_pickle_path, "wb") as f:
            pickle.dump(grand_traj_l, f)
        with open(seg_numpy_pickle_path, "wb") as f:
            pickle.dump(grand_traj_np, f)

        print(f"Duplicated both pickles to: {segment_dir}")
