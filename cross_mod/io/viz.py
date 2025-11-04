# refactored module (auto-generated)


# ---- imports from original modules ----
from  utils.RLBenchFunctions.template_sensor_views import compute_camera_pose

from PIL import Image

from matplotlib.colors import ListedColormap     # ← add this

from matplotlib.image import imread

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

from torchvision.utils import save_image

from transforms3d.euler import quat2euler, euler2quat

from types import MethodType

from typing import List

from typing import List, Tuple, Dict

from utils.Classes.policy_trajectory_class import PolicyTrajectory

from utils.RLBenchFunctions.add_time_stamps_to_drawings import add_time_stamps_to_drawings

from utils.RLBenchFunctions.custom_action_modes import EndEffectorPoseViaPlanning_Record,IVKPlanningBounds,IVKPlanningBounds_NonDiscrete

from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds,MoveArmThenGripperWithBoundsDelta_IVK, EndEffectorPoseViaPlanning_Custom, MoveArmThenGripperWithBoundsDelta

from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import save_sample_trajectories

from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import visualize_trajectories

from utils.RLBenchFunctions.trajectory_generator import rbf_kernel,trajectory_model

import FrEIA.framework as Ff

import FrEIA.modules as Fm

import concurrent.futures

import copy

import cv2

import cv2                                   # ← switched to OpenCV

import gc

import glob

import json

import matplotlib.pyplot as plt

import numpy as np

import os

import os, pickle, math

import os, re, ast

import os, re, ast, copy

import os, re, ast, cv2, numpy as np

import os, re, glob

import os, shutil   # ←‑‑ NEW

import pickle

import random

import re

import torch

import torch.nn as nn



def plot_trajectory_grid(config,root_path,
                         fname="full_ivk_trajectories.pkl",
                         colours=("tab:red", "tab:green"),
                         limit_to_correction_indices=None,   # (fail, success)
                         fig_kw=None):
    
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

    lengths, success, max_cols = [], [], 0
    for d in folders:
        with open(os.path.join(root_path, d, fname), "rb") as fh:
            data = pickle.load(fh)
        trajs, succ_flags = data["full_trajectories"], data["successful"]
        row_len = [len(t.observations) for t in trajs]
        lengths.append(row_len)
        success.append(succ_flags)
        max_cols = max(max_cols, len(row_len))

    pad = lambda seq, fill: seq + [fill] * (max_cols - len(seq))
    len_mat  = np.array([pad(r, np.nan) for r in lengths], dtype=float)
    succ_mat = np.array([pad(r, False)  for r in success ], dtype=bool)

    # ── plot ───────────────────────────────────────────────────────────
    cmap = ListedColormap(colours)                 # ← fix here
    fig_kw = dict(figsize=(max_cols * .6, len(folders) * .6), **(fig_kw or {}))
    fig, ax = plt.subplots(**fig_kw)

    ax.imshow(succ_mat.astype(int), cmap=cmap, aspect="equal", vmin=0, vmax=1)  # cast → int

    for r in range(len(folders)):
        for c in range(max_cols):
            if not math.isnan(len_mat[r, c]):
                ax.text(c, r, int(len_mat[r, c]),
                        ha="center", va="center",
                        color="white" if succ_mat[r, c] else "black",
                        fontsize=8)

    ax.set_xticks(range(max_cols))
    ax.set_yticks(range(len(folders)))
    ax.set_yticklabels(folders)
    ax.set_xlabel("trajectory idx"); ax.set_ylabel("folder idx")
    ax.set_title("Demo trajectory success (green) / failure (red)\n"
                 "numbers = trajectory length")
    plt.tight_layout(); plt.show()

def add_trajectory_line(traj, line_thickness=0.005, color=[1.0, 0.0, 0.0]):
    points = traj[:, :3]
    segments = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        
        cyl = Shape.create(
            PrimitiveShape.CYLINDER,
            size=[line_thickness, line_thickness, np.linalg.norm(end - start)],
            mass=0,
            color=color,
            static=True,
            respondable=False
        )

        set_cylinder_between_points(cyl, start, end)
        segments.append(cyl)
    return segments

def set_cylinder_between_points(cyl, start_point, end_point):
    """
    Orients and positions a cylinder between two points.
    """
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    center = (start_point + end_point) / 2
    diff = end_point - start_point
    length = np.linalg.norm(diff)

    # Default cylinder orientation is along Z-axis in PyRep
    default_dir = np.array([0, 0, 1])

    # Compute rotation axis and angle
    rotation_axis = np.cross(default_dir, diff)
    if np.linalg.norm(rotation_axis) < 1e-6:
        # Vectors are parallel
        if np.dot(default_dir, diff) > 0:
            quat = [0, 0, 0, 1]  # No rotation needed
        else:
            quat = R.from_euler('x', np.pi).as_quat()
    else:
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(default_dir, diff) / length, -1.0, 1.0))
        quat = R.from_rotvec(rotation_axis * angle).as_quat()

    cyl.set_orientation(quat, relative_to=None)
    cyl.set_position(center)

def build_overview_grids(config, limit_to_correction_indices=None, dpi=200,segment_idx=0):
    iteration_working_dir = config.iteration_working_dir
    if limit_to_correction_indices is None:
        limit_to_correction_indices = config.seeds

    for seed in limit_to_correction_indices:
        image_dir = os.path.join(iteration_working_dir, str(seed))
        pattern   = re.compile(r"camera_angle_(\d+)_traj_(\d+)\.png$")
        files     = glob.glob(os.path.join(image_dir, "camera_angle_*_traj_*.png"))

        traj_map = {}
        view_map = {v:{} for v in range(4)}
        for fp in files:
            m = pattern.search(os.path.basename(fp))
            if not m: continue
            view, idx = map(int, m.groups())
            traj_map.setdefault(idx, {})[view] = fp
            view_map.setdefault(view, {})[idx] = fp

        if not traj_map:
            print(f"[Seed {seed}] No matching trajectory images found.")
            continue

        blank = np.zeros((1200,1200,3), dtype=np.uint8)

        # -------- 1) per-trajectory (unchanged) ------------------------------
        for idx, vdict in traj_map.items():
            fig,ax = plt.subplots(2,2,figsize=(6,6),dpi=dpi)
            for v,(r,c) in zip(range(4),[(0,0),(0,1),(1,0),(1,1)]):
                a=ax[r][c]; a.axis("off")
                img = imread(vdict.get(v,"")) if v in vdict else blank
                a.imshow(img)
            fig.tight_layout(pad=0)
            fig.savefig(os.path.join(image_dir,f"trajectory_overview_traj_{idx}.png"),
                        bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        # -------- 2) all-trajectories overlay -------------------------------
        fig,ax = plt.subplots(2,2,figsize=(6,6),dpi=dpi)
        for v,(r,c) in zip(range(4),[(0,0),(0,1),(1,0),(1,1)]):
            a=ax[r][c]; a.axis("off")
            if not view_map.get(v):
                a.imshow(blank); continue

            # start with first trajectory in this view
            paths = list(view_map[v].values())
            canvas = imread(paths[0]).copy()

            for fp in paths[1:]:
                overlay = imread(fp)
                # adaptive mask (float or uint8)
                scale = 1.0 if overlay.dtype.kind == 'f' else 255.0
                mask  = (overlay[...,0] > 0.8*scale) & (overlay[...,1] < 0.2*scale)
                canvas[mask] = overlay[mask]

            a.imshow(canvas)

        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(image_dir, "trajectory_overview_all.png"),
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"[Seed {seed}] grids + composite saved")

        # ------------------------------------------------------------------
        # 3) ALSO copy the overview images into segment_<segment_idx>/
        # ------------------------------------------------------------------
        segment_dir = os.path.join(image_dir, f"segment_{segment_idx}")
        os.makedirs(segment_dir, exist_ok=True)

        # copy per‑trajectory grids
        for idx in traj_map.keys():
            src = os.path.join(image_dir, f"trajectory_overview_traj_{idx}.png")
            if os.path.exists(src):
                shutil.copy(src, segment_dir)

        # copy the combined overlay
        overlay_src = os.path.join(image_dir, "trajectory_overview_all.png")
        if os.path.exists(overlay_src):
            shutil.copy(overlay_src, segment_dir)

        print(f"[Seed {seed}] grids + composite saved (and duplicated to {segment_dir})")

def render_trajectories(
        config,
        limit_to_correction_indices=None,
        *,
        json_name       : str = "grand_traj_narrow.json",
        img_tmpl        : str = "camera_angle_{}.png",
        overlay_tmpl    : str = "camera_angle_{}_traj_{}.png",
        grid_tmpl       : str = "trajectory_overview_json_traj_{}.png",
        radius          : int  = 2,
) -> None:
    """
    •  Reads <seed>/<json_name>   (grand_traj_l structure)
    •  Draws every trajectory on the raw images → <seed>/<overlay_tmpl>
    •  Builds one 2×2 overview grid per trajectory index
         TL=view-0 | TR=view-1
         BL=view-2 | BR=view-3 (white if missing)
       → <seed>/<grid_tmpl>
    Nothing is returned; PNGs are written in-place.
    """

    # --------------- which seeds to process ------------------------------
    if limit_to_correction_indices is None:
        limit_to_correction_indices = config.seeds

    for seed in limit_to_correction_indices:
        seed_dir  = os.path.join(config.iteration_working_dir, str(seed))
        json_path = os.path.join(seed_dir, json_name)
        if not os.path.exists(json_path):
            raise FileNotFoundError(json_path)

        with open(json_path, "r") as f:
            grand = json.load(f)

        n_views = len(grand)
        if n_views == 0:
            print(f"[seed {seed}] JSON empty, skipping.")
            continue

        # ---------------- first pass: draw overlays ----------------------
        for v in range(n_views):
            raw_path = os.path.join(seed_dir, img_tmpl.format(v))
            if not os.path.exists(raw_path):
                raise FileNotFoundError(f"Missing raw image: {raw_path}")

            img_raw = cv2.imread(raw_path)
            if img_raw is None:
                raise ValueError(f"Couldn’t load {raw_path}")

            H, W = img_raw.shape[:2]
            for j, traj_norm in enumerate(grand[v]):
                arr = np.asarray(traj_norm, dtype=np.float32)
                if (arr < 0).any() or (arr > 1).any():
                    raise ValueError(f"Coords out of range (seed {seed} view {v} traj {j})")

                px = (arr[:, 0] * W).round().astype(int)
                py = (1.0 - arr[:, 1]) * H
                py = py.round().astype(int)

                overlay = img_raw.copy()
                for x, y in zip(px, py):
                    cv2.circle(overlay, (int(x), int(y)),
                               radius, (0, 0, 255), -1, lineType=cv2.LINE_8)

                out_path = os.path.join(seed_dir, overlay_tmpl.format(v, j))
                cv2.imwrite(out_path, overlay)

        print(f"[seed {seed}] overlay PNGs saved.")

        # ---------------- second pass: 2×2 overview grids ----------------
        # infer number of trajectories from view-0 overlays
        idx = 0
        while os.path.exists(os.path.join(seed_dir, overlay_tmpl.format(0, idx))):
            idx += 1
        n_trajs = idx

        blank = 255 * np.ones((H, W, 3), dtype=np.uint8)

        for j in range(n_trajs):
            fig, ax = plt.subplots(2, 2, figsize=(6, 6), dpi=200)
            for v, (r, c) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
                path_overlay = os.path.join(seed_dir, overlay_tmpl.format(v, j))
                path_raw     = os.path.join(seed_dir, img_tmpl.format(v))
                if os.path.exists(path_overlay):
                    img = cv2.cvtColor(cv2.imread(path_overlay), cv2.COLOR_BGR2RGB)
                elif os.path.exists(path_raw):
                    img = cv2.cvtColor(cv2.imread(path_raw), cv2.COLOR_BGR2RGB)
                else:
                    img = blank
                ax[r][c].imshow(img); ax[r][c].axis("off")

            plt.tight_layout(pad=0)
            grid_path = os.path.join(seed_dir, grid_tmpl.format(j))
            plt.savefig(grid_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        print(f"[seed {seed}] overview grids saved.")

def draw_traj(
        view_str="bullet_view_{}.png",
        n_images=2,
        n_trajs=1,
        max_y=None,
        traj_len_est=6,
        ix=-1,
        iy=-1,
        ts=20,
        save_drawings=True,
        cursor_size=3,
        set_normal_size=False):
    """
    Draw trajectories on images with color–toggle and erase / undo support.

    KEYBOARD CONTROLS  (when the OpenCV window is active)
    -----------------------------------------------------
      r  : red           g : green        b : blue
      y  : yellow        k : black        w : white
      e  : erase all     z : undo stroke  ESC : finish current image
    """
    import cv2
    import torch
    import copy
    import matplotlib.pyplot as plt

    global img  # required for the OpenCV callback

    # ------------------------------------------------------------------
    # Colour map (BGR tuples – OpenCV uses BGR, not RGB)
    # ------------------------------------------------------------------
    color_map = {
        'r': (0,   0, 255),
        'g': (0, 255,   0),
        'b': (255, 0,   0),
        'y': (0, 255, 255),
        'k': (0,   0,   0),
        'w': (255, 255, 255)
    }
    current_color = color_map['r']   # default = red

    # ------------------------------------------------------------------
    # Storage for all trajectories
    # ------------------------------------------------------------------
    traj_all_list = []

    # ------------------------------------------------------------------
    # Mouse-callback: draw circles while L-mouse is held
    # ------------------------------------------------------------------
    def make_callback(strokes, drawing_state):
        """
        Returns a callback bound to the local strokes list and drawing flag.
        This trick avoids having to declare them global / nonlocal everywhere.
        """
        def draw_curve(event, x, y, flags, param):
            drawing = drawing_state['drawing']
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing_state['drawing'] = True
                drawing_state['current_stroke'] = []
            elif event == cv2.EVENT_MOUSEMOVE and drawing_state['drawing']:
                cv2.circle(img, (x, y), cursor_size, current_color, -1)
                drawing_state['current_stroke'].append((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                drawing_state['drawing'] = False
                if drawing_state['current_stroke']:
                    strokes.append((drawing_state['current_stroke'][:], current_color))
        return draw_curve

    # ==================================================================
    # MAIN LOOP OVER IMAGES / TRAJECTORIES
    # ==================================================================
    for ii in range(n_images):
        traj_list = []

        for jj in range(n_trajs):
            # ----------------------------------------------------------
            # Load image and set up state for this trajectory
            # ----------------------------------------------------------
            img = cv2.imread(view_str.format(ii))
            if img is None:
                raise FileNotFoundError(f"Could not read {view_str.format(ii)}")

            img_original = img.copy()           # pristine copy for erase
            h_px, w_px   = img.shape[:2]
            strokes      = []                   # list of (points, colour)
            drawing_state = {'drawing': False,
                             'current_stroke': []}

            cv2.namedWindow("Curve Window")
            cv2.setMouseCallback("Curve Window", make_callback(strokes, drawing_state))

            # ----------------------------------------------------------
            # Interactive drawing loop
            # ----------------------------------------------------------
            while True:
                cv2.imshow("Curve Window", img)
                key = cv2.waitKey(10) & 0xFF

                if key == 27:           # ESC → finish this trajectory
                    break

                # -------- Colour toggle ----------
                if key in (ord('r'), ord('g'), ord('b'),
                           ord('y'), ord('k'), ord('w')):
                    current_color = color_map[chr(key)]

                # -------- Erase all --------------
                if key == ord('e'):
                    img[:] = img_original.copy()
                    strokes.clear()

                # -------- Undo last stroke -------
                if key == ord('z') and strokes:
                    strokes.pop()               # remove last stroke
                    img[:] = img_original.copy()
                    for pts, col in strokes:
                        for x, y in pts:
                            cv2.circle(img, (x, y), cursor_size, col, -1)

            cv2.destroyAllWindows()

            # ----------------------------------------------------------
            # Save annotated image (optional)
            # ----------------------------------------------------------
            if save_drawings:
                out_name = view_str.format(ii).replace('.png', f'_traj_{jj}.png')
                cv2.imwrite(out_name, img)

            # ----------------------------------------------------------
            # Build torch trajectory tensor from accumulated points
            # ----------------------------------------------------------
            list_xy = [pt for stroke, _ in strokes for pt in stroke]
            if not list_xy:
                traj_list.append(torch.empty((0, 2)))
                continue

            xy_tor = torch.tensor(list_xy, dtype=torch.float32)
            xy_tor[:, 1] = h_px - xy_tor[:, 1]   # flip y
            xy_tor[:, 0] /= w_px                # normalise x
            xy_tor[:, 1] /= h_px                # normalise y

            traj_list.append(xy_tor.clone())

        traj_all_list.append(copy.deepcopy(traj_list))

    # ------------------------------------------------------------------
    # Quick preview (scatter) & package return
    # ------------------------------------------------------------------
    grand_traj_l = []
    for im in range(n_images):
        plt.figure()
        traj_list_s = []
        for i in range(n_trajs):
            traj_all = traj_all_list[im][i]
            if traj_all.numel() > 0:
                plt.scatter(traj_all[:, 0], traj_all[:, 1])
            traj_list_s.append(traj_all[None])
        grand_traj_l.append(copy.deepcopy(traj_list_s))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title(f"Normalised Trajectories – image {im}")
        plt.show(block=False)

    return grand_traj_l

def draw_traj_DEPRECATED(view_str="bullet_view_{}.png",n_images=2,n_trajs=1,max_y=None,traj_len_est=6,ix=-1,iy=-1,ts=20,save_drawings=True,cursor_size=3,set_normal_size=False):
#def draw_traj(view_str="bullet_view_{}.png", n_images=2,ts=20):
    global img
    list_xy = []

    

    # define mouse callback function to draw circle
    def draw_curve(event, x, y, flags, param):
        global ix, iy, drawing, img
        if event == cv2.EVENT_LBUTTONDOWN:
            if drawing == False:
                drawing = True
            else:
                drawing = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.circle(img, (x, y), cursor_size, (0, 0, 255), -1)
                list_xy.append([x, y])
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                cv2.circle(img, (x, y), cursor_size, (0, 0, 255), -1)

    traj_all_list = []
    for ii in range(0, n_images):
        traj_list = []
        for jj in range(n_trajs):
            list_xy = []
            img = cv2.imread(view_str.format(ii))
            h_px, w_px = img.shape[:2]          # NEW

            cv2.namedWindow("Curve Window")
            cv2.setMouseCallback("Curve Window", draw_curve)

            while True:
                cv2.imshow("Curve Window", img)
                cv2.moveWindow("Curve Window", 500, 100)
                if cv2.waitKey(10) == 27:
                    break
            cv2.destroyAllWindows()

            #save the image
            if save_drawings==True:
                cv2.imwrite(view_str.format(ii).replace('.png', f'_traj_{jj}.png'), img)
            
            """
            xy_tor = torch.tensor(list_xy)
            """
            xy_tor = torch.tensor(list_xy, dtype=torch.float32)
            # print(xy_tor.shape)
            # flip y
            xy_tor[:, 1] = h_px - xy_tor[:, 1]

            # normalise
            xy_tor[:, 0] /= w_px            # NEW
            xy_tor[:, 1] /= h_px            # CHANGED (was /max_y)
            traj_list.append(xy_tor.clone())
            """
            xy_tor[:, 1] = max_y - xy_tor[:, 1]
            traj_list.append(xy_tor.detach().clone())
            """
        traj_all_list.append(copy.deepcopy(traj_list))
    grand_traj_l = []

    for im in range(n_images):
        traj_list_s = []
        plt.figure()
        for i in range(n_trajs):
            # traj_len = int(len(traj_all_list[im][i]) / (ts))
            # traj_sketch = traj_all_list[im][i][::traj_len, :][:ts]
            # traj_sketch_end = traj_all_list[im][i][-traj_len_est:-1, :]
            # traj_all = torch.vstack([traj_sketch, traj_sketch_end])
            """
            traj_all = traj_all_list[im][i] / max_y
            """
            traj_all = traj_all_list[im][i]
            plt.scatter(traj_all[:, 0], traj_all[:, 1])
            # print(len(traj_all))
            traj_list_s.append(traj_all[None])
        # traj_tor = traj_all  # torch.vstack(traj_list_s)[None]
        grand_traj_l.append(copy.deepcopy(traj_list_s))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.savefig("bullet_traj_table{}.png".format(im))

    # normalise to 1
    # grand_traj_tor = torch.vstack(grand_traj_l)
    # grand_traj_tor_r = grand_traj_tor / max_y
    return grand_traj_l
