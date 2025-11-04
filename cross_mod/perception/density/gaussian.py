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



def densify_waypoints(waypoints, step_px=10):
    """
    Insert `n_extra` evenly‑spaced points between each pair of waypoints.

    • waypoints: list of (x, y) tuples
    • n_extra:   how many points to add between every successive pair
                 (defaults to 1 if left as None)

    Returns a new list starting with the first original waypoint and
    ending with the last, but densified in between.
    """

    n_extra = int(step_px)

    #if n_extra is None:
    #    n_extra = 1            # sensible default

    if len(waypoints) < 2 or n_extra <= 0:
        return list(waypoints)  # nothing to do

    dense = [tuple(waypoints[0])]

    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        # add interior samples
        for i in range(1, n_extra + 1):
            alpha = float(i) / (n_extra + 1)  # 0 < alpha < 1
            dense.append((x0 + alpha * (x1 - x0),
                          y0 + alpha * (y1 - y0)))
        dense.append((x1, y1))

    return dense

def densify_waypoints_DEPRECATED(
    waypoints: List[Tuple[float, float]],
    step_px: float = 10.0,
) -> List[Tuple[float, float]]:
    """Linear interpolation so successive samples are ≤ step_px apart."""
    if len(waypoints) < 2:
        return waypoints.copy()

    dense = [tuple(map(float, waypoints[0]))]
    for p0, p1 in zip(waypoints[:-1], waypoints[1:]):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        seg_len = np.hypot(*(p1 - p0))
        n_samples = max(int(np.floor(seg_len / step_px)) - 1, 0)
        if n_samples:
            alphas = np.linspace(0, 1, n_samples + 2, dtype=float)[1:-1]
            dense.extend([tuple(p0 + a * (p1 - p0)) for a in alphas])
        dense.append(tuple(map(float, p1)))
    return dense

def generate_waypoint_gaussians(
        config,
        limit_to_correction_indices,
        traj_idx=0,          # now used to filter lines
        segment_idx=0,       # now used to filter lines
        noise_std=1.0,
        print_paths=False):  # observability toggle
    """
    For each index in ``limit_to_correction_indices`` create RGB(ish) heat‑maps
    for waypoints belonging **only** to
        camera_angle_*_traj_<traj_idx>_segment_<segment_idx>
    lines inside ``trajectory_points.txt``.

    – Images are (h, w, 3) uint8 with identical values in all 3 channels.
    – Saved to   .../<idx>/traj_imgs/img_<view>_<wp>.png
    – If ``print_paths`` is True, prints paths and filter parameters.
    """

    # ─────────────── helpers ────────────────────────────────────────────────
    w, h = config.density_generation_params["img_len"]

    def _gaussian_img(cx, cy):
        if noise_std == 0:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= ix < w and 0 <= iy < h:
                img[iy, ix] = (255, 255, 255)
            return img
        yy, xx = np.ogrid[:h, :w]
        d2 = (xx - cx)**2 + (yy - cy)**2
        g  = np.exp(-d2 / (2 * noise_std**2))
        g  = (g / g.max() * 255).astype(np.uint8)
        return np.dstack([g, g, g])

    # pattern extracts view, traj, segment, then the point list
    header_pat = re.compile(
        r"camera_angle_(\d+)_traj_(\d+)_segment_(\d+)\s*=\s*\[(.*)\]", re.I
    )

    if print_paths:
        print(f"[INFO] Regex pattern : {header_pat.pattern}")
        print(f"[INFO] traj_idx      : {traj_idx}")
        print(f"[INFO] segment_idx   : {segment_idx}")

    # ─────────────── iterate over correction indices ───────────────────────
    for idx in limit_to_correction_indices:
        iter_dir = os.path.join(config.iteration_working_dir, str(idx))
        pts_path = os.path.join(iter_dir, "trajectory_points.txt")
        out_dir  = os.path.join(iter_dir, "traj_imgs")
        os.makedirs(out_dir, exist_ok=True)

        if print_paths:
            print(f"[INFO] idx={idx}")
            print(f"       pts_path: {pts_path}")
            print(f"       out_dir : {out_dir}")

        if not os.path.isfile(pts_path):
            print(f"[!] {pts_path} not found — skipping")
            continue

        # {view_idx: [ (x, y), … ]} for the chosen traj & segment only
        cam_data = {}

        with open(pts_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                m = header_pat.match(line)
                if not m:
                    continue

                view_i  = int(m.group(1))
                traj_i  = int(m.group(2))
                seg_i   = int(m.group(3))
                if traj_i != traj_idx or seg_i != segment_idx:
                    continue  # ignore other traj/segment combinations

                rhs_str = m.group(4)
                try:
                    pts = ast.literal_eval(f"[{rhs_str}]")
                except Exception:
                    continue

                cam_data.setdefault(view_i, []).extend(pts)

        if not cam_data:
            print(f"[!] No matching traj={traj_idx}, segment={segment_idx} in {pts_path}")
            continue

        # ─────────────── generate & save Gaussian images ───────────────────
        total = 0
        for v_idx in sorted(cam_data.keys()):
            waypoints = cam_data[v_idx]
            for wp_idx, (x, y) in enumerate(waypoints):
                img = _gaussian_img(x, y)
                cv2.imwrite(os.path.join(out_dir, f"img_{v_idx}_{wp_idx}.png"), img)
            total += len(waypoints)

        print(f"[✓] {iter_dir}: {total} RGB images saved for traj={traj_idx}, seg={segment_idx}")

def generate_waypoint_gaussians_from_pointing(
        config,
        limit_to_correction_indices,
        noise_std=0):
    """
    Creates RGB(ish) heat‑maps for every waypoint listed in points.txt
    under each <iteration_working_dir>/<idx>/ directory.

    – Images are (h, w, 3) uint8 with identical values in all 3 channels
      (so they look gray but satisfy “RGB” format).
    – Saved to   .../<idx>/traj_imgs/img_<view>_<wp>.png
    """

    pt_pat = re.compile(r"x\s*=\s*([0-9.]+)\s*,?\s*y\s*=\s*([0-9.]+)", re.I)
    w, h = config.density_generation_params["img_len"]   # width, height

    def _gaussian_img(cx, cy):
        """Return an (h, w, 3) uint8 image with a Gaussian bump at (cx, cy)."""
        if noise_std == 0:                          # single white pixel
            img = np.zeros((h, w, 3), dtype=np.uint8)
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= ix < w and 0 <= iy < h:
                img[iy, ix] = (255, 255, 255)
            return img

        yy, xx = np.ogrid[:h, :w]
        d2 = (xx - cx)**2 + (yy - cy)**2
        g  = np.exp(-d2 / (2 * noise_std**2))
        g  = (g / g.max() * 255).astype(np.uint8)      # shape (h, w)

        # Broadcast the grayscale map into 3 identical channels
        img = np.dstack([g, g, g])                    # shape (h, w, 3)
        return img

    for idx in limit_to_correction_indices:
        iter_dir = os.path.join(config.iteration_working_dir, str(idx))
        pts_path = os.path.join(iter_dir, "points.txt")
        out_dir  = os.path.join(iter_dir, "traj_imgs")
        os.makedirs(out_dir, exist_ok=True)

        with open(pts_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        views, cur = [], []
        for line in lines:
            if line.lower().startswith("view"):
                if cur:
                    views.append(cur)
                cur = []
            m = pt_pat.search(line)
            if m:
                x, y = map(float, m.groups())
                cur.append((x, y))
        if cur:
            views.append(cur)

        for v_idx, waypoints in enumerate(views):
            for wp_idx, (x, y) in enumerate(waypoints):
                img    = _gaussian_img(x, y)
                fname  = f"img_{v_idx}_{wp_idx}.png"
                cv2.imwrite(os.path.join(out_dir, fname), img)

        print(f"[✓] {iter_dir}: {sum(len(v) for v in views)} RGB images saved")
