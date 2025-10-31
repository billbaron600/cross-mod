# Wzhi: Extract 2d densities from drawings
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import copy
import pickle

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn as nn
import os
import pickle
import torch
import re
from typing import List

from torchvision.utils import save_image
from utils.RLBenchFunctions.add_time_stamps_to_drawings import add_time_stamps_to_drawings

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os, re, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import json


#drawing = False  # true if mouse is pressed
#ix, iy = -1, -1
#max_y = 600  # image_size_max. WAS 600
#max_y = 1200

#n_images = 3  # we have 2 images
#n_trajs = 2  # we have 3 trajectories
#traj_len_est = 6
drawing = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


import os, re, ast, copy
from typing import List, Tuple, Dict
import cv2
import torch
import numpy as np

# -------------------------------------------------------------------------
# 1)  Densifier (same as we discussed, kept local for self‑contained file)
# -------------------------------------------------------------------------
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

"""
# -------------------------------------------------------------------------
# 2)  Loader that emulates draw_traj()
# -------------------------------------------------------------------------
def load_grand_traj_from_file_DEPRECATED(
    config,
    segment_idx: int,
    working_dir: str,
    step_px: float = 1.0,
    render_trajectories: bool = False,
    dot_radius: int = 2,
    dot_color: Tuple[int, int, int] = (0, 0, 255),
    create_continuous_path = True
) -> List[List[torch.Tensor]]:

    # ----------------------------------------------------------------
    # (1) Load base resolution from view 0 (any view works).
    # ----------------------------------------------------------------
    sample_im = cv2.imread(os.path.join(working_dir, "camera_angle_0.png"))
    if sample_im is None:
        raise FileNotFoundError("camera_angle_0.png missing in working_dir")
    h_px, w_px = sample_im.shape[:2]

    # ----------------------------------------------------------------
    # (2) Parse trajectory_points.txt  →  dict[view][traj] = [(x,y),...]
    # ----------------------------------------------------------------
    txt_path = os.path.join(working_dir, "trajectory_points.txt")
    with open(txt_path, "r") as f:
        txt = f.read()

    patt = re.compile(r"view_(\d+)_traj_(\d+)\s*=\s*\[(.*?)\]", re.S)
    raw_views: Dict[int, Dict[int, List[Tuple[float, float]]]] = {}
    for vid, tid, rhs in patt.findall(txt):
        raw_views.setdefault(int(vid), {})[int(tid)] = ast.literal_eval("[" + rhs + "]")

    # ----------------------------------------------------------------
    # (3) Iterate over *all* requested views
    # ----------------------------------------------------------------
    view_ids = config.ray_tracing_params["n_views"]
    grand_traj_l: List[List[torch.Tensor]] = []

    for v in view_ids:
        # Prepare base image (copy each loop to avoid carry‑over drawings)
        view_img_path = os.path.join(working_dir, f"camera_angle_{v}.png")
        view_img = cv2.imread(view_img_path)
        if view_img is None:
            raise FileNotFoundError(f"{view_img_path} not found")

        traj_list: List[torch.Tensor] = []
        trajs_for_view = raw_views.get(v, {})          # may be empty

        for t in sorted(trajs_for_view.keys()):
            if create_continuous_path == True:
                dense_px = densify_waypoints(trajs_for_view[t], step_px=step_px)
            else:
                dense_px = trajs_for_view[t]

            print("Traj Size: " + str(len(dense_px)))
            # -------- render (unnormalised, original pixel coords)
            if render_trajectories:
                for x, y in dense_px:
                    cv2.circle(view_img, (int(round(x)), int(round(y))),
                               dot_radius, dot_color, -1)

            # -------- flip + normalise for return tensor
            xy = torch.tensor(dense_px, dtype=torch.float32)   # (N,2)
            xy[:, 1] = h_px - xy[:, 1]        # flip Y
            xy[:, 0] /= w_px                  # normalise
            xy[:, 1] /= h_px
            traj_list.append(xy[None])        # (1, N, 2)

        grand_traj_l.append(copy.deepcopy(traj_list))

        # -------- save rendered image (drawn or original)
        if render_trajectories:
            out_path = os.path.join(working_dir, f"camera_angle_{v}_rendered.png")
            cv2.imwrite(out_path, view_img)

    return grand_traj_l
"""

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

import os, shutil   # ←‑‑ NEW
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
    
from utils.RLBenchFunctions.trajectory_generator import rbf_kernel,trajectory_model

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

import os
import re
import numpy as np
import cv2                                   # ← switched to OpenCV

import os, re, ast
import cv2
import numpy as np

import os, re, ast, cv2, numpy as np

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