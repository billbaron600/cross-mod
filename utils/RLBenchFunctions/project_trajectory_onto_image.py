import os
import pickle
import math
from typing import Union

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from itertools import cycle, islice
import pickle


def overlay_sampled_trajectories(
        config,
        limit_to_correction_indices=None,
        real_demo=True,
        use_intrinsics=True,
):
    """
    • If generate_trajectories is None → projects only the mean trajectory
      found in ray_tracing_results.pkl (like the old helper).
    • Otherwise projects every trajectory (NaN-padded) as a **scatter plot**,
      saving one PNG per (trajectory × view).
    """

    if limit_to_correction_indices is None:
        limit_to_correction_indices = config.seeds

    n_views = config.ray_tracing_params["n_views"]

    # ── helper: perspective projection ──────────────────────────────────
    def project_points(pts_xyz, K_np, T_cam_from_world_np):
        P = pts_xyz.shape[0]
        Xw_h = np.concatenate([pts_xyz, np.ones((P, 1), np.float32)], axis=1).T
        Xc_h = T_cam_from_world_np @ Xw_h
        Z    = Xc_h[2]
        mask = Z > 1e-6
        Xc   = Xc_h[:3, mask]
        xy   = Xc[:2] / Xc[2]                # perspective divide
        uv_h = K_np @ np.vstack((xy, np.ones_like(xy[:1])))
        return uv_h[:2].T, mask              # (Q,2)  , (P,)

    # ────────────────────────────────────────────────────────────────────
    for idx in limit_to_correction_indices:
        base_path = os.path.join(config.iteration_working_dir, str(idx))
        os.makedirs(base_path, exist_ok=True)

        # ---------- load trajectories -----------------------------------
        #if generate_trajectories is None:
        #    with open(os.path.join(base_path, "ray_tracing_results.pkl"), "rb") as f:
        #        rt_results = pickle.load(f)
        #    traj_list = [rt_results["mean_tor"].astype(np.float32)]
        #else:
        with open(os.path.join(base_path, "generated_trajectories.pkl"), "rb") as f:
            generated_trajectories = pickle.load(f)


        if isinstance(generated_trajectories, torch.Tensor):
            traj_tensor = generated_trajectories.detach().cpu()          # (M,N,3)
        else:  # e.g. it was pickled as list/ndarray
            traj_tensor = torch.as_tensor(generated_trajectories).detach().cpu()

        M = traj_tensor.shape[0]
        traj_list = [
            traj_tensor[i][~torch.isnan(traj_tensor[i][:, 0])].numpy().astype(np.float32)
            for i in range(M)
        ]

        # ---------- camera bundle ---------------------------------------
        cam_bundle = torch.load(os.path.join(base_path, "poses_mobile.tar"))
        extrinsics = cam_bundle["extrinsics"]
        intrinsics = cam_bundle["intrinsics"]

        # colour per trajectory (seismic gradient)
        cmap = plt.get_cmap("seismic")
        traj_colors_b = (cmap(np.linspace(0, 1, M))[:, :3] * 255).astype(np.uint8)

        # ---------- iterate over views ----------------------------------
        for v in n_views:
            # intrinsics --------------------------------------------------
            if use_intrinsics:
                K = intrinsics[v].numpy().copy()
            else:
                FOCAL = config.ray_tracing_params["focal"]
                H, W  = cv2.imread(os.path.join(base_path, f"camera_angle_{v}.png")).shape[:2]
                K = np.array([[FOCAL, 0, W * 0.5],
                              [0, FOCAL, H * 0.5],
                              [0,     0,     1]], dtype=np.float32)

            # extrinsics --------------------------------------------------
            T_w_c = extrinsics[v].numpy()
            if not real_demo:                              # flip-Z for sim
                T_w_c[:3, :3] = T_w_c[:3, :3] @ np.diag([1, -1, -1])
            T_c_w = np.linalg.inv(T_w_c)

            # base RGB ----------------------------------------------------
            img_path = os.path.join(base_path, f"camera_angle_{v}.png")
            img_base = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            H, W     = img_base.shape[:2]

            # --------------- per-trajectory scatter ---------------------
            for t_idx, pts in enumerate(traj_list):
                uv, _ = project_points(pts, K, T_c_w)
                in_img = (
                    (uv[:, 0] >= 0) & (uv[:, 0] < W) &
                    (uv[:, 1] >= 0) & (uv[:, 1] < H)
                )
                if not np.any(in_img):
                    continue

                img = img_base.copy()
                colour = traj_colors_b[t_idx].tolist()
                for (u, v_) in uv[in_img].astype(int):
                    cv2.circle(img, (u, v_), 3, colour, -1)

                out_root = os.path.join(base_path, "sampled_trajectories_scattered")
                os.makedirs(out_root, exist_ok=True)

                # 2. write every PNG here
                out_file = os.path.join(
                    out_root, f"projected_ray_traced_traj{t_idx}_view_{v}.png"
                )
                cv2.imwrite(out_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"saved  {out_file}")



# -----------------------------------------------------------------------------
# main helper
# -----------------------------------------------------------------------------
def project_ray_traced_mean(
        config,
        limit_to_correction_indices=None,
        real_demo=False,
        use_intrinsics=True,
        segment_idx=None
        #base_path: Union[str, os.PathLike] = "run_results/rubish_in_bin/demos/3/",
    ):
    """
    Projects the mean ray-traced trajectory onto each camera image and saves:
      •  projected_ray_traced_mean_view_{i}.png  for i in 0…n_views-1
      •  projected_ray_traced_mean_all_views.png (2×2 montage)
      •  projected_points_rgb.pt  ▶  {'points_3d': (N,3) float32 tensor,
                                      'rgb':       (N,3) float32 tensor}
    
    Parameters
    ----------
    config : object
        Must contain  config.ray_tracing_params["n_views"]
    base_path : str or Path
        Folder that contains  ray_tracing_results.pkl ,
                             poses_mobile.tar           ,
                             camera_angle_{i}.png
    """
    if limit_to_correction_indices is None:
        limit_to_correction_indices = config.seeds
    
    for i in limit_to_correction_indices:
        base_path = os.path.join(config.iteration_working_dir,str(i))
        base_path = os.fspath(base_path)  # in case Path object
        n_views   = config.ray_tracing_params["n_views"]

        # ------------------------------------------------------------------ I/O --
        with open(os.path.join(base_path, "ray_tracing_results.pkl"), "rb") as f:
            ray_tracing_results = pickle.load(f)
        mean_tor   = ray_tracing_results["mean_tor"]        # (N,3)  np.float32
        cam_bundle = torch.load(os.path.join(base_path, "poses_mobile.tar"))
        extrinsics = cam_bundle["extrinsics"]               # (V,4,4)
        intrinsics = cam_bundle["intrinsics"]               # list/tuple len V

    
        
        N = mean_tor.shape[0]

        # ---------------------------------------------------------------- gradient
        #cmap      = plt.get_cmap("viridis")
        cmap = plt.get_cmap("seismic")
        colors_f  = cmap(np.linspace(0, 1, N))[:, :3]       # (N,3) floats 0-1
        colors_b  = (colors_f * 255).astype(np.uint8)       # (N,3) uint8 for cv2

        # store later
        torch_dict = {
            "points_3d": torch.as_tensor(mean_tor, dtype=torch.float32),
            "rgb":       torch.as_tensor(colors_f, dtype=torch.float32),
        }

        # -------------- helper to project once -----------------------------------
        def project_points(K_np, T_cam_from_world_np):
            """Return (M,2) pixel coords for valid (z>0) points; same index order."""
            Xw_h = np.concatenate([mean_tor, np.ones((N, 1))], axis=1).T       # 4×N
            Xc_h = T_cam_from_world_np @ Xw_h                                  # 4×N
            Z    = Xc_h[2]
            mask = Z > 1e-6
            Xc   = Xc_h[:3, mask]
            # perspective divide
            xy = Xc[:2] / Xc[2]
            uv_h = K_np @ np.vstack((xy, np.ones_like(xy[:1])))
            uv   = uv_h[:2].T  # (M,2)
            return uv, mask

        # -------------------- loop over every view -------------------------------
        combined_imgs = []
        for v in n_views:
            # load / fix intrinsics ----------------------------------------------
            K  = intrinsics[v].numpy().copy()
            
                


            #K[0, 0] = abs(K[0, 0])
            #K[1, 1] = abs(K[1, 1])

            # extrinsics ----------------------------------------------------------
            T_world_from_cam = extrinsics[v].numpy()
            if real_demo is False:
                flip_z = np.diag([1, -1, -1])
                T_world_from_cam[0:3,0:3] = T_world_from_cam[0:3,0:3] @ flip_z  # Now the "forward" direction is flipped.
            
            T_cam_from_world = np.linalg.inv(T_world_from_cam)

            # project -------------------------------------------------------------
            uv, mask = project_points(K, T_cam_from_world)

            # read image ----------------------------------------------------------
            img_path = os.path.join(base_path, f"camera_angle_{v}.png")
            img      = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            # draw ---------------------------------------------------------------
            H, W = img.shape[:2]
            #if use_intrinsics==False:
            #    FOCAL = config.ray_tracing_params["focal"]
            #    K = np.array([[FOCAL,     0, W * 0.5],     # fx  0  cx
            #        [    0,  FOCAL, H * 0.5],     # 0  fy  cy
            #        [    0,     0,       1]],     # 0   0   1
            #        dtype=np.float32)


            in_img = (
                (uv[:, 0] >= 0) & (uv[:, 0] < W) &
                (uv[:, 1] >= 0) & (uv[:, 1] < H)
            )
            for (u, v_), col in zip(uv[in_img].astype(int), colors_b[mask][in_img]):
                cv2.circle(img, (u, v_), 3, col.tolist(), -1)

            # save single-view image ---------------------------------------------
            if segment_idx is None:
                out_single = os.path.join(base_path, f"projected_ray_traced_mean_view_{v}.png")
            else:
                out_single = os.path.join(base_path,"segment_"+str(segment_idx), f"projected_ray_traced_mean_view_{v}.png")
            cv2.imwrite(out_single, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"saved  {out_single}")

            combined_imgs.append(img)

        # ---------------- montage (2×2 or ceil grid) -----------------------------
        cols = int(math.ceil(math.sqrt(len(n_views))))
        rows = int(math.ceil(len(n_views) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.ravel(axes)

        for idx, (ax, im) in enumerate(zip(axes, combined_imgs)):
            ax.imshow(im)
            ax.set_title(f"View {idx}")
            ax.axis("off")

        # hide unused axes
        for ax in axes[len(combined_imgs):]:
            ax.axis("off")

        combined_path = os.path.join(base_path, "projected_ray_traced_mean_all_views.png")
        plt.tight_layout()
        plt.savefig(combined_path, dpi=200)
        plt.close(fig)
        print(f"saved  {combined_path}")

        # ---------------- save colour dictionary ---------------------------------
        #torch.save(torch_dict, os.path.join(base_path, "projected_points_rgb.pt"))
        # build dict as NumPy                               ↓  ↓
        data_dict = {
            "points_3d": mean_tor.astype(np.float32),   # (N,3)  np.ndarray
            "rgb":       colors_f.astype(np.float32),   # (N,3)  np.ndarray
        }

        # save with pickle                                ↓  ↓
        out_pickle = os.path.join(base_path, "projected_points_rgb.pkl")
        with open(out_pickle, "wb") as f:
            pickle.dump(data_dict, f)
        print(f"saved  {out_pickle}")
        print("saved  projected_points_rgb.pkl")


def project_ray_traced_trajectories(
        config,
        limit_to_correction_indices=None,
        real_demo=False,
        use_intrinsics=True,
        generate_trajectories=None          # (M, N, 3) tensor or None
):
    """
    • If generate_trajectories is None → behaves like the old
      project_ray_traced_mean (uses ray_tracing_results.pkl).
    • If generate_trajectories is a tensor (M×N×3, NaN-padded) →
      projects every trajectory as a coloured poly-line.
    """

    if limit_to_correction_indices is None:
        limit_to_correction_indices = config.seeds

    n_views = config.ray_tracing_params["n_views"]

    # --------------------------------------------------------------------------
    # helper ─ project arbitrary (K,N,3) pts with any 3×3 intrinsic
    # --------------------------------------------------------------------------
    def project_points(pts_xyz, K_np, T_cam_from_world_np):
        """
        pts_xyz : (P,3) float32 array (world coords)
        K_np    : (3,3)
        T_cam_from_world_np : (4,4)
        Returns
        -------
        uv  : (Q,2) float32 pixel coords  (only for pts with z>0)
        mask: (P,)  bool  → True if point projected (z>0)
        """
        P = pts_xyz.shape[0]
        Xw_h = np.concatenate([pts_xyz, np.ones((P, 1), np.float32)], axis=1).T  # 4×P
        Xc_h = T_cam_from_world_np @ Xw_h
        Z    = Xc_h[2]
        mask = Z > 1e-6
        Xc   = Xc_h[:3, mask]
        xy   = Xc[:2] / Xc[2]                         # perspective divide
        uv_h = K_np @ np.vstack((xy, np.ones_like(xy[:1])))
        return uv_h[:2].T, mask                      # (Q,2) , (P,)

    # --------------------------------------------------------------------------
    for idx in limit_to_correction_indices:
        base_path = os.path.join(config.iteration_working_dir, str(idx))
        os.makedirs(base_path, exist_ok=True)        # ensure folder exists

        # ------------------------------------------------------------------ I/O
        if generate_trajectories is None:
            with open(os.path.join(base_path, "ray_tracing_results.pkl"), "rb") as f:
                ray_tracing_results = pickle.load(f)
            traj_list = [ray_tracing_results["mean_tor"].astype(np.float32)]
        else:
            # strip NaNs per-trajectory, detach, move to CPU, cast to np.float32
            traj_list = [
                t.detach()[~torch.isnan(t[:, 0])].cpu().numpy().astype(np.float32)
                for t in generate_trajectories
            ]
        M = len(traj_list)                           # how many trajectories

        cam_bundle = torch.load(os.path.join(base_path, "poses_mobile.tar"))
        extrinsics = cam_bundle["extrinsics"]        # (V,4,4) torch
        intrinsics = cam_bundle["intrinsics"]        # list / tuple len V

        # ---------------------------------------------------------------- colors
        cmap = plt.get_cmap("seismic")
        traj_colors_f = cmap(np.linspace(0, 1, M))[:, :3]          # (M,3) 0-1
        traj_colors_b = (traj_colors_f * 255).astype(np.uint8)     # uint8 BGR

        combined_imgs = []
        for v in n_views:
            # ---------------- intrinsics --------------------------------------
            if use_intrinsics:
                K = intrinsics[v].numpy().copy()
            else:
                # synthetic pin-hole with scalar focal
                FOCAL = config.ray_tracing_params["focal"]
                img_shape = cv2.imread(
                    os.path.join(base_path, f"camera_angle_{v}.png")
                ).shape[:2]   # (H,W)
                H, W = img_shape
                K = np.array([[FOCAL, 0, W * 0.5],
                              [0, FOCAL, H * 0.5],
                              [0, 0, 1]], dtype=np.float32)

            # ---------------- extrinsics --------------------------------------
            T_world_from_cam = extrinsics[v].numpy()
            if not real_demo:                            # flip Z for sim views
                T_world_from_cam[:3, :3] = (
                    T_world_from_cam[:3, :3] @ np.diag([1, -1, -1])
                )
            T_cam_from_world = np.linalg.inv(T_world_from_cam)

            # ---------------- read & draw image -------------------------------
            img_path = os.path.join(base_path, f"camera_angle_{v}.png")
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]

            for t_idx, pts in enumerate(traj_list):
                uv, _ = project_points(pts, K, T_cam_from_world)
                in_img = (
                    (uv[:, 0] >= 0) & (uv[:, 0] < W) &
                    (uv[:, 1] >= 0) & (uv[:, 1] < H)
                )
                if np.count_nonzero(in_img) >= 2:
                    cv2.polylines(
                        img,
                        [uv[in_img].astype(int)],
                        isClosed=False,
                        color=traj_colors_b[t_idx].tolist(),
                        thickness=2,
                    )

            out_single = os.path.join(
                base_path, f"projected_ray_traced_trajectories_view_{v}.png"
            )
            cv2.imwrite(out_single, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"saved  {out_single}")
            combined_imgs.append(img)

        # ---------------- 2×2 (or ceil) montage -------------------------------
        cols = int(math.ceil(math.sqrt(len(n_views))))
        rows = int(math.ceil(len(n_views) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.ravel(axes)
        for ax, im in zip(axes, combined_imgs):
            ax.imshow(im)
            ax.axis("off")
        for ax in axes[len(combined_imgs):]:
            ax.axis("off")

        combined_path = os.path.join(base_path, "projected_ray_traced_mean_all_views.png")
        plt.tight_layout()
        plt.savefig(combined_path, dpi=200)
        plt.close(fig)
        print(f"saved  {combined_path}")