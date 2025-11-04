# refactored module (auto-generated)


# ---- imports from original modules ----
from rendering_helpers import *

from torch import nn

from utils.RLBenchFunctions.combine_density_views import combine_density_views

from utils.RLBenchFunctions.plottingFunctions.plot_traced_points import plot_and_save

import numpy as np

import os, shutil

import pickle

import torch



class trajectory_model:
    def __init__(
        self,
        focal=None,
        height=100,
        width=100,
        near=1,
        far=6,
        n_weights=20,
        n_views=[0,1], #what camera views to use in the ray tracing
        n_times=50,
        n_samples=150,
        perturb=True,
        gamma=50,
        ray_dist_threshold=0.01,
        density_threshold=0.25,
        #inducing_points=[-0.2,1.2],
        inducing_points=[0,1],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        if focal == None:
            # default focal from the lego nerf data
            focal = torch.tensor(138.8888789).to(device)
        self.focal = focal
        self.height = height
        self.width = width
        self.focal_lengths = None
        self.near = near
        self.far = far
        self.n_views = n_views
        self.n_times = n_times
        self.n_samples = n_samples
        self.perturb = perturb
        self.ray_dist_threshold = ray_dist_threshold
        self.density_threshold = density_threshold
        self.n_weights = n_weights
        self.gamma = gamma
        self.device = device
        # initialise weights means and weights standard deviation
        self.weights = torch.zeros((n_weights, 3)).to(device)
        self.std_weights = torch.zeros((n_weights, 3)).to(device)
        # set the time inducing points +-0.2 over the boundaries
        self.t_s = torch.linspace(inducing_points[0], inducing_points[1], n_weights).reshape((-1, 1)).to(device)
        # query resolution set as n_times
        self.t_check = torch.linspace(0, 1, n_times).reshape((-1, 1)).to(device)
        # precompute the features
        self.feats = rbf_kernel(self.t_check, self.t_s, self.gamma)

    def get_focal_length_for_each_view(self, intrinsic_mats: torch.Tensor):
        """
        Given a stack of camera intrinsic matrices (shape [N, 3, 3]),
        derive a single scalar focal length for each view.

        We assume square-ish pixels so fx ≈ fy and store the mean of the two.
        The resulting tensor is cached in `self.focal_lengths` and returned.
        """
        if intrinsic_mats.ndim != 3 or intrinsic_mats.shape[-2:] != (3, 3):
            raise ValueError("intrinsic_mats must have shape [N, 3, 3]")

        # fx = K[0,0], fy = K[1,1]
        fx = intrinsic_mats[:, 0, 0]
        fy = intrinsic_mats[:, 1, 1]

        # simple average yields a robust scalar focal per view
        self.focal_lengths = ((fx + fy) / 2.0).to(self.device)

        return self.focal_lengths
    
    def extract_mean_std_from_images(self, im_list_tor_all, poses,print_outputs=False,intrinsic_mats=None,negative_z=True):
        """
        Here we input the images (time,views,width,height),
        along with poses of each view
        """

        intrinsic_mats = None
        mean_l = []
        var_l = []
        # loop through the times and check for intersecting rays
        kept_rows = []
        for ctime in range(self.n_times):
            im_list_ti = im_list_tor_all[ctime]
            rel_points_list = []
            for i_pose in range(len(self.n_views)):
                cur_img = im_list_ti[i_pose]
                target_pose = poses[i_pose].to(self.device)
                if intrinsic_mats is not None:
                    K = intrinsic_mats[i_pose].to(self.device)        # shape (3,3)
                    fx, fy = K[0, 0], K[1, 1]                         # focal lengths
                    cx, cy = K[0, 2], K[1, 2]                         # principal point
                    # ---------------------------------------------------------

                    # use a helper that understands fx / fy / cx / cy
                    focal = (fx+fy)/2
                    #print(focal)
                    rays_o, rays_d = get_rays(
                        self.height, self.width,focal,target_pose,negative_z=False
                    )

                    query_points, z_vals = sample_stratified(
                        rays_o, rays_d,
                        self.near, self.far,
                        n_samples=self.n_samples,
                        perturb=self.perturb,
                    )
                else:
                    if isinstance(self.focal, List):
                        focal_use = self.focal[i_pose]
                    else:
                        focal_use = self.focal

                    rays_o, rays_d = get_rays(
                        self.height, self.width, focal_use, target_pose,negative_z=negative_z
                    )
                    query_points, z_vals = sample_stratified(
                        rays_o,
                        rays_d,
                        self.near,
                        self.far,
                        n_samples=self.n_samples,
                        perturb=self.perturb,
                    )
                # Add debug print right here, after getting query_points but before creating rel_points
                if print_outputs==True:
                    print(f"Time {ctime}, View {i_pose} - Number of points above density threshold:", (cur_img > self.density_threshold).sum())
                rel_points = query_points[cur_img > self.density_threshold, :, :]
                rel_points_list.append(rel_points.detach().clone())
            points_0 = rel_points_list[0].reshape((-1, 3))
            points_1 = rel_points_list[1].reshape((-1, 3))
            #dists = torch.cdist(points_0, points_1)

            try:
                # fast path – stays on GPU
                dists = torch.cdist(points_0, points_1)                    # (|P0|,|P1|)

            except RuntimeError as err:
                if "out of memory" in str(err):                            # CUDA OOM caught
                    print(f"[extract_mean_std] CUDA OOM for cdist – "
                        f"falling back to CPU, size={points_0.shape}×{points_1.shape}")

                    # free as much GPU cache as possible before continuing
                    torch.cuda.empty_cache()

                    # slow path – compute on CPU, then copy back
                    dists = torch.cdist(points_0.cpu(), points_1.cpu())    # CPU RAM
                    dists = dists.to(points_0.device) 

            vals_0, inds_0 = torch.min(dists, dim=-1)
            vals_1, inds_1 = torch.min(dists, dim=0)

            # After computing distances
            if print_outputs==True:
                print("Min distance between points:", torch.min(dists))
                print("Number of close points:", (vals_0 < self.ray_dist_threshold).sum())

            inter_points_0 = points_0[vals_0 < self.ray_dist_threshold]
            inter_points_1 = points_1[vals_1 < self.ray_dist_threshold]
            inter_points = torch.cat([inter_points_0, inter_points_1])

            #ADDED NEW
            if inter_points.numel() == 0:
                # skip this time step or fill with last valid estimate
                continue        # or append a placeholder

            kept_rows.append(ctime) #ADDED
            mean_l.append(inter_points.mean(dim=0))
            var_l.append(inter_points.var(dim=0))
        mean_tor = torch.vstack(mean_l)
        var_tor = torch.vstack(var_l)
        # OLD return (mean_tor, var_tor)
        #NEW
        return torch.vstack(mean_l), torch.vstack(var_l), kept_rows

    def fit_continuous_function(
        self, mean_tor, var_tor, kept_rows=None,n_iter=50000, n_display=5000, lr_mean=1e-4, lr_std=1e-4
    ):
        self.weights.requires_grad_()
        opt = torch.optim.Adam({self.weights}, lr=lr_mean)
        print("Matching trajectory distribution mean")
        for i in range(n_iter):
            opt.zero_grad()
            xyz = self.feats[kept_rows] @ self.weights
            #xyz = self.feats @ self.weights
            loss = (torch.norm(xyz - mean_tor, dim=-1) ** 2).mean()
            loss.backward()
            opt.step()
            if i % n_display == 0:
                print("{}:{}".format(i, loss))

        self.std_weights.requires_grad_()
        opt = torch.optim.Adam({self.std_weights}, lr=lr_std)
        print("Matching trajectory distribution std")
        for i in range(n_iter):
            opt.zero_grad()
            xyz_vars = self.feats[kept_rows] @ nn.Softplus()(self.std_weights)
            loss = (torch.norm(xyz_vars - torch.sqrt(var_tor), dim=-1) ** 2).mean()
            loss.backward()
            opt.step()
            if i % n_display == 0:
                print("{}:{}".format(i, loss))

    def generate_trajectories(self, n_traj_gen=50, number_t_steps=200, print_outputs=False):
        # we can have more dense trajectory timesteps here
        t_check_test = torch.linspace(0, 1, number_t_steps).reshape((-1, 1))
        t_check_test = t_check_test.to(self.device)
        feats_test = rbf_kernel(t_check_test, self.t_s, self.gamma)
        xyz_list = []
        for ii in range(n_traj_gen):
            weights_tr = self.weights + nn.Softplus()(
                self.std_weights
            ) * torch.randn_like(self.std_weights)
            xyz = feats_test @ weights_tr
            xyz_list.append(xyz.detach().clone()[None])
        trajs_generated = torch.vstack(xyz_list)
        return trajs_generated

    def sample_weights(self, n_traj_gen=50):
        std_weights_tiled = torch.tile(self.std_weights[None], (n_traj_gen, 1, 1))
        weights_drawn = torch.tile(
            self.weights[None], (n_traj_gen, 1, 1)
        ) + nn.Softplus()(std_weights_tiled) * torch.randn_like(std_weights_tiled)
        return weights_drawn

    def sample_controlled_weights(
        self,
        n_traj_gen: int = 50,
        variance_scale: float = 1.0,   # ← new knob
    ):
        """
        Draw weight samples from the posterior.

        variance_scale = 0   ➔ deterministic mean
        variance_scale = 1   ➔ original behaviour
        variance_scale = k   ➔ std multiplied by k
        """
        mean_w = torch.tile(self.weights[None], (n_traj_gen, 1, 1))          # (N,D,3)

        if variance_scale == 0:
            return mean_w                                                     # deterministic

        std_w  = nn.Softplus()(
                    torch.tile(self.std_weights[None], (n_traj_gen, 1, 1))
                ) * variance_scale                                            # scaled σ
        return mean_w + std_w * torch.randn_like(std_w)

    def resample_by_arclength(
        self,
        trajs: torch.Tensor,          # (T,3) or (N,T,3)
        n_points: int = 50,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """
        Return `n_points` positions that are *evenly spaced along the
        path length* (not the time parameter).

        Parameters
        ----------
        trajs : (T,3) **or** (N,T,3) tensor
            Trajectory/ies in xyz space, ordered in time.
        n_points : int
            Number of output way-points on each path.
        eps : float
            Small value to avoid division-by-zero on degenerate segments.

        Returns
        -------
        out : (n_points,3) **or** (N,n_points,3) tensor
            Re-sampled trajectory/ies.
        """
        # ------------------------------------------------------------------
        # Handle batch↔single cases uniformly
        # ------------------------------------------------------------------
        if trajs.dim() == 2:                     # (T,3) ➔ (1,T,3)
            trajs = trajs.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        N, T, _ = trajs.shape
        resampled = []

        for i in range(N):
            P = trajs[i]                         # (T,3)

            # --- cumulative arc-length ------------------------------------
            seg_len = torch.norm(P[1:] - P[:-1], dim=-1)   # (T-1,)
            cum_len = torch.cat(
                [torch.zeros(1, device=P.device), seg_len.cumsum(0)]
            )                                            # (T,)
            total_len = cum_len[-1]

            # --- target arc-lengths ---------------------------------------
            s_target = torch.linspace(
                0, total_len, n_points, device=P.device
            )                                           # (n_points,)

            # --- locate neighbouring segments -----------------------------
            idx = torch.searchsorted(cum_len, s_target, right=True) - 1
            idx = idx.clamp(0, T - 2)                   # keep in range

            s0, s1 = cum_len[idx], cum_len[idx + 1]
            w = (s_target - s0) / (s1 - s0 + eps)       # (n_points,)

            # --- linear interpolation -------------------------------------
            P0, P1 = P[idx], P[idx + 1]                 # (n_points,3)
            resampled.append(P0 + w.unsqueeze(1) * (P1 - P0))

        out = torch.stack(resampled)                    # (N,n_points,3)
        return out.squeeze(0) if squeeze_back else out


    def condition_start(
        self,
        start_pos: torch.Tensor,
        n_traj_gen: int = 50,
        number_t_steps: int = 200,          # ← kept for API-compat
        include_variance_in_samples: bool = True,
        variance_scale = None,
        # --- new knobs ------------------------------------------------
        arc_points = None,      # output way-points along path
        dense_time_steps = None # internal temporal resolution
    ) -> torch.Tensor:
        """
        Generate trajectories that

            • start at `start_pos` (t = 0), *and*
            • are returned on `arc_points` positions that are
              uniformly spaced **along the path length**.

        All old kwargs are preserved so existing calls keep working.

        Parameters
        ----------
        start_pos : (3,) tensor
            Desired xyz start position.
        n_traj_gen, number_t_steps, include_variance_in_samples,
        variance_scale : same semantics as the original method.
        arc_points : int, optional
            Number of arc-length-equidistant way-points to return.
            Defaults to `number_t_steps` for backward compatibility.
        dense_time_steps : int, optional
            Temporal resolution of the *intermediate* dense curve from
            which we re-sample.  Default is `max(arc_points*5,
            number_t_steps)`—high enough to make arc-length sampling
            smooth even when the output has few points.

        Returns
        -------
        trajs_even : (n_traj_gen, arc_points, 3) tensor
            Trajectories conditioned on `start_pos` and re-indexed by
            arc-length.
        """

        # ------------------------------------------------------------------
        # 0. House-keeping defaults
        # ------------------------------------------------------------------
        if arc_points is None:
            arc_points = number_t_steps                       # keep old behaviour
        if dense_time_steps is None:
            dense_time_steps = max(arc_points * 5,            # over-sample
                                   number_t_steps)

        # ------------------------------------------------------------------
        # 1. Draw weights (identical logic to the old version)
        # ------------------------------------------------------------------
        if include_variance_in_samples or variance_scale is None:
            weights_drawn = self.sample_weights(n_traj_gen)                # full var
        else:
            weights_drawn = self.sample_controlled_weights(
                n_traj_gen=n_traj_gen,
                variance_scale=variance_scale,                             # scaled σ
            )

        # ------------------------------------------------------------------
        # 2. Build *dense* feature matrix on a fine t-grid
        # ------------------------------------------------------------------
        t_dense = torch.linspace(
            0, 1, dense_time_steps, device=self.device).view(-1, 1)        # (Tdense,1)
        feats_dense = rbf_kernel(t_dense, self.t_s, self.gamma)            # (Tdense,D)
        feats_dense = feats_dense.unsqueeze(0).expand(n_traj_gen, -1, -1)  # (N,Td,D)

        # ------------------------------------------------------------------
        # 3. Condition first weight so x(0) = start_pos
        # ------------------------------------------------------------------
        start_pos_tiled = start_pos.repeat(n_traj_gen, 1)                  # (N,3)
        trajs_no_first  = torch.bmm(feats_dense[:, :, 1:],                 # (N,Td,D-1)
                                    weights_drawn[:, 1:, :])               # (N,D-1,3)

        start_diff      = start_pos_tiled - trajs_no_first[:, 0, :]        # (N,3)
        weight_0        = start_diff / feats_dense[:, 0, 0].unsqueeze(1)   # (N,3)
        weights_drawn[:, 0, :] = weight_0

        # ------------------------------------------------------------------
        # 4. Produce *dense* trajectories
        # ------------------------------------------------------------------
        trajs_dense = torch.bmm(feats_dense, weights_drawn)                # (N,Td,3)

        # ------------------------------------------------------------------
        # 5. Re-sample each trajectory at equal arc-length spacing
        # ------------------------------------------------------------------
        trajs_even = self.resample_by_arclength(
            trajs_dense, n_points=arc_points
        )                                                                  # (N,arc,3)

        return trajs_even

    def condition_start_time(
        self,
        start_pos,
        n_traj_gen: int = 50,
        number_t_steps: int = 200,
        include_variance_in_samples: bool = True,
        variance_scale = None,          # knob used when include_variance_in_samples=False
    ):
        """
        Generate trajectories that are forced to pass through `start_pos` at t = 0.

        Parameters
        ----------
        start_pos : (3,) tensor
            Desired xyz start position.
        n_traj_gen : int
            Number of trajectories to draw.
        number_t_steps : int
            Temporal resolution of each trajectory.
        include_variance_in_samples : bool
            • True  – use `sample_weights`  (original behaviour: full variance)
            • False – use `sample_controlled_weights` with `variance_scale`
        variance_scale : float
            Multiplier applied to the posterior std in `sample_controlled_weights`
            (ignored if include_variance_in_samples=True).
            0   → deterministic mean  
            1   → original variance  
            k   → std × k

        Returns
        -------
        trajs_cond : (n_traj_gen, number_t_steps, 3) tensor
            Trajectories conditioned on the start point.
        """
        # ------------------------------------------------------------------
        # 1. Draw weights according to the requested variance regime
        # ------------------------------------------------------------------
        if include_variance_in_samples or variance_scale is None:
            weights_drawn = self.sample_weights(n_traj_gen)                       # full variance
        else:
            weights_drawn = self.sample_controlled_weights(                       # scaled variance
                n_traj_gen=n_traj_gen,
                variance_scale=variance_scale,
            )

        # ------------------------------------------------------------------
        # 2. Prepare feature matrix
        # ------------------------------------------------------------------
        t_check_test = torch.linspace(0, 1, number_t_steps, device=self.device).view(-1, 1)
        feats_test   = rbf_kernel(t_check_test, self.t_s, self.gamma)             # (T,D)
        feats_test_tiled = feats_test.unsqueeze(0).expand(n_traj_gen, -1, -1)     # (N,T,D)

        # ------------------------------------------------------------------
        # 3. Condition first-row weight so that t=0 equals start_pos
        # ------------------------------------------------------------------
        start_pos_tiled = start_pos.repeat(n_traj_gen, 1)                         # (N,3)
        trajs_no_first  = torch.bmm(feats_test_tiled[:, :, 1:],                   # (N,T,D-1)
                                    weights_drawn[:, 1:, :])                      # (N,D-1,3)

        start_diff      = start_pos_tiled - trajs_no_first[:, 0, :]               # (N,3)
        weight_0        = start_diff / feats_test_tiled[:, 0, 0].unsqueeze(1)     # (N,3)
        weights_drawn[:, 0, :] = weight_0

        # ------------------------------------------------------------------
        # 4. Final conditioned trajectories
        # ------------------------------------------------------------------
        trajs_cond = torch.bmm(feats_test_tiled, weights_drawn)                   # (N,T,3)
        return trajs_cond

    def condition_start_DEPRECATED(self, start_pos, n_traj_gen=50, number_t_steps=200,include_variance_in_samples=True):
        weights_drawn = self.sample_weights(n_traj_gen)
        start_pos_tiled = torch.tile(start_pos, (n_traj_gen, 1))
        t_check_test = torch.linspace(0, 1, number_t_steps).reshape((-1, 1))
        t_check_test = t_check_test.to(self.device)
        feats_test = rbf_kernel(t_check_test, self.t_s, self.gamma)
        feats_test_tiled = torch.tile(feats_test[None], (n_traj_gen, 1, 1))
        trajs_no_cond = torch.bmm(feats_test_tiled[:, :, 1:], weights_drawn[:, 1:, :])
        # condition now, by adding the first row weights accordingly
        start_diff = start_pos_tiled - trajs_no_cond[:, 0, :]
        weight_diff = start_diff / feats_test_tiled[:, 0, 0][:, None]
        weights_drawn[:, 0, :] = weight_diff
        trajs_cond = torch.bmm(feats_test_tiled, weights_drawn)
        return trajs_cond

    def condition_start_end(
        self, start_pos, end_pos, n_traj_gen=50, number_t_steps=200
    ):
        weights_drawn = self.sample_weights(n_traj_gen)
        start_pos_tiled = torch.tile(start_pos, (n_traj_gen, 1))
        end_pos_tiled = torch.tile(end_pos, (n_traj_gen, 1))
        t_check_test = torch.linspace(0, 1, number_t_steps).reshape((-1, 1))
        t_check_test = t_check_test.to(self.device)
        feats_test = rbf_kernel(t_check_test, self.t_s, self.gamma)
        feats_test_tiled = torch.tile(feats_test[None], (n_traj_gen, 1, 1))
        trajs_no_cond = torch.bmm(
            feats_test_tiled[:, :, 1:-1], weights_drawn[:, 1:-1, :]
        )
        # condition now, by adding the first row weights accordingly
        start_diff = start_pos_tiled - trajs_no_cond[:, 0, :]
        end_diff = end_pos_tiled - trajs_no_cond[:, -1, :]
        start_weight_diff = start_diff / feats_test_tiled[:, 0, 0][:, None]
        end_weight_diff = end_diff / feats_test_tiled[:, -1, -1][:, None]
        weights_drawn[:, 0, :] = start_weight_diff
        weights_drawn[:, -1, :] = end_weight_diff
        trajs_cond = torch.bmm(feats_test_tiled, weights_drawn)
        return trajs_cond
