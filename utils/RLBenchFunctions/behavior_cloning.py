from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import copy
import math

class ReplayBufferDataset_DEP(Dataset):
    """
    Wrap (a subset of) an SB3 replay buffer as a PyTorch Dataset.

    Each sample yields:
        obs_dict   – dict[str, Tensor]   with shape (..., M)
        act_tensor – Tensor              with shape (A,)
    Any leading singleton dimension (1, M) is squeezed away.
    """

    def __init__(self, buf, indices, device="cpu"):
        self.buf = buf
        self.indices = np.asarray(indices, dtype=np.int64)
        self.device = device

    def __len__(self):
        return len(self.indices)

    # --------------------------------------------------------------------- #
    # Helpers                                                               #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _to_tensor(x, device):
        """Convert ndarray / Tensor to torch.Tensor on `device`."""
        if torch.is_tensor(x):
            return x.to(device)
        return torch.as_tensor(x, device=device)

    @staticmethod
    def _squeeze_leading_unit_dim(t):
        """
        If `t` has shape (1, M, …) or (1, M) squeeze the first axis,
        else return unchanged.
        """
        return t.squeeze(0) if (t.ndim >= 2 and t.shape[0] == 1) else t

    # --------------------------------------------------------------------- #
    # Dataset API                                                           #
    # --------------------------------------------------------------------- #
    def __getitem__(self, idx):
        """
        1) pull the idx-th demo transition
        2) run it through _get_samples -> ReplayBufferSamples
        3) compute policy features (already on self.device)
        """
        # A. locate the *buffer* index that corresponds to the idx-th demo
        demo_indices = np.flatnonzero(self.buf.is_demo)     # 1-D np.array
        i_buf        = int(demo_indices[self.indices[idx]]) # absolute index

        # B. grab that single transition via the buffer’s internal helper
        samples = self.buf._get_samples(np.array([i_buf]), env=None)  # returns ReplayBufferSamples
        obs     = samples.observations                          # dict or tensor
        act     = samples.actions                               # tensor (B=1, A)

        # C. convert obs to tensor(s) on the right device
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        if isinstance(obs_tensor, dict):
            obs_tensor = {k: v.to(self.device) for k, v in obs_tensor.items()}
        else:
            obs_tensor = obs_tensor.to(self.device)

        # D. extract policy features
        features = self.model.policy.extract_features(
            obs_tensor, self.model.policy.feature_extractor
        )

        # E. flatten action to shape (A,)
        act = act.squeeze(0).to(self.device)

        return features, act

def add_outer_edge_noise_rows(
    buf,
    percent        : float = 0.05,           # 5 % new rows
    method         : str   = "gaussian_scaled",
    k              : float = 2.0,
    margin         : float = 1.0,
    action_eps_pct : float = 0.00001,           # N % of action range
    rng=np.random,
):
    """
    Adds ~`percent` × pos synthetic observations (see `method`), *and*
    writes a small action into `buf.actions` for each synthetic row.

    The action for dimension j is sampled i.i.d. from
        U(−eps_j, +eps_j)  where  eps_j = action_eps_pct * (max_j − min_j)

    Parameters
    ----------
    model            SB3 model with DictReplayBuffer
    percent          Fraction of existing rows to add
    method/k/margin  Same semantics as previous helper
    action_eps_pct   Size of the perturbation relative to each dim’s range
    """
    old_pos = pos = buf.pos
    Nnew = max(1, int(round(pos * percent)))

    # --- stats of existing ACTIONS (0 … pos-1) ---
    act_arr = buf.actions            # shape  (T, 1, A)  or  (T, A)
    act_slice = act_arr[:pos]        # (pos, …)
    act_max   = act_slice.max(axis=0, keepdims=True)
    act_min   = act_slice.min(axis=0, keepdims=True)
    act_range = act_max - act_min
    eps       = action_eps_pct * act_range

    # helper to grow an array if we run out of room
    def _ensure_capacity(arr, extra):
        if pos + Nnew > arr.shape[0]:
            grow_by = max(extra, arr.shape[0] // 2)
            bigger  = np.empty((arr.shape[0] + grow_by, *arr.shape[1:]),
                               dtype=arr.dtype)
            bigger[:pos] = arr[:pos]
            return bigger
        return arr

    # --- OBSERVATIONS ---
    for key, arr in buf.observations.items():
        data  = arr[:pos]
        mu    = data.mean(axis=0, keepdims=True)
        sigma = np.maximum(data.std(axis=0, keepdims=True), 1e-6)
        mins  = data.min(axis=0, keepdims=True)
        maxs  = data.max(axis=0, keepdims=True)

        if method == "gaussian_shell":
            eps_obs = rng.randn(Nnew, *arr.shape[1:])
            eps_obs /= np.linalg.norm(eps_obs.reshape(Nnew, -1), axis=1, keepdims=True)
            new_rows = mu + eps_obs * (k * sigma)

        elif method == "gaussian_scaled":
            new_rows = mu + rng.randn(Nnew, *arr.shape[1:]) * (k * sigma)

        elif method == "uniform_box":
            low  = mins - margin * sigma
            high = maxs + margin * sigma
            new_rows = rng.uniform(low, high, size=(Nnew, *arr.shape[1:])).astype(arr.dtype)

        else:
            raise ValueError(f"Unknown method {method}")

        # ensure capacity + write
        buf.observations[key] = _ensure_capacity(arr, Nnew)
        buf.observations[key][pos : pos + Nnew] = new_rows

    # --- ACTIONS: tiny random perturbations ---
    #perturb = rng.uniform(-eps, eps, size=(Nnew, *act_arr.shape[1:])).astype(act_arr.dtype)
    #buf.actions = _ensure_capacity(act_arr, Nnew)
    #buf.actions[pos : pos + Nnew] = perturb

    # --- ACTIONS: zeros for synthetic rows ----------------------------
    
    zero_rows = np.zeros((Nnew, *act_arr.shape[1:]), dtype=act_arr.dtype)
    buf.actions = _ensure_capacity(act_arr, Nnew)
    buf.actions[pos : pos + Nnew] = zero_rows
    
    # bump cursor
    buf.pos += Nnew
    print(f"[{method}] added {Nnew} obs rows and small actions "
          f"(±{action_eps_pct*100:.1f}% range) → len {buf.pos}")





def _sample_outside_demo(joint_obs, Nnew, margin=0.30,rng=np.random):
    """
    joint_obs : (pos, …, 7)  (works for (pos,7) or (pos,1,7))
    returns   : (Nnew, …, 7) synthetic rows, each OOD in ≥1 joint
    """

    _J_LOW  = np.array([-166, -101, -166, -176, -166,  -1, -166], dtype=np.float32) * np.pi / 180
    _J_HIGH = np.array([ 166,  101,  166,   -4,  166, 215,  166], dtype=np.float32) * np.pi / 180
    # ---- collapse all extra dims except the 7-DOF joint axis ----------
    flat = joint_obs.reshape(-1, 7)          # (pos*…, 7)

    pos_min = flat.min(axis=0)               # (7,)
    pos_max = flat.max(axis=0)               # (7,)
    delta   = margin * (pos_max - pos_min)   # (7,)

    rows = np.empty((Nnew, 7), dtype=joint_obs.dtype)

    for n in range(Nnew):
        j = rng.randint(7)                   # joint that will be OOD

        # gap intervals for joint j
        gaps = []
        low_gap_hi = pos_min[j] - delta[j]
        up_gap_lo  = pos_max[j] + delta[j]

        if low_gap_hi > _J_LOW[j]:
            gaps.append((_J_LOW[j], low_gap_hi))
        if _J_HIGH[j] > up_gap_lo:
            gaps.append((up_gap_lo, _J_HIGH[j]))

        if gaps:                             # pick a gap, sample inside it
            lo, hi = gaps[rng.randint(len(gaps))]
            rows[n, j] = rng.uniform(lo, hi)
        else:                                # fallback: anywhere in phys range
            rows[n, j] = rng.uniform(_J_LOW[j], _J_HIGH[j])

        # remaining joints: inside observed range
        for k in range(7):
            if k != j:
                rows[n, k] = rng.uniform(pos_min[k], pos_max[k])

    # ---- re-expand to match original trailing shape (…,7) -------------
    extra_shape = joint_obs.shape[1:-1]      # ()  or  (1,)
    if extra_shape:                          # e.g. (1,) for (T,1,7)
        rows = rows.reshape((Nnew,)+extra_shape+(7,))  # (Nnew,1,7)
    return rows


def add_noise_rows_to_obs(buf, model, percent: float = 0.05, action_value = 0.05,sigma_factor=2.0, use_joint_ranges=False,rng=np.random):
    
    
    pos = buf.pos
    Nnew = max(1, int(round(pos * percent)))
    original_pos = copy.deepcopy(pos)

    for key, arr in buf.observations.items():

        if key == "joint_positions" and use_joint_ranges==True:
            # --- uniform sample inside physical joint limits ------------
            noise_rows = _sample_outside_demo(arr[:pos], Nnew, margin=0.10, rng=rng)

        else:
            # --- Gaussian noise around demo mean/std --------------------
            mu    = arr[:pos].mean(axis=0, keepdims=True)
            sigma = np.maximum(arr[:pos].std(axis=0, keepdims=True), 1e-6)
            sigma = sigma * sigma_factor
            noise_rows = mu + sigma * rng.randn(Nnew, *arr.shape[1:]).astype(arr.dtype)

        # append rows, growing array if necessary -----------------------
        if pos + Nnew > arr.shape[0]:
            grow_by = max(Nnew, arr.shape[0] // 2)
            new_arr = np.empty((arr.shape[0] + grow_by, *arr.shape[1:]),
                               dtype=arr.dtype)
            new_arr[:pos] = arr[:pos]
            buf.observations[key] = new_arr
            arr = new_arr

        arr[pos : pos + Nnew] = noise_rows

    # -- actions: keep zeros for newly added rows -----------------------

    # -- bump cursor & mark as demos ------------------------------------
    buf.pos += Nnew
    print(f"Injected {Nnew} synthetic rows (≈ {percent*100:.1f} %) → new buffer length = {buf.pos}")

    if hasattr(buf, "is_demo"):
        buf.is_demo[:buf.pos] = True

    # --- pre-compute stats of the *scaled* quaternion channels already in the buffer ---
    q_mean = buf.actions[:buf.pos, :, 3:7].mean(axis=(0, 1))          # shape (4,)
    q_std  = buf.actions[:buf.pos, :, 3:7].std(axis=(0, 1)) + 1e-8    # avoid zero-std

    for t in range(original_pos, buf.pos):
        pose   = buf.observations["gripper_pose"][t, :, :]            # (N, 7)
        grip   = np.full((pose.shape[0], 1), 0.04, dtype=pose.dtype)  # (N, 1)
        act    = np.concatenate((pose, grip), axis=-1)                # (N, 8)

        scaled = model.policy.scale_action(act)                       # scale pos + grip
        scaled[:, 3:7] = q_mean + q_std * np.random.randn(*scaled[:, 3:7].shape)  # sample quat (already scaled)

        buf.actions[t, :, :] = scaled

    #action_limit = action_value
    #buf.actions[original_pos:buf.pos, :, :6] = np.random.uniform(-action_limit, action_limit, (buf.pos - original_pos, 1, 6))
    #buf.actions[original_pos:buf.pos, :, :6] = buf.actions[:original_pos, :, :6].mean(axis=0, keepdims=True)/reduction_factor


class ReplayBufferDataset(Dataset):
    """
    Wrap (a subset of) an SB3 replay buffer as a PyTorch Dataset that
    produces observations in the *exact* format expected by the policy.

    Each sample yields:
        obs_tensor  dict[str, Tensor]  *or* Tensor (whatever the policy takes)
        act_tensor  Tensor  (A,)
    """

    def __init__(self, buf, indices, model, device="cpu"):
        self.buf     = buf
        self.indices = np.asarray(indices, dtype=np.int64)
        self.model   = model                  # <-- needed for obs_to_tensor
        self.device  = device

    def __len__(self):
        return len(self.indices)

    # ------------------------------------------------------------------ #
    # Dataset API                                                        #
    # ------------------------------------------------------------------ #
    """
    def __getitem__(self, idx):
        i = int(self.indices[idx])

        # ---------- raw observation from replay buffer -----------------
        if isinstance(self.buf.observations, dict):
            obs_raw = {k: self.buf.observations[k][i] for k in self.buf.observations}
        else:
            obs_raw = self.buf.observations[i]

        # ---------- SB3 helper → torch tensor(s) on requested device ---
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs_raw)
        if isinstance(obs_tensor, dict):
            obs_tensor = {k: v.to(self.device) for k, v in obs_tensor.items()}
        else:
            obs_tensor = obs_tensor.to(self.device)

        # ---------- actions -------------------------------------------
        act = torch.as_tensor(self.buf.actions[i], device=self.device).float()

        return obs_tensor, act
    """
    def __getitem__(self, idx):
        i_buf = int(self.indices[idx])
        samples = self.buf._get_samples(np.array([i_buf]), env=None)
        obs_t   = samples.observations          # dict[str, Tensor]  (B=1)
        #act_t   = samples.actions.squeeze(0)    # (A,)
        act_t   = samples.actions

        # move to the requested device
        if isinstance(obs_t, dict):
            #obs_t = {k: v.to(self.device) for k, v in obs_t.items()}
            #print("DICTIONARY")
            pass
        else:
            obs_t = obs_t.to(self.device)
        act_t = act_t.to(self.device)

        return obs_t, act_t

def make_demo_loader(
    model,
    batch_size=256,
    device="cpu",
    rng_seed=42,
    val_fraction=0.01,
    shuffle=True,
    post_done_window = 5,      # ⬅ how many steps after each done to up-weight
    dup_factor       = 15,      # ⬅ how many extra copies of each such step
    percent_noise    = None
):
    buf = model.replay_buffer

    if percent_noise is not None:
        add_noise_rows_to_obs(buf, model,percent = percent_noise)
        #add_outer_edge_noise_rows(buf,percent=percent_noise/2)

    pos = buf.pos

    # pick demo indices (using the `is_demo` flag if it exists)
    if hasattr(buf, "is_demo"):
        demo_idx = np.where(buf.is_demo[:pos])[0]
    else:
        demo_idx = np.arange(pos)          # fallback: all transitions
    if len(demo_idx) == 0:
        raise ValueError("No demonstrations found in replay buffer.")

     # ---------- collect “after-done” indices --------------------------
    # ---------- collect “before-and-at-done” indices -------------------
    dones     = buf.dones[:pos]              # 1-D bool array
    done_pos  = np.where(dones)[0]           # where done == True

    dup_idx = []
    for p in done_pos:
        # 0 → the done step itself, 1..k → k steps before the done
        dup_idx.extend(p - np.arange(0, post_done_window + 1))

    # keep only valid buffer positions
    dup_idx = [i for i in dup_idx if 0 <= i < pos]

    # duplicate each selected index dup_factor times
    extra = np.repeat(dup_idx, dup_factor)


    #dataset = ReplayBufferDataset(buf, demo_idx, model,device=device)
    # ----------- build DataLoaders -------------------------------------
    # ----------- train / val split -------------------------------------
    rng = np.random.RandomState(rng_seed)
    perm = rng.permutation(len(demo_idx))
    split_at = int(len(demo_idx) * (1.0 - val_fraction))
    train_idx, val_idx = perm[:split_at], perm[split_at:]

    # add duplicated “after-done” samples to the training set
    if dup_factor>0:
        train_idx = np.concatenate([train_idx, extra])

    train_set = ReplayBufferDataset(buf, train_idx, model, device=device)
    val_set   = ReplayBufferDataset(buf, val_idx,   model, device=device)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              drop_last=False,
                              generator=torch.Generator().manual_seed(torch.seed()))
    val_loader   = DataLoader(val_set,
                              batch_size=batch_size,
                              shuffle=False,         # keep validation deterministic
                              drop_last=False)

    return train_loader, val_loader
# --------------------------------------------------------------------------
# 2. NEW: make_demo_loaders → returns (train_loader, val_loader)
# --------------------------------------------------------------------------
def make_demo_loader_DEP(
    model,
    *,
    batch_size: int = 256,
    device: str = "cpu",
    shuffle: bool = True,
    val_fraction: float = 0.20,                 # 20 % validation split
    rng_seed: int = 42,                         # reproducible shuffle
    # ---------- weighting schedule --------------------------------------
    weight_schedule: tuple = ((1, 1),(1,1)),
    default_weight: int = 1,
    # ---------- trajectory filter ---------------------------------------
    gripper_pose_key: str = "gripper_pose",
    skip_if_2nd_elem_below = None,
):
    
    #Return *(train_loader, val_loader)* where validation gets `val_fraction`
    #of the weighted data indices (default 20 %).

    #All other semantics identical to the original make_demo_loader.
    
    buf, pos = model.replay_buffer, model.replay_buffer.pos

    # ----------- locate demonstration indices --------------------------
    if hasattr(buf, "is_demo"):
        is_demo_mask = buf.is_demo[:pos]
    else:
        is_demo_mask = np.ones(pos, dtype=bool)
    if not is_demo_mask.any():
        raise ValueError("No demonstrations found in replay buffer.")

    # ----------- build weighted index list -----------------------------
    weighted_idx: list = []
    traj_start = 0
    for i in range(pos):
        if buf.dones[i] or i == pos - 1:                       # trajectory ends
            traj_end   = i
            traj_slice = slice(traj_start, traj_end + 1)

            # optional gripper-pose filter
            if (skip_if_2nd_elem_below is not None
                and gripper_pose_key in buf.observations):
                first_gp = buf.observations[gripper_pose_key][traj_start].squeeze()
                if first_gp[2] < skip_if_2nd_elem_below:
                    traj_start = i + 1
                    continue

            # gather demo indices in this traj & apply weighting
            traj_demos = np.where(is_demo_mask[traj_slice])[0] + traj_start
            for local_step, idx in enumerate(traj_demos):
                cumulative, repeat = 0, default_weight
                for window, factor in weight_schedule:
                    if local_step < cumulative + window:
                        repeat = factor
                        break
                    cumulative += window
                weighted_idx.extend([idx] * repeat)

            traj_start = i + 1

    if not weighted_idx:
        raise ValueError("Weighted index list is empty.")

    weighted_idx = np.asarray(weighted_idx, dtype=np.int64)

    # ----------- train / val split -------------------------------------
    rng = np.random.RandomState(rng_seed)
    perm = rng.permutation(len(weighted_idx))
    split_at = int(len(weighted_idx) * (1.0 - val_fraction))
    train_idx, val_idx = perm[:split_at], perm[split_at:]

    # ----------- build DataLoaders -------------------------------------
    train_set = ReplayBufferDataset(buf, weighted_idx[train_idx], model, device=device)
    val_set   = ReplayBufferDataset(buf, weighted_idx[val_idx],   model, device=device)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              drop_last=True)
    val_loader   = DataLoader(val_set,
                              batch_size=batch_size,
                              shuffle=False,         # keep validation deterministic
                              drop_last=False)

    return train_loader, val_loader



def make_demo_loader_DEPRECATED(
    model,
    batch_size: int = 256,
    device: str = "cpu",
    shuffle: bool = True,
    *,
    # ---------- weighting schedule --------------------------------------
    weight_schedule=((1, 1), (1, 1)),
    default_weight: int = 1,
    # ---------- trajectory filter ---------------------------------------
    gripper_pose_key: str = "gripper_pose",
    skip_if_2nd_elem_below: float =0.5,   # ← set e.g. to 0.15
):
    """
    Return a DataLoader that oversamples early-timestep demos and
    *optionally* drops trajectories whose first gripper-pose[1] is
    below `skip_if_2nd_elem_below`.

    Parameters
    ----------
    skip_if_2nd_elem_below : float or None
        • None  → keep every trajectory (default).
        • float → discard any trajectory whose very first transition has
                   observations[gripper_pose_key][1] < this value.
    All other arguments are as before.
    """
    buf, pos = model.replay_buffer, model.replay_buffer.pos

    # ----------- pick demonstration indices -----------------------------
    if hasattr(buf, "is_demo"):
        is_demo_mask = buf.is_demo[:pos]
    else:
        is_demo_mask = np.ones(pos, dtype=bool)
    if not is_demo_mask.any():
        raise ValueError("No demonstrations found in replay buffer.")

    # ----------- build weighted index list ------------------------------
    weighted_idx = []
    traj_start = 0
    for i in range(pos):
        if buf.dones[i] or i == pos - 1:                   # trajectory ends
            traj_end = i
            traj_slice = slice(traj_start, traj_end + 1)

            # --------- optional gripper-pose filter ---------------------
            if skip_if_2nd_elem_below is not None and gripper_pose_key in buf.observations:
                first_gp = buf.observations[gripper_pose_key][traj_start]
                first_gp = first_gp.squeeze()              # handle (1,7) or (7,)
                if first_gp[2] < skip_if_2nd_elem_below:
                    traj_start = i + 1                     # skip this trajectory
                    continue

            # --------- gather demo indices & apply weighting ------------
            traj_demos = np.where(is_demo_mask[traj_slice])[0] + traj_start
            for local_step, idx in enumerate(traj_demos):
                cumulative = 0
                repeat = default_weight
                for window, factor in weight_schedule:
                    if local_step < cumulative + window:
                        repeat = factor
                        break
                    cumulative += window
                weighted_idx.extend([idx] * repeat)

            traj_start = i + 1                             # next trajectory

    # ----------- DataLoader ---------------------------------------------
    dataset = ReplayBufferDataset(buf, weighted_idx, model,device=device)
    loader  = DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=True)
    return loader
