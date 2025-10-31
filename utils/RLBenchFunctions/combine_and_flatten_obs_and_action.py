import numpy as np

def angles_to_sincos(theta: np.ndarray) -> np.ndarray:
    """θ (N,) → [sinθ₁, cosθ₁, …, sinθₙ, cosθₙ] (2 N,)"""
    return np.stack([np.sin(theta), np.cos(theta)], axis=-1).ravel()


def flatten_observation_NO_STANDARD(
        obs: dict,
        feat_stats: dict = None,
        *,
        use_sincos: bool = True,   # ← default: sin / cos
        use_low_dim_delta: bool = True,
    ):
    """
    Convert an RLBench observation to a 1-D float32 vector.

    Keys expected in *obs*:
      'joint_velocities', 'joint_positions', 'joint_forces',
      'gripper_open', 'gripper_pose', 'gripper_joint_positions',
      'gripper_touch_forces', 'task_low_dim_state'

    Parameters
    ----------
    use_sincos : bool, default True
        If True, each joint angle θ is replaced by [sin θ, cos θ] (14-dim for Panda).
    feat_stats : dict or None
        Unused placeholder (no standardization in this version).
    use_low_dim_delta : bool, default True
        If True, appends ‖pos[:3]−pos[3:6]‖ as a one-element array.
    """
    # 1) Joint positions (optionally sin/cos embedding)
    jp = obs["joint_positions"].ravel()
    if use_sincos:
        jp = angles_to_sincos(jp)        # leave your implementation of angles_to_sincos
        jp_key = "joint_positions_sincos"
    else:
        jp_key = "joint_positions"

    # 2) Collect all the raw parts
    parts = {
        "joint_velocities":         obs["joint_velocities"].ravel(),
        jp_key:                     jp,
        "joint_forces":             obs["joint_forces"].ravel(),
        "gripper_open":             obs["gripper_open"].ravel(),
        "gripper_pose":             obs["gripper_pose"].ravel(),
        "gripper_joint_positions":  obs["gripper_joint_positions"].ravel(),
        "gripper_touch_forces":     obs["gripper_touch_forces"].ravel(),
        "task_low_dim_state":       obs["task_low_dim_state"].ravel(),
    }

    # 3) Optionally compute low-dim delta
    if use_low_dim_delta:
        ld = obs["task_low_dim_state"].ravel()
        delta = np.linalg.norm(ld[:3] - ld[3:6])
        parts["task_low_dim_state_delta"] = np.array([delta], dtype=np.float32)

    # 4) Concatenate and return
    return np.concatenate(list(parts.values())).astype(np.float32)

def flatten_observation(
        obs: dict,
        feat_stats: dict = None,
        *,
        use_sincos: bool = True,   # ← default: sin / cos
        eps: float = 1e-8,
        use_low_dim_delta: bool = True,
        skip_normalization: list = ["gripper_open","gripper_pose","gripper_joint_positions","task_low_dim_state","task_low_dim_state_delta"],
        #skip_normalization: list = [],
        clip_limit: float = 6.0
    ):
    """
    Convert an RLBench observation to a 1-D float32 vector.

    Keys expected in *obs*:
      'joint_velocities', 'joint_positions', 'gripper_pose',
      'gripper_joint_positions', 'task_low_dim_state'

    Parameters
    ----------
    use_sincos : bool, default False
        If True, each joint angle θ is replaced by [sin θ, cos θ] (14-dim for Panda).
        If False, use the raw 7-dim radians.
    feat_stats : dict or None
        Per-field mean / std dict for optional z-score normalisation.
        Should come from the batch `compute_stats` function
        (keys: 'mean', 'std').
    """
    # ------------------------------------------------------------------
    jp_vec = obs["joint_positions"].ravel()
    if use_sincos:
        jp_vec = angles_to_sincos(jp_vec)
        jp_key = "joint_positions_sincos"
    else:
        jp_key = "joint_positions"

    
    # before building `parts`, do:
    gjp_raw   = obs["gripper_joint_positions"].ravel()
    # assume raw range is [0, 0.04] → scale into [0, 1]:
    gjp_norm  = np.clip(gjp_raw / 0.04, 0.0, 1.0)
    #print(gjp_norm)

    # --- scale gripper_pose so that:
    #     x in [0.1, 0.8] → [0, 1]
    #     y in [–0.5, 0.5] → [0, 1]
    #     z in [0.7, 1.6] → [0, 1]
    gp_raw = obs["gripper_pose"].ravel()           # shape (7,)
    gp_scaled = gp_raw.copy()
    gp_scaled[0] = (gp_raw[0] - 0.1) / (0.8 - 0.1)   # x
    gp_scaled[1] = (gp_raw[1] - (-0.5)) / (0.5 - (-0.5))# y
    gp_scaled[2] = (gp_raw[2] - 0.7) / (1.6 - 0.7)   # z
    # scale quaternion components (indices 3–6) from [-1,1] → [0,1]
    gp_scaled[3:] = (gp_raw[3:] + 1.0) / 2.0

    # gp_scaled[3:] remains the quaternion unchanged

    # --- scale task_low_dim_state so that both the “object” and “target” xyz
    #     segments use the same bounds:
    #     x in [0.1, 0.8] → [0, 1]
    #     y in [–0.5, 0.5] → [0, 1]
    #     z in [0.7, 1.6] → [0, 1]
    tlds_raw = obs["task_low_dim_state"].ravel()    # shape (N,), first 6 entries are xyz’s
    tlds_scaled = tlds_raw.copy()
    # object position (indices 0,1,2)
    tlds_scaled[0] = (tlds_raw[0] - 0.1) / 0.7
    tlds_scaled[1] = (tlds_raw[1] - (-0.5)) / 1.0
    tlds_scaled[2] = (tlds_raw[2] - 0.7) / 0.9
    # target position (indices 3,4,5)
    tlds_scaled[3] = (tlds_raw[3] - 0.1) / 0.7
    tlds_scaled[4] = (tlds_raw[4] - (-0.5)) / 1.0
    tlds_scaled[5] = (tlds_raw[5] - 0.7) / 0.9
    # tlds_scaled[6:] (if any) remain unchanged

    parts = {
        #"joint_velocities":         obs["joint_velocities"].ravel(),
        #jp_key:                     jp_vec,
        #"joint_forces":             obs["joint_forces"].ravel(),
        #"gripper_open":             obs["gripper_open"].ravel(),
        "gripper_pose":             gp_scaled,
        "gripper_joint_positions":  gjp_norm,
        #"gripper_touch_forces":     obs["gripper_touch_forces"].ravel(),
        "task_low_dim_state":       tlds_scaled,
    }

    
    if use_low_dim_delta is True:
        low_dim = obs["task_low_dim_state"].ravel()      # [obj_xyz, target_xyz, ...]
        gripper_xyz = obs["gripper_pose"].ravel()[:3]    # first 3 entries → gripper position

        # --- distances --------------------------------------------------
        #d_obj_target_raw = np.linalg.norm(low_dim[:3] - low_dim[3:6])
        # min–max scale to [0, 1] with min=0, max=0.2
        #d_obj_target = np.clip(d_obj_target_raw / 0.2, 0.0, 1.0)
        # compute raw distance
        d_obj_target_raw = np.linalg.norm(low_dim[:3] - low_dim[3:6])

        # min–max scale over [0, 0.2], but allow values >1 if raw > 0.2
        d_obj_target = d_obj_target_raw / 0.2

        # 2) gripper-to-object distance
        d_gripper_obj  = np.linalg.norm(gripper_xyz - low_dim[:3])

        # 3) gripper-to-target distance
        d_gripper_tgt  = np.linalg.norm(gripper_xyz - low_dim[3:6])

        # collect the three distances in order
        parts["task_low_dim_state_delta"] = np.array(
            [d_obj_target, d_gripper_obj, d_gripper_tgt], dtype=np.float32
        )

        #
    # -------- optional z-score scaling --------------------------------
    if feat_stats is not None:
        for k, v in parts.items():
            if k in feat_stats:
                if k not in skip_normalization:
                    mu  = feat_stats[k]["mean"]
                    std = feat_stats[k]["std"]
                    if v.shape == mu.shape:               # broadcast-safe
                        parts[k] = (v - mu) / (std + eps)

    # -------- concatenate,clip & return ------------------------------------
    vec = np.concatenate(list(parts.values())).astype(np.float32)
    #vec = np.clip(vec, -clip_limit, clip_limit)
    #print(vec)

    return vec



def combine_observation_and_action(obs: dict, action,include_action=False,feat_stats = None) -> np.ndarray:
    """
    Combines a single observation and action into a flattened NumPy array.

    Args:
        obs (dict): Observation dictionary with predefined keys.
        action: Action associated with the observation.

    Returns:
        np.ndarray: Flattened combined array of observation and action.
    """

    #if feat_stats is None:
        #print("feat_stats is set to None. Input features to reward will not be normalized (PROBLEM)")
        
    
    flattened_obs = flatten_observation(obs,feat_stats = feat_stats)
    if include_action == True:
        flattened_action = np.asarray(action).ravel().astype(np.float32)
        combined = np.concatenate([flattened_obs, flattened_action])
    else:
        combined = np.concatenate([flattened_obs])

    return combined

#Used to convert rlbench demos to dictionary observations
def obs_to_dict(obs):
    """Flatten each numerical field and return a pure-Python dict."""
    return {
        "joint_positions":        obs.joint_positions.ravel(),
        "joint_velocities":       obs.joint_velocities.ravel(),
        "joint_forces":           obs.joint_forces.ravel(),
        "gripper_open":           np.asarray([obs.gripper_open]).ravel(),   # bool → 0/1
        "gripper_pose":           obs.gripper_pose.ravel(),
        "gripper_joint_positions":obs.gripper_joint_positions.ravel(),
        "gripper_touch_forces":   obs.gripper_touch_forces.ravel(),
        "task_low_dim_state":     obs.task_low_dim_state.ravel(),
    }

def demos_to_nested_dicts(demos):
    """
    demos: List[List[Observation]] from RLBench.
    returns: List[List[dict]] – same nesting, but every Observation is now a dict.
    """
    nested = []
    for demo in demos:
        # RLBench sometimes wraps a demo in a DemoEpisode object (with ._observations)
        observations = getattr(demo, "_observations", demo)
        nested.append([obs_to_dict(o) for o in observations])
    return nested