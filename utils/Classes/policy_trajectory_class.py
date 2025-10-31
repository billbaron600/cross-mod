import torch
from utils.RLBenchFunctions.combine_and_flatten_obs_and_action import combine_observation_and_action
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

class PolicyTrajectory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.next_observations = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.dones = []
        self.info = []
        self.environment_states = []
        self.ee_positions = []
        self.env_seed = None
        self.trajectory_tensor = None
        self.video_path = None
        # segment-sampling bookkeeping (used only by sample_pairs_BALD)
        self.is_segment = False                  # True ⇒ this is a trimmed view
        self.segment_meta = None       # e.g. {'start': 50, 'length': 75}

    def append(self, obs, action, next_obs, reward, terminated, truncated, done, info, env_state,ee_position=None):
        self.observations.append(obs)
        self.actions.append(action)
        self.next_observations.append(next_obs)
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.dones.append(done)
        self.info.append(info)
        self.environment_states.append(env_state)
        if ee_position is not None:
            self.ee_positions.append(ee_position)

    def generate_tensor_from_trajectory(self,feat_stats=None,source = "observations",):
        """
        Generate a PyTorch tensor where each row is [flattened observation, action]
        """
        obs_iter = getattr(self, source)

        trajectory_tensor = [
            combine_observation_and_action(obs, action,feat_stats=feat_stats)
            for obs, action in zip(obs_iter, self.actions)
        ]

        trajectory_tensor = torch.tensor(np.vstack(trajectory_tensor), dtype=torch.float32)
        self.trajectory_tensor = trajectory_tensor

        return trajectory_tensor
    
    def limit_to_segment(self, start_idx=0, end_idx=None):
        """
        Trim the trajectory to a specific segment [start_idx:end_idx].
        If end_idx is None, it will be set to the end of the trajectory.
        """
        if end_idx is None:
            end_idx = len(self.observations)

        # Slice all relevant lists
        self.observations = self.observations[start_idx:end_idx]
        self.actions = self.actions[start_idx:end_idx]
        self.next_observations = self.next_observations[start_idx:end_idx]
        self.rewards = self.rewards[start_idx:end_idx]
        self.terminated = self.terminated[start_idx:end_idx]
        self.truncated = self.truncated[start_idx:end_idx]
        self.dones = self.dones[start_idx:end_idx]
        self.info = self.info[start_idx:end_idx]
        self.environment_states = self.environment_states[start_idx:end_idx]

        if self.ee_positions:
            self.ee_positions = self.ee_positions[start_idx:end_idx]

        # Invalidate cached trajectory tensor
        self.trajectory_tensor = None

    def downsample_to_length(self, target_length=200):
        """
        Downsamples the trajectory to the specified target length by evenly selecting indices.
        This operation modifies all trajectory fields accordingly.
        """
        original_length = len(self.observations)
        if target_length >= original_length or target_length <= 0:
            print(f"[Warning] Requested target length {target_length} is invalid or exceeds original length {original_length}. No downsampling applied.")
            return

        # Compute evenly spaced indices to keep
        indices = np.linspace(0, original_length - 1, num=target_length, dtype=int)

        # Apply to all relevant lists
        self.observations = [self.observations[i] for i in indices]
        self.actions = [self.actions[i] for i in indices]
        self.next_observations = [self.next_observations[i] for i in indices]
        self.rewards = [self.rewards[i] for i in indices]
        self.terminated = [self.terminated[i] for i in indices]
        self.truncated = [self.truncated[i] for i in indices]
        self.dones = [self.dones[i] for i in indices]
        self.info = [self.info[i] for i in indices]
        self.environment_states = [self.environment_states[i] for i in indices]

        if self.ee_positions:
            self.ee_positions = [self.ee_positions[i] for i in indices]

        # Invalidate cached trajectory tensor
        self.trajectory_tensor = None


def plot_actions_vs_index(trajectories, joint_labels=None):
    """
    Plot each joint's action value against the timestep index for every trajectory.

    Parameters
    ----------
    trajectories : list
        List of trajectory objects.  Each trajectory must expose
        `trajectory.actions`, a sequence of N × D numpy vectors.
    joint_labels : list[str] | None
        Optional custom labels for joints (default: "joint 0" … "joint D-1").
    """
    if not trajectories:
        raise ValueError("`trajectories` is empty")

    # Determine joint count from the first action vector
    first_vec = np.asarray(trajectories[0].actions[0])
    n_joints   = first_vec.size
    if joint_labels is None:
        joint_labels = [f"joint {j}" for j in range(n_joints)]

    for t_idx, traj in enumerate(trajectories):
        acts = np.asarray(traj.actions)          # shape: (T, n_joints)
        timesteps = np.arange(acts.shape[0])

        plt.figure(figsize=(10, 4))
        for j in range(n_joints):
            plt.plot(timesteps, acts[:, j], label=joint_labels[j])

        plt.title(f"Trajectory {t_idx} – actions vs. timestep")
        plt.xlabel("Step index")
        plt.ylabel("Action value")
        plt.legend(ncol=4, fontsize=8, frameon=False)
        plt.tight_layout()

    plt.show()