from stable_baselines3 import SAC
import numpy as np
from scipy.spatial.transform import Rotation as R

class SACQuatSafe(SAC):
    """
    SAC with a predict() that auto-normalises the quaternion slice
    action[..., 3:7]  -> unit quaternion.
    Assumes the action vector has >= 7 dims.
    """

    @staticmethod
    def _unit_quat(q):
        # q: (...,4) ndarray
        n = np.linalg.norm(q, axis=-1, keepdims=True)
        # avoid NaN if norm is tiny
        q_unit = np.where(n < 1e-6, np.array([0, 0, 0, 1], dtype=q.dtype), q / n)
        return q_unit

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # get raw action from parent
        action, new_state = super().predict(
            observation, state, episode_start, deterministic)

        #yaw180   = [0., 0., 1., 0.]

        # handle both single action (1D) and batch (2D+)
        if action.shape[-1] >= 7:                     # safety check
            quat = action[..., 3:7]                   # slice last dim
            #action[..., 3:7] = (R.from_quat(self._unit_quat(quat)) * R.from_quat(yaw180)).as_quat()
            action[..., 3:7] = self._unit_quat(quat)

        return action, new_state