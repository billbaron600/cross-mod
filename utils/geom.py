# refactored module (auto-generated)


# ---- imports from original modules ----
from  utils.RLBenchFunctions.template_sensor_views import compute_camera_pose

from matplotlib.colors import ListedColormap     # ← add this

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

from transforms3d.euler import quat2euler, euler2quat

from types import MethodType

from utils.Classes.policy_trajectory_class import PolicyTrajectory

from utils.RLBenchFunctions.custom_action_modes import EndEffectorPoseViaPlanning_Record,IVKPlanningBounds,IVKPlanningBounds_NonDiscrete

from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds,MoveArmThenGripperWithBoundsDelta_IVK, EndEffectorPoseViaPlanning_Custom, MoveArmThenGripperWithBoundsDelta

from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import save_sample_trajectories

from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import visualize_trajectories

import concurrent.futures

import copy

import cv2

import gc

import matplotlib.pyplot as plt

import numpy as np

import os

import os, pickle, math

import pickle

import random

import re

import torch



def quaternion_distance(q1, q2):
    """
    Computes a simple distance between two quaternions (cosine distance on the unit sphere).
    """
    q1 = np.array(q1)
    q2 = np.array(q2)
    return 1 - abs(np.dot(q1, q2))

def quaternion_slerp(q1, q2, t):
    """
    Perform spherical linear interpolation (slerp) between two quaternions q1 and q2.
    t ranges from 0 to 1, where:
    t = 0 returns q1
    t = 1 returns q2
    """
    key_times = [0, 1]
    key_rots = R.from_quat([q1, q2])
    slerp = Slerp(key_times, key_rots)
    interpolated_rotation = slerp([t])[0]
    interpolated_quat = interpolated_rotation.as_quat()
    return interpolated_quat / np.linalg.norm(interpolated_quat)

def euler_to_quaternions_use(eulers: np.ndarray,
                         seq: str = 'xyz',
                         degrees: bool = True) -> np.ndarray:
    """
    Convert an (N×3) array of Euler angles (roll, pitch, yaw) to an (N×4) array of unit quaternions (w, x, y, z)
    using SciPy's Rotation, with an explicit normalization step.

    Parameters
    ----------
    eulers : np.ndarray, shape (N, 3)
        Each row is [roll, pitch, yaw].
    seq : str, optional
        Axis sequence for the Euler angles, by default 'xyz'.
    degrees : bool, optional
        If True, interprets input angles as degrees (default is radians).

    Returns
    -------
    np.ndarray, shape (N, 4)
        Unit quaternions in (w, x, y, z) order.
    """
    # Build rotation and get quaternions as (x, y, z, w)
    rot = R.from_euler(seq, eulers, degrees=degrees)
    xyzw = rot.as_quat()

    # Reorder to (w, x, y, z)
    wxyz = xyzw[:, [0, 1, 2, 3]]

    # Explicitly normalize to unit length
    norms = np.linalg.norm(wxyz, axis=1, keepdims=True)
    return wxyz / norms
