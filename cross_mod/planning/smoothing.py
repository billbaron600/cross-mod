# refactored module (auto-generated)


# ---- imports from original modules ----
from  utils.RLBenchFunctions.template_sensor_views import compute_camera_pose

from matplotlib.colors import ListedColormap     # ← add this

from pathlib import Path

from pyrep.objects.dummy import Dummy

from pyrep.objects.shape import Shape, PrimitiveShape

from pyrep.objects.vision_sensor import VisionSensor

from rendering_helpers import *

from rlbench.action_modes.action_mode import MoveArmThenGripper

from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK,EndEffectorPoseViaPlanning, JointPosition

from rlbench.action_modes.gripper_action_modes import Discrete,GripperJointPosition

from rlbench.gym import RLBenchEnv

from rlbench.tasks import SlideBlockToTarget

from scipy.interpolate import interp1d

from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation as R,RotationSpline

from scipy.spatial.transform import Slerp

from torch import nn

from transforms3d.euler import quat2euler, euler2quat

from types import MethodType

from utils.Classes.policy_trajectory_class import PolicyTrajectory

from utils.RLBenchFunctions.combine_density_views import combine_density_views

from utils.RLBenchFunctions.custom_action_modes import EndEffectorPoseViaPlanning_Record,IVKPlanningBounds,IVKPlanningBounds_NonDiscrete

from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds,MoveArmThenGripperWithBoundsDelta_IVK, EndEffectorPoseViaPlanning_Custom, MoveArmThenGripperWithBoundsDelta

from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import save_sample_trajectories

from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import visualize_trajectories

from utils.RLBenchFunctions.plottingFunctions.plot_traced_points import plot_and_save

import concurrent.futures

import copy

import cv2

import gc

import matplotlib.pyplot as plt

import numpy as np

import os

import os, pickle, math

import os, shutil

import pickle

import random

import re

import torch



def interpolate_points(p0, p1, step_size=0.005):
    """
    Given two points p0, p1 in R^N, interpolate between them
    in small increments of `step_size`.
    Returns a list of points (including p1).
    """
    p0, p1 = np.array(p0), np.array(p1)
    dist = np.linalg.norm(p1 - p0)
    # Number of steps (excluding p0, but including p1).
    n_steps = max(int(np.ceil(dist / step_size)), 1)
    # Interpolate [p0, ..., p1].
    t_vals = np.linspace(0, 1, n_steps + 1)[1:]
    return np.array([p0 + t * (p1 - p0) for t in t_vals])

def build_interpolated_trajectory(points, step_size=0.001):
    """
    Takes an array of shape (T, N) and returns a new array with
    interpolated points for each consecutive pair of the original.
    """
    interp_points = [points[0]]
    for i in range(len(points) - 1):
        seg = interpolate_points(points[i], points[i + 1], step_size)
        interp_points.extend(seg)
    return np.array(interp_points)

def build_smooth_trajectory(points, num_interp=1000, kind='cubic'):
    """
    Given (T, N) points, interpolate a smooth trajectory using
    spline interpolation. `kind` can be 'linear', 'quadratic', or 'cubic'.
    """
    points = np.array(points)
    T, N = points.shape
    x_vals = np.linspace(0, 1, T)

    # Create a spline for each dimension
    splines = [interp1d(x_vals, points[:, d], kind=kind) for d in range(N)]

    # Interpolate with more points
    x_interp = np.linspace(0, 1, num_interp)
    interpolated = np.stack([s(x_interp) for s in splines], axis=1)

    return interpolated

def interpolate_trajectory(trajectory, init_quat, interpolate_positions=False,desired_quat=None, num_interpolate=200, steps_to_interp_quaternion=50):
    """
    Interpolates the trajectory's positions and orientations.

    Parameters:
    - trajectory: Nx3 array of position points.
    - init_quat: Initial orientation as a quaternion [x, y, z, w].
    - desired_quat: Desired orientation as a quaternion [x, y, z, w]. If None, orientation remains constant.
    - num_interpolate: Number of interpolation steps between consecutive points.
    - steps_to_interp_quaternion: Number of steps over which to interpolate from init_quat to desired_quat.

    Returns:
    - interp_traj: Mx7 array of interpolated positions and orientations.
    """
    # Ensure quaternions are numpy arrays
    init_quat = np.array(init_quat)
    if desired_quat is not None:
        desired_quat = np.array(desired_quat)

    # Interpolate positions
    if interpolate_positions is True:
        interp_positions = build_interpolated_trajectory(trajectory,step_size = 1/num_interpolate)
    else:
        interp_positions = trajectory
    
    #interp_positions = build_smooth_trajectory(trajectory)

    # Interpolate orientations
    interp_quats = np.zeros((len(interp_positions), 4))

    # --- decide whether we actually need SLERP --------------------------------
    need_slerp = (
        desired_quat is not None and
        not (np.allclose(init_quat,  desired_quat, atol=1e-8) or     # same sign
            np.allclose(init_quat, -desired_quat, atol=1e-8))       # opposite sign
    )

    # --- make sure desired_quat really differs ---
    if desired_quat is not None and need_slerp==True:
        # 1) bring both quats to the same "hemisphere"
        if np.dot(init_quat, desired_quat) < 0.0:
            desired_quat = -desired_quat                    # flip sign

        # 2) if they’re (still) the same, skip SLERP entirely
        if np.allclose(init_quat, desired_quat, atol=1e-8):
            interp_quats[:] = init_quat
        else:
            key_times = [0, steps_to_interp_quaternion - 1]
            key_rots  = R.from_quat([init_quat, desired_quat])
            slerp     = Slerp(key_times, key_rots)
            for i in range(len(interp_positions)):
                if i < steps_to_interp_quaternion:
                    t            = i / (steps_to_interp_quaternion - 1)
                    interp_quats[i] = slerp([i])[0].as_quat()
                else:
                    interp_quats[i] = desired_quat
    else:
        print("Repeated Quaternion")
        interp_quats[:] = init_quat

    quats = interp_quats             # slice out the Nx4 quaternion block

    #print("init≈desired?", np.allclose(init_quat, desired_quat, atol=1e-8) or np.allclose(init_quat, -desired_quat, atol=1e-8), "‖diff‖=", np.linalg.norm(init_quat - desired_quat))
    # ── check whether every quaternion equals `init_quat` (within tolerance) ──
    #if np.allclose(quats, init_quat, atol=1e-8):
    #    print("✅  Every quaternion is identical to init_quat.")
    #else:
    #    print("⚠️  At least one quaternion differs from init_quat.")

    # Combine positions and orientations
    interp_traj = np.hstack((interp_positions, interp_quats))
    print(interp_traj.shape)

    return interp_traj

def rbf_kernel(t, t_s, gamma=50):
    feat_dists = torch.exp(-gamma * (torch.cdist(t, t_s, p=2)) ** 2)
    return feat_dists

def interpolate_segment(tensor, idx=0, num_points=10):
    """
    Overwrites values between tensor[idx] and tensor[idx + num_points]
    with interpolated values (inclusive), keeping tensor size the same.

    Parameters:
    - tensor: (N, 3) torch tensor
    - idx: start index of interpolation
    - num_points: number of interpolation steps (so the span is idx → idx+num_points)

    Returns:
    - tensor with modified segment
    """
    N = tensor.shape[0]
    assert idx >= 0 and idx + num_points < N, "Invalid interpolation range"

    start = tensor[idx]
    end = tensor[idx + num_points]

    # Interpolation weights from 0 to 1 (inclusive), num_points + 1 total points
    t_vals = torch.linspace(0, 1, num_points + 1, device=tensor.device, dtype=tensor.dtype)

    interpolated = torch.stack([start + t * (end - start) for t in t_vals])

    # Overwrite the segment
    tensor[idx:idx + num_points + 1] = interpolated

    return tensor
