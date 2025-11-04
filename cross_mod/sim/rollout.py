# refactored module (auto-generated)


def _prepare_env_and_scene(task, action_mode_builder):
    """Create RLBench env and expose scene and robot handles."""
    env = action_mode_builder()
    scene = env.rlbench_env._scene
    robot = scene.robot
    return env, scene, robot

def _load_seed_artifacts(seed_dir, use_orientations_file=True, use_gripper_action_file=True, kept_rows=None):
    """Load euler_orientations.npy and gripper_actions.npy (optionally mask by kept_rows)."""
    import numpy as np, os
    eulers = None
    gripper = None
    if use_orientations_file:
        p = os.path.join(seed_dir, "euler_orientations.npy")
        if os.path.exists(p):
            eulers = np.load(p)
            if kept_rows is not None:
                eulers = eulers[kept_rows]
    if use_gripper_action_file:
        p = os.path.join(seed_dir, "gripper_actions.npy")
        if os.path.exists(p):
            gripper = np.load(p)
            if kept_rows is not None:
                gripper = gripper[kept_rows]
            if gripper.ndim == 1:
                gripper = gripper[:, None]
    return eulers, gripper

def _euler_deg_to_unit_quat(eulers_deg):
    """Convert (N,3) euler degrees (xyz order) to unit quaternions [x,y,z,w]."""
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    if eulers_deg is None:
        return None
    r = R.from_euler('xyz', eulers_deg, degrees=True)
    q = r.as_quat()
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q[n==0] = [0,0,0,1]
    return q / (n + 1e-8)

def _pack_action_seq(xyz, quats, grip):
    """Pack [xyz + quat + grip] per step."""
    import numpy as np
    N = xyz.shape[0]
    if quats is None:
        quats = np.tile(np.array([0,0,0,1.0], dtype=float), (N,1))
    if grip is None:
        grip = np.zeros((N,1), dtype=float)
    return np.concatenate([xyz, quats, grip], axis=1)

def _rollout_actions(env, actions, record_video=False):
    """Step the env with prepacked actions; return sequences and optional frames."""
    frames = []
    obs_seq, rew_seq, done_seq, info_seq = [], [], [], []
    for a in actions:
        try:
            obs, env_reward, done, info = env.step(a)
        except Exception as e:
            info_seq.append({"error": str(e)})
            done_seq.append(True)
            rew_seq.append(0.0)
            obs_seq.append(None)
            break
        obs_seq.append(obs); rew_seq.append(env_reward); done_seq.append(done); info_seq.append(info)
        if record_video and hasattr(env, "capture_frame"):
            frame = env.capture_frame()
            if frame is not None:
                frames.append(frame)
        if done:
            break
    return obs_seq, rew_seq, done_seq, info_seq, frames

# ---- explicit cross-mod imports ----
from cross_mod.utils.geom import quaternion_distance, quaternion_slerp, euler_to_quaternions_use
from cross_mod.planning.smoothing import interpolate_points, build_interpolated_trajectory, build_smooth_trajectory, interpolate_trajectory
from cross_mod.sim.envs import instantiate_environment, set_task_objects_in_scene, get_object_quaternion
from cross_mod.io.viz import plot_trajectory_grid, add_trajectory_line, set_cylinder_between_points


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



def execute_trajectory(task=None,discrete_gripper=False,gripper_pause=False,use_gripper_action_file=False,kept_rows=None,min_vals=None,max_vals=None,use_gripper_orientation_file=False ,perturbation_size=0.02,trials=500,steps_to_interp_quaternion=100,video_filename="trajectory_execution.mp4",starting_env_state=None,trajectory=None, render_mode='rgb_array', record_video=False, starting_positions=None, seed=0, working_dir=None,resolution=[1200,1200],cam_pose = None,frame_rate=1,interpolate=False,num_interpolate=200,interpolate_positions=False,show_traj_line=True,desired_quaternion=None,open_gripper=1.0,offset_by=None):
    
    import numpy as np
    seed_dir = kwargs.get("seed_dir", None)
    kept_rows = kwargs.get("kept_rows", None)
    record_video = kwargs.get("record_video", False)

    eulers_deg, grip = _load_seed_artifacts(
        seed_dir,
        use_orientations_file=kwargs.get("use_gripper_orientation_file", True),
        use_gripper_action_file=kwargs.get("use_gripper_action_file", True),
        kept_rows=kept_rows
    )
    quats = _euler_deg_to_unit_quat(eulers_deg)

    action_mode_builder = kwargs.get("action_mode_builder", None)
    if action_mode_builder is None:
        from cross_mod.sim.envs import instantiate_environment as action_mode_builder
    env, scene, robot = _prepare_env_and_scene(task, action_mode_builder)

    actions = _pack_action_seq(trajectory, quats, grip)
    obs_seq, rew_seq, done_seq, info_seq, frames = _rollout_actions(env, actions, record_video=record_video)
    return {"obs": obs_seq, "rewards": rew_seq, "done_flags": done_seq, "info": info_seq, "frames": frames}


def execute_trajectory_from_file(trajectory,cam_pose=None,gripper_pause=False,resolution=[1200,1200],record_video=True,task=None,discrete_gripper=True,seed=None,working_dir=None,video_filename="trajectory_execution.mp4"):
    

    show_traj_line = False


    if trajectory is None:
        raise ValueError("Trajectory must be provided.")



    """    
    # Create an RLBench environment with our desired action mode.
    action_mode = MoveArmThenGripperWithBoundsDelta(
        #arm_action_mode=EndEffectorPoseViaPlanning_Record(),
        arm_action_mode = JointPosition(absolute_mode=False),
        gripper_action_mode=GripperJointPosition()
    )
    """



    absolute_mode = True
    interp_traj = trajectory

    if discrete_gripper==False:
        action_mode = IVKPlanningBounds_NonDiscrete(
            arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=absolute_mode),
            gripper_action_mode=GripperJointPosition()
        )


    else:
        action_mode = IVKPlanningBounds(
            arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=absolute_mode),
            gripper_action_mode=Discrete()
        )


    env = RLBenchEnv(
        task_class=task,
        observation_mode='state',
        #render_mode=render_mode,
        render_mode="rgb_array",
        action_mode=action_mode)

    env.rlbench_task_env._shaped_rewards = False
    result = env.reset(seed=seed)
    current_obs = result[0]


    # Video recording setup
    video = None
    video_path = os.path.join(working_dir, "trajectory_0",video_filename) if record_video else None

    if record_video:
        # Use implicit handling: the sensor is updated automatically on each sim step.
        vision_sensor = VisionSensor.create(resolution)  # <-- no explicit_handling=True
        if cam_pose is None:
            cam_pose = compute_camera_pose([1.5, 0, 1.5])
        else:
            cam_pose = compute_camera_pose(np.array(cam_pose),
                                        center_point=np.array([0.3, 0.0, 1.0]))
            print("Set Camera Pose for Video")
        vision_sensor.set_pose(cam_pose)

        scene = env.rlbench_task_env._scene
        pr = scene.pyrep
        orig_pr_step = pr.step

        sim_dt = pr.get_simulation_timestep()
        sim_fps = max(1, int(round(1.0 / sim_dt)))
        target_fps = frame_rate if ('frame_rate' in globals() and frame_rate) else sim_fps

        def _pr_step_and_record(self, *args, **kwargs):
            nonlocal video
            ret = orig_pr_step(*args, **kwargs)  # advance one physics tick

            # Sensor is auto-updated under implicit handling
            img = (vision_sensor.capture_rgb() * 255).astype(np.uint8)

            if video is None:
                h, w, _ = img.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_path, fourcc, target_fps, (w, h))

            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return ret

        # Patch the *PyRep* step so we catch all internal stepping
        pr.step = MethodType(_pr_step_and_record, pr)

    # Execute the trajectory step by step.
    ee_positions = []  # to store actual end effector positions
    trajectory_data = PolicyTrajectory()
    trajectory_data.env_seed = seed
    traj_ivk_failed=False

    
    from pyrep.const import ConfigurationPathAlgorithms as Algos, ObjectType


    #if absolute_mode == False:
    #    interp_traj = interp_traj[1:] - interp_traj[:-1]
    if gripper_pause:
        # ── 1. locate every transition in the 1‑D action stream
        #     (we reshaped earlier, so column‑0 holds the scalar command)
        gripper_actions=[]
        for quickAct in interp_traj:
            gripper_actions.append(quickAct[-1])

        gripper_actions = np.array(gripper_actions).reshape(-1,1)

        transition_idx = (
            np.where(gripper_actions[:-1, 0] != gripper_actions[1:, 0])[0] + 1
        )

        if gripper_actions[0]!=0.04:
            transition_idx = np.append(np.array([0]),transition_idx)

        # ── 2. insert three duplicates of the transition row to create a pause
        offset = 0                                      # tracks growth of the arrays
        for idx in transition_idx:
            idx += offset                               # update for rows already inserted
            pose_row = interp_traj[idx:idx + 1]         # (1 × 3+4) slice
            act_row  = gripper_actions[idx:idx + 1]     # (1 × 1)   slice

            # repeat each slice n_dup× and insert *before* the change
            n_dup = 4
            interp_traj     = np.insert(
                interp_traj, idx, np.repeat(pose_row, n_dup, axis=0), axis=0
            )
            gripper_actions = np.insert(
                gripper_actions, idx, np.repeat(act_row, n_dup, axis=0), axis=0
            )
            offset += n_dup                                 # keep indices aligned
    

    for i, waypoint in enumerate(interp_traj):
        # Combine the 7D waypoint with a 1D gripper action (e.g. 1.0 = open).
        # refresh observation after the move, then proceed to i = 1
        #current_obs = env.rlbench_env._scene.get_observation()
        try:
            action = waypoint
            obs, reward, done, truncated, info = env.step(action)
            next_obs = obs
            terminated = False
            env_state = None
            ee_position = waypoint
            trajectory_data.append(current_obs, action, next_obs, reward, terminated, truncated, done, info, env_state,ee_position=ee_position)
            current_obs = next_obs
            successful_ivk=True
        except Exception as e:
            if i<len(interp_traj)-1:
                print("Failed waypoint " + str(i) + ", continuting on")
                continue
            else:
                #print(f"IK solver failed at step {i} with waypoint={waypoint}: {e}")
                print(e)
                print("Failed at Waypoint = " + str(i))
                traj_ivk_failed = True
                reward = 0.0
                done=True
            
        if done==True and record_video==True:
            # Unpatch and close writer
            if record_video:
                pr.step = orig_pr_step
                if video is not None:
                    video.release()
                    print("Video Saved to: " + video_path)
                    trajectory_data.video_path = video_path

        # Record the actual tip position after the step.
        tip = env.rlbench_task_env._robot.arm.get_tip()
        actual_pos = np.array(tip.get_position())
        ee_positions.append(actual_pos)

        if reward>0.0:
            print(" ✅  Task completed early!")
            success_status = True
            break
        else:
            success_status = False

        if traj_ivk_failed:
            print("Ivk Failed")
            break

    #print("Total Number of Points:")
    #print(len(env.action_mode.arm_action_mode.action_commands))
    env.close()
    del env
    gc.collect()
    print()

    return trajectory_data, done,success_status

def execute_trajectory_original(task=None,discrete_gripper=False,gripper_pause=False,use_gripper_action_file=False,kept_rows=None,min_vals=None,max_vals=None,use_gripper_orientation_file=False ,perturbation_size=0.02,trials=500,steps_to_interp_quaternion=100,video_filename="trajectory_execution.mp4",starting_env_state=None,trajectory=None, render_mode='rgb_array', record_video=False, starting_positions=None, seed=0, working_dir=None,resolution=[1200,1200],cam_pose = None,frame_rate=1,interpolate=False,num_interpolate=200,interpolate_positions=False,show_traj_line=True,desired_quaternion=None,open_gripper=1.0,offset_by=None):
    

    show_traj_line = False


    if trajectory is None:
        raise ValueError("Trajectory must be provided.")

    if np.isnan(trajectory).any():

        #NEW
        # Assume `trajectory` is an (N×3) NumPy array
        # First, build a mask of rows that have no NaNs:
        mask = ~np.isnan(trajectory).any(axis=1)

        # Apply the mask to keep only the rows without NaNs:
        trajectory = trajectory[mask]

        # If, after filtering, there’s nothing left, handle it as before:
        if trajectory.size == 0:
            print("All rows contained NaNs. Skip")
            trajectory_data = PolicyTrajectory()
            done = False
            return trajectory_data, done

        #END NEW

    """    
    # Create an RLBench environment with our desired action mode.
    action_mode = MoveArmThenGripperWithBoundsDelta(
        #arm_action_mode=EndEffectorPoseViaPlanning_Record(),
        arm_action_mode = JointPosition(absolute_mode=False),
        gripper_action_mode=GripperJointPosition()
    )
    """

    if trajectory.shape[1]>3:
        print("Gripper Action Specfied")
        gripper_action = trajectory[:,3]
        trajectory = trajectory[:,0:3]
    else:
        gripper_action = None
        print("Gripper action not specified")


    absolute_mode = True

    if discrete_gripper==False:
        action_mode = IVKPlanningBounds(
            arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=absolute_mode),
            gripper_action_mode=GripperJointPosition()
        )
    else:
        action_mode = IVKPlanningBounds(
            arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=absolute_mode),
            gripper_action_mode=Discrete()
        )


    env = RLBenchEnv(
        task_class=task,
        observation_mode='state',
        #render_mode=render_mode,
        render_mode="rgb_array",
        action_mode=action_mode)

    env.rlbench_task_env._shaped_rewards = False
    result = env.reset(seed=seed)
    current_obs = result[0]

    if starting_env_state is not None:
        starting_joint_positions = starting_env_state["robot_state"]
        gripper_state = starting_env_state["gripper_state"]
        env.rlbench_env._scene.robot.arm.set_joint_target_positions(starting_joint_positions)
        env.rlbench_env._scene.robot.gripper.set_joint_target_positions(gripper_state)
        set_task_objects_in_scene(env,starting_env_state)
        for _ in range(20):
            result = env.rlbench_env._scene.pyrep.step()
        
        #current_obs = result[0]


        print("x")
    
    #Get the first pose
    
    #first_pose = env.rlbench_env._scene.robot.arm.get_tip().get_pose()
    #first_pose=env.rlbench_env._scene.robot.gripper.get_pose()
    #first_position = np.array(first_pose[:3])  # Only the xyz
    
    # Compute distances to all trajectory points
    #dists = np.linalg.norm(trajectory - first_position, axis=1)

    # Get the index with the minimum distance
    #min_idx = np.argmin(dists)

    # Trim the trajectory
    #trajectory = trajectory[min_idx:]

    #trajectory = np.insert(trajectory,0,first_pose[:3],axis=0)
    
    

    
    if interpolate and use_gripper_orientation_file==False:
        init_quat = env.rlbench_env._scene.robot.arm.get_tip().get_quaternion()
        init_quat = env.rlbench_env._scene.robot.arm.get_tip().get_orientation()
        
        scene   = env.rlbench_env._scene          # or however you store it
        ee      = scene.robot.arm.get_tip()       # Shape handle of the gripper tool
        ee = scene.robot.gripper
        # ------------------------------------------------------------------

        # 1) Desired orientation *relative to the gripper frame*
        #    (roll, pitch, yaw) in **radians**
        #delta_rpy = np.deg2rad([ 0.0,   -30.0,  90.0])   # example values
        delta_rpy = offset_by
        delta_rot  = R.from_euler('xyz', delta_rpy)     # local → quaternion/rot obj

        # 2) Current end-effector orientation in the *world* frame
        ee_quat_world = ee.get_quaternion()             # [x, y, z, w]  (world frame)
        ee_rot_world  = R.from_quat(ee_quat_world)      # Rotation object

        # 3) Compose rotations:   world_target = current_world ∘ delta_local
        world_target_rot  = ee_rot_world * delta_rot
        world_target_quat = world_target_rot.as_quat()  # still [x, y, z, w]
        desired_quaternion = world_target_quat

        interp_traj = interpolate_trajectory(trajectory, init_quat,interpolate_positions=interpolate_positions,desired_quat=desired_quaternion,num_interpolate=num_interpolate, steps_to_interp_quaternion=steps_to_interp_quaternion)
    else:
        interp_traj = trajectory


    #APPLY WORKSPACE LIMITS
    if min_vals is not None:
        interp_traj = np.maximum(interp_traj, min_vals)

    if max_vals is not None:
        interp_traj = np.minimum(interp_traj, max_vals)

    #USE ORIENTATION FILE
    if use_gripper_orientation_file==True:
        # make a copy of working_dir with the last segment removed
        base_dir = Path(working_dir).resolve().parent    # "x/y/z/"  ➜  "x/y"

        # build the orientations‑file path in that parent folder
        gripper_orientations = np.load(base_dir / "euler_orientations.npy")
        #print("x")

        #Position of the end effector
        #tip = env.rlbench_task_env._robot.arm.get_tip().get_orientation()
        

        
        #gripper_orientations[gripper_orientations[:, 2] < 179, 2] = 120
        #gripper_orientations[gripper_orientations[:, 0] < 90, 0] = 270
        

        #gripper_orientations[1:,0] = 270 #CHANGE THIS BACK

        #quickShape = interp_traj.shape[0]
        #gripper_orientations = gripper_orientations[:quickShape,:]

        #Only keep the rows whose indices are in kept_rows
        #if kept_rows is not None:
            #gripper oreintatoins is Nx3
        #    gripper_orientations = gripper_orientations[kept_rows]
        
        if gripper_orientations.shape[0] != interp_traj.shape[0]:
            gripper_orientations = np.vstack([gripper_orientations, gripper_orientations[-1]])
        # convert to quaternions
        gripper_orientations = euler_to_quaternions_use(eulers=gripper_orientations)

        if kept_rows is not None:
            gripper_orientations = gripper_orientations[kept_rows]

        

        interp_traj = np.concatenate((interp_traj, gripper_orientations), axis=1)

        if use_gripper_action_file is True:
            # load the action array from the same parent folder as the orientations
            gripper_actions = np.load(base_dir / "gripper_actions.npy")      # shape (N,) or (N, 1)
            

            if kept_rows is not None:
                gripper_actions = gripper_actions[kept_rows]

            # keep only the desired rows (if a mask was supplied)
            #if kept_rows is not None:
            #    gripper_actions = gripper_actions[kept_rows]

            # pad with the last entry if the length is off‑by‑one (same rule as above)
            if gripper_actions.shape[0] != interp_traj.shape[0]:
                gripper_actions = np.append(gripper_actions, gripper_actions[-1])

            # ensure a 2‑D column so concatenation works
            gripper_actions = gripper_actions.reshape(-1, 1)                 # shape (N, 1)

            if discrete_gripper==True:
                gripper_actions[gripper_actions > 0.02] = 1.0
                gripper_actions[gripper_actions <= 0.02] = 0.0


            # append as an extra column to interp_traj   →   shape becomes (N, 3 + 4 + 1)
            #interp_traj = np.concatenate((interp_traj, gripper_actions), axis=1)

            if gripper_pause:
                # ── 1. locate every transition in the 1‑D action stream
                #     (we reshaped earlier, so column‑0 holds the scalar command)
                transition_idx = (
                    np.where(gripper_actions[:-1, 0] != gripper_actions[1:, 0])[0] + 1
                )

                # ── 2. insert three duplicates of the transition row to create a pause
                offset = 0                                      # tracks growth of the arrays
                for idx in transition_idx:
                    idx += offset                               # update for rows already inserted
                    pose_row = interp_traj[idx:idx + 1]         # (1 × 3+4) slice
                    act_row  = gripper_actions[idx:idx + 1]     # (1 × 1)   slice

                    # repeat each slice n_dup× and insert *before* the change
                    n_dup = 2
                    interp_traj     = np.insert(
                        interp_traj, idx, np.repeat(pose_row, n_dup, axis=0), axis=0
                    )
                    gripper_actions = np.insert(
                        gripper_actions, idx, np.repeat(act_row, n_dup, axis=0), axis=0
                    )
                    offset += n_dup                                 # keep indices aligned

        else:
            gripper_actions = None


    if np.isnan(interp_traj).any():
        print("Error with: "+working_dir+" -> NaN values in the trajectory")
        env.close()
        trajectory_data=None
        success=False
        return trajectory_data,success
    

    #visualize the desired trajectory as specified in interp_traj
    if show_traj_line==True:
        # Add visual line for entire trajectory
        visual_traj_obj = add_trajectory_line(interp_traj)


    # Video recording setup
    frames = []
    if record_video:
        vision_sensor = VisionSensor.create(resolution)
        if cam_pose==None:
            cam_pose = compute_camera_pose([1.5,0,1.5])
        else:
            cam_pose = compute_camera_pose(np.array(cam_pose),center_point=np.array([0.3,0.0,1.0]))
            print("Set Camera Pose for Video")
        vision_sensor.set_pose(cam_pose)
        #self.rlbench_task_env.pyrep.step()
        env.rlbench_task_env._scene.pyrep.step()
        frame = vision_sensor.capture_rgb()
        frame = (frame * 255).astype(np.uint8)

        # Convert to BGR for OpenCV (once!)
        #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bgr = frame
        
        # Now append the correctly modified frame_bgr
        frames.append(frame_bgr)

    # Execute the trajectory step by step.
    ee_positions = []  # to store actual end effector positions
    
    trajectory_data = PolicyTrajectory()
    trajectory_data.env_seed = seed
    traj_ivk_failed=False

    
    from pyrep.const import ConfigurationPathAlgorithms as Algos, ObjectType


    if absolute_mode == False:
        interp_traj = interp_traj[1:] - interp_traj[:-1]

    for i, waypoint in enumerate(interp_traj):
        # Combine the 7D waypoint with a 1D gripper action (e.g. 1.0 = open).
        # refresh observation after the move, then proceed to i = 1
        #current_obs = env.rlbench_env._scene.get_observation()
        try:
            if absolute_mode==False:
                # waypoint: 1-D torch tensor length-7  [x y z qx qy qz qw]
                norm = np.linalg.norm(waypoint[3:])                # ‖q‖₂

                if norm < 1e-6:                                    # degenerate → identity
                    waypoint[3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=waypoint.dtype)
                    #pass
                else:                                              # normalise in-place
                    waypoint[3:] /= norm
                
            if gripper_actions is not None:
                grip = gripper_actions[i]
                #if grip>0.9:
                #    grip = 0.0
                #else:
                #    grip = 0.04
            
            else:
                grip = np.array([0.04 * open_gripper])
                if discrete_gripper==True:
                    if grip[0]>=0.02:
                        grip[0] = 1.0
                    else:
                        grip[0] = 0.0

            #print(grip)
            #print(grip)
            action = np.concatenate([waypoint, grip])
            obs, reward, done, truncated, info = env.step(action)
            next_obs = obs
            terminated = False
            env_state = None
            ee_position = waypoint
            trajectory_data.append(current_obs, action, next_obs, reward, terminated, truncated, done, info, env_state,ee_position=ee_position)
            current_obs = next_obs
            successful_ivk=True
        except Exception as e:
            if i<len(interp_traj)-1:
                print("Failed waypoint " + str(i) + ", continuting on")
                continue
            else:
                #print(f"IK solver failed at step {i} with waypoint={waypoint}: {e}")
                print(e)
                print("Failed at Waypoint = " + str(i))
                traj_ivk_failed = True
                reward = 0.0
                done=True
            

        if record_video:
            frame = vision_sensor.capture_rgb()
            frame = (frame * 255).astype(np.uint8)

            # Convert to BGR for OpenCV (once!)
            #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = frame

            # Overlay text in upper-right corner onto frame_bgr
            
            # Now append the correctly modified frame_bgr
            frames.append(frame_bgr)

        # Record the actual tip position after the step.
        tip = env.rlbench_task_env._robot.arm.get_tip()
        actual_pos = np.array(tip.get_position())
        ee_positions.append(actual_pos)

        if reward>0.0:
            print(" ✅  Task completed early!")
            success_status = True
            break
        else:
            success_status = False

        if traj_ivk_failed:
            print("Ivk Failed")
            break
    if record_video and len(frames)>0:
        frame = vision_sensor.capture_rgb()
        frame = (frame * 255).astype(np.uint8)

        # Convert to BGR for OpenCV (once!)
        #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bgr = frame

        # Overlay text in upper-right corner onto frame_bgr
        

        # Now append the correctly modified frame_bgr
        frames.append(frame_bgr)
        frames.append(frame_bgr)



        video_path = os.path.join(working_dir, video_filename)
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("Frame Rate: " + str(frame_rate))
        video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
        
        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        video.release()
        trajectory_data.video_path = video_path
        print(f"Video saved at {video_path}")

    #print("Total Number of Points:")
    #print(len(env.action_mode.arm_action_mode.action_commands))
    env.close()
    del env
    gc.collect()
    print()

    return trajectory_data, done,success_status