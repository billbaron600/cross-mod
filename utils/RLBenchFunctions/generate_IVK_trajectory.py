import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import cv2
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape, PrimitiveShape

from rlbench.gym import RLBenchEnv
from rlbench.tasks import SlideBlockToTarget

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK,EndEffectorPoseViaPlanning, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete,GripperJointPosition
from  utils.RLBenchFunctions.template_sensor_views import compute_camera_pose
from scipy.spatial.transform import Rotation as R,RotationSpline
import gc
from utils.Classes.policy_trajectory_class import PolicyTrajectory
from rlbench.tasks import SlideBlockToTarget
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from rlbench.tasks import SlideBlockToTarget
from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import visualize_trajectories
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from pyrep.objects.dummy import Dummy
from scipy.interpolate import interp1d
from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds,MoveArmThenGripperWithBoundsDelta_IVK, EndEffectorPoseViaPlanning_Custom, MoveArmThenGripperWithBoundsDelta
import random
import copy
from utils.RLBenchFunctions.custom_action_modes import EndEffectorPoseViaPlanning_Record,IVKPlanningBounds,IVKPlanningBounds_NonDiscrete
from transforms3d.euler import quat2euler, euler2quat
import os, pickle, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap     # ← add this
import torch
from pathlib import Path
from types import MethodType



def merge_full_trajectories(
        config,
        results_root: str,                       # e.g. "run_results/reach_target/demos"
        fname: str = "full_ivk_trajectories.pkl",
        save: bool = True,
        limit_to_correction_indices = None,
        limit_to_first=True
    ):
    """
    Combine the per-folder <fname> files into one dictionary.

    Parameters
    ----------
    config        : your experiment/config object (must have .iteration_working_dir)
    results_root  : path that contains the numbered sub-folders (ignored if
                    limit_to_correction_indices is provided)
    fname         : pickle file to load from each sub-folder
    save          : if True, write the merged dict as <config.iteration_working_dir>/<fname>
    limit_to_correction_indices : ordered list of integers.  
                    When provided, only these indices are loaded and the folder
                    paths are built as:
                        os.path.join(config.iteration_working_dir, str(idx))

    Returns
    -------
    merged : dict with keys "full_trajectories" and "successful"
    """
    # ---------------------------------------------------------------------
    # 1) determine which sub-folders to load and where they live
    # ---------------------------------------------------------------------
    if limit_to_correction_indices is not None:
        # Use exactly the indices the caller specified, in order
        base_dir = getattr(config, "iteration_working_dir")
        folders = [str(i) for i in limit_to_correction_indices]
    else:
        # Fall back to scanning results_root for numeric sub-folders
        base_dir = os.path.abspath(results_root)
        folders = sorted(
            [d for d in os.listdir(base_dir) if d.isdigit()],
            key=int
        )
        if not folders:
            raise FileNotFoundError(f"No numeric sub-folders in {base_dir}")

    # ---------------------------------------------------------------------
    # 2) load, concatenate
    # ---------------------------------------------------------------------
    merged = {"full_trajectories": [], "successful": []}

    for d in folders:
        pkl_path = os.path.join(base_dir, d, fname)
        if not os.path.isfile(pkl_path):
            continue
            #raise FileNotFoundError(f"Expected file not found: {pkl_path}")

        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        if len(data["successful"])>0 and limit_to_first is True:
            data["full_trajectories"] = [data["full_trajectories"][0]]
            data["successful"] = [data["successful"][0]]

        merged["full_trajectories"].extend(data["full_trajectories"])
        merged["successful"].extend(data["successful"])

    # ---------------------------------------------------------------------
    # 3) optionally save the merged dict in iteration_working_dir
    # ---------------------------------------------------------------------
    if save:
        out_dir = getattr(config, "iteration_working_dir")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "wb") as fh:
            pickle.dump(merged, fh)
        print(f"merged dict saved → {out_path}")

    total = len(merged["successful"])
    successes = sum(merged["successful"])
    pct_success = 100.0 * successes / total if total > 0 else 0.0

    print(f"merged {len(merged['full_trajectories'])} trajectories "
          f"from {len(folders)} folders")

    print(f"{successes}/{total} successful "
          f"({pct_success:.2f}%)")

    return merged


def merge_full_trajectories_DEPRECATED(config,
                            results_root,                 # e.g. "run_results/reach_target/demos"
                            fname="full_ivk_trajectories.pkl",
                            save=True):
    """
    Combine the per-folder <fname> files under <results_root>/0,1,2,… into one
    dictionary and (optionally) save it in <config.iteration_working_dir>.

    Parameters
    ----------
    config        : your experiment/config object (must have .iteration_working_dir)
    results_root  : path that contains the numbered sub-folders
    fname         : pickle file to load from each sub-folder  [default: "full_ivk_trajectories.pkl"]
    save          : if True, write the merged dict as <config.iteration_working_dir>/<fname>

    Returns
    -------
    merged : dict with keys "full_trajectories" and "successful"
    """
    results_root = os.path.abspath(results_root)

    # 1) collect numeric sub-folders in ascending order
    folders = sorted([d for d in os.listdir(results_root) if d.isdigit()],
                     key=int)
    if not folders:
        raise FileNotFoundError(f"No numeric sub-folders in {results_root}")

    merged = {"full_trajectories": [], "successful": []}

    # 2) load and extend
    for d in folders:
        pkl_path = os.path.join(results_root, d, fname)
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        merged["full_trajectories"].extend(data["full_trajectories"])
        merged["successful"].extend(data["successful"])

    # 3) optionally save into the config’s working dir
    if save:
        out_dir = getattr(config, "iteration_working_dir")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "wb") as fh:
            pickle.dump(merged, fh)
        print(f"merged dict saved → {out_path}")

    print(f"merged {len(merged['full_trajectories'])} trajectories "
          f"from {len(folders)} folders")
    return merged




def plot_trajectory_grid(config,root_path,
                         fname="full_ivk_trajectories.pkl",
                         colours=("tab:red", "tab:green"),
                         limit_to_correction_indices=None,   # (fail, success)
                         fig_kw=None):
    
    if limit_to_correction_indices is not None:
        # Use exactly the indices the caller specified, in order
        base_dir = getattr(config, "iteration_working_dir")
        folders = [str(i) for i in limit_to_correction_indices]
    else:
        # Fall back to scanning results_root for numeric sub-folders
        base_dir = os.path.abspath(results_root)
        folders = sorted(
            [d for d in os.listdir(base_dir) if d.isdigit()],
            key=int
        )
        if not folders:
            raise FileNotFoundError(f"No numeric sub-folders in {base_dir}")

    lengths, success, max_cols = [], [], 0
    for d in folders:
        with open(os.path.join(root_path, d, fname), "rb") as fh:
            data = pickle.load(fh)
        trajs, succ_flags = data["full_trajectories"], data["successful"]
        row_len = [len(t.observations) for t in trajs]
        lengths.append(row_len)
        success.append(succ_flags)
        max_cols = max(max_cols, len(row_len))

    pad = lambda seq, fill: seq + [fill] * (max_cols - len(seq))
    len_mat  = np.array([pad(r, np.nan) for r in lengths], dtype=float)
    succ_mat = np.array([pad(r, False)  for r in success ], dtype=bool)

    # ── plot ───────────────────────────────────────────────────────────
    cmap = ListedColormap(colours)                 # ← fix here
    fig_kw = dict(figsize=(max_cols * .6, len(folders) * .6), **(fig_kw or {}))
    fig, ax = plt.subplots(**fig_kw)

    ax.imshow(succ_mat.astype(int), cmap=cmap, aspect="equal", vmin=0, vmax=1)  # cast → int

    for r in range(len(folders)):
        for c in range(max_cols):
            if not math.isnan(len_mat[r, c]):
                ax.text(c, r, int(len_mat[r, c]),
                        ha="center", va="center",
                        color="white" if succ_mat[r, c] else "black",
                        fontsize=8)

    ax.set_xticks(range(max_cols))
    ax.set_yticks(range(len(folders)))
    ax.set_yticklabels(folders)
    ax.set_xlabel("trajectory idx"); ax.set_ylabel("folder idx")
    ax.set_title("Demo trajectory success (green) / failure (red)\n"
                 "numbers = trajectory length")
    plt.tight_layout(); plt.show()


def quaternion_distance(q1, q2):
    """
    Computes a simple distance between two quaternions (cosine distance on the unit sphere).
    """
    q1 = np.array(q1)
    q2 = np.array(q2)
    return 1 - abs(np.dot(q1, q2))

"""

def get_object_quaternion(env=None,init_quat=None,object_name="Panda_gripper",relative_to_object=False,offset_by=None):
   
    if offset_by == [0,0,0]:
        print("No offset")

        return init_quat
    #gripper = env.rlbench_env._scene.robot.arm.get_tip()

    #gripper = env.rlbench_env._scene.robot.gripper
    #euler = gripper.get_orientation()
    # 1) Build a Rotation from your initial quaternion
    rot_initial = R.from_quat(init_quat)    
    # 2) Extract its Euler angles (in radians). 'xyz' = roll, pitch, yaw
    euler = rot_initial.as_euler('xyz')  

    euler_orig = copy.deepcopy(euler)

    if offset_by is not None:
        for angle in range(len(offset_by)):
            #secure_rng = random.SystemRandom()
            euler[angle] = euler[angle] + np.deg2rad(offset_by[angle])   #np.deg2rad(secure_rng.uniform(-offset_by[1][angle], offset_by[1][angle])) #
            #+ np.deg2rad(np.random.uniform(-offset_by[1][angle],offset_by[1][angle]))
        print("Changed euler angle")
            
        print("Euler Change:")
        print(np.rad2deg(euler)-np.rad2deg(euler_orig))

        #print(euler)
        rot_new = R.from_euler('xyz', euler)

        # 4) Export back to quaternion (x, y, z, w)
        quat = rot_new.as_quat()
        return quat
"""



def get_object_quaternion(env=None,init_quat=None,object_name="Panda_gripper",relative_to_object=False,offset_by=None):
   
    if offset_by == [0,0,0]:
        print("No offset")

        return init_quat
    #gripper = env.rlbench_env._scene.robot.arm.get_tip()

    #gripper = env.rlbench_env._scene.robot.gripper
    #euler = gripper.get_orientation()
    # 1) Build a Rotation from your initial quaternion
    rot_initial = R.from_quat(init_quat)    
    # 2) Extract its Euler angles (in radians). 'xyz' = roll, pitch, yaw
    euler= rot_initial.as_euler('xyz') 
    

    euler_orig = copy.deepcopy(euler)

    if offset_by is not None:
        for angle in range(len(offset_by)):
            #secure_rng = random.SystemRandom()
            euler[angle] = euler[angle] + np.deg2rad(offset_by[angle])   #np.deg2rad(secure_rng.uniform(-offset_by[1][angle], offset_by[1][angle])) #
            #+ np.deg2rad(np.random.uniform(-offset_by[1][angle],offset_by[1][angle]))
        print("Changed euler angle")
            
        print("Euler Change:")
        print(np.rad2deg(euler_orig)-np.rad2deg(euler))
    
        #BILL
        euler = euler_orig + np.array([45,0,0])
        #print(euler)
        dummy = Dummy.create()
        dummy.set_orientation(euler)
        quat = dummy.get_quaternion()
        # 3) Build a new Rotation from the modified Euler angles
        
        return quat


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
# --------------------------------------------------------------------------
# Helper functions to interpolate between points.
# If you don't need interpolation, you can remove this section.
# --------------------------------------------------------------------------

"""
class MoveArmThenGripperWithBounds(MoveArmThenGripper):


    def action_bounds(self):
        #value = 0.01 #was initially 0.1
        value=5
        low = np.array(7 * [-value] + [0])   # shape (8,)
        high = np.array(7 * [value] + [0.04])  # shape (8,)
        return (low, high)

"""


def set_cylinder_between_points(cyl, start_point, end_point):
    """
    Orients and positions a cylinder between two points.
    """
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    center = (start_point + end_point) / 2
    diff = end_point - start_point
    length = np.linalg.norm(diff)

    # Default cylinder orientation is along Z-axis in PyRep
    default_dir = np.array([0, 0, 1])

    # Compute rotation axis and angle
    rotation_axis = np.cross(default_dir, diff)
    if np.linalg.norm(rotation_axis) < 1e-6:
        # Vectors are parallel
        if np.dot(default_dir, diff) > 0:
            quat = [0, 0, 0, 1]  # No rotation needed
        else:
            quat = R.from_euler('x', np.pi).as_quat()
    else:
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(default_dir, diff) / length, -1.0, 1.0))
        quat = R.from_rotvec(rotation_axis * angle).as_quat()

    cyl.set_orientation(quat, relative_to=None)
    cyl.set_position(center)

def add_trajectory_line(traj, line_thickness=0.005, color=[1.0, 0.0, 0.0]):
    points = traj[:, :3]
    segments = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        
        cyl = Shape.create(
            PrimitiveShape.CYLINDER,
            size=[line_thickness, line_thickness, np.linalg.norm(end - start)],
            mass=0,
            color=color,
            static=True,
            respondable=False
        )

        set_cylinder_between_points(cyl, start, end)
        segments.append(cyl)
    return segments

def set_task_objects_in_scene(env,starting_env_state):
    objects_in_tree = env.rlbench_env._scene.pyrep.get_objects_in_tree()
    set_objects = starting_env_state["scene_objects"]

    for i in objects_in_tree:
        object_name = i.get_name()
        if set_objects.get(object_name) is not None:
            pose = set_objects[object_name]["pose"]
            i.set_pose(pose)
            env.rlbench_env._scene.pyrep.step()
            print("set pose for " + object_name)


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



def execute_trajectory(task=None,discrete_gripper=False,gripper_pause=False,use_gripper_action_file=False,kept_rows=None,min_vals=None,max_vals=None,use_gripper_orientation_file=False ,perturbation_size=0.02,trials=500,steps_to_interp_quaternion=100,video_filename="trajectory_execution.mp4",starting_env_state=None,trajectory=None, render_mode='rgb_array', record_video=False, starting_positions=None, seed=0, working_dir=None,resolution=[1200,1200],cam_pose = None,frame_rate=1,interpolate=False,num_interpolate=200,interpolate_positions=False,show_traj_line=True,desired_quaternion=None,open_gripper=1.0,offset_by=None):
    """
    Executes a given trajectory in RLBench.
    
    Parameters:
        task: The task to be used in the RLBench environment.
        trajectory (numpy array): The trajectory to be executed.
        render_mode (str): The rendering mode (default: 'rgb_array').
        record_video (bool): Whether to record a video of the scene (default: False).
        starting_positions: Not used for now (default: None).
        seed (int): The seed for environment reset (default: None).
    """

    show_traj_line = False


    if trajectory is None:
        raise ValueError("Trajectory must be provided.")

    if np.isnan(trajectory).any():
        # OLD
        #print("Trajectory has None values. SKip")
        #trajectory_data = PolicyTrajectory()
        #done = False
        #return trajectory_data,done

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
        """
        #init_quat =  env.rlbench_task_env._robot.arm.get_tip().get_pose()
        init_quat = env.rlbench_task_env._robot.gripper.get_quaternion()

        # --- quaternion → Euler (roll, pitch, yaw) ---
        roll, pitch, yaw = R.from_quat(init_quat).as_euler('xyz', degrees=False)

        # --- modify yaw ---
        yaw *= -1.0                      # flip the sign

        # --- back to quaternion ---
        init_quat = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_quat()
        """
        #init_quat = env.rlbench_task_env._robot.gripper.get_pose()
        #init_quat = init_quat[3:]
        #yaw180     = [0., 0., 1., 0.]                              # 180° about Z
        #init_quat = (R.from_quat(init_quat) * R.from_quat(yaw180)).as_quat()
        #BILL
        #if desired_quaternion is None:
        #desired_quaternion = get_object_quaternion(env=env,init_quat=init_quat,offset_by=offset_by)
        # Handles
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
        init_quat = euler_to_quaternions_use(eulers=np.rad2deg(init_quat.reshape(1,3)))
        init_quat = init_quat.squeeze(0)
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
    video = None
    video_path = os.path.join(working_dir, video_filename) if record_video else None

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
    
    #record the full trajectory data (after IVK)
    #trajectory_data = {
    #'observations': [],
    #'actions': [],
    #'rewards': [],
    #'ee_positions': [],
    #'dones': []}
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
            
        if done==True and record_video==True:
            # Unpatch and close writer
            if record_video:
                pr.step = orig_pr_step
                if video is not None:
                    video.release()
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


def execute_trajectory_from_file(trajectory,cam_pose=None,gripper_pause=False,resolution=[1200,1200],record_video=True,task=None,discrete_gripper=True,seed=None,working_dir=None,video_filename="trajectory_execution.mp4"):
    """
    Executes a given trajectory in RLBench.
    
    Parameters:
        task: The task to be used in the RLBench environment.
        trajectory (numpy array): The trajectory to be executed.
        render_mode (str): The rendering mode (default: 'rgb_array').
        record_video (bool): Whether to record a video of the scene (default: False).
        starting_positions: Not used for now (default: None).
        seed (int): The seed for environment reset (default: None).
    """

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
    """
    Executes a given trajectory in RLBench.
    
    Parameters:
        task: The task to be used in the RLBench environment.
        trajectory (numpy array): The trajectory to be executed.
        render_mode (str): The rendering mode (default: 'rgb_array').
        record_video (bool): Whether to record a video of the scene (default: False).
        starting_positions: Not used for now (default: None).
        seed (int): The seed for environment reset (default: None).
    """

    show_traj_line = False


    if trajectory is None:
        raise ValueError("Trajectory must be provided.")

    if np.isnan(trajectory).any():
        # OLD
        #print("Trajectory has None values. SKip")
        #trajectory_data = PolicyTrajectory()
        #done = False
        #return trajectory_data,done

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
        """
        #init_quat =  env.rlbench_task_env._robot.arm.get_tip().get_pose()
        init_quat = env.rlbench_task_env._robot.gripper.get_quaternion()

        # --- quaternion → Euler (roll, pitch, yaw) ---
        roll, pitch, yaw = R.from_quat(init_quat).as_euler('xyz', degrees=False)

        # --- modify yaw ---
        yaw *= -1.0                      # flip the sign

        # --- back to quaternion ---
        init_quat = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_quat()
        """
        #init_quat = env.rlbench_task_env._robot.gripper.get_pose()
        #init_quat = init_quat[3:]
        #yaw180     = [0., 0., 1., 0.]                              # 180° about Z
        #init_quat = (R.from_quat(init_quat) * R.from_quat(yaw180)).as_quat()
        #BILL
        #if desired_quaternion is None:
        #desired_quaternion = get_object_quaternion(env=env,init_quat=init_quat,offset_by=offset_by)
        # Handles
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
        """
        # Overlay text in upper-right corner onto frame_bgr
        text = f"Waypoint: {i+1}/{len(interp_traj)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # white
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        position = (frame_bgr.shape[1] - text_width - 10, text_height + 10)
        cv2.putText(frame_bgr, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        # Add another text in the lower-left corner
        # Add another text just under the waypoint text in upper-right corner
        seed_text = f"Seed: {seed}"
        seed_size, _ = cv2.getTextSize(seed_text, font, font_scale, thickness)
        seed_width, seed_height = seed_size
        seed_position = (frame_bgr.shape[1] - seed_width - 10, position[1] + text_height + 10)

        cv2.putText(frame_bgr, seed_text, seed_position, font, font_scale, color, thickness, cv2.LINE_AA)

        """
        # Now append the correctly modified frame_bgr
        frames.append(frame_bgr)

    # Execute the trajectory step by step.
    ee_positions = []  # to store actual end effector positions
    
    #record the full trajectory data (after IVK)
    #trajectory_data = {
    #'observations': [],
    #'actions': [],
    #'rewards': [],
    #'ee_positions': [],
    #'dones': []}
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
            """
            text = f"Waypoint: {i+1}/{len(interp_traj)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)  # white
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size
            position = (frame_bgr.shape[1] - text_width - 10, text_height + 10)
            cv2.putText(frame_bgr, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

            # Add another text in the lower-left corner
            # Add another text just under the waypoint text in upper-right corner
            seed_text = f"Seed: {seed}"
            seed_size, _ = cv2.getTextSize(seed_text, font, font_scale, thickness)
            seed_width, seed_height = seed_size
            seed_position = (frame_bgr.shape[1] - seed_width - 10, position[1] + text_height + 10)

            cv2.putText(frame_bgr, seed_text, seed_position, font, font_scale, color, thickness, cv2.LINE_AA)

            """
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
        """
        text = f"Waypoint: {i+1}/{len(interp_traj)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # white
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        position = (frame_bgr.shape[1] - text_width - 10, text_height + 10)
        cv2.putText(frame_bgr, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        # Add another text in the lower-left corner
        # Add another text just under the waypoint text in upper-right corner
        seed_text = f"Seed: {seed}"
        seed_size, _ = cv2.getTextSize(seed_text, font, font_scale, thickness)
        seed_width, seed_height = seed_size
        seed_position = (frame_bgr.shape[1] - seed_width - 10, position[1] + text_height + 10)

        cv2.putText(frame_bgr, seed_text, seed_position, font, font_scale, color, thickness, cv2.LINE_AA)
        """

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




from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import save_sample_trajectories

def generate_interpolated_trajectories(
        config,
        init_orientation,                 # [roll,pitch,yaw]  (deg, XYZ)
        target_orientations,              # list  [[roll,pitch,yaw], …]
        target_indices,                   # list  [idx0, idx1, …]  same length
        seed_idx=0):
    """
    Linear Euler-angle interpolation (yaw-pitch-roll, ZYX).

    * Segment 0 :   init_orientation  →  target_orientations[0]
                    over indices   0 … target_indices[0]
    * Segment k :   target_orientations[k-1]  →  target_orientations[k]
                    over   target_indices[k-1]+1 … target_indices[k]

    XYZ positions are copied untouched.
    Saves all trajectories + one overlay plot.
    """

    # ── safety checks ──────────────────────────────────────────────────────
    if len(target_orientations) != len(target_indices):
        raise ValueError("target_orientations and target_indices must match")
    if sorted(target_indices) != list(target_indices):
        raise ValueError("target_indices must be monotonically increasing")

    # ── load xyz positions ────────────────────────────────────────────────
    folder = os.path.join(config.iteration_working_dir, str(seed_idx))
    with open(os.path.join(folder, "generated_trajectories.pkl"), "rb") as f:
        traj_xyz = pickle.load(f)                       # (M,N,3) torch
    traj_xyz = traj_xyz.clone()
    M, N, _ = traj_xyz.shape

    init_orientation   = np.asarray(init_orientation,  dtype=float)
    tgt_oris           = [np.asarray(o, dtype=float) for o in target_orientations]
    tgt_idxs           = target_indices

    # ── container for output ──────────────────────────────────────────────
    traj_out = torch.full(
        (M, N, 6), float("nan"),
        dtype=traj_xyz.dtype,
        device=traj_xyz.device,
    )
    traj_out[:, :, :3] = traj_xyz                       # copy xyz

    all_eulers = []                                     # for overlay plot

    # ── build interpolation segments once (index mapping) ────────────────
    segments = []
    prev_idx  = -1
    prev_ori  = init_orientation
    for idx, tgt_idx in enumerate(tgt_idxs):
        start = prev_idx + 1
        end   = tgt_idx
        if start > end:
            raise ValueError(f"segment {idx} has invalid indices")
        segments.append((start, end, prev_ori, tgt_oris[idx]))
        prev_idx, prev_ori = end, tgt_oris[idx]
    # handle any remaining tail (hold last orientation)
    if prev_idx + 1 < N:
        segments.append((prev_idx + 1, N - 1, prev_ori, prev_ori))

    # ── fill each trajectory ──────────────────────────────────────────────
    for m in range(M):
        xyz   = traj_xyz[m]
        valid = ~torch.isnan(xyz[:, 0])
        if valid.sum() < 2:
            all_eulers.append(None)
            continue

        eul_full = np.full((N, 3), np.nan, float)

        for (s, e, ori_a, ori_b) in segments:
            n  = e - s + 1
            alpha = np.linspace(0.0, 1.0, n)[:, None]
            eul_seg = ori_a + alpha * (ori_b - ori_a)   # (n,3)
            eul_full[s : e + 1] = eul_seg

        traj_out[m, :, 3:] = torch.tensor(
            eul_full, dtype=traj_out.dtype, device=traj_out.device
        )
        all_eulers.append(eul_full)

    # ── save trajectories ────────────────────────────────────────────────
    save_sample_trajectories(
        os.path.join(folder, "trajectories_with_orientations"), traj_out
    )

    # ── overlay plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    labels = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    x = np.arange(N)

    for eul in all_eulers:
        if eul is None:
            continue
        for i, ax in enumerate(axes):
            ax.plot(x, eul[:, i], linewidth=1, alpha=0.8)

    for ax, lab in zip(axes, labels):
        ax.set_ylabel(lab)
        ax.grid(True)
    axes[-1].set_xlabel('Index')
    fig.suptitle('All Trajectories: Euler Angles vs Index', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = os.path.join(folder, "trajectories_with_orientations",
                             "euler_angles_overlay.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"saved  {plot_path}")

    return traj_out


import re

def apply_shifts_to_trajs(
    config,
    seed_idx_demo = None,
    cartesian_shift_amount = (0.0, 0.0, 0.0),
    indices_shift= (0,),
):
    """
    For every *.npy* trajectory in
        <iteration_working_dir>/<seed_idx_demo>/sample_trajectories/

    • Rename the current file  foo.npy  →  foo_shifted_<nextIdx>.npy
      where <nextIdx> is 1 + highest existing suffix for that base name.
    • Add cartesian_shift_amount to columns 0-2 at each row in indices_shift.
    • Save the shifted result back to the original name  foo.npy.
    """

    path = os.path.join(
        config.iteration_working_dir, str(seed_idx_demo), "trajectories_with_orientations","sample_trajectories"
    )
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    shift_vec = np.asarray(cartesian_shift_amount, dtype=np.float32).ravel()
    if shift_vec.size != 3:
        print("⚠️  cartesian_shift_amount must have exactly 3 numbers; nothing done.")
        return

    base_re = re.compile(r"^(?P<name>.+?)(?:_shifted_(?P<idx>\d+))?\.npy$")

    all_files  = [f for f in os.listdir(path) if f.endswith(".npy")]
    base_files = [f for f in all_files if "_shifted_" not in f]

    for fname in base_files:
        base, _ = os.path.splitext(fname)          # 'foo' from 'foo.npy'

        # -------- find the next backup index ---------------------------
        existing = [
            int(m.group("idx"))
            for f in all_files
            if (m := base_re.match(f))
            and m.group("name") == base
            and m.group("idx") is not None
        ]
        next_idx = max(existing, default=-1) + 1

        full_path = os.path.join(path, fname)
        traj = np.load(full_path)                  # load BEFORE renaming

        # -------- rename original to backup -----------------------------
        backup_name = f"{base}_shifted_{next_idx}.npy"
        os.rename(full_path, os.path.join(path, backup_name))
        print(f"original renamed → {backup_name}")

        # -------- apply the shift --------------------------------------
        for row in indices_shift:
            if row >= traj.shape[0]:
                print(f"⚠️  index {row} out of bounds for {fname}; skipping.")
                continue
            traj[row, :3] += shift_vec            # shift cols 0-2

        # -------- save shifted array back to original filename ----------
        np.save(full_path, traj)
        print(f"saved shifted array → {fname}")


"""
def apply_shifts_to_trajs(
    config,
    seed_idx_demo = None,
    cartesian_shift_amount = (0,0,0),
    indices_shift = (0,),
):
    

    path = os.path.join(
        config.iteration_working_dir, str(seed_idx_demo), "sample_trajectories"
    )
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    #shift_vec = np.asarray(cartesian_shift_amount, dtype=np.float32).reshape(1, 3)
    shift_vec = np.asarray(cartesian_shift_amount, dtype=np.float32).ravel()
    base_re   = re.compile(r"^(?P<name>.+?)(?:_shifted_(?P<idx>\d+))?\.npy$")

    all_files  = [f for f in os.listdir(path) if f.endswith(".npy")]
    base_files = [f for f in all_files if "_shifted_" not in f]

    for fname in base_files:
        base, _ = os.path.splitext(fname)              # without ".npy"

        # ---------- determine next shift index --------------------------
        existing = []
        for f in all_files:
            m = base_re.match(f)
            if m and m.group("name") == base and m.group("idx") is not None:
                existing.append(int(m.group("idx")))
        next_idx = max(existing, default=-1) + 1

        traj = np.load(os.path.join(path, fname))      # (N, ≥3)
        N    = traj.shape[0]

        for row in indices_shift:
            if row >= N:
                print(f"⚠️  index {row} out of bounds for {fname}; skipping.")
                continue

            shifted = traj.copy()
            shifted[row, :3] += shift_vec

            out_name = f"{base}_shifted_{next_idx}.npy"
            np.save(os.path.join(path, out_name), shifted)
            print(f"saved  {out_name}")
            next_idx += 1

"""

def generate_IVK_trajectories(config,discrete_gripper=False,gripper_pause=False,use_gripper_action_file=False,min_vals=None,max_vals=None,shift_mean=None,use_mean=False,use_gripper_orientation_file=False,gripper_provided=False,task=SlideBlockToTarget,limit_to_correction_indices=None,trajectory_indices = None,min_z=None):
    
    # Get the working directories we will be loading data in from
    working_dirs = config.working_dirs

    if limit_to_correction_indices is None:
        limit_to_correction_indices = config.seeds


    # Define common kwargs once, then reuse it:
    ivk_generation_kwargs = config.ivk_generation_kwargs

    full_trajectories = {"full_trajectories": [],
                            "successful":[]}

    #quaternion offsets
    #offset_by = config.execute_trajectory_kwargs.pop('offset_by')
    offset_by = config.execute_trajectory_kwargs['offset_by']

    #for seed_idx,working_dir in enumerate(working_dirs):
    for idx in limit_to_correction_indices:

        seed_idx = config.seeds[idx]
        working_dir = config.working_dirs[seed_idx]
        # Load the generated trajectoreis from the pickle file
        #with open(working_dir+'generated_trajectories.pkl', "rb") as f:
        #    trajs_generated = pickle.load(f)
        
        if use_mean==False:
            if gripper_provided==True:
                with open(working_dir+'generated_trajectories_numpy_with_gripper.pkl', "rb") as f:
                    trajs_generated = pickle.load(f)
                trajs_generated = torch.from_numpy(trajs_generated)
            else:
                with open(working_dir+'generated_trajectories.pkl', "rb") as f:
                    trajs_generated = pickle.load(f)


        #JUST USE MEAN FROM RAY TRACING
        if use_mean is True:
            with open(working_dir+'ray_tracing_results.pkl', "rb") as f:
                ray_tracing_results = pickle.load(f)
                trajs_generated = ray_tracing_results["mean_tor"]
                try:
                    kept_rows = ray_tracing_results["kept_rows"]
                except Exception as e:
                    print("No 'kept_rows' attribute in the ray tracing results pickle")
                    kept_rows = None

                if shift_mean is not None:
                    trajs_generated = trajs_generated + shift_mean
                trajs_generated = np.expand_dims(trajs_generated, axis=0)
                trajs_generated = torch.from_numpy(trajs_generated)
        else:
            kept_rows = None



        

        seed_of_working_dir = config.seeds[seed_idx]    

        #dictionary to hold:
        #   -full IVK traejctory for each end effector trajectory in trajs_geenrated
        #   -whether it succedded or failed
        full_trajectories_seed = {"full_trajectories": [],
                            "successful":[]}
        ivk_generation_kwargs["seed"]=seed_of_working_dir

        if trajectory_indices is None:
            trajectory_indices = list(range(trajs_generated.shape[0]))
            print("Number of Trajectories: " + str(len(trajectory_indices)))

        for traj_idx in range(0,len(trajectory_indices)):    
            
            i = trajectory_indices[traj_idx]
            #if i>0:
            #    break
            #create directory to save htis to
            save_dir = working_dir+"trajectory_"+str(i)+"/"
            os.makedirs(save_dir,exist_ok=True)
            # Update kwargs with iteration-specific parameters
            kwargs = {
                **ivk_generation_kwargs,
                'trajectory': trajs_generated[i].cpu().detach().numpy(),
                'working_dir': save_dir,
                'task':task,
                'cam_pose':config.execute_trajectory_kwargs["cam_pose"],
                "open_gripper":config.execute_trajectory_kwargs["open_gripper"]
            }

            
            #full_trajectory,success = execute_trajectory(**kwargs)
            # Execute the trajectory with a timeout

            full_trajectory, done,success = execute_trajectory(**kwargs,discrete_gripper=discrete_gripper,gripper_pause=gripper_pause,min_vals=min_vals,max_vals=max_vals,use_gripper_action_file=use_gripper_action_file,use_gripper_orientation_file=use_gripper_orientation_file,kept_rows=kept_rows,offset_by=offset_by[seed_idx])
            print(f"Trajectory {str(i)} executed successfully: {success}")

            #append the key information to the full_trajectories dictionary
            full_trajectories['full_trajectories'].append(full_trajectory)
            full_trajectories['successful'].append(success)
            full_trajectories_seed['full_trajectories'].append(full_trajectory)
            full_trajectories_seed['successful'].append(success)
        
        with open(working_dir+'full_ivk_trajectories.pkl', 'wb') as f:
            pickle.dump(full_trajectories_seed, f)
        print("Saved to: " + working_dir+'full_ivk_trajectories.pkl')
    
    with open(config.iteration_working_dir+'full_ivk_trajectories.pkl', 'wb') as f:
            pickle.dump(full_trajectories, f)


def generate_IVK_trajectories_from_file(config,gripper_pause=False,task=SlideBlockToTarget,cam_pose=None,limit_to_correction_indices=None,record_video=True,discrete_gripper=True,working_dir=None,video_filename="trajectory_execution.mp4"):
    
    # Get the working directories we will be loading data in from
    working_dirs = config.working_dirs

    if limit_to_correction_indices is None:
        limit_to_correction_indices = config.seeds

    #for seed_idx,working_dir in enumerate(working_dirs):
    for idx in limit_to_correction_indices:
        seed_idx = config.seeds[idx]
        working_dir = config.working_dirs[seed_idx]
        seed_of_working_dir = config.seeds[seed_idx]    


        full_trajectories_seed = {"full_trajectories": [],
                            "successful":[]}

        with open(working_dir+'full_ivk_trajectories.pkl', "rb") as f:
            full_trajs = pickle.load(f)
        
        trajectory = full_trajs["full_trajectories"][0].actions

        full_trajectory, done,success = execute_trajectory_from_file(trajectory,gripper_pause=gripper_pause,cam_pose=cam_pose,record_video=record_video,seed=seed_of_working_dir,task=task,discrete_gripper=discrete_gripper,working_dir=working_dir,video_filename=video_filename)
        print(f"Trajectory {str(idx)} executed successfully: {success}")

        #append the key information to the full_trajectories dictionary
        full_trajectories_seed['full_trajectories'].append(full_trajs["full_trajectories"][0])
        full_trajectories_seed['successful'].append(success)
        
        with open(working_dir+'full_ivk_trajectories.pkl', 'wb') as f:
            pickle.dump(full_trajectories_seed, f)
            print("Saved to: " + working_dir+'full_ivk_trajectories.pkl')




def run_ivk_for_corrections(config,task=SlideBlockToTarget,limit_to_correction_indices=None):
    #This function generates the full IVK trajectoris for all the corrections made in this iteration
    from utils.Classes.preference_database import Correction
    working_dirs = config.working_dirs

    
    with open(config.path_to_current_trajectories,'rb') as file:
        current_policy_trajectories = pickle.load(file)

    if limit_to_correction_indices is None:
        #current_policy_trajectories = [current_policy_trajectories[i] for i in limit_to_correction_indices]
        #working_dirs = 
        limit_to_correction_indices = list(range(len(working_dirs)))

    
    execute_trajectory_kwargs = config.execute_trajectory_kwargs
    all_corrections = [] #list to hold all corrections

    #quaternion offsets
    offset_by = execute_trajectory_kwargs.pop('offset_by')

    for traj_index in limit_to_correction_indices:
        working_dir = working_dirs[traj_index]
        with open(working_dir+"generated_trajectories.pkl", 'rb') as file:
            trajectories = pickle.load(file)
        
        starting_env_state = current_policy_trajectories[traj_index].environment_states[config.frame_correction_indices[traj_index]]
        execute_trajectory_kwargs["starting_env_state"] = starting_env_state
        execute_trajectory_kwargs["seed"] = current_policy_trajectories[traj_index].env_seed
        count = 0
        
        #Create the creaction object for this correction
        correction_object = Correction(current_policy_trajectory=current_policy_trajectories[traj_index])

        #Set the offset_by parameter in the execture trajectory kwargs


        for idx, trajectory in enumerate(trajectories):
            #print("x")
            #execute_trajectory_kwargs["trajectory"]=trajectory.cpu().numpy()
            execute_trajectory_kwargs["video_filename"] = "correction"+str(traj_index)+"_trajectory"+str(count)+".mp4"
            execute_trajectory_kwargs["trajectory"] = trajectory.detach().cpu().numpy()
            correction_working_dir = os.path.join(working_dir, f"corrected_trajectory_{idx}")
            os.makedirs(correction_working_dir, exist_ok=True)
            execute_trajectory_kwargs["working_dir"]=correction_working_dir
            trajectory_data,done,success = execute_trajectory(task=task,**execute_trajectory_kwargs,offset_by=offset_by[traj_index])
            
            #Apppend to our correction object
            correction_object.append_corrections_list(trajectory_data,success_status=success)
            count+=1
            # Full file path to save the trajectory to (with the correct name)
            file_path = os.path.join(correction_working_dir, 'trajectory_data.pkl')
            #print(trajectory_data.video_path)

            # Pickle the variable
            with open(file_path, 'wb') as f:
                pickle.dump(trajectory_data, f)
        #Append the correcion object to the list
        all_corrections.append(correction_object)
        #Save the corrections object to the working_dir path
        file_path = os.path.join(working_dir,"original_trajectory_and_corrections.pkl")
        with open(file_path,'wb') as f:
            pickle.dump(correction_object,f)
    
    #Save all corrections
    save_dir = config.iteration_working_dir
    file_path = os.path.join(save_dir,"all_corrections.pkl")
    with open(file_path,'wb') as f:
        pickle.dump(all_corrections,f)

def run_ivk_for_corrections_DEPRECATED(config,task=SlideBlockToTarget,limit_to_correction_indices=None):
    #This function generates the full IVK trajectoris for all the corrections made in this iteration
    from utils.Classes.preference_database import Correction
    working_dirs = config.working_dirs

    with open(config.path_to_current_trajectories,'rb') as file:
        current_policy_trajectories = pickle.load(file)

    if limit_to_correction_indices is not None:
        current_policy_trajectories = [current_policy_trajectories[i] for i in limit_to_correction_indices]

    
    execute_trajectory_kwargs = config.execute_trajectory_kwargs
    all_corrections = [] #list to hold all corrections

    #quaternion offsets
    offset_by = execute_trajectory_kwargs.pop('offset_by')

    for traj_index in range(0,len(working_dirs)):
        working_dir = working_dirs[traj_index]
        with open(working_dir+"generated_trajectories.pkl", 'rb') as file:
            trajectories = pickle.load(file)
        
        starting_env_state = current_policy_trajectories[traj_index].environment_states[config.frame_correction_indices[traj_index]]
        execute_trajectory_kwargs["starting_env_state"] = starting_env_state
        execute_trajectory_kwargs["seed"] = current_policy_trajectories[traj_index].env_seed
        count = 0
        
        #Create the creaction object for this correction
        correction_object = Correction(current_policy_trajectory=current_policy_trajectories[traj_index])

        #Set the offset_by parameter in the execture trajectory kwargs


        for idx, trajectory in enumerate(trajectories):
            #print("x")
            #execute_trajectory_kwargs["trajectory"]=trajectory.cpu().numpy()
            execute_trajectory_kwargs["video_filename"] = "correction"+str(traj_index)+"_trajectory"+str(count)+".mp4"
            execute_trajectory_kwargs["trajectory"] = trajectory.detach().cpu().numpy()
            correction_working_dir = os.path.join(working_dir, f"corrected_trajectory_{idx}")
            os.makedirs(correction_working_dir, exist_ok=True)
            execute_trajectory_kwargs["working_dir"]=correction_working_dir
            trajectory_data,done = execute_trajectory(task=task,**execute_trajectory_kwargs,offset_by=offset_by[traj_index])
            
            #Apppend to our correction object
            correction_object.append_corrections_list(trajectory_data,success_status=done)
            count+=1
            # Full file path to save the trajectory to (with the correct name)
            file_path = os.path.join(correction_working_dir, 'trajectory_data.pkl')

            # Pickle the variable
            with open(file_path, 'wb') as f:
                pickle.dump(trajectory_data, f)
        #Append the correcion object to the list
        all_corrections.append(correction_object)
        #Save the corrections object to the working_dir path
        file_path = os.path.join(working_dir,"original_trajectory_and_corrections.pkl")
        with open(file_path,'wb') as f:
            pickle.dump(correction_object,f)
    
    #Save all corrections
    save_dir = config.iteration_working_dir
    file_path = os.path.join(save_dir,"all_corrections.pkl")
    with open(file_path,'wb') as f:
        pickle.dump(all_corrections,f)



