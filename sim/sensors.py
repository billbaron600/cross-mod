# refactored module (auto-generated)


# ---- imports from original modules ----
from PIL import Image

from pyrep.const import RenderMode

from pyrep.objects.vision_sensor import VisionSensor

from rlbench.gym import RLBenchEnv

from rlbench.tasks import SlideBlockToTarget

from scipy.spatial.transform import Rotation

from scipy.spatial.transform import Rotation as R

from stable_baselines3 import PPO

from utils.RLBenchFunctions.custom_envs import instantiate_environment

import matplotlib.pyplot as plt

import numpy as np

import os

import pybullet as p  # For computeViewMatrix

import torch



def save_camera_images(angles, save_folder=None,save_prefix="camera_angle"):
    """
    Takes a list of camera angle images and saves them as PNG files.
    
    Parameters:
    - angles (list of np.array): List of images captured from different camera angles.
    - save_prefix (str): Prefix for saving image files (default: "camera_angle").
    """
    for i, angle in enumerate(angles):
        # Convert float32 to uint8
        angle_uint8 = (angle * 255).astype(np.uint8)
        
        # Convert to PIL and save
        img = Image.fromarray(angle_uint8)
        img.save(save_folder + f"{save_prefix}_{i}.png")
        print(f"Saved {save_prefix}_{i}.png")

def cam_pose_to_transform(cam_pose):
    """
    Convert a 7-element camera pose [x, y, z, qx, qy, qz, qw] to a 4x4 
    homogeneous transformation matrix representing the camera-to-world transform.
    
    This version accounts for the fact that a PyRep VisionSensor looks along the -Z axis.
    """
    pos = np.array(cam_pose[:3], dtype=np.float32)
    quat = np.array(cam_pose[3:], dtype=np.float32)  # [qx, qy, qz, qw]
    
    # Get the rotation matrix from the quaternion.
    rot_matrix = R.from_quat(quat).as_matrix()  # 3x3
    
    # Correction: Flip the z-axis.
    flip_z = np.diag([1, -1, -1])
    #flip_z = np.diag([-1, 1, 1])
    rot_matrix = rot_matrix @ flip_z  # Now the "forward" direction is flipped.
    
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot_matrix
    T[:3, 3] = pos
    return T

def compute_camera_pose(new_position, center_point=np.array([0,0,1])):
    """
    Same as before: compute a 7D [pos + quaternion] that looks at `center_point`.
    """
    from transforms3d.quaternions import mat2quat
   
    forward_vector = center_point - new_position
    forward_vector = forward_vector / np.linalg.norm(forward_vector)

    world_up = np.array([0, 0, 1])
   
    new_right = np.cross(world_up, forward_vector)
    new_right = new_right / np.linalg.norm(new_right)
   
    new_up = np.cross(forward_vector, new_right)
    new_up = new_up / np.linalg.norm(new_up)
   
    R_new = np.column_stack((new_right, new_up, forward_vector))
   
    quat_wxyz = mat2quat(R_new)
    quat_new = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
   
    return np.concatenate((new_position, quat_new))

def generate_camera_view(env=None,camera_positions=None, task=SlideBlockToTarget,frames=1, environment_information=None,resolution=[1200,1200], seed=0,start_pose=None,save_folder=None,robot_starting_position=None,correction_index=None,save_with_seed_name=True):
    """
    Generate camera views with RLBench/PyRep, and store data in 'poses_mobile.tar'
    similarly to your first (pure PyBullet) code.

    We'll store:
      - 'view_matrices':  The 16-float PyBullet "view matrix" (T_{camera←world}) for each sensor
      - 'extrinsics':     The 4x4 matrix T_{world←camera} (the inverse of the view matrix,
                          matching what you did in your first code with pose_mat).
      - 'intrinsics':     The 3x3 pinhole intrinsics from PyRep’s sensor


    """

    if env is None:
        env = instantiate_environment(task=task,seed=seed)

    #if not camera positions are set by user, set them here
    # If no camera positions are set by the user, set them here
    if camera_positions is None:
        camera_position1 = compute_camera_pose(np.array([2, 1.25, 1.25]))  # No center_point for this one
        camera_position2 = compute_camera_pose(np.array([2, -1.25, 1.25]), center_point=np.array([0.3, 0, 1]))
        camera_position3 = compute_camera_pose(np.array([-1, -1.25, 1.5]), center_point=np.array([0.3, 0, 1]))
        camera_position4 = compute_camera_pose(np.array([-1, 1.25, 1.5]),center_point=np.array([0.3, 0, 1]))  # No center_point for this one
        camera_positions = [camera_position1, camera_position2, camera_position3, camera_position4]

    #create the RLBench environment
    #env = RLBenchEnv(task,**environment_information)
    #turn on reward shaping
    #env.rlbench_task_env._shaped_rewards = True
    #env.rlbench_task_env._randomize_every_episode = False
    #set the seed
    #env.rlbench_task_env._variation_index = seed

    # Start an episode.
    #obs, info = env.reset(seed=seed)
    #done = False
    iter_counter = 0

    # Explicitly set robot starting positions right after reset
    #if robot_starting_position is not None:
    #    env.rlbench_task_env._robot.arm.set_joint_positions(robot_starting_position)
    #    env.rlbench_task_env._robot.arm.set_joint_target_positions(robot_starting_position)

        # Step the simulator to update physics and visuals
    #    for _ in range(10):
    #        env.rlbench_task_env._scene.step()

    # Create a list of VisionSensor objects from the provided camera positions.
    sensors = []
    intrinsic_mats = []
    dummy_sensor_pose_1 = compute_camera_pose(np.array([2, 1.25, 1.25]))
    dummy_sensor = VisionSensor.create(resolution, perspective_mode=True)
    dummy_sensor.set_pose(dummy_sensor_pose_1)
    

    for i, cam_pose in enumerate(camera_positions):
        sensor = VisionSensor.create(resolution, perspective_mode=True)
        sensor.set_pose(cam_pose)  # pose is [x,y,z, qx,qy,qz,qw]
        sensors.append(sensor)
        # The 3x3 intrinsics:
        K = sensor.get_intrinsic_matrix()
        intrinsic_mats.append(torch.tensor(K, dtype=torch.float32))

        # Just printing to confirm:
        #fx = K[0, 0]
        #fy = K[1, 1]
        #print(f"Sensor {i}: Focal length (fx, fy) = ({fx:.2f}, {fy:.2f})")

    num_sensors = len(sensors)

    # For each sensor, store captured frames.
    frames_all = [[] for _ in range(num_sensors)]
    done=False

    while not done and iter_counter < frames:
        if start_pose is not None:
            joint_positions = env.rlbench_env._scene.robot.arm.get_joint_positions()
            gripper_positions = env.rlbench_env._scene.robot.gripper.get_joint_positions()[1]
            combined_action = np.array(joint_positions + [gripper_positions])
            result = env.step(combined_action)
        else:
            env.rlbench_task_env._scene.step()
        # Capture frames
        for idx, sensor in enumerate(sensors):
            frame = sensor.capture_rgb()  # shape [H, W, 3] float32 in [0,1]
            frames_all[idx].append(frame)
        
        #obs = next_obs
        iter_counter += 1

    #save the extrinsics of the sensors
    #extrinsic_list = []
    #for cam_pose in camera_positions:
    #    T = cam_pose_to_transform(cam_pose)  # returns a numpy array of shape (4,4)
    #    extrinsic_list.append(torch.tensor(T, dtype=torch.float32))
    # Save the extrinsics of the sensors
    extrinsic_list = []
    for sensor in sensors:
        # Get the current pose of the sensor directly from PyRep
        cam_pose = sensor.get_pose()  # This returns [x, y, z, qx, qy, qz, qw]
        
        # Convert to transformation matrix using your function
        T = cam_pose_to_transform(cam_pose)  # returns a numpy array of shape (4,4)
        extrinsic_list.append(torch.tensor(T, dtype=torch.float32))

    # Stack them into a single tensor of shape [4, 4, 4]
    extrinsics_tensor = torch.stack(extrinsic_list)  # shape: (4, 4, 4)

    #get the name of the task here
    task_name = env.rlbench_task_env._task.get_name()
    seed_name = str(seed)
    #save_folder = save_folder + task_name + "/" + seed_name + "/"
    if save_with_seed_name:
        save_folder = save_folder + "/" + seed_name + "/"
    elif correction_index is not None:
        save_folder = save_folder + "/" + "correction_"+str(correction_index)+"/"

    os.makedirs(save_folder,exist_ok=True)

    # Save the extrinsics tensor in a file.
    torch.save({
    "extrinsics": extrinsics_tensor,  # T_world<-camera for each sensor, shape [4, 4, 4]
    "intrinsics": intrinsic_mats}, save_folder+"poses_mobile.tar")

    #close the RLBench Env
    env.close()

    # Return the first frame from each sensor
    first_frames = [frames_list[0] for frames_list in frames_all if len(frames_list) > 0]
    
    #save the first frames
    save_camera_images(first_frames, save_folder=save_folder,save_prefix="camera_angle")
    

    #return the first frames, extrincis tensor, and folder where everything is saed
    return first_frames,extrinsics_tensor, save_folder
