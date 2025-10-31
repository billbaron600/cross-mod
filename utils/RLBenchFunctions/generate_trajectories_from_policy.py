from utils.PreferenceModelFunctions.preference_transformer import PreferenceTransformer
from rlbench.gym import RLBenchEnv
from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds
from rlbench.action_modes.arm_action_modes import JointVelocity,JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from stable_baselines3 import PPO
import cv2
from pyrep.objects.vision_sensor import VisionSensor
from  utils.RLBenchFunctions.template_sensor_views import compute_camera_pose
import numpy as np
import pickle
import os
from utils.RLBenchFunctions.template_sensor_views import cam_pose_to_transform
import torch

def create_next_episode_dir(base_dir):
    """
    Creates a new directory with an incremented numerical name.

    Args:
        base_dir (str): The directory in which to look for existing directories.

    Returns:
        str: The path to the newly created directory.
    """
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    existing = [name for name in os.listdir(base_dir) if name.isdigit()]

    numbers = [int(name) for name in existing]

    episode_index = max(numbers) + 1 if numbers else 1

    new_dir = os.path.join(base_dir, f"{episode_index}")
    os.makedirs(new_dir, exist_ok=True)

    return new_dir


def generate_trajectories_from_policy(task=None,model_path=None,action_mode=None,episodes=5,max_iterations=100,record_videos=False,resolution=[1200,1200],camera_positions=None,render_mode='rgb_array',output_path="",seed=0,frame_rate=20):
    #Step 1: Instantaite the environemnt
    if action_mode is None:
        # Instantiate action modese
        arm_action_mode = JointVelocity()  
        gripper_action_mode = Discrete()  

        # Create custom action mode instance
        action_mode = MoveArmThenGripperWithBounds(arm_action_mode, gripper_action_mode)
    

    # Create the base environment
    env = RLBenchEnv(task,observation_mode='state',action_mode=action_mode,render_mode=render_mode)
    
    # Enable shaped rewards
    env.rlbench_task_env._shaped_rewards = True
    env.reset()

    #Step 2: Load in teh model
    model = PPO.load(model_path, env=env)

    #Step 3: If record_videos is True, create PyRep visison for each pose in camera_views
    sensors = []
    if record_videos==True:
        if camera_positions==None:
            camera_position1 = compute_camera_pose(np.array((2, 0, 1))) #,center_point=np.array([0.3,0,0.7]))
            camera_position2 = compute_camera_pose(np.array((0.35, 0, 2)),center_point=np.array([0.3,0,1]))
            camera_position3 = compute_camera_pose(np.array((-1.5,-0.5,1)),center_point=np.array([0.3,0,1]))
            camera_position4 = compute_camera_pose(np.array((-0.5, 1, 0.8)))
            camera_positions = [camera_position1,camera_position2,camera_position3,camera_position4]
        for cam_pose in camera_positions:
            vision_sensor = VisionSensor.create(resolution)
            vision_sensor.set_pose(cam_pose)
            sensors.append(vision_sensor)



    #Step 4: Simulate "episodes" using the current policy
    trajectories = []
    all_frames = []
    base_path = output_path
    for ep in range(episodes):
        obs, info = env.reset(seed=seed)
        done = False
        iteration_count = 0
        all_frames = [[] for _ in sensors]
        output_path = create_next_episode_dir(base_path)

        #record the full trajectory data (after IVK)
        trajectory_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'ee_positions': [],
        'dones': []}
        print("Running Episode:")
        print(ep)
        while not done:
            # Predict the next action from the PPO policy
            action, _ = model.predict(obs, deterministic=False)
            
            # Step in the RLBench environment
            next_obs, reward, done, truncated, info = env.step(action)
            tip = env.rlbench_task_env._robot.arm.get_tip()
            
            # Record trajectory data
            trajectory_data['observations'].append(obs)
            trajectory_data['actions'].append(action)
            trajectory_data['rewards'].append(reward)
            trajectory_data['ee_positions'].append(tip.get_position())
            trajectory_data['dones'].append(done)

            obs = next_obs
            iteration_count += 1

            if record_videos==True:
                for idx,sensor in enumerate(sensors):
                    frame = sensor.capture_rgb()
                    frame = (frame * 255).astype(np.uint8)
                    frame_bgr = frame

                    # Overlay text in upper-right corner onto frame_bgr
                    text = f"Waypoint: {iteration_count}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    color = (255, 255, 255)  # white
                    thickness = 2
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_width, text_height = text_size
                    position = (frame_bgr.shape[1] - text_width - 10, text_height + 10)
                    cv2.putText(frame_bgr, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

                    # Now append the correctly modified frame_bgr
                    all_frames[idx].append(frame_bgr)
            #print(iteration_count)
            if iteration_count > max_iterations:
                break
        #save the trajectories in a pkl
        traj_path = os.path.join(output_path, "ppo_trajs.pkl")
        trajectory_data["seed"] = seed
        with open(traj_path,"wb") as file:
            pickle.dump(trajectory_data,file)

        if record_videos==True:
            
            for idx,frames in enumerate(all_frames):
                video_path = os.path.join(output_path, str(idx)+"_policy_video"+".mp4")
                height, width, _ = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
            
                for frame in frames:
                    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
                video.release()
                print(f"Video saved at {video_path}")
            #pickle the all_frames object
            episode_pickle_file_name = os.path.join(output_path, "all_video_frames_episode"+str(ep)+".pkl")
            with open(episode_pickle_file_name,"wb") as file:
                pickle.dump(all_frames,file)
            
            
            #save the extrsincis of the cameras
            extrinsic_list = []
            for sensor in sensors:
                # Get the current pose of the sensor directly from PyRep
                cam_pose = sensor.get_pose()  # This returns [x, y, z, qx, qy, qz, qw]
                
                # Convert to transformation matrix using your function
                T = cam_pose_to_transform(cam_pose)  # returns a numpy array of shape (4,4)
                extrinsic_list.append(torch.tensor(T, dtype=torch.float32))

            # Stack them into a single tensor of shape [4, 4, 4]
            extrinsics_tensor = torch.stack(extrinsic_list)  # shape: (4, 4, 4)

            # Save the extrinsics tensor in a file.
            torch.save({
            "extrinsics": extrinsics_tensor  # T_world<-camera for each sensor, shape [4, 4, 4]
            }, os.path.join(output_path,"poses_mobile.tar"))


            
            
        #Add this episodes trajectory to the trajectories list
        trajectories.append(trajectory_data)

    env.close()