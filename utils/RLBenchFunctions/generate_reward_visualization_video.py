from utils.RLBenchFunctions.train_preference_model import train_preference_ensemble
import pickle
from utils.RLBenchFunctions.custom_envs import instantiate_environment
from utils.RLBenchFunctions.preference_database_utils import generate_preference_database
import torch
import numpy as np
from pyrep.objects.vision_sensor import VisionSensor
from utils.RLBenchFunctions.template_sensor_views import compute_camera_pose
from pyrep.const import RenderMode
import os
import cv2


def generate_reward_visualization(config,episode_length=200,number_to_generate=5,random_seeds=True):
    # Load in the config file
    #path_to_config_pickle = "run_results/slide_block_to_target/demos/config_instance.pkl"
    #with open(path_to_config_pickle, 'rb') as file:
    #    config = pickle.load(file)

    #with open("run_results/slide_block_to_target/demos/0/full_ivk_trajectories.pkl","rb") as f:
    #    full_trajectories = pickle.load(f)

    #trajectory = full_trajectories['full_trajectories'][0]
    #actions = trajectory.actions

    #load in the reward model
    path_to_reward = os.path.join(config.iteration_working_dir, "reward_model.pt")
    reward_model = torch.load(path_to_reward)
    reward_model.eval()  # Optional: switch to eval mode

    #load in current trajectories
    path_to_trajectories = config.path_to_current_trajectories
    with open(path_to_trajectories, "rb") as f:
        current_trajectories = pickle.load(f)

    generated = 0

    for traj_idx in range(len(current_trajectories)):
        traj = current_trajectories[traj_idx]
        #env_start_state = traj.environment_states[config.frame_correction_indices[traj_idx]]
        env = instantiate_environment(pref_model=reward_model) #,starting_env_state=env_start_state)
        if random_seeds is True:
            seed=np.random.randint(0, 200)
        else:
            seed = traj.env_seed
        env.reset(seed=seed)

        resolution = [1200,1200]
        vision_sensor = VisionSensor.create(resolution)
        cam_pose = compute_camera_pose([1,0,1.5])
        vision_sensor.set_pose(cam_pose)

        # Video recording setup
        frames = []

        actions = traj.actions
        for action in actions:
            # Generate a random action within bounds
            #action = np.random.uniform(env.action_space.low, env.action_space.high)
            #action = actions[j]
            obs, reward, terminated, truncated, info = env.step(action)

            frame = vision_sensor.capture_rgb()
            frame = (frame * 255).astype(np.uint8)

            # Convert to BGR for OpenCV (once!)
            #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = frame

            # Overlay text in upper-right corner onto frame_bgr
            text = f"Reward: {reward:.4f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)  # white
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size
            position = (frame_bgr.shape[1] - text_width - 10, text_height + 10)
            cv2.putText(frame_bgr, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

            # Now append the correctly modified frame_bgr
            frames.append(frame_bgr)

        #video_path = os.path.join(working_dir, video_filename)
        #video_path = "visualize_reward.mp4"
        video_path = os.path.join(config.iteration_working_dir, "visualize_reward_"+str(traj_idx)+".mp4")
        frame_rate = 4
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video.release()
        print(f"Video saved at {video_path}")

        env.close()

        #append the generated list
        generated+=1
        if generated>number_to_generate:
            break