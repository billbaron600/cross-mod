# Import necessary libraries
from rlbench.gym import RLBenchEnv
from rlbench.tasks import SlideBlockToTarget,ReachTarget
from stable_baselines3 import SAC,PPO
from rlbench.action_modes.arm_action_modes import JointPosition,EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds
import numpy as np
import torch
from pyrep.backend import sim
from time import sleep
from scipy.spatial.transform import Rotation as R
from utils.RLBenchFunctions.custom_envs import MaxStepWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import secrets


import cv2
import numpy as np
from time import sleep
from pyrep.backend import sim
from pyrep.objects.vision_sensor import VisionSensor
from  utils.RLBenchFunctions.template_sensor_views import compute_camera_pose
from utils.RLBenchFunctions.custom_envs import instantiate_environment
from utils.Classes.policy_trajectory_class import PolicyTrajectory
import os
import re
from utils.RLBenchFunctions.custom_envs import instantiate_environment
import pickle
from utils.RLBenchFunctions.generate_IVK_trajectory import add_trajectory_line
from datetime import datetime
import copy
import numpy as np
from collections import OrderedDict
import numpy as np
from typing import Any, Dict
from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds, EndEffectorPoseViaPlanning_Custom, MoveArmThenGripperWithBoundsDelta
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK,EndEffectorPoseViaPlanning, JointPosition
from rlbench.gym import RLBenchEnv
from rlbench.action_modes.gripper_action_modes import Discrete,GripperJointPosition


# ---- desired key order -----------------------------------------------------
_LOW_DIM_KEYS = [
    "gripper_joint_positions",
    "gripper_open",
    "gripper_pose",
    "gripper_touch_forces",
    "joint_forces",
    "joint_positions",
    "joint_velocities",
    "task_low_dim_state",
]




def low_dim_observation_to_dict(obs) -> Dict[str, Any]:
    """
    Convert an RLBench `Observation` into an OrderedDict whose keys
    appear in the precise order required for your policy input.

    * Scalars (gripper_open) are wrapped into shape-(1,) arrays.
    * Keys that are `None` in the source observation are skipped.

    Returns
    -------
    OrderedDict
        Keys in the order:
        ['gripper_joint_positions', 'gripper_open', 'gripper_pose',
         'gripper_touch_forces', 'joint_forces', 'joint_positions',
         'joint_velocities', 'task_low_dim_state']
    """
    out = OrderedDict()

    # ---- desired key order -----------------------------------------------------
    _LOW_DIM_KEYS = [
        "gripper_joint_positions",
        "gripper_open",
        "gripper_pose",
        "gripper_touch_forces",
        "joint_forces",
        "joint_positions",
        "joint_velocities",
        "task_low_dim_state",
    ]

    for key in _LOW_DIM_KEYS:
        value = getattr(obs, key, None)
        if value is None:
            continue
        # Wrap scalar gripper_open into (1,) array
        if key == "gripper_open" and np.isscalar(value):
            value = np.array([value], dtype=np.float32)
        else:
            value = np.asarray(value, dtype=np.float32)
        out[key] = value

    return out



def evaluate_policy(
    model, 
    env=None,
    close_env=True, 
    task=None,
    safety_factor = 1.0,
    print_rewards = False,
    num_episodes=5,
    deterministic=True,
    max_steps=200,
    shaped_rewards=True,
    render_mode='rgb_array',
    print_info=False,
    record_video=False,
    video_path=None,
    seed=None,
    seeds=None,
    show_traj_line = True,
    fps=1,
    min_vals=None,
    max_vals=None
):
    """
    Evaluate a trained policy.

    If record_video=True, each frame will be recorded via a PyRep VisionSensor
    and written to an OpenCV video at `video_path` (by default, .m4a).
    """
    
    """
    # Create an RLBench environment with our desired action mode.
    action_mode = MoveArmThenGripperWithBoundsDelta(
        #arm_action_mode=EndEffectorPoseViaPlanning_Record(),
        arm_action_mode = JointPosition(absolute_mode=False),
        gripper_action_mode=GripperJointPosition()
    )

    env.close()
    env = RLBenchEnv(
        task_class=ReachTarget,
        observation_mode='state',
        render_mode="human",
        #render_mode="human",
        action_mode=action_mode)
    model.set_env(env)
    """
    env.rlbench_task_env._shaped_rewards = False
    result = env.reset(seed=seed)

    if record_video and video_path is None:
        raise ValueError("If record_video is True, you must provide a video_path.")
    
    video_path = os.path.join(video_path,"policy_videos", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(video_path,exist_ok=True)

    if task is None:
        task = SlideBlockToTarget

    #if env is None:
        # Construct environment
    #    env = instantiate_environment(safety_factor=safety_factor,task=task,max_steps=max_steps,shaped_rewards=shaped_rewards)
    
    #model.set_env(env)


    # Optional video recording setup
    if record_video:
        # Create/OpenCV VideoWriter
        # Adjust the camera name below to match a camera in your RLBench scene
        cinematic_cam = VisionSensor.create([1200,1200])  # Example name; must exist in scene
        w, h = cinematic_cam.get_resolution()
        cam_pose = compute_camera_pose([1,0,1.5])
        cinematic_cam.set_pose(cam_pose)

        # Decide on FPS (either from simulation time step or a fixed number)
        #sim_dt = sim.simGetSimulationTimeStep()  # e.g. 0.05 => 20 FPS
        #fps = int(round(1.0 / sim_dt))
        #fps defaults to 20
        
        #if fps <= 0:
            #fps = 20  # fallback if the sim time step is large

        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Save as .m4a per your request (but normally you'd do .mp4 or .avi)
        #frame_size = (int(w), int(h))          # cast/convert just in case
        #fourcc      = cv2.VideoWriter_fourcc(*'mp4v')   # example codec
        #fps         = 1.0                     # or whatever you need
        #video_writer = cv2.VideoWriter(ep_path, fourcc, fps, frame_size)
        #cinematic_cam.set_pose(compute_camera_pose(np.array((1.5, 0, 2))))

    episode_rewards = []
    successes = 0
    trajectories = []

    
    if seeds is not None:
        num_episodes = len(seeds)
    else:
        num_episodes = num_episodes

        
    for episode in range(num_episodes):
        
        if seeds is None:
            seed = np.random.randint(1, 21)
        else:
            seed = seeds[episode]
            #seed_use = np.random.randint(0,501)
        #else:
        #    seed = seed
        
        seed_use = seed
        
        result = env.reset(seed=seed_use)
        obs = result[0]
        #first_arm_action      = np.zeros(7)
        #first_action = np.concatenate([first_arm_action, [0.04 * 1]])
        #obs, _, _, _, _ = env.step(first_action)
        #obs = result[0]
        #obs = result[0]
        done = False
        episode_reward = 0
        step_count = 0
        current_trajectory = PolicyTrajectory()
        current_trajectory.env_seed = seed_use
        #print(current_trajectory.env_seed)
        total_trajectory_reward = 0 # this is will hold the cumulative reward for this trajectory

        if record_video:
            # Optional video recording setup

            # Create/OpenCV VideoWriter
            # Adjust the camera name below to match a camera in your RLBench scene
            cinematic_cam = VisionSensor.create([1200,1200])  # Example name; must exist in scene
            w, h = cinematic_cam.get_resolution()
            cam_pose = compute_camera_pose([1,0,1.5])
            cinematic_cam.set_pose(cam_pose)
            ep_name = "traj_{:02d}_seed_{}.mp4".format(episode, seed_use)
            ep_path = os.path.join(video_path, ep_name)
            print(ep_path)

            frame_size = (int(w), int(h))          # cast/convert just in case
            fourcc      = cv2.VideoWriter_fourcc(*'mp4v')   # example codec
            fps         = 1.0                     # or whatever you need

            video_writer = cv2.VideoWriter(ep_path, fourcc, fps, frame_size)
            current_trajectory.video_path = ep_path      # ⇦ record the path on the object

            # Capture an RGB image from our chosen camera
            rgb = cinematic_cam.capture_rgb()
            # Convert [0,1] float array to [0,255] uint8
            rgb = (rgb * 255).astype(np.uint8)
            # OpenCV uses BGR instead of RGB
            bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # Add text to the frame
            text = "Episode: " + str(episode) + " Step: " + str(step_count)  # replace with your desired text
            position = (10, 30)  # (x, y) coordinates for the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (255, 255, 255)  # White color in BGR
            thickness = 2
            cv2.putText(bgr_frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

            # Calculate height of the first line
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_height = text_size[1]

            # Add second line of text just under the first one
            seed_text = "Seed: " + str(seed_use)
            seed_position = (10, position[1] + text_height + 10)  # 10px gap below the first line
            cv2.putText(bgr_frame, seed_text, seed_position, font, font_scale, color, thickness, cv2.LINE_AA)
            video_writer.write(bgr_frame)

        if print_info:
            print(f"\nStarting episode {episode + 1}/{num_episodes}")

        
        while not done:
            #get the current obs
            #current_obs = env.extract_obs(obs)
            current_obs = obs
            #observation = {}
            #for key in current_obs.keys():
            #    observation[key] = current_obs[key]
            #current_obs = low_dim_observation_to_dict(env.rlbench_env._scene.get_observation())
            # Get action from the policy
            #if first_step==True:
            #    action = np.zeros(8)
            #    first_step=False
            #else:
            
            # Example: Force the last dimension to 1.0 if that's your "open gripper" action
            #action[-1] = 1.0
            #print(action)

            # Step the environment
            for i in range(1):
                try:
                    if record_video is True:
                        # Capture an RGB image from our chosen camera
                        
                        rgb = cinematic_cam.capture_rgb()
                        # Convert [0,1] float array to [0,255] uint8
                        rgb = (rgb * 255).astype(np.uint8)
                        # OpenCV uses BGR instead of RGB
                        bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        # Add text to the frame
                        text = "Episode: " + str(episode) + " Step: " + str(step_count)  # replace with your desired text
                        position = (10, 30)  # (x, y) coordinates for the text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        color = (255, 255, 255)  # White color in BGR
                        thickness = 2
                        cv2.putText(bgr_frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

                        # Calculate height of the first line
                        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                        text_height = text_size[1]

                        # Add second line of text just under the first one
                        seed_text = "Seed: " + str(seed_use)
                        seed_position = (10, position[1] + text_height + 10)  # 10px gap below the first line
                        cv2.putText(bgr_frame, seed_text, seed_position, font, font_scale, color, thickness, cv2.LINE_AA)
                        video_writer.write(bgr_frame)

                    action, _ = model.predict(current_obs, deterministic=deterministic)
                    if min_vals is not None:
                        action[:3] = np.maximum(action[:len(min_vals)], min_vals)      # any value below its min is raised to that min
                        action[:3] = np.minimum(action[:len(max_vals)], max_vals)  # any value above its max is clipped down

                    #print(action[:3])

                    if record_video is True:
                        # Capture an RGB image from our chosen camera
                        #env.rlbench_task_env.pyrep.step()
                        env.rlbench_env._scene.pyrep.step()
                        rgb = cinematic_cam.capture_rgb()
                        # Convert [0,1] float array to [0,255] uint8
                        rgb = (rgb * 255).astype(np.uint8)
                        # OpenCV uses BGR instead of RGB
                        bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        # Add text to the frame
                        text = "Episode: " + str(episode) + " Step: " + str(step_count)  # replace with your desired text
                        position = (10, 30)  # (x, y) coordinates for the text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        color = (255, 255, 255)  # White color in BGR
                        thickness = 2
                        cv2.putText(bgr_frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

                        # Calculate height of the first line
                        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                        text_height = text_size[1]

                        # Add second line of text just under the first one
                        seed_text = "Seed: " + str(seed_use)
                        seed_position = (10, position[1] + text_height + 10)  # 10px gap below the first line
                        cv2.putText(bgr_frame, seed_text, seed_position, font, font_scale, color, thickness, cv2.LINE_AA)
                        video_writer.write(bgr_frame)


                    

                    # arr7: 7-element vector, last 4 entries are the quaternion
                    q = action[3:7]                      # (w, x, y, z)  ← change if yours is (x, y, z, w)

                    # q: (...,4) ndarray
                    n = np.linalg.norm(q, axis=-1, keepdims=True)
                    # avoid NaN if norm is tiny
                    q_unit = np.where(n < 1e-6, np.array([0, 0, 0, 1], dtype=q.dtype), q / n)
                    
                    action[3:7] = q_unit

                    w, x, y, z = q                    # unpack

                    # SciPy expects (x, y, z, w) order:
                    rot = R.from_quat([x, y, z, w])

                    # ‘xyz’ gives roll (X), pitch (Y), yaw (Z); request degrees=True for ° output
                    #roll, pitch, yaw = rot.as_euler('xyz', degrees=True)

                    #print(f"roll = {roll:.2f}°, pitch = {pitch:.2f}°, yaw = {yaw:.2f}°")


                    #take the action    
                    obs, reward, terminated, truncated, info = env.step(action)

                    done = terminated or truncated
                    #current_env_state = env.env_states[-1]
                    current_env_state = None
                    #append this stpes info to the current_trajectory object
                    current_trajectory.append(obs=current_obs, action=action, next_obs=obs, reward=reward, 
                                terminated=terminated, truncated=truncated, done=done, info=info, 
                                env_state=current_env_state)
                    failed_solve = False
                    break
                except Exception as e:
                    failed_solve = True
                    continue
                    
            
            if failed_solve==True:
                done = True
                break


            # Record the frame if requested
            if record_video:
                # Capture an RGB image from our chosen camera
                #env.rlbench_task_env.pyrep.step()
                #env.rlbench_env._scene.pyrep.step()
                env.rlbench_env._scene.pyrep.step()
                rgb = cinematic_cam.capture_rgb()
                # Convert [0,1] float array to [0,255] uint8
                rgb = (rgb * 255).astype(np.uint8)
                # OpenCV uses BGR instead of RGB
                bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                # Add text to the frame
                text = "Episode: " + str(episode) + " Step: " + str(step_count)  # replace with your desired text
                position = (10, 30)  # (x, y) coordinates for the text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                color = (255, 255, 255)  # White color in BGR
                thickness = 2
                cv2.putText(bgr_frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

                # Calculate height of the first line
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_height = text_size[1]

                # Add second line of text just under the first one
                seed_text = "Seed: " + str(seed_use)
                seed_position = (10, position[1] + text_height + 10)  # 10px gap below the first line
                cv2.putText(bgr_frame, seed_text, seed_position, font, font_scale, color, thickness, cv2.LINE_AA)
                video_writer.write(bgr_frame)
                
                if print_rewards is True:
                    # Add: reward mean, reward_std, step_reward and cumulative reward on the frame
                    # in the upper right hand corner
                    # ---------- overlay reward metrics (upper-right) ----------
                    reward_mean = info["reward_mean"]
                    reward_std = info["reward_std"]
                    step_reward = reward_mean - env.safety_factor * reward_std
                    total_trajectory_reward+=step_reward
                    metric_lines = [
                        f"mean reward:   {reward_mean:+.3f}",
                        f"std reward:   {reward_std:.3f}",
                        f"step reward:{step_reward:+.3f}",
                        f"cumulative reward: {total_trajectory_reward:+.3f}",
                    ]

                    margin_x  = 10                      # distance from the right edge
                    base_y    = 30                      # matches the left-side “Episode” text
                    line_gap  = 10                      # same 10-px gap you already use

                    # We only need the height once (all lines use same font + scale)
                    (_, text_h), _ = cv2.getTextSize("Tg", font, font_scale, thickness)

                    for i, txt in enumerate(metric_lines):
                        (txt_w, _), _ = cv2.getTextSize(txt, font, font_scale, thickness)
                        x = w - txt_w - margin_x                     # right-align
                        y = base_y + i * (text_h + line_gap)         # stack downward
                        cv2.putText(bgr_frame, txt, (x, y),
                                    font, font_scale, color, thickness, cv2.LINE_AA)

                # 1) Extract the (x,y,z) from each obs.gripper_pose
                #eef_xyz = [obs["gripper_pose"][:3] for obs in current_trajectory.observations]

                # 2) Stack into an (N,3) torch.Tensor
                #ee_mat = torch.tensor(eef_xyz, dtype=torch.float32)
                #visual_traj_obj = add_trajectory_line(ee_mat)

                #video_writer.write(bgr_frame)

            episode_reward += reward
            step_count += 1
            #print("Step: " + str(step_count))
            #print("Max Steps: " + str(max_steps))

            # Optional small delay to synchronize with the sim's time step
            sleep(sim.simGetSimulationTimeStep())

            success = info.get('success', False)
            if success:
                successes += 1
                #if print_info:
                print(f" ✅  Episode {episode+1} successful at step {step_count}!")
                if record_video is True:
                    
                    # Capture an RGB image from our chosen camera
                    #env.rlbench_task_env.pyrep.step()
                    env.rlbench_env._scene.pyrep.step()
                    rgb = cinematic_cam.capture_rgb()
                    # Convert [0,1] float array to [0,255] uint8
                    rgb = (rgb * 255).astype(np.uint8)
                    # OpenCV uses BGR instead of RGB
                    bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    # Add text to the frame
                    text = "Episode: " + str(episode) + " Step: " + str(step_count)  # replace with your desired text
                    position = (10, 30)  # (x, y) coordinates for the text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    color = (255, 255, 255)  # White color in BGR
                    thickness = 2
                    cv2.putText(bgr_frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

                    # Calculate height of the first line
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_height = text_size[1]

                    # Add second line of text just under the first one
                    seed_text = "Seed: " + str(seed_use)
                    seed_position = (10, position[1] + text_height + 10)  # 10px gap below the first line
                    cv2.putText(bgr_frame, seed_text, seed_position, font, font_scale, color, thickness, cv2.LINE_AA)
                    video_writer.write(bgr_frame)
                    video_writer.write(bgr_frame)
                    video_writer.write(bgr_frame)
                    video_writer.release()
                break
                #break

            if print_info and (step_count % 10 == 0):
                print(f"  Step {step_count}, Reward: {reward:.4f}")

            if step_count >= max_steps:
                if print_info:
                    print(f"Episode {episode+1} truncated after {step_count} steps")
                break

        if record_video:
            video_writer.release()
        episode_rewards.append(episode_reward)
        if print_info:
            print(f"Episode {episode + 1} finished with total reward: {episode_reward:.4f}")

        #append the PolicyTrajectory object to our list
        #current_trajectory.env_seed = env.rlbench_env._scene.task._variation_index
        trajectories.append(current_trajectory)

    mean_reward = np.mean(episode_rewards)
    success_rate = (successes / num_episodes) * 100

    if print_info:
        print("\nEvaluation Results:")
        print(f"  Mean Reward: {mean_reward:.4f}")
        print(f"  Success Rate: {success_rate:.2f}%")

    #if record_video:
    #    video_writer.release()

    if close_env is True:
        env.close()

    #Save the policy samples to the same video path
    with open(os.path.join(video_path, 'policy_samples.pkl'), 'wb') as f:
        pickle.dump(trajectories, f)

    
    return success_rate,trajectories

def get_latest_model_path(folder_path):
    """
    Given a folder path, find the model file with the highest number in the pattern 'model_{number}.zip'
    and return its full path.
    """
    model_files = [f for f in os.listdir(folder_path) if re.match(r'model_\d+\.zip', f)]

    if not model_files:
        raise FileNotFoundError("No model_{}.zip files found in the folder.")

    # Extract numbers and find the highest
    max_model = max(model_files, key=lambda f: int(re.search(r'\d+', f).group()))
    
    return os.path.join(folder_path, max_model)




def run_evaluate_policy(config,model=None,close_env=True,env=None,min_vals=None,max_vals=None,limit_to_correction_indices=None):
    # IF the user did not specify a model too evaluate, set it to the most recent one the path to the desired model/buffer
    
    if model is None:
        #model_path = get_latest_model_path(os.path.join(config.iteration_working_dir,"policy_models"))
        #buffer_path = "run_results/slide_block_to_target/iteration_1/policy_models/buffer_50000"
        print("Need to input a model to evaluate")
        # Load the trained model
        #model = SAC.load(model_path)
        #model.load_replay_buffer(buffer_path)
        #print("Model loaded successfully!")

    # Instatiate the envrionemtn
    #env = instantiate_environment()


    # Get the desire parameters for evaluate policy kwargs
    evaluate_policy_kwargs = config.evaluate_policy_kwargs



    #Run evaluate policy and get success rate
    success_rate,trajectories = evaluate_policy(model,env=env,close_env=close_env,min_vals=min_vals,max_vals=max_vals,seeds=limit_to_correction_indices,**evaluate_policy_kwargs)

    # Specify the file path where you want to save the trajectories
    file_path = os.path.join(config.iteration_working_dir,"policy_samples.pkl")


    # Pickle the trajectories
    with open(file_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Trajectories successfully saved to {file_path}")

    return success_rate,trajectories



# Test the trained policy
def evaluate_policy_DEPRECATED(model, env=None, task=None,num_episodes=5,deterministic=False,max_steps=200,shaped_rewards=True,render_mode='rgb_array',print_info=False):
    
    if task==None:
        task=SlideBlockToTarget


    if env==None:
        action_mode = MoveArmThenGripperWithBounds(
        arm_action_mode=JointPosition(),
        gripper_action_mode=Discrete())
        #Create env
        base_env = RLBenchEnv(
            task,
            observation_mode='state',
            action_mode=action_mode,
            render_mode=render_mode
        )
        # Enable shaped rewards
        base_env.rlbench_task_env._shaped_rewards = shaped_rewards
        base_env.reset()
        # Set max steps per episode
        MAX_EPISODE_STEPS = max_steps  # Change this based on your task
        #Instatite the environemnt
        env = MaxStepWrapper(base_env,MAX_EPISODE_STEPS)
    else:
        pass

    
    episode_rewards = []
    successes = 0

    for episode in range(num_episodes):
        # Gymnasium-style reset -> returns (obs, info)
        obs, _ = env.reset()  
        done = False
        episode_reward = 0
        step_count = 0

        if print_info:
            print(f"\nStarting episode {episode + 1}/{num_episodes}")

        while not done:
            # Get action from the policy
            action, _ = model.predict(obs, deterministic=deterministic)
            # Example: force the last dimension to 1.0 if that's your "open gripper" action
            action[-1] = 1.0
            

            # Step the environment (Gymnasium 5-tuple: obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = env.step(action)

            # Convert it to old Gym style done if needed
            done = terminated or truncated

            # Optional slow-down or synchronization
            sleep(sim.simGetSimulationTimeStep())

            episode_reward += reward
            step_count += 1

            # Check custom success signal in info
            success = info.get('success', False)
            if success:
                successes += 1
                if print_info:
                    print(f"Episode {episode+1} successful at step {step_count}!")
                break

            if print_info and (step_count % 10 == 0):
                print(f"  Step {step_count}, Reward: {reward:.4f}")

            # Additional manual time-limit check if desired
            if step_count >= max_steps:
                if print_info:
                    print(f"Episode {episode+1} truncated after {step_count} steps")
                break

        episode_rewards.append(episode_reward)
        if print_info:
            print(f"Episode {episode + 1} finished with total reward: {episode_reward:.4f}")

    mean_reward = np.mean(episode_rewards)
    success_rate = (successes / num_episodes) * 100

    print("\nEvaluation Results:")
    print(f"  Mean Reward: {mean_reward:.4f}")
    print(f"  Success Rate: {success_rate:.2f}%")

    env.close()
    return success_rate