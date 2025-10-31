import os
import pickle
import numpy as np
from collections import defaultdict
from rlbench.gym import RLBenchEnv
from rlbench.tasks import SlideBlockToTarget
from rlbench.action_modes.arm_action_modes import JointPosition,EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds

def convert_actions_from_ivk_function(actions,dones):
    # Instantiate action modes (same as during training)
    action_mode = MoveArmThenGripperWithBounds(
        arm_action_mode=JointPosition(),
        gripper_action_mode=Discrete())

    # Create the environment (same as during training)
    env = RLBenchEnv(
        SlideBlockToTarget,
        observation_mode='state',
        action_mode=action_mode,
        #device="cuda"
        #render_mode='human'  # Enable rendering
    )

    orig_actions = actions
    
    #actions = np.squeeze(actions,axis=1)

    fixed_actions = np.zeros_like(actions)
    scene = env.unwrapped.rlbench_task_env._scene
    for action_idx in range(0,actions.shape[0]):
        action = actions[action_idx,:]
        # Step environment
        #try:
        try:
            joint_positions = scene.robot.arm.solve_ik_via_jacobian(action[:3], quaternion=action[3:7]) #, relative_to=relative_to)
            action_use = np.append(joint_positions,1.0)
            obs, reward, terminated, truncated, info = env.step(action_use)
            done = terminated or truncated
            fixed_actions[action_idx,:] = action_use
        except Exception as e:
            print("IVK Failed")
        
        if dones[action_idx] == True:
            env.reset()
    
    #fixed_actions = np.expand_dims(fixed_actions,axis=1)
    env.close()
    return fixed_actions


def load_all_trajectories_DEPRECATED(base_dir="run_results/slide_block_to_target/", convert_actions_from_ivk=False,include_non_experts=False, verbose=False,sparse_reward=False):
    """
    Load and process all trajectories from the specified directory structure.
    
    Args:
        base_dir (str): Base directory containing numbered folders with trajectory data
        include_non_experts (bool): If False, only include trajectories marked as 'successful'
        verbose (bool): If True, print progress updates and summary information
    
    Returns:
        dict: Contains processed observations, next_observations, actions, and rewards
    """
    # Initialize containers for combined data
    all_observations = defaultdict(list)
    all_next_observations = defaultdict(list)
    all_actions = []
    all_rewards = []
    all_dones = []
    
    # Get all subdirectories in the base directory
    try:
        folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    except FileNotFoundError:
        print(f"Base directory {base_dir} not found!")
        return None
    
    if verbose:
        print(f"Found {len(folders)} folders to process")
    
    # Process each folder
    for folder in folders:
        pkl_path = os.path.join(base_dir, folder, "full_ivk_trajectories.pkl")
        
        # Skip if the pickle file doesn't exist
        if not os.path.exists(pkl_path):
            if verbose:
                print(f"No trajectory file found in {folder}, skipping...")
            continue
            
        if verbose:
            print(f"Processing folder: {folder}")
        
        # Load the pickle file
        with open(pkl_path, "rb") as file:
            data = pickle.load(file)
        
        # Get the trajectories and their success status
        trajectories = data['full_trajectories']
        successful = data['successful']
        
        # Process each trajectory
        for idx, (trajectory, is_expert) in enumerate(zip(trajectories, successful)):
            # Skip if we only want expert trajectories and this isn't one
            if not include_non_experts and not is_expert:
                continue
            
            if verbose:
                print(f"  Processing trajectory {idx} (expert: {is_expert})")
            
            # Process observations first
            observations_list = trajectory['observations']
            if not observations_list or len(observations_list) <= 1:  # Skip if empty or has only one observation
                if verbose:
                    print(f"    Skipping trajectory with insufficient observations: {len(observations_list)}")
                continue
            
            # Convert observations to dictionary of arrays
            observations_dict = defaultdict(list)
            for obs in observations_list:
                for key, value in obs.items():
                    observations_dict[key].append(value)
            
            # Create observations and next_observations
            # We'll use this to determine how many transitions we can create from this trajectory
            transitions_from_trajectory = 0
            for key, value_list in observations_dict.items():
                if not value_list:  # Skip if empty
                    continue
                
                # Convert to numpy array with shape [N, 1, feature_dim]
                value_array = np.array(value_list, dtype=np.float32)
                feature_dim = value_array[0].shape[0] if hasattr(value_array[0], "shape") else 1
                value_array = value_array.reshape(-1, 1, feature_dim)
                
                # Create observations (excluding the last element)
                obs_array = value_array[:-1]
                all_observations[key].append(obs_array)
                
                # Create next_observations (excluding the first element)
                next_obs_array = value_array[1:]
                all_next_observations[key].append(next_obs_array)
                
                # Record the number of transitions this trajectory gives us
                # We only need to do this once since all keys should have same length
                if transitions_from_trajectory == 0:
                    transitions_from_trajectory = len(obs_array)
            
            # Now process actions, rewards, and dones
            # But make sure to only take as many elements as we have valid transitions
            
            # Process actions
            actions_list = trajectory['actions']
            if actions_list and transitions_from_trajectory > 0:
                # Convert to numpy array with shape [N, 1, action_dim]
                all_actions_for_traj = np.array(actions_list, dtype=np.float32).reshape(-1, 1, len(actions_list[0]))
                # Only take as many actions as we have transitions (obs, next_obs pairs)
                actions = all_actions_for_traj[:transitions_from_trajectory]
                all_actions.append(actions)
            
            # Process rewards
            # Process rewards
            rewards_list = trajectory['rewards']
            if rewards_list and transitions_from_trajectory > 0:
                # If using sparse rewards, create a new reward list based on success
                if sparse_reward:
                    # Zero rewards for all steps except possibly the last
                    sparse_rewards = np.zeros(len(rewards_list), dtype=np.float32)
                    
                    # Set the last reward to 1 if this is a successful trajectory
                    if is_expert:
                        sparse_rewards[-1] = 1.0
                        sparse_rewards[-2] = 1.0
                        
                    # Convert to numpy array with shape [N, 1]
                    all_rewards_for_traj = sparse_rewards.reshape(-1, 1)
                else:
                    # Use original dense rewards
                    all_rewards_for_traj = np.array(rewards_list, dtype=np.float32).reshape(-1, 1)
                
                # Only take as many rewards as we have transitions
                rewards = all_rewards_for_traj[:transitions_from_trajectory]
                all_rewards.append(rewards)
                
            # Process dones
            dones_list = trajectory['dones']
            if dones_list and transitions_from_trajectory > 0:
                # Convert to numpy array with shape [N, 1]
                all_dones_for_traj = np.array(dones_list, dtype=bool).reshape(-1, 1)
                # Only take as many dones as we have transitions
                dones = all_dones_for_traj[:transitions_from_trajectory]
                dones[-1]=True
                all_dones.append(dones)
    
    # Combine all data
    combined_data = {}
    
    # Combine observations and next_observations
    observations = {}
    next_observations = {}
    
    for key in all_observations.keys():
        if all_observations[key]:  # Skip if empty
            observations[key] = np.concatenate(all_observations[key], axis=0)
            next_observations[key] = np.concatenate(all_next_observations[key], axis=0)
    
    # Combine actions, rewards, and dones
    if all_actions:
        actions = np.concatenate(all_actions, axis=0)
    else:
        actions = np.array([], dtype=np.float32)
    
    if all_rewards:
        rewards = np.concatenate(all_rewards, axis=0)
    else:
        rewards = np.array([], dtype=np.float32)
        
    if all_dones:
        dones = np.concatenate(all_dones, axis=0)
    else:
        dones = np.zeros_like(rewards, dtype=bool)

    if convert_actions_from_ivk==True:
        ivk_actions = actions
        actions = convert_actions_from_ivk_function(ivk_actions,dones)
    
    # Final check to ensure all arrays have the same first dimension
    if observations:
        sample_key = list(observations.keys())[0]
        obs_length = observations[sample_key].shape[0]
        
        # Verify all arrays have the same first dimension
        if actions.shape[0] != obs_length:
            if verbose:
                print(f"WARNING: Actions shape {actions.shape[0]} doesn't match observations shape {obs_length}. Truncating.")
            actions = actions[:obs_length]
            
        if rewards.shape[0] != obs_length:
            if verbose:
                print(f"WARNING: Rewards shape {rewards.shape[0]} doesn't match observations shape {obs_length}. Truncating.")
            rewards = rewards[:obs_length]
            
        if dones.shape[0] != obs_length:
            if verbose:
                print(f"WARNING: Dones shape {dones.shape[0]} doesn't match observations shape {obs_length}. Truncating.")
            dones = dones[:obs_length]
    
    # Create final result dictionary
    combined_data['observations'] = observations
    combined_data['next_observations'] = next_observations
    combined_data['actions'] = actions
    combined_data['rewards'] = rewards
    combined_data['dones'] = dones
    
    # Print summary
    if verbose:
        print("\nData processing complete!")
        print(f"Total trajectories processed: {len(all_actions)}")
        if observations:
            sample_key = list(observations.keys())[0]
            print(f"Final data shape - observations: {observations[sample_key].shape}")
        if actions.size > 0:
            print(f"Final data shape - actions: {actions.shape}")
        if rewards.size > 0:
            print(f"Final data shape - rewards: {rewards.shape}")
        if dones.size > 0:
            print(f"Final data shape - dones: {dones.shape}")
    
    return combined_data



def load_all_trajectories_DEPRECATED(base_dir="run_results/slide_block_to_target/", include_non_experts=False, verbose=False):
    """
    Load and process all trajectories from the specified directory structure.
    
    Args:
        base_dir (str): Base directory containing numbered folders with trajectory data
        include_non_experts (bool): If False, only include trajectories marked as 'successful'
        verbose (bool): If True, print progress updates and summary information
    
    Returns:
        dict: Contains processed observations, next_observations, actions, and rewards
    """
    # Initialize containers for combined data
    all_observations = defaultdict(list)
    all_next_observations = defaultdict(list)
    all_actions = []
    all_rewards = []
    all_dones = []
    
    # Get all subdirectories in the base directory
    try:
        folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    except FileNotFoundError:
        print(f"Base directory {base_dir} not found!")
        return None
    
    if verbose:
        print(f"Found {len(folders)} folders to process")
    
    # Process each folder
    for folder in folders:
        pkl_path = os.path.join(base_dir, folder, "full_ivk_trajectories.pkl")
        
        # Skip if the pickle file doesn't exist
        if not os.path.exists(pkl_path):
            if verbose:
                print(f"No trajectory file found in {folder}, skipping...")
            continue
            
        if verbose:
            print(f"Processing folder: {folder}")
        
        # Load the pickle file
        with open(pkl_path, "rb") as file:
            data = pickle.load(file)
        
        # Get the trajectories and their success status
        trajectories = data['full_trajectories']
        successful = data['successful']
        
        # Process each trajectory
        for idx, (trajectory, is_expert) in enumerate(zip(trajectories, successful)):
            # Skip if we only want expert trajectories and this isn't one
            if not include_non_experts and not is_expert:
                continue
            
            if verbose:
                print(f"  Processing trajectory {idx} (expert: {is_expert})")
            
            # Process actions
            actions_list = trajectory['actions']
            if actions_list:  # Check if the list is not empty
                # Convert to numpy array with shape [N, 1, action_dim]
                actions = np.array(actions_list).reshape(-1, 1, len(actions_list[0]))
                all_actions.append(actions)
            
            # Process rewards
            rewards_list = trajectory['rewards']
            if rewards_list:  # Check if the list is not empty
                # Convert to numpy array with shape [N, 1]
                rewards = np.array(rewards_list).reshape(-1, 1)
                all_rewards.append(rewards)
                
            # Process dones
            dones_list = trajectory['dones']
            if dones_list:  # Check if the list is not empty
                # Convert to numpy array with shape [N, 1]
                dones = np.array(dones_list, dtype=bool).reshape(-1, 1)
                all_dones.append(dones)
            
            # Process observations
            observations_list = trajectory['observations']
            if not observations_list:  # Skip if empty
                continue
                
            # Convert observations to dictionary of arrays
            observations_dict = defaultdict(list)
            for obs in observations_list:
                for key, value in obs.items():
                    observations_dict[key].append(value)
            
            # Create observations and next_observations
            for key, value_list in observations_dict.items():
                if not value_list:  # Skip if empty
                    continue
                
                # Convert to numpy array with shape [N, 1, feature_dim]
                value_array = np.array(value_list)
                feature_dim = value_array[0].shape[0] if hasattr(value_array[0], "shape") else 1
                value_array = value_array.reshape(-1, 1, feature_dim)
                
                # Create observations (excluding the last element)
                obs_array = value_array[:-1]
                all_observations[key].append(obs_array)
                
                # Create next_observations (excluding the first element)
                next_obs_array = value_array[1:]
                all_next_observations[key].append(next_obs_array)
    
    # Combine all data
    combined_data = {}
    
    # Combine observations and next_observations
    observations = {}
    next_observations = {}
    
    for key in all_observations.keys():
        if all_observations[key]:  # Skip if empty
            observations[key] = np.concatenate(all_observations[key], axis=0)
            next_observations[key] = np.concatenate(all_next_observations[key], axis=0)
    
    # Combine actions and rewards
    if all_actions:
        actions = np.concatenate(all_actions[:-1], axis=0)  # Exclude last action to match observations
    else:
        actions = np.array([])
    
    if all_rewards:
        rewards = np.concatenate(all_rewards[:-1], axis=0)  # Exclude last reward to match observations
    else:
        rewards = np.array([])
        
    if all_dones:
        dones = np.concatenate(all_dones[:-1], axis=0)  # Exclude last done to match observations
    else:
        dones = np.zeros_like(rewards, dtype=bool)
    
    # Create final result dictionary
    combined_data['observations'] = observations
    combined_data['next_observations'] = next_observations
    combined_data['actions'] = actions
    combined_data['rewards'] = rewards
    combined_data['dones'] = dones
    
    # Print summary
    if verbose:
        print("\nData processing complete!")
        print(f"Total trajectories processed: {len(all_actions)}")
        if observations:
            sample_key = list(observations.keys())[0]
            print(f"Final data shape - observations: {observations[sample_key].shape}")
        if actions.size > 0:
            print(f"Final data shape - actions: {actions.shape}")
        if rewards.size > 0:
            print(f"Final data shape - rewards: {rewards.shape}")
        if dones.size > 0:
            print(f"Final data shape - dones: {dones.shape}")
    
    return combined_data

# Example usage:
# dataset = load_all_trajectories(include_non_experts=False, verbose=True)

def fill_replay_buffer(data, replay_buffer):
    """
    Fill a replay buffer with pre-processed trajectory data.
    
    Args:
        data (dict): Output from load_all_trajectories function
        replay_buffer: The replay buffer object from the RL model
        
    Returns:
        The updated replay buffer
    """
    # Set the replay buffer attributes directly
    replay_buffer.observations = data['observations']
    replay_buffer.next_observations = data['next_observations']
    replay_buffer.actions = data['actions']
    replay_buffer.rewards = data['rewards']
    replay_buffer.dones = data['dones']
    
    # Update buffer position and size attributes
    n_samples = data['actions'].shape[0]
    replay_buffer.pos = n_samples % replay_buffer.buffer_size
    replay_buffer.full = (n_samples >= replay_buffer.buffer_size)
    #if not replay_buffer.full:
    replay_buffer.buffer_size = n_samples
    replay_buffer.size = n_samples
    replay_buffer.full = True
    
    print(f"Successfully populated replay buffer with {n_samples} transitions")
    return replay_buffer