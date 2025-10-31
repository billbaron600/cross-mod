import numpy as np
from utils.Classes.preference_database import PreferenceDatabase
import pickle
from utils.RLBenchFunctions.custom_envs import instantiate_environment
import os
import copy
from utils.RLBenchFunctions.combine_and_flatten_obs_and_action import flatten_observation, combine_observation_and_action

"""

def flatten_observation(obs: dict) -> np.ndarray:
    
    Flattens a dictionary observation from RLBench/Gym into a single 1D float32 array.
    
    The order of concatenation is fixed to ensure consistency across calls.
    It assumes the dict contains the following keys:
      'joint_velocities', 'joint_positions', 'joint_forces',
      'gripper_open', 'gripper_pose', 'gripper_joint_positions',
      'gripper_touch_forces', 'task_low_dim_state'.
      
    Args:
        obs (dict): A single timestep observation with the above keys,
                    each mapping to a NumPy array.
                    
    Returns:
        np.ndarray: A 1D float32 array with all fields concatenated in the
                    same fixed order every time.
    

    # 1) Extract each sub-array and ravel it.
    joint_velocities = obs["joint_velocities"].ravel()
    joint_positions = obs["joint_positions"].ravel()
    joint_forces = obs["joint_forces"].ravel()
    gripper_open = obs["gripper_open"].ravel()
    gripper_pose = obs["gripper_pose"].ravel()
    gripper_joint_positions = obs["gripper_joint_positions"].ravel()
    gripper_touch_forces = obs["gripper_touch_forces"].ravel()
    task_low_dim_state = obs["task_low_dim_state"].ravel()

    # 2) Concatenate into one vector, forcing float32 dtype.
    flattened = np.concatenate([
        joint_velocities,
        joint_positions,
        joint_forces,
        gripper_open,
        gripper_pose,
        gripper_joint_positions,
        gripper_touch_forces,
        task_low_dim_state,
    ]).astype(np.float32)

    return flattened

def combine_observation_and_action(obs: dict, action) -> np.ndarray:
    
    Combines a single observation and action into a flattened NumPy array.

    Args:
        obs (dict): Observation dictionary with predefined keys.
        action: Action associated with the observation.

    Returns:
        np.ndarray: Flattened combined array of observation and action.
    
    flattened_obs = flatten_observation(obs)
    flattened_action = np.asarray(action).ravel().astype(np.float32)
    combined = np.concatenate([flattened_obs, flattened_action])

    return combined
"""
def generate_noise_trajectories(number_noise_trajectories=1,length_per_traj=100):
    from utils.Classes.policy_trajectory_class import PolicyTrajectory
    
    #create list to store noise trajectoires in, and the enviornment
    noise_trajectories = []
    env = instantiate_environment()
    
    for i in range(number_noise_trajectories):
        traj = PolicyTrajectory()
        env.seed = np.random.randint(0, 11)
        obs,_ = env.reset()
        for j in range(length_per_traj):
            # Generate a random action within bounds
            action = np.random.uniform(env.action_space.low, env.action_space.high)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            reward = None
            done = False
            env_state = None
            traj.append(obs, action, next_obs, reward, terminated, truncated, done, info, env_state,ee_position=None)
            obs = next_obs
        noise_trajectories.append(traj)
    
    return noise_trajectories
        

def generate_preference_database(config,number_noise_trajectories=None,save_to_working_dir=True):
    # Get the working directoreis
    working_dirs = config.working_dirs

    # Specify expert list
    expert_demos = []
    non_expert_demos = []

    #Iterate through all our demos, and get expert and non expert demos based on if they were successful or not
    for working_dir in working_dirs:
        with open(working_dir+"full_ivk_trajectories.pkl","rb") as f:
            full_trajectories = pickle.load(f)
        
        #Get the trajectories from the dict, as well as if they were successful or not
        trajectories = full_trajectories['full_trajectories']
        success_status = full_trajectories['successful']

        #Iterate through the trajectories, and append to the correct list depeending on if they were successful or not
        for traj_idx in range(0,len(trajectories)):
            if success_status[traj_idx]==True:
                expert_demos.append(trajectories[traj_idx])
            elif len(trajectories[traj_idx].observations)>0:
                non_expert_demos.append(trajectories[traj_idx])

    if number_noise_trajectories is not None:
        noise_trajectories = generate_noise_trajectories(number_noise_trajectories=number_noise_trajectories)
    else:
        noise_trajectories = []

    #Create the preference database
    preference_db = PreferenceDatabase(
        expert_demos=expert_demos,
        non_expert_demos=non_expert_demos,
        noise_trajectories=noise_trajectories
    )

    #generate the preference comparisons
    preference_db.generate_initial_comparisons()

    #save to pickle
    if save_to_working_dir:
        with open(config.preference_database_path, "wb") as f:
            pickle.dump(preference_db, f)

    return preference_db

def add_corrections_to_database(config, compare_with_current_expert_demos=True,db_path=None,limit_policy_traj_corrected_segment=True, downsample_ivk=False, print_additions=True, number_of_duplicates=10,include_negative_demos=False,add_policy_vs_noise=None):
    from utils.Classes.preference_database import Correction, PreferenceDatabase
    from utils.Classes.policy_trajectory_class import PolicyTrajectory

    # Load in the current trajectories and their corrections
    path_to_corrections = os.path.join(config.iteration_working_dir, "all_corrections.pkl")
    with open(path_to_corrections, 'rb') as file:
        all_corrections = pickle.load(file)

    # Get the path to human preferences over those trajectories
    path_to_human_preferences = os.path.join(config.iteration_working_dir, "user_preferences.pkl")
    with open(path_to_human_preferences, 'rb') as file:
        human_preference_labels = pickle.load(file)

    # Load in the current human preference database
    if db_path is None:
        with open(config.preference_database_path, 'rb') as file:
            current_database = pickle.load(file)
    else:
        with open(db_path,'rb') as file:
            current_database = pickle.load(file)

    # Create the updated database copy
    updated_database = copy.deepcopy(current_database)

    for policy_traj_idx in range(len(all_corrections)):
        policy_traj = all_corrections[policy_traj_idx]

        #Save this beeore limiting it. We will train against noise so as to not bias away too much from the current policy
        full_policy_traj = copy.deepcopy(policy_traj.current_policy_trajectory)

        # Limit the policy trajectory to only include the part we corrected (not the entire thing)
        if limit_policy_traj_corrected_segment is True:
            policy_traj.current_policy_trajectory.limit_to_segment(
                start_idx=config.frame_correction_indices[policy_traj_idx], end_idx=config.frame_correction_indices_end[policy_traj_idx]
            )

        preference_row = human_preference_labels[policy_traj_idx]
        expert_indices = [i for i, val in enumerate(preference_row) if val == 2]
        semi_expert_indices = [i for i, val in enumerate(preference_row) if val == 1]
        negative_indices = [i for i, val in enumerate(preference_row) if val == -1]

        expert_trajs = [policy_traj.corrections[i] for i in expert_indices]
        semi_expert_trajs = [policy_traj.corrections[i] for i in semi_expert_indices]
        negative_trajs = [policy_traj.corrections[i] for i in negative_indices]

        # Apply downsampling if needed
        if downsample_ivk:
            target_length = len(policy_traj.current_policy_trajectory.observations)
            for traj in expert_trajs + semi_expert_trajs + negative_trajs:
                traj.downsample_to_length(target_length=target_length)

        # Add expert & semi-expert as comparisons against the policy trajectory
        combined_corrections = expert_trajs + semi_expert_trajs
        combined_corrections = number_of_duplicates * combined_corrections
        updated_database.add_corrections_to_database(policy_traj.current_policy_trajectory, combined_corrections)

        for expert_temp_add in expert_trajs:
            updated_database.expert_demos.append(expert_temp_add)
            
        for semi_expert_temp_add in semi_expert_trajs:
            updated_database.non_expert_demos.append(semi_expert_temp_add)

        # Add comparisons between semi-expert and expert
        if len(expert_trajs) > 0 and len(semi_expert_trajs) > 0:
            for semi_expert in semi_expert_trajs:
                updated_database.add_corrections_to_database(semi_expert, expert_trajs)

        # Add negative demonstrations: policy trajectory is preferred
        if include_negative_demos is True:
            for neg_demo in negative_trajs:
                updated_database.add_corrections_to_database(neg_demo, [policy_traj.current_policy_trajectory])

        if compare_with_current_expert_demos is True:
            expert_demos = updated_database.expert_demos
            for expert_demo in expert_demos:
                updated_database.add_corrections_to_database(policy_traj.current_policy_trajectory,[expert_demo])
        
        if add_policy_vs_noise is True:
            end_of_correction_segment = config.frame_correction_indices_end[policy_traj_idx]
            if len(full_policy_traj.observations)-1>end_of_correction_segment:
                full_policy_traj.limit_to_segment(start_idx = end_of_correction_segment,end_idx=len(full_policy_traj.observations)-1)
                if len(combined_corrections)>0:
                    noise_trajectories = updated_database.noise_trajectories
                    noise_traj = noise_trajectories[0]
                    #for noise_traj in noise_trajectories:
                    updated_database.add_corrections_to_database(noise_traj,[full_policy_traj])

    if print_additions is True:
        print("# of Corrections Before:")
        print(len(current_database.pairwise_comparisons))
        print("# of Corrections After:")
        print(len(updated_database.pairwise_comparisons))
        print("# of Comparisons Added:")
        print(len(updated_database.pairwise_comparisons) - len(current_database.pairwise_comparisons))

    # Save the preference database
    path_to_save = os.path.join(config.iteration_working_dir, "preference_database.pkl")
    with open(path_to_save, 'wb') as f:
        pickle.dump(updated_database, f)

    print(f"Preference database saved to: {path_to_save}")

    return updated_database


def add_corrections_to_database_DEPRECATED(config,limit_policy_traj_corrected_segment=True,downsample_ivk=True,print_additions=True,number_of_duplicates = 10):
    from utils.Classes.preference_database import Correction,PreferenceDatabase
    from utils.Classes.policy_trajectory_class import PolicyTrajectory
    #Load in the current trajectoreis and their correctiosn
    path_to_corrections = os.path.join(config.iteration_working_dir,"all_corrections.pkl")
    with open(path_to_corrections,'rb') as file:
        all_corrections = pickle.load(file)

    
    #Get the path to humans preferences over those trajectories
    path_to_human_preferences = os.path.join(config.iteration_working_dir,"user_preferences.pkl")
    with open(path_to_human_preferences, 'rb') as file:
        human_preference_labels = pickle.load(file)

    #Load in the current human preference database
    with open(config.preference_database_path,'rb') as file:
        current_database = pickle.load(file)
    
    #Create the updated database copy
    updated_database = copy.deepcopy(current_database)
    for policy_traj_idx in range(len(all_corrections)):
        policy_traj = all_corrections[policy_traj_idx]
        
        #Limit the policy trajectory to only include the part we corrected (not the entire thing)
        if limit_policy_traj_corrected_segment is True:
            policy_traj.current_policy_trajectory.limit_to_segment(start_idx=config.frame_correction_indices[policy_traj_idx],end_idx=None)
        
        preference_row = human_preference_labels[policy_traj_idx]
        expert_indices = [i for i, val in enumerate(preference_row) if val == 2]
        semi_expert_indices = [i for i, val in enumerate(preference_row) if val == 1]

        #Actualyl get the expert and semi expert trajectories now
        expert_trajs = [policy_traj.corrections[i] for i in expert_indices]
        semi_expert_trajs = [policy_traj.corrections[i] for i in semi_expert_indices]

        #Apply downsampling if needed
        if downsample_ivk:
            for traj in expert_trajs + semi_expert_trajs:
                target_length = len(policy_traj.current_policy_trajectory.observations)
                traj.downsample_to_length(target_length=target_length)
        
        # Add all the expert and semi expert trajectoreis as comparisons to the original policy traj
        combined_corrections = expert_trajs + semi_expert_trajs
        combined_corrections = number_of_duplicates * combined_corrections
        updated_database.add_corrections_to_database(policy_traj.current_policy_trajectory,combined_corrections)
        
        #Add teh comparison between expert and semi expert trajectories to the database
        if len(expert_trajs)>0 and len(semi_expert_trajs)>0:
            for semi_expert in semi_expert_trajs:
                updated_database.add_corrections_to_database(semi_expert,expert_trajs)
    
    if print_additions is True:
        print("# of Corrections Before:")
        print(len(current_database.pairwise_comparisons))
        print("# of Corrections After:")
        print(len(updated_database.pairwise_comparisons))
        print("# of Comparisons Added:")
        print(len(updated_database.pairwise_comparisons)-len(current_database.pairwise_comparisons))

    #Save the preference databse
    path_to_save = os.path.join(config.iteration_working_dir,"preference_database.pkl")
    with open(path_to_save, 'wb') as f:
        pickle.dump(updated_database, f)

    print(f"Preference database saved to: {path_to_save}")
    #Return the updated database
    return updated_database