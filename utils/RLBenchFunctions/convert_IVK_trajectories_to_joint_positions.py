import os
import pickle
import numpy as np
from imitation.data.types import Trajectory
from utils.RLBenchFunctions.fill_replay_buffer import convert_actions_from_ivk_function

def load_ivk_trajectories(base_path: str):
    """
    Loads trajectories from subfolders of `base_path` where folder names are integer seeds.
    For each seed folder, it reads `full_ivk_trajectories.pkl`, converts each trajectory
    into an `imitation.data.types.Trajectory`, and sorts them into expert vs. non-expert
    based on the 'successful' field.

    Returns:
        dict:
            {
               "seeds": {
                   seed_int: {
                       "expert_trajectories": [Trajectory, ...],
                       "non_expert_trajectories": [Trajectory, ...]
                   },
                   ...
               }
            }
    """
    trajectory_data = {"seeds": {}}

    # 1. List all items in base_path
    for folder_name in os.listdir(base_path):
        # 2. Skip any folder that isn't an integer
        try:
            seed_int = int(folder_name)
        except ValueError:
            continue  # skip non-integer folder names

        folder_path = os.path.join(base_path, folder_name)
        pkl_path = os.path.join(folder_path, "full_ivk_trajectories.pkl")

        # 3. If the pickle file doesn't exist, skip this folder
        if not os.path.isfile(pkl_path):
            continue

        # 4. Load the pickle file
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        full_trajectories = data["full_trajectories"]
        successes = data["successful"]

        # Prepare storage for expert vs. non-expert
        expert_trajs = []
        non_expert_trajs = []

        # 5. Convert each trajectory in full_trajectories
        for i, traj_dict in enumerate(full_trajectories):
            # Extract fields
            observations = traj_dict["observations"]
            actions = traj_dict["actions"]
            dones = traj_dict["dones"]

            #for/experiences that fail immediat
            if len(observations)<2:
                print("Bad Trajectory " + str(i) + " of file " +  pkl_path)
                continue
            # Rewards, ee_positions, etc. are available too, but typically not
            # needed directly for creating the imitation Trajectory.

            # Following your snippet: remove the last action to match obs dimension
            actions.pop()  # remove final action
            actions = np.array(actions)

            # Convert actions from IVK action mode to JointPosition action mode
            actions = convert_actions_from_ivk_function(actions, dones)

            # `infos` is optional; here we just store empty dicts
            infos = [{} for _ in range(len(actions))]

            # The final done in `dones` tells us if the episode actually ended
            terminal = dones[-1]

            # Construct the imitation Trajectory
            traj = Trajectory(
                obs=observations,
                acts=actions,
                infos=infos,
                terminal=terminal,
            )

            # 6. Sort into expert vs. non-expert
            if successes[i]:
                expert_trajs.append(traj)
            else:
                non_expert_trajs.append(traj)

        # 7. Store results in the dictionary
        trajectory_data["seeds"][seed_int] = {
            "expert_trajectories": expert_trajs,
            "non_expert_trajectories": non_expert_trajs,
        }

    return trajectory_data