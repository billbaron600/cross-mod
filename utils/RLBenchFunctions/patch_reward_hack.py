from utils.RLBenchFunctions.evaluate_policy import evaluate_policy
from rlbench.tasks import SlideBlockToTarget
import os
import pickle

def patch_reward_hack(model, config, trajectories=None,n_patch_trajectories=200, evaluate_policy_dict={}, save_database=False):
    
    if trajectories is None:
        # Run evaluate_policy to get the current reward hacked samples
        success_rate,trajectories = evaluate_policy(model=model, **evaluate_policy_dict)

    # Load in the current preference database
    path_to_pref_db = os.path.join(config.iteration_working_dir, "preference_database.pkl")
    with open(path_to_pref_db, 'rb') as file:
        current_database = pickle.load(file)

    original_size = len(current_database.pairwise_comparisons)

    # Prepare reversed and truncated list of corrections
    all_corrections = list(reversed(current_database.corrections))[:n_patch_trajectories]

    num_comparisons_added = 0

    for hacked_traj in trajectories:
        for correction in all_corrections:
            for expert_traj in correction.corrections:
                current_database.pairwise_comparisons.append((expert_traj, hacked_traj, 0))
                num_comparisons_added += 1

    new_size = len(current_database.pairwise_comparisons)

    print(f"Original comparisons: {original_size}")
    print(f"New comparisons: {new_size}")
    print(f"Number of comparisons added: {num_comparisons_added}")

    if save_database:
        with open(path_to_pref_db, "wb") as f:
            pickle.dump(current_database, f)
        print("Updated preference database saved to disk.")
    else:
        print("Preference database NOT saved (save_database=False).")