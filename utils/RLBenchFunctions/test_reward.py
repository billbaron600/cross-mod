import pickle
import torch
import os

def flatten_and_prune(list_of_lists):
    out = []
    for sub in list_of_lists:
        for traj in sub if isinstance(sub, list) else [sub]:
            # skip if no observations
            if hasattr(traj, 'observations') and len(traj.observations) > 0:
                out.append(traj)
    return out

def test_reward_function(config, full_reward_averages=True, database_index=None,beta=0.0,last_n_policy_samples = 10,safety_factor=1.0):
    # 1) pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2) load the ensemble
    reward_path = os.path.join(config.iteration_working_dir, "reward_model_ensemble.pt")
    ensemble = torch.load(reward_path, map_location=device)
    ensemble.to(device).eval()
    
    # 3) load the DB
    #db_path = os.path.join(config.iteration_working_dir, "preference_database.pkl")
    # 1) Load the updated preference database
    if database_index is None:
        db_path = os.path.join(config.iteration_working_dir, "preference_database.pkl")
    else:
        quickpath = "preference_database_"+str(database_index)+".pkl"
        db_path = os.path.join(config.iteration_working_dir, quickpath)
    
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    
    #expert_trajs    = db.expert_demos
    #nonexpert_trajs = db.non_expert_demos
    expert_trajs = flatten_and_prune(db.expert_demos)
    nonexpert_trajs = flatten_and_prune(db.non_expert_demos)
    noise_trajs = [traj for traj in db.noise_trajectories
               if hasattr(traj, 'observations') and len(traj.observations) > 0]

    policy_trajs = flatten_and_prune(db.policy_samples[-last_n_policy_samples:])
            
               
    def compute_stats(name, trajectories):
        cum_rewards = []
        all_frame_rewards = []
        lengths = []
        
        for traj in trajectories:
            t = traj.generate_tensor_from_trajectory(feat_stats = db.feat_stats).to(device)  # (T,D)
            with torch.no_grad():
                # ensemble returns a list of M tensors, each (T,1)
                outs = ensemble(t, update_stats=False)               # [ (T,1), … ]
                stacked = torch.stack(outs, dim=0).squeeze(-1)       # (M,T)
                mean_r = stacked.mean(0)                            # (T,)
                std_r  = stacked.std(0, unbiased=False)             # (T,)
                
                # optional risk-penalized reward per step
                if beta > 0:
                    mean_r = mean_r - beta * std_r

            if full_reward_averages:
                # sum of mean rewards
                R = mean_r.sum().item()
                cum_rewards.append(R)
            else:
                # flatten all mean rewards for frame-wise stats
                all_frame_rewards.append(mean_r)
            
            lengths.append(t.shape[0])

        # aggregate
        if full_reward_averages:
            if cum_rewards:
                avg_R   = sum(cum_rewards) / len(cum_rewards)
                std_R   = torch.tensor(cum_rewards).std(unbiased=False).item()
                net_combined_reward = avg_R - safety_factor*std_R
            else:
                avg_R = std_R = float('nan')
            print(f"{name} Trajectories (cumulative):")
            print(f"  Avg   CumReward: {avg_R:.4f}")
            print(f"  Std   CumReward: {std_R:.4f}")
            print(f"  Net   CumReward: {net_combined_reward:.4f}")
        else:
            if all_frame_rewards:
                flat = torch.cat(all_frame_rewards)
                avg_R = flat.mean().item()
                std_R = flat.std(unbiased=False).item()
            else:
                avg_R = std_R = float('nan')
            print(f"{name} Trajectories (frame-wise):")
            print(f"  Avg   FrameReward: {avg_R:.4f}")
            print(f"  Std   FrameReward: {std_R:.4f}")
        
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            std_len = torch.tensor(lengths, dtype=torch.float32).std(unbiased=False).item()
        else:
            avg_len = std_len = float('nan')
        print(f"  Avg   Length: {avg_len:.1f}")
        print(f"  Std   Length: {std_len:.1f}\n")

    # run through each set
    compute_stats("Expert   ", expert_trajs)
    compute_stats("NonExpert", nonexpert_trajs)
    compute_stats("Noise    ", noise_trajs)
    compute_stats("Policy Samples", policy_trajs)

def test_reward_function_DEPRECATED(config, full_reward_averages=True):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the reward model
    current_reward_path = os.path.join(config.iteration_working_dir, "reward_model.pt")
    reward_model = torch.load(current_reward_path, map_location=device)
    reward_model = reward_model.to(device)
    reward_model.eval()

    # Load the updated preference database
    path_to_load = os.path.join(config.iteration_working_dir, "preference_database.pkl")
    with open(path_to_load, 'rb') as f:
        updated_database = pickle.load(f)

    # Get trajectory classes
    expert_trajectories = updated_database.expert_demos
    non_expert_trajectories = updated_database.non_expert_demos
    noise_trajectories = updated_database.noise_trajectories

    def compute_stats(name, trajectories):
        cumulative_rewards = []
        all_rewards = []
        trajectory_lengths = []

        for traj in trajectories:
            traj_vector = traj.generate_tensor_from_trajectory().to(device)
            with torch.no_grad():
                rewards = reward_model(traj_vector)  # shape: (N, 1) or (N,)
                if rewards.dim() > 1:
                    rewards = rewards.squeeze(-1)
            if full_reward_averages:
                cumulative_reward = rewards.sum().item()
                cumulative_rewards.append(cumulative_reward)
            else:
                all_rewards.append(rewards)
            trajectory_lengths.append(traj_vector.shape[0])

        if full_reward_averages:
            if cumulative_rewards:
                avg_reward = sum(cumulative_rewards) / len(cumulative_rewards)
                std_reward = torch.std(torch.tensor(cumulative_rewards)).item()
            else:
                avg_reward = std_reward = float('nan')
        else:
            if all_rewards:
                all_rewards_cat = torch.cat(all_rewards)
                avg_reward = all_rewards_cat.mean().item()
                std_reward = all_rewards_cat.std().item()
            else:
                avg_reward = std_reward = float('nan')

        if trajectory_lengths:
            avg_length = sum(trajectory_lengths) / len(trajectory_lengths)
        else:
            avg_length = float('nan')

        print(f"{name} Trajectories:")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Std Dev Reward: {std_reward:.4f}")
        print(f"  Avg Length: {avg_length:.2f}\n")

    # Run for each class
    compute_stats("Expert", expert_trajectories)
    compute_stats("Non-Expert", non_expert_trajectories)
    compute_stats("Noise", noise_trajectories)



