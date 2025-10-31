import os
from utils.Classes.preference_database import Correction,PreferenceDatabase
import torch
import pickle
import matplotlib.pyplot as plt

def calculate_reward_ensemble_accuracy(config,reward_model=None,plot_cumulative_error=True,database_index = None,eps = 0.05,device="cuda"):
    #If the reward model is None, load in the model in this iteratiosn working dir
    if reward_model is None:
        reward_path = os.path.join(config.iteration_working_dir, "reward_model_ensemble.pt")
        ensemble = torch.load(reward_path, map_location=device)
        ensemble.to(device).eval()
    
    # Load in the preference database
    quickpath = "preference_database_"+str(database_index)+".pkl"
    db_path = os.path.join(config.iteration_working_dir, quickpath)

    with open(db_path, 'rb') as f:
        pref_db = pickle.load(f)

    #get the pairwise comparisons
    pairwise_comparisons = pref_db.pairwise_comparisons

    #Keep track of total and correct
    correct = 0
    total = 0

    # --- running counters -------------------------------------------------
    incorrect_cumsum = []    # y-axis
    cum_incorrect    = 0

    #Iterate through
    for idx, (trajA, trajB, label) in enumerate(pairwise_comparisons):
        #Create vectors from the PolicyTrajectory classes
        trajA_vec = trajA.generate_tensor_from_trajectory(feat_stats = pref_db.feat_stats).to(device) # [60x37 tensor]
        trajB_vec = trajB.generate_tensor_from_trajectory(feat_stats = pref_db.feat_stats).to(device) # [60x37 tensor]
        
        #Score the trajectories using the senemble
        outs_A = ensemble.score_trajectory(trajA_vec)
        outs_B = ensemble.score_trajectory(trajB_vec)
        
        #Get the mean across the ensemble heads
        score_A = outs_A.mean().item()
        score_B = outs_B.mean().item()
        diff    = score_A - score_B

        # Predicted preference: 0 if A>B else 1
        # ----- model’s predicted preference ----------------------------
        if abs(diff) <= eps:
            pred_label = 0.5           # tie
        else:
            pred_label = 0 if diff > 0 else 1
        #label is an int, either 0, 1, or 0.5. If 0, then A should be prefereed, if 1, then B, if both, then 0.5
        
        # ----- compare to ground truth ---------------------------------
        if pred_label == label:
            correct += 1
        else:
            cum_incorrect+=1
        incorrect_cumsum.append(cum_incorrect)
        total += 1
    accuracy = correct / total if total else float("nan")
    print(f"Pref-prediction accuracy: {accuracy:.3f}  "
          f"(correct={correct} / total={total})")

    if plot_cumulative_error is True:
        # --- plot -------------------------------------------------------------
        plt.figure(figsize=(6,4))
        plt.plot(incorrect_cumsum, lw=2)
        plt.xlabel("Pair index")
        plt.ylabel("Cumulative incorrect predictions")
        plt.title("Reward-ensemble mistakes over evaluation set")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return accuracy