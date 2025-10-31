from utils.RLBenchFunctions.fill_buffer import preload_from_config
import os
import pickle
from utils.RLBenchFunctions.custom_envs import instantiate_environment
from utils.Classes.custom_replay_buffer import ScaledRewardBuffer
from rlbench.tasks import ReachTarget
from stable_baselines3 import SAC
from utils.RLBenchFunctions.fill_buffer import preload_from_config
import numpy as np

def test_policy_on_buffer(config,env=None,print_info=False,results_path=None,n_trajs_to_keep=None,model_path=None,buffer_path=None,fill_buffer=True,device="cuda"):
    #Load in teh SAC model
    model = SAC.load(model_path)

    #Fill teh buffer with unnoramlized actions
    model = preload_from_config(model,config,results_path=results_path,n_trajs_to_keep=n_trajs_to_keep,plot_success_from_demos=False,normalize_actions=True,include_non_expert=False)
    
    #just a poointer to vec_env
    current_idx = 0

    for seed in config.seeds:
        for numberPerSeed in range(n_trajs_to_keep):
            env.reset(seed=seed)
            print("")
            print("Current Seed: " + str(seed))
            print("")
            traj_done = False
            while traj_done==False:
                if current_idx>model.replay_buffer.pos:
                    print("CurrentIdx Exceed Buffer Position")
                    break

                #Get the action from teh filled buffer
                action_use = model.replay_buffer.actions[current_idx,0,:]
                action_use = model.policy.unscale_action(action_use)

                #Get the boservatoin for current_idx
                observation = {}
                for key in model.replay_buffer.observations.keys():
                    observation[key] = model.replay_buffer.observations[key][current_idx,0,:]

                #See the models prediction for this index
                action, _ = model.predict(observation, deterministic=True)

                #step, and get the observation
                results = env.step(action)
                real_obs = results[0]

                # 4) per-element percent error
                eps = 1e-8  # avoid division by zero
                denom = np.where(action_use == 0, eps, action_use)
                percent_error = np.abs((action - action_use) / denom) * 100


                # 5) nice printout
                if print_info is True:
                    print(f"True expert action:       {action_use}")
                    print(f"Model prediction:         {action}")
                    errors = [f"{e:.2f}%" for e in percent_error]
                    print(f"Percent error per element: [{', '.join(errors)}]")
                    print()

                #if this is the last point in the trajectory, traj_done to ture
                if model.replay_buffer.dones[current_idx] == True:
                    traj_done = True

                current_idx=current_idx+1
