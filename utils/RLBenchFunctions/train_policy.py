from rlbench.gym import RLBenchEnv
from rlbench.tasks import SlideBlockToTarget
from stable_baselines3 import PPO, SAC
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper,JointPositionActionMode, ActionMode
from rlbench.action_modes.arm_action_modes import JointVelocity,JointPosition,ArmActionMode
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import SlideBlockToTarget
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.backend.scene import Scene
#from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
import random
from gymnasium import Wrapper, ObservationWrapper
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds
from utils.RLBenchFunctions.custom_envs import MaxStepWrapper
import torch
from stable_baselines3 import SAC
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
#from utils.RLBenchFunctions.fill_replay_buffer import load_all_trajectories,fill_replay_buffer
from rlbench.action_modes.arm_action_modes import JointPosition,EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds
import numpy as np
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle
import os

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch
from torch.optim import Adam
import torch as th
from stable_baselines3 import SAC,PPO
import torch as th
import numpy as np
from scipy.spatial.transform import Rotation as R
#from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import math
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor   import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from utils.RLBenchFunctions.custom_envs import instantiate_environment
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from utils.Classes.custom_replay_buffer import ScaledRewardBuffer
#from utils.RLBenchFunctions.vec_env_utils import make_env

class RLBenchLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # You can access env here: self.training_env
        # If using Monitor wrapper with info dicts:
        info = self.locals.get("infos", [{}])[0]
        if "success" in info:
            self.logger.record("rollout/success_rate", float(info["success"]))
        return True

class EntCoefScheduleCallback(BaseCallback):
    def __init__(self, schedule_fn, verbose=0):
        """
        :param schedule_fn: A function that takes current timestep and returns new entropy coefficient
        """
        super().__init__(verbose)
        self.schedule_fn = schedule_fn

    def _on_training_start(self) -> None:
        self._update_ent_coef(self.schedule_fn(0))

    def _on_step(self) -> bool:
        new_ent_coef = self.schedule_fn(self.num_timesteps)
        self._update_ent_coef(new_ent_coef)
        return True

    def _update_ent_coef(self, new_alpha: float):
        if hasattr(self.model, "ent_coef_tensor"):
            with torch.no_grad():
                self.model.ent_coef_tensor.fill_(new_alpha)
        else:
            raise AttributeError("SAC model does not have `ent_coef_tensor` in this version.")





def train_policy(env,model = None,model_starting_index=0,max_steps_std=0,reset_initial_timesteps=True,model_hps = {}, task=SlideBlockToTarget,shaped_rewards=False,policy_kwargs=None,log_dir="sac_tensorboard",action_mode=None,max_steps = 100,render_mode='rgb_array',total_train_steps = 10000,log_every = 2000,log_buffer_every=100000,run_name=None,output_path="SAC_models",print_paths=True, extra_callbacks=None):
    # Set action mode
    # Instatie envirionemntlearning_rate
    # Create model if not passed in
    # Find loop iteartion step: total_timesteps/log_every
    # Itearte, and call .learn each time
    # log the model there
    #import globals
    
    
    #Step 1: Wrap the env in a Moniator objeect for loggins
    #env = Monitor(env)
    #env = Monitor(env)

    #Step 3: Create model if one is not passed in
    
    if model is None:
        #model_hps["policy_kwargs"] = policy_kwargs
        #model = PPO(
         #   policy="MultiInputPolicy",
         #   env=env,
         #   tensorboard_log=log_dir,
         #   device="cuda",
         #   **model_hps
        #)
        #model_hps["policy_kwargs"] = policy_kwargs
        #model = SAC(
        #    policy="MultiInputPolicy",
        #    env=env,
        #    tensorboard_log=log_dir,
        #    device="cuda",
        #    **model_hps)
        print("Model not set")
    else:
        print("Set model env")
        #model.set_env(env)
        
        model.set_env(env)
    
    #Set the policy model attribute
    #globals.MODEL = model
    
    # Step 4: Do the outer iterations
    iterations = math.ceil(total_train_steps/log_every)
    
    #Make the model directoy if it does not exist
    os.makedirs(output_path,exist_ok=True)

    # Step 5: Iteartae and train
    # Build the final callback list once (inside the for‑loop we just reuse it)
    base_cb = RLBenchLoggingCallback()
    if extra_callbacks is None:
        cb_list = base_cb
    else:
        # ensure user‑supplied callbacks run **as well** as the built‑in one
        cb_items = [base_cb] + ([extra_callbacks] if not isinstance(extra_callbacks, list) else extra_callbacks)
        cb_list  = CallbackList(cb_items)

    #get the max steps mean
    max_steps_mean = env.max_steps    


    for i in range(0,iterations):
        #model.policy.reset_noise()
    #for i in range(0,iterations):

        if max_steps_std>0:
            env.max_steps = max(int(round(np.random.normal(max_steps_mean, max_steps_std))),20)
            
        if i==0 and reset_initial_timesteps==True:
            reset_timesteps = True
        else:
            reset_timesteps=False
        #Train the model
        if run_name==None:
            #model.learn(total_timesteps=log_every,log_interval=1,progress_bar=True,callback=RLBenchLoggingCallback(),reset_num_timesteps=reset_timesteps)
            model.learn(total_timesteps=log_every,log_interval=1,progress_bar=True,callback=cb_list, reset_num_timesteps=reset_timesteps)
        else:
            #model.learn(total_timesteps=log_every,log_interval=1,progress_bar=True,callback=RLBenchLoggingCallback(),reset_num_timesteps=reset_timesteps,tb_log_name=run_name)
            model.learn(total_timesteps=log_every,log_interval=1,progress_bar=True,callback=cb_list, reset_num_timesteps=reset_timesteps,tb_log_name=run_name)

        
        # Paths to save to
        save_path_model = os.path.join(output_path,"model_"+str(i*log_every + log_every + model_starting_index))
        save_path_buffer = os.path.join(output_path,"buffer_"+str(i*log_every + log_every + model_starting_index))
        
        
        # 2) Save the VecNormalize stats
        #env.save(output_path+"buffer_"+str(i*log_every + log_every)+"_vec_normalize_stats.pkl")
        
        # SAve model
        model.save(save_path_model)
        if hasattr(model,"save_replay_buffer"):
            model.save_replay_buffer(save_path_buffer)
        

        if print_paths==True:
            print(f"Model saved to: {save_path_model}")
            print(f"Replay buffer saved to: {save_path_buffer}")

    #final model and replay buffer save into general file
    model.save("sac_slide_block_expert")
    #model.save_replay_buffer("sac_replay_buffer.pkl")
    #Shutdown environemtn
    env.close()

    #return the model
    return model



def run_train_policy(config,env=None,pref_model=None,model=None,db_index=None,max_steps_std=0,safety_factor=1.0,model_starting_index=0,max_steps=200,include_replay_buffer=True,divide_rewards_by=1,success_bonus=10,render_mode="rgb_array",extra_callbacks = None,shaped_rewards=True):
    #load in the reward model
    
    train_policy_kwargs = config.train_policy_kwargs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref_model is not None:
        current_reward_path = os.path.join(config.iteration_working_dir,"reward_model_ensemble.pt")
        reward_model = torch.load(current_reward_path)
        reward_model = reward_model.to(device)
        reward_model.eval()  # Optional: switch to eval mode
        
    else:
        train_policy_kwargs["model"] = None

    # Set the train policy kwargs
    
    train_policy_kwargs["model"]=model
    # Instatiate the environemtn, with the desire kwargs
    #env = instantiate_environment(config=config,pref_model=reward_model,db_index=db_index,safety_factor=safety_factor,divide_rewards_by=divide_rewards_by,success_bonus=success_bonus,render_mode=render_mode,max_steps=max_steps)
    
    
    #model = PPO.load(model)
    

    
    #Train the policy train the policy
    policy_model = train_policy(env,model_starting_index=model_starting_index,max_steps_std=max_steps_std,extra_callbacks = extra_callbacks,**train_policy_kwargs)
    return policy_model