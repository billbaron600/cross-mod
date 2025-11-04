# refactored module (auto-generated)


# ---- imports from original modules ----
from  utils.RLBenchFunctions.template_sensor_views import compute_camera_pose

from gymnasium import Wrapper, ObservationWrapper

from gymnasium.spaces import Box

from gymnasium.wrappers import TimeLimit

from imitation.data.types import Trajectory

from matplotlib.colors import ListedColormap     # ← add this

from pathlib import Path

from pyrep.objects.dummy import Dummy

from pyrep.objects.shape import Shape, PrimitiveShape

from pyrep.objects.vision_sensor import VisionSensor

from rlbench.action_modes.action_mode import MoveArmThenGripper

from rlbench.action_modes.action_mode import MoveArmThenGripper,JointPositionActionMode, ActionMode

from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK,EndEffectorPoseViaPlanning, JointPosition

from rlbench.action_modes.arm_action_modes import JointPosition

from rlbench.action_modes.arm_action_modes import JointVelocity,JointPosition,ArmActionMode,EndEffectorPoseViaIK

from rlbench.action_modes.gripper_action_modes import Discrete,GripperJointPosition

from rlbench.backend.scene import Scene

from rlbench.environment import Environment

from rlbench.gym import RLBenchEnv

from rlbench.tasks import SlideBlockToTarget

from scipy.interpolate import interp1d

from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation as R,RotationSpline

from scipy.spatial.transform import Slerp

from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from stable_baselines3 import PPO, SAC

from stable_baselines3 import SAC

from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from transforms3d.euler import quat2euler, euler2quat

from types import MethodType

from utils.Classes.policy_trajectory_class import PolicyTrajectory

from utils.RLBenchFunctions.combine_and_flatten_obs_and_action import flatten_observation, combine_observation_and_action

from utils.RLBenchFunctions.custom_action_modes import EndEffectorPoseViaPlanning,MoveArmThenGripperWithBounds,MoveArmThenGripperWithBoundsDelta_IVK, EndEffectorPoseViaPlanning_Custom, MoveArmThenGripperWithBoundsDelta,IVKPlanningBounds, IVKPlanningBounds_Delta

from utils.RLBenchFunctions.custom_action_modes import EndEffectorPoseViaPlanning_Record

from utils.RLBenchFunctions.custom_action_modes import EndEffectorPoseViaPlanning_Record,IVKPlanningBounds,IVKPlanningBounds_NonDiscrete

from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds, MoveArmThenGripperWithBoundsDelta

from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds,MoveArmThenGripperWithBoundsDelta_IVK, EndEffectorPoseViaPlanning_Custom, MoveArmThenGripperWithBoundsDelta

from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import save_sample_trajectories

from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import visualize_trajectories

import concurrent.futures

import copy

import cv2

import gc

import matplotlib.pyplot as plt

import numpy as np

import os

import os, pickle, math

import pickle

import random

import re

import secrets

import time

import torch



def set_task_objects_in_scene(env,starting_env_state):
    objects_in_tree = env.rlbench_env._scene.pyrep.get_objects_in_tree()
    set_objects = starting_env_state["scene_objects"]

    for i in objects_in_tree:
        object_name = i.get_name()
        if set_objects.get(object_name) is not None:
            pose = set_objects[object_name]["pose"]
            i.set_pose(pose)
            env.rlbench_env._scene.pyrep.step()
            print("set pose for " + object_name)

def get_object_quaternion(env=None,init_quat=None,object_name="Panda_gripper",relative_to_object=False,offset_by=None):
   
    if offset_by == [0,0,0]:
        print("No offset")

        return init_quat
    #gripper = env.rlbench_env._scene.robot.arm.get_tip()

    #gripper = env.rlbench_env._scene.robot.gripper
    #euler = gripper.get_orientation()
    # 1) Build a Rotation from your initial quaternion
    rot_initial = R.from_quat(init_quat)    
    # 2) Extract its Euler angles (in radians). 'xyz' = roll, pitch, yaw
    euler= rot_initial.as_euler('xyz') 
    

    euler_orig = copy.deepcopy(euler)

    if offset_by is not None:
        for angle in range(len(offset_by)):
            #secure_rng = random.SystemRandom()
            euler[angle] = euler[angle] + np.deg2rad(offset_by[angle])   #np.deg2rad(secure_rng.uniform(-offset_by[1][angle], offset_by[1][angle])) #
            #+ np.deg2rad(np.random.uniform(-offset_by[1][angle],offset_by[1][angle]))
        print("Changed euler angle")
            
        print("Euler Change:")
        print(np.rad2deg(euler_orig)-np.rad2deg(euler))
    
        #BILL
        euler = euler_orig + np.array([45,0,0])
        #print(euler)
        dummy = Dummy.create()
        dummy.set_orientation(euler)
        quat = dummy.get_quaternion()
        # 3) Build a new Rotation from the modified Euler angles
        
        return quat

class MaxStepWrapper(Wrapper):
    """
    A wrapper that enforces a maximum number of steps per episode.
    Also scales rewards and ends the episode early on success (with a bonus).
    
    Returns Gymnasium-style outputs:
      - reset() -> (obs, info)
      - step() -> (obs, reward, terminated, truncated, info)
    """

    def __init__(self, env, max_steps=2000,action_repeat=3,config=None,safety_factor=0.0,seed=0,pref_model = None, save_env_states = False,starting_env_state = None,divide_rewards_by=1,reward_scale=0.0, success_bonus=10.0,objects_in_env_to_log=None,db_index=None):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        self.reward_scale = reward_scale
        self.success_bonus = success_bonus
        self.safety_factor = safety_factor
        self.action_repeat = action_repeat

        if objects_in_env_to_log is None:
            self.objects_in_env_to_log = {"block":1}

        
        #If we do not pass in a preference based reward model, we will just use the scalar reward function defined in the task file
        if pref_model is None:
            self.reward_model = None
            #self.divide_preference_output_by = divide_preference_output_by # the output of the model is logits. We just divide by this to scale it if need be
        else:
            #self.reward_model = torch.jit.load(pref_model_path)
            self.reward_model = pref_model

        #Set the environment state replay buffer if we want to save the env state at every step. Important for corrections
        self.save_env_states = save_env_states
        if self.save_env_states is True:
            self.env_states = []
        

        #set the path to the env state we want to initialize to
        self.starting_env_state = starting_env_state

        self.divide_rewards_by = divide_rewards_by

        

        self.seed = seed
        self.current_seed = None
        self.number_rollouts = 0

        #self.current_action = None
        #self.action_steps = 0
        #self.action_step_lag = 10


        # 1) Load the updated preference database
        if config is not None:
            if db_index is not None:
                db_path = os.path.join(config.iteration_working_dir, "preference_database_"+str(db_index)+".pkl")
                with open(db_path, "rb") as f:
                    pref_db = pickle.load(f)
                self.feat_stats = pref_db.feat_stats
                print("Feature normalization limits set in Max Step Wrapper")
        else:
            print("Config set to None in MaxStepWrapper")
            self.feat_stats = None

        #self.policy_model = None

    def _randomise_start_config(self,
                            delta: float = 0.50,       # ≤ rad or ≤ fraction
                            settle_steps: int = 30):
        
        arm   = self.env.rlbench_env._scene.robot.arm
        joints = arm.joints
        start_joint_position = []

        for j in joints:
            q0 = j.get_joint_position()
            #print("Original:")
            #print(q0)
            

            _, (low, high) = j.get_joint_interval()

            low_limit  = max(low,  q0 - delta)
            high_limit = min(high, q0 + delta)

            #print("Set to:")
            set_to = random.uniform(low_limit, high_limit)
            #print(set_to)
            start_joint_position.append(set_to)

        start_joint_position = np.asarray(start_joint_position, np.float32)
        arm.set_joint_positions(start_joint_position)
        arm.set_joint_target_positions(start_joint_position)
        for _ in range(settle_steps):
            self.env.rlbench_env._scene.pyrep.step()

    def reset(self, **kwargs):
        """Gymnasium-style reset: returns (obs, info)."""
        
        self.current_step = 0
        self.grasped_yet = False
        self.total_steps_in_traj = 0
        
        
        
        seed = kwargs["seed"]
        print(seed)
        result = self.env.reset(**kwargs)
        if self.reward_model is not None:
            reward_model_present=True
            reward_model_holder = self.reward_model
            self.reward_model = None
        else:
            reward_model_present=False

            gripper_state = self.starting_env_state["gripper_state"][1]
            combined_action = np.append(start_joint_position,gripper_state)
            self.env.rlbench_env._scene.robot.arm.set_joint_positions(start_joint_position)
            for _ in range(5):
                self.env.rlbench_env._scene.pyrep.step()
            for _ in range(5):
                result=self.step(combined_action)
                
                self.current_step-=1
            for _ in range(10):
                result=self.step(combined_action)
                self.env.rlbench_task_env._scene.robot.arm.set_joint_target_velocities(np.zeros(7))
                self.current_step-=1
            
            scene_objects = self.starting_env_state["scene_objects"]
            objects_in_scene = self.env.rlbench_env._scene.pyrep.get_objects_in_tree()
            for i in objects_in_scene:
                if scene_objects.get(i.get_name()) is not None:
                    i.set_pose(scene_objects[i.get_name()]['pose'])

            result = (result[0],result[4])

        self.restart_action = np.concatenate([self.rlbench_task_env._scene.robot.arm.get_tip().get_pose(),np.array([1.0])])
        self.current_obs = result[0]
        
        
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}

        if reward_model_present is True:
            self.reward_model = reward_model_holder

        
        
        return obs, info

    def step(self, action):
    try:
        obs, env_reward, done, info = self._env.step(action)
        self.current_step += 1
        info = info if isinstance(info, dict) else {}
        if done:
            info.setdefault("terminated", True)
        return obs, env_reward, done, info
    except Exception as e:
        info = {"error": str(e)}
        return None, 0.0, True, info

    
    
    def _get_current_state(self):
       # Get the unwrapped environment
       unwrapped = self.env.unwrapped
      
       # Access the RLBench environment
       rlbench_env = unwrapped.rlbench_env
      
       # Find the robot
       robot = None
       if hasattr(rlbench_env, '_robot'):
           robot = rlbench_env._robot
       elif hasattr(rlbench_env, 'robot'):
           robot = rlbench_env.robot
       elif hasattr(rlbench_env, '_scene') and hasattr(rlbench_env._scene, 'robot'):
           robot = rlbench_env._scene.robot
      
       if robot is None:
           raise ValueError("Could not find robot in environment structure")
      
       # Extract state
       state = {
           'robot_state': robot.arm.get_joint_positions(),
           'gripper_state': robot.gripper.get_open_amount() if hasattr(robot, 'gripper') else None,
           'tip_pose':robot.arm.get_tip().get_pose(),
           'scene_objects': self._capture_scene_objects(rlbench_env)
       }
      
       return state


    def _capture_scene_objects(self, rlbench_env):
        #Capture scene objects based on environment structure.
        objects = {}

        # Try to find scene and objects
        scene = None
        if hasattr(rlbench_env, '_scene'):
            scene = rlbench_env._scene
        elif hasattr(rlbench_env, 'scene'):
            scene = rlbench_env.scene

        objects_list = self.env.rlbench_env._scene.pyrep.get_objects_in_tree()
        objects_list = scene.pyrep.get_objects_in_tree()
        for i in objects_list:
            if i.get_name() in self.objects_in_env_to_log:
                objects[i.get_name()] = {"pose": i.get_pose(),"orientation":i.get_orientation()}


        return objects

class MaxStepWrapper_Ablations(Wrapper):
    """
    A wrapper that enforces a maximum number of steps per episode.
    Also scales rewards and ends the episode early on success (with a bonus).
    
    Returns Gymnasium-style outputs:
      - reset() -> (obs, info)
      - step() -> (obs, reward, terminated, truncated, info)
    """

    def __init__(self, env, max_steps=50,action_repeat=3,config=None,safety_factor=0.0,seed=0,pref_model = None, save_env_states = False,starting_env_state = None,divide_rewards_by=1,reward_scale=0.0, success_bonus=10.0,objects_in_env_to_log=None,db_index=None):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        self.reward_scale = reward_scale
        self.success_bonus = success_bonus
        self.safety_factor = safety_factor
        self.action_repeat = action_repeat

        if objects_in_env_to_log is None:
            self.objects_in_env_to_log = {"block":1}

        
        #If we do not pass in a preference based reward model, we will just use the scalar reward function defined in the task file
        if pref_model is None:
            self.reward_model = None
            #self.divide_preference_output_by = divide_preference_output_by # the output of the model is logits. We just divide by this to scale it if need be
        else:
            #self.reward_model = torch.jit.load(pref_model_path)
            self.reward_model = pref_model

        #Set the environment state replay buffer if we want to save the env state at every step. Important for corrections
        self.save_env_states = save_env_states
        if self.save_env_states is True:
            self.env_states = []
        

        #set the path to the env state we want to initialize to
        self.starting_env_state = starting_env_state

        self.divide_rewards_by = divide_rewards_by

        

        self.seed = seed
        self.current_seed = None
        self.number_rollouts = 0

        #self.current_action = None
        #self.action_steps = 0
        #self.action_step_lag = 10


        # 1) Load the updated preference database
        if config is not None:
            if db_index is not None:
                db_path = os.path.join(config.iteration_working_dir, "preference_database_"+str(db_index)+".pkl")
                with open(db_path, "rb") as f:
                    pref_db = pickle.load(f)
                self.feat_stats = pref_db.feat_stats
                print("Feature normalization limits set in Max Step Wrapper")
        else:
            print("Config set to None in MaxStepWrapper")
            self.feat_stats = None

        #self.policy_model = None

    def _randomise_start_config(self,
                            delta: float = 0.50,       # ≤ rad or ≤ fraction
                            settle_steps: int = 30):
        
        arm   = self.env.rlbench_env._scene.robot.arm
        joints = arm.joints
        start_joint_position = []

        for j in joints:
            q0 = j.get_joint_position()
            #print("Original:")
            #print(q0)
            

            _, (low, high) = j.get_joint_interval()

            low_limit  = max(low,  q0 - delta)
            high_limit = min(high, q0 + delta)

            #print("Set to:")
            set_to = random.uniform(low_limit, high_limit)
            #print(set_to)
            start_joint_position.append(set_to)

        start_joint_position = np.asarray(start_joint_position, np.float32)
        arm.set_joint_positions(start_joint_position)
        arm.set_joint_target_positions(start_joint_position)
        for _ in range(settle_steps):
            self.env.rlbench_env._scene.pyrep.step()

    def reset(self, **kwargs):
        """Gymnasium-style reset: returns (obs, info)."""
        
        self.current_step = 0
        self.grasped_yet = False
        self.total_steps_in_traj = 0
        
        
        result = self.env.reset(**kwargs)
        self.current_obs = result[0]
        
        
        if isinstance(result, tuple):
            # If it is (obs, info)
            obs, info = result
        else:
            # If it is just obs
            obs, info = result, {}

        #print(self.env.rlbench_env._scene.task._variation_index)
        return obs, info

    def step(self, action):
        """
        Gymnasium-style step: returns (next_obs, reward, terminated, truncated, info).
        """



        


        
        result = self.env.step(action)
        self.current_step += 1
        self.total_steps_in_traj+=1
        next_obs, env_reward, terminated, truncated, info = result
        


        #print(reward)
        # Check RLBench success condition
        # (commonly: self.env.rlbench_task_env._task.success())
        done_flag, success = self.env.rlbench_task_env._task.success()
        if success:
            #reward += self.success_bonus
            #print("Got reward")
            env_reward=1.0
            terminated = True
            info["success"] = True
        else:
            info["success"] = False

        # Increment step count, possibly truncate
        #if (self.current_step >= self.max_steps):
        #    terminated = True

        # Save the current state of the environemtn
        if self.save_env_states is True:
            self.env_states.append(self._get_current_state())
            
        #Always leave truncated as false
        #truncated=False

        #set next obs to current obs
        self.current_obs = next_obs



        reward = env_reward
        return next_obs, reward, terminated, truncated, info
        # Get the unwrapped environment
    
    
    def _get_current_state(self):
       # Get the unwrapped environment
       unwrapped = self.env.unwrapped
      
       # Access the RLBench environment
       rlbench_env = unwrapped.rlbench_env
      
       # Find the robot
       robot = None
       if hasattr(rlbench_env, '_robot'):
           robot = rlbench_env._robot
       elif hasattr(rlbench_env, 'robot'):
           robot = rlbench_env.robot
       elif hasattr(rlbench_env, '_scene') and hasattr(rlbench_env._scene, 'robot'):
           robot = rlbench_env._scene.robot
      
       if robot is None:
           raise ValueError("Could not find robot in environment structure")
      
       # Extract state
       state = {
           'robot_state': robot.arm.get_joint_positions(),
           'gripper_state': robot.gripper.get_open_amount() if hasattr(robot, 'gripper') else None,
           'tip_pose':robot.arm.get_tip().get_pose(),
           'scene_objects': self._capture_scene_objects(rlbench_env)
       }
      
       return state


    def _capture_scene_objects(self, rlbench_env):
        #Capture scene objects based on environment structure.
        objects = {}

        # Try to find scene and objects
        scene = None
        if hasattr(rlbench_env, '_scene'):
            scene = rlbench_env._scene
        elif hasattr(rlbench_env, 'scene'):
            scene = rlbench_env.scene

        objects_list = self.env.rlbench_env._scene.pyrep.get_objects_in_tree()
        objects_list = scene.pyrep.get_objects_in_tree()
        for i in objects_list:
            if i.get_name() in self.objects_in_env_to_log:
                objects[i.get_name()] = {"pose": i.get_pose(),"orientation":i.get_orientation()}


        return objects

def make_env(rank: int,
             config,
             seed: int,
             **instantiate_kwargs):
    """
    Returns a thunk  _init()  that creates ONE environment.
    Needed by SubprocVecEnv / AsyncVectorEnv.
    `rank` ensures a unique seed per worker.
    """
    def _init():
        env = instantiate_environment(
            config=config,
            seed=seed + rank,
            **instantiate_kwargs
        )
        env = Monitor(env)            # keeps episode stats per worker
        return env
    return _init

def instantiate_environment(config=None,absolute_joints=False,action_repeat=1,task=SlideBlockToTarget,action_limits=None,safety_factor=0.0,pref_model=None,seed=None,render_mode="rgb_array",max_steps=200,shaped_rewards=True,save_env_states = True ,starting_env_state = None, divide_rewards_by=1,success_bonus=10,db_index=None):
    # Construct environment
    # Create an RLBench environment with our desired action mode.


    #action_mode = IVKPlanningBounds_Delta(
    #    arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=False),
    #    gripper_action_mode=GripperJointPosition()
    #)

    action_mode = IVKPlanningBounds(
        arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True),
        gripper_action_mode=Discrete()
    )
    
    base_env = RLBenchEnv(
        task,
        observation_mode='state', 
        action_mode=action_mode,
        render_mode=render_mode
    )
    # Enable shaped rewards
    base_env.rlbench_task_env._shaped_rewards = False
    base_env.reset(seed=seed)

    # Wrap with a max-step environment
    MAX_EPISODE_STEPS = max_steps
    env = MaxStepWrapper(base_env, action_repeat=action_repeat,config=config,safety_factor=safety_factor,max_steps=MAX_EPISODE_STEPS,pref_model=pref_model,save_env_states = save_env_states ,starting_env_state = starting_env_state,seed=seed,divide_rewards_by=divide_rewards_by,success_bonus=success_bonus,db_index=db_index)
    env.reset(seed=seed)
    return env

def instantiate_environment_ablations(config=None,absolute_joints=False,action_repeat=1,task=SlideBlockToTarget,action_limits=None,safety_factor=0.0,pref_model=None,seed=None,render_mode="rgb_array",max_steps=200,shaped_rewards=True,save_env_states = True ,starting_env_state = None, divide_rewards_by=1,success_bonus=10,db_index=None):
    # Construct environment
    # Create an RLBench environment with our desired action mode.


    #action_mode = IVKPlanningBounds_Delta(
    #    arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=False),
    #    gripper_action_mode=GripperJointPosition()
    #)

    
    #render_mode = "human"

    action_mode = MoveArmThenGripperWithBounds(
        arm_action_mode=JointPosition(),
        gripper_action_mode=GripperJointPosition(),
        action_limits=action_limits
        )
    
    base_env = RLBenchEnv(
        task,
        observation_mode='state', 
        action_mode=action_mode,
        render_mode=render_mode
    )
    # Enable shaped rewards
    base_env.rlbench_task_env._shaped_rewards = False
    base_env.reset(seed=seed)

    # Wrap with a max-step environment
    MAX_EPISODE_STEPS = max_steps
    env = MaxStepWrapper_Ablations(base_env,action_repeat=action_repeat,config=config,safety_factor=safety_factor,max_steps=MAX_EPISODE_STEPS,pref_model=pref_model,save_env_states = save_env_states ,starting_env_state = starting_env_state,seed=seed,divide_rewards_by=divide_rewards_by,success_bonus=success_bonus,db_index=db_index)
    env.reset(seed=seed)
    return env

def create_vectorized_env(config,n_envs = 8, base_seed = 42,render_mode="rgb_array",shaped_rewards=True,max_steps=100,db_index=None,safety_factor=0.0,clip_limit = (-8.0,8.0),divide_rewards_by = 1.0,success_bonus = 0.0):
    #Load in the reward model from the config files specified path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_reward_path = os.path.join(config.iteration_working_dir,"reward_model_ensemble.pt")
    reward_model = torch.load(current_reward_path)
    reward_model = reward_model.to(device)
    reward_model.eval()  # Optional: switch to eval mode

    #Create the vectorized environemnt
    env_fns = [
        make_env(rank=i,
                config=config,
                seed=base_seed,
                max_steps=max_steps,
                render_mode=render_mode,
                shaped_rewards=shaped_rewards,
                safety_factor=safety_factor,
                pref_model=reward_model,   # ← pass whatever kwargs you need
                divide_rewards_by=divide_rewards_by,
                success_bonus=success_bonus,
                db_index=db_index)
        for i in range(n_envs)
    ]

    vec_env = SubprocVecEnv(env_fns, start_method="spawn")   # safer with PyRep
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False)

    #Return the vectorized environemt
    return vec_env


class MaxStepWrapper_DEPRECATED(Wrapper):
    """
    A wrapper that enforces a maximum number of steps per episode.
    Also scales rewards and ends the episode early on success (with a bonus).
    
    Returns Gymnasium-style outputs:
      - reset() -> (obs, info)
      - step() -> (obs, reward, terminated, truncated, info)
    """

    def __init__(self, env, max_steps=2000,action_repeat=3,config=None,safety_factor=0.0,seed=0,pref_model = None, save_env_states = False,starting_env_state = None,divide_rewards_by=1,reward_scale=0.0, success_bonus=10.0,objects_in_env_to_log=None,db_index=None):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        self.reward_scale = reward_scale
        self.success_bonus = success_bonus
        self.safety_factor = safety_factor
        self.action_repeat = action_repeat

        if objects_in_env_to_log is None:
            self.objects_in_env_to_log = {"block":1}

        
        #If we do not pass in a preference based reward model, we will just use the scalar reward function defined in the task file
        if pref_model is None:
            self.reward_model = None
            #self.divide_preference_output_by = divide_preference_output_by # the output of the model is logits. We just divide by this to scale it if need be
        else:
            #self.reward_model = torch.jit.load(pref_model_path)
            self.reward_model = pref_model

        #Set the environment state replay buffer if we want to save the env state at every step. Important for corrections
        self.save_env_states = save_env_states
        if self.save_env_states is True:
            self.env_states = []
        

        #set the path to the env state we want to initialize to
        self.starting_env_state = starting_env_state

        self.divide_rewards_by = divide_rewards_by

        

        self.seed = seed
        self.current_seed = None
        self.number_rollouts = 0

        #self.current_action = None
        #self.action_steps = 0
        #self.action_step_lag = 10


        # 1) Load the updated preference database
        if config is not None:
            if db_index is not None:
                db_path = os.path.join(config.iteration_working_dir, "preference_database_"+str(db_index)+".pkl")
                with open(db_path, "rb") as f:
                    pref_db = pickle.load(f)
                self.feat_stats = pref_db.feat_stats
                print("Feature normalization limits set in Max Step Wrapper")
        else:
            print("Config set to None in MaxStepWrapper")
            self.feat_stats = None

        #self.policy_model = None

    def _randomise_start_config(self,
                            delta: float = 0.50,       # ≤ rad or ≤ fraction
                            settle_steps: int = 30):
        
        arm   = self.env.rlbench_env._scene.robot.arm
        joints = arm.joints
        start_joint_position = []

        for j in joints:
            q0 = j.get_joint_position()
            #print("Original:")
            #print(q0)
            

            _, (low, high) = j.get_joint_interval()

            low_limit  = max(low,  q0 - delta)
            high_limit = min(high, q0 + delta)

            #print("Set to:")
            set_to = random.uniform(low_limit, high_limit)
            #print(set_to)
            start_joint_position.append(set_to)

        start_joint_position = np.asarray(start_joint_position, np.float32)
        arm.set_joint_positions(start_joint_position)
        arm.set_joint_target_positions(start_joint_position)
        for _ in range(settle_steps):
            self.env.rlbench_env._scene.pyrep.step()

    def reset(self, **kwargs):
        """Gymnasium-style reset: returns (obs, info)."""
        
        self.current_step = 0
        self.grasped_yet = False
        self.total_steps_in_traj = 0
        
        
        #seed = secrets.randbelow(20) + 1
        #seeds = [3,9,10,15,19]
        #seed = seeds[secrets.randbelow(5)]
        
        #kwargs["seed"] = seed
        seed = kwargs["seed"]
        #seed = kwargs["seed"]
        print(seed)
        result = self.env.reset(**kwargs)
        #sampled_joints, sampled_grip = self._randomise_start_config()
        #self._randomise_start_config()
        #self.env.rlbench_env._scene.task._variation_index = self.seed
        if self.reward_model is not None:
            reward_model_present=True
            reward_model_holder = self.reward_model
            self.reward_model = None
        else:
            reward_model_present=False

        if self.starting_env_state is not None:
            #self.restore_state(self.starting_env_state)
            start_joint_position = self.starting_env_state["robot_state"]
            gripper_state = self.starting_env_state["gripper_state"][1]
            combined_action = np.append(start_joint_position,gripper_state)
            self.env.rlbench_env._scene.robot.arm.set_joint_positions(start_joint_position)
            for _ in range(5):
                self.env.rlbench_env._scene.pyrep.step()
            #self.env.rlbench_env._scene.robot.arm.set_joint_target_positions(gripper_state)
            for _ in range(5):
                result=self.step(combined_action)
                
                self.current_step-=1
            for _ in range(10):
                result=self.step(combined_action)
                self.env.rlbench_task_env._scene.robot.arm.set_joint_target_velocities(np.zeros(7))
                self.current_step-=1
            
            # set the position of the block
            scene_objects = self.starting_env_state["scene_objects"]
            objects_in_scene = self.env.rlbench_env._scene.pyrep.get_objects_in_tree()
            for i in objects_in_scene:
                if scene_objects.get(i.get_name()) is not None:
                    i.set_pose(scene_objects[i.get_name()]['pose'])

            result = (result[0],result[4])

            #start_gripper_position = self.gripper_pose[]
        self.restart_action = np.concatenate([self.rlbench_task_env._scene.robot.arm.get_tip().get_pose(),np.array([1.0])])
        self.current_obs = result[0]
        
        
        if isinstance(result, tuple):
            # If it is (obs, info)
            obs, info = result
        else:
            # If it is just obs
            obs, info = result, {}

        # Store obs if you need it
        # self.current_observation = obs
        #Set the reward model back to what is was
        if reward_model_present is True:
            self.reward_model = reward_model_holder

        
        #self.starting_pose = env.rlbench_task_env
        #self.first_obs = obs
        
        #print(self.env.rlbench_env._scene.task._variation_index)
        return obs, info

    def step(self, action):
        """
        Gymnasium-style step: returns (next_obs, reward, terminated, truncated, info).
        """


        
        
        
        #action[3:7] = q_unit

        #min_vals = np.array([-0.3249,-0.4550,0.7]) #min xyz values in the bounding box. Just dont have waypoitns outside this
        #max_vals = np.array([0.3249,0.4550,1.45])
        #min_vals = np.array([-1.0,-1.0,0.7]) #min xyz values in the bounding box. Just dont have waypoitns outside this
        #max_vals = np.array([1.0,1.0,1.45])

        #action[:3] = np.maximum(action[:len(min_vals)], min_vals)      # any value below its min is raised to that min
        #action[:3] = np.minimum(action[:len(max_vals)], max_vals)  # any value above its max is clipped down

        #print(action)

        


        try:
            result = self.env.step(action)
            self.current_step += 1
            self.total_steps_in_traj+=1
            next_obs, env_reward, terminated, truncated, info = result

            
            #if len(grasped_objects)>0:
            #    self.current_step-=1

            #get key poses
            ee_pose = self.rlbench_task_env._scene.robot.arm.get_tip().get_pose()

            for obj in self.rlbench_task_env._scene.pyrep.get_objects_in_tree():
                name = obj.get_name()
                if name == "success_centre":
                    success_center_pose = obj.get_pose()
                elif name == "pillar0":
                    pillar0_pose = obj.get_pose()
                elif name == "square_ring":
                    square_ring_pose = obj.get_pose()

            top_pillar_pose = success_center_pose
            top_pillar_pose[2] = pillar0_pose[2]

            #reward = - (distance from ee to square ring + distance from squarewring to top_pillar_pose)
            #first three elemtns of each is xyz, ignore quaternion stuff
            dist_ee_to_ring = np.linalg.norm(ee_pose[:3] - square_ring_pose[:3])
            dist_ring_to_top_pillar = np.linalg.norm(square_ring_pose[:3] - top_pillar_pose[:3])

            env_reward = 1/(dist_ee_to_ring + dist_ring_to_top_pillar)
            env_reward = env_reward/1000

            grasped_objects = self.rlbench_task_env._scene.robot.gripper.get_grasped_objects()
            if len(grasped_objects)>0 and self.grasped_yet == False:
                env_reward += 0.2
                self.grasped_yet = True
                print("Grasped")


            env_reward = min(env_reward,0.2)
            

        except Exception as e:

            next_obs = self.current_obs
            terminated = True
            truncated = True
            env_reward = 0.0
            info = {}
            self.current_step += 1
            #next_obs = self.first_obs

            """
            next_obs = self.current_obs
            env_reward = -0.01
            terminated = False
            truncated = False
            info = {}
            """

            
        #If preference based reward
        
        reward = env_reward
        #print(reward)
        # Check RLBench success condition
        # (commonly: self.env.rlbench_task_env._task.success())
        done_flag, success = self.env.rlbench_task_env._task.success()
        if success:
            #reward += self.success_bonus
            print("Got reward")
            env_reward+=1.0
            terminated = True
            info["success"] = True
        else:
            info["success"] = False

        # Increment step count, possibly truncate
        if (self.current_step >= self.max_steps) or self.total_steps_in_traj>50:
            terminated = True

        # Save the current state of the environemtn
        if self.save_env_states is True:
            self.env_states.append(self._get_current_state())
            
        #Always leave truncated as false
        #truncated=False

        #set next obs to current obs
        self.current_obs = next_obs



        
        return next_obs, reward, terminated, truncated, info
        # Get the unwrapped environment
    
    
    def _get_current_state(self):
       # Get the unwrapped environment
       unwrapped = self.env.unwrapped
      
       # Access the RLBench environment
       rlbench_env = unwrapped.rlbench_env
      
       # Find the robot
       robot = None
       if hasattr(rlbench_env, '_robot'):
           robot = rlbench_env._robot
       elif hasattr(rlbench_env, 'robot'):
           robot = rlbench_env.robot
       elif hasattr(rlbench_env, '_scene') and hasattr(rlbench_env._scene, 'robot'):
           robot = rlbench_env._scene.robot
      
       if robot is None:
           raise ValueError("Could not find robot in environment structure")
      
       # Extract state
       state = {
           'robot_state': robot.arm.get_joint_positions(),
           'gripper_state': robot.gripper.get_open_amount() if hasattr(robot, 'gripper') else None,
           'tip_pose':robot.arm.get_tip().get_pose(),
           'scene_objects': self._capture_scene_objects(rlbench_env)
       }
      
       return state


    def _capture_scene_objects(self, rlbench_env):
        #Capture scene objects based on environment structure.
        objects = {}

        # Try to find scene and objects
        scene = None
        if hasattr(rlbench_env, '_scene'):
            scene = rlbench_env._scene
        elif hasattr(rlbench_env, 'scene'):
            scene = rlbench_env.scene

        objects_list = self.env.rlbench_env._scene.pyrep.get_objects_in_tree()
        objects_list = scene.pyrep.get_objects_in_tree()
        for i in objects_list:
            if i.get_name() in self.objects_in_env_to_log:
                objects[i.get_name()] = {"pose": i.get_pose(),"orientation":i.get_orientation()}


        return objects