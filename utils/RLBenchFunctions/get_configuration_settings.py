import json
import numpy as np
from rlbench.tasks import SlideBlockToTarget
from utils.RLBenchFunctions.template_sensor_views import generate_camera_view,compute_camera_pose
import pickle
from utils.RLBenchFunctions.custom_envs import instantiate_environment
import os
import torch


class Configuration:
    def __init__(self, json_path="config.json",task=None,real_demo=False):
        with open(json_path, 'r') as file:
            cfg = json.load(file)

        # Load seeds
        self.real_demo = real_demo
        self.seeds = cfg.get('seeds', [])
        self.number_corrections = cfg.get('number_corrections',None)
        self.frame_correction_indices = cfg.get("frame_correction_indices",[])
        self.frame_correction_indices_end = cfg.get("frame_correction_indices_end",[])
        self.path_to_current_trajectories = cfg.get("path_to_current_trajectories",None)
        self.iteration_working_dir = cfg.get("iteration_working_dir",None)
        self.path_to_current_policy_model = cfg.get("path_to_current_policy_model",None)
        self.path_to_current_replay_buffer = cfg.get("path_to_current_replay_buffer",None)
        self.preference_model_kwargs = cfg.get("preference_model_kwargs",None)

        # Load and reconstruct camera_view_config
        if real_demo == True:
            self.generate_camera_param_files(seeds=self.seeds)
            self.camera_view_config = {}
            self.camera_view_config["save_folder"] = cfg['camera_view_config']['save_folder']
        else:
            cam_cfg = cfg['camera_view_config']
            camera_positions = []
            for cam in cam_cfg['camera_positions']:
                position = np.array(cam['position'])
                center_point = np.array(cam['center_point']) if cam['center_point'] else None
                pose = compute_camera_pose(position, center_point=center_point) if center_point is not None else compute_camera_pose(position)
                camera_positions.append(pose)

            cam_cfg['camera_positions'] = camera_positions
            cam_cfg['task'] = task
            self.camera_view_config = cam_cfg

        #set other params
        self.density_generation_params = cfg['density_generation_params']
        self.ray_tracing_params = cfg['ray_tracing_params']
        self.train_policy_kwargs = cfg["train_policy_kwargs"]
        self.evaluate_policy_kwargs = cfg["evaluate_policy_kwargs"]

        # Load draw_traj_config
        self.draw_traj_config = cfg['draw_traj_config']
        self.ivk_generation_kwargs = cfg['ivk_generation_kwargs']
        self.preference_database_path = cfg["preference_database_path"]
        self.path_to_reward_model = cfg["path_to_reward_model"]

        self.execute_trajectory_kwargs = cfg.get("execute_trajectory_kwargs",None)
        
        
    
    def generate_camera_param_files(self,seeds=[]):
        for seed in seeds:
            path_to_seed = os.path.join(self.iteration_working_dir,str(seed))
            extrinsic_data = os.path.join(path_to_seed,"cam_extrinsics.npy")
            extrinsics = np.load(extrinsic_data)
            num_cams = extrinsics.shape[0]
            intrinsics_list = []
            for i in range(num_cams):
                intrinsic_path = os.path.join(path_to_seed,"cam_intrinsic_"+str(i)+".npy")
                instrinsic = np.load(intrinsic_path)
                intrinsics_list.append(instrinsic)
            
            intrinsics = np.stack(intrinsics_list, axis=0)
            # convert to torch.Tensor
            extrinsics_tensor = torch.from_numpy(extrinsics)            # dtype will match the numpy dtype
            extrinsics_tensor[:, :3, 3] /= 1000.0 #mm to meters

            #flip_yz = torch.tensor([[ 1.,  0.,  0., 0.],
            #            [ 0., -1.,  0., 0.],
            #            [ 0.,  0., -1., 0.],
            #            [ 0.,  0.,  0., 1.]], 
            #           dtype=extrinsics_tensor.dtype,
            #           device=extrinsics_tensor.device)

            #extrinsics_tensor_flipped = extrinsics_tensor @ flip_yz
            #extrinsics_tensor = extrinsics_tensor_flipped

            #perm = torch.tensor([1, 0, 2])
            #0,1,2
            #2,1,0
            #1,2,0
            #0,2,1
            #2,0,1
            #1,0,2

            #extrinsics_tensor[:, :3, :3] = extrinsics_tensor[:, perm][:, :, perm]  # rot rows & cols
            #extrinsics_tensor[:, :3, 3]  = extrinsics_tensor[:, perm, 3]           # translation

            #flip_z = torch.tensor([1, -1, -1],dtype=torch.float64)
            #for i in range(extrinsics_tensor.shape[0]):
                #extrinsics_tensor[i,:3,:3] = extrinsics_tensor[i,:3,:3] @ flip_z
            #    extrinsics_tensor[i,:3,:3] = torch.linalg.inv(extrinsics_tensor[i,:3,:3])

            intrinsic_mats     = torch.from_numpy(intrinsics)

            # build save dict and write out
            save_path = os.path.join(path_to_seed, "poses_mobile.tar")
            torch.save({
                "extrinsics": extrinsics_tensor,   # shape [num_cams, 4, 4]
                "intrinsics": intrinsic_mats       # shape [num_cams, 3, 3]
            }, save_path)

            print(f"  → saved poses to {save_path}")



    def create_working_dirs(self, save_pkl=False, limit_to_correction_indices=None,pkl_filename='working_dirs.pkl',generate_camera_views=True):
        self.working_dirs = []


        if self.real_demo is True:
            for i in self.seeds:
                seed_path = os.path.join(self.iteration_working_dir,str(i))
                seed_path = seed_path + "/"
                self.working_dirs.append(seed_path)
        else:
            if len(self.seeds)>0:
                if generate_camera_views is True:
                    if limit_to_correction_indices is None:
                        limit_to_correction_indices = self.seeds
                    for seed in limit_to_correction_indices:
                        # Set seed in camera_view_config
                        self.camera_view_config['seed'] = seed

                        # Generate camera views
                        frames, extrinsics, working_folder = generate_camera_view(**self.camera_view_config)
                        self.working_dirs.append(working_folder)
                #else:
                    #for seed in self.seeds:
                    #    working_folder = os.path.join(self.iteration_working_dir,str(seed))
                    #    working_folder = working_folder + "/"
                    #    self.working_dirs.append(working_folder)
            else:
                self.create_working_dirs_for_corrections()

        self.working_dirs = []
        for seed in self.seeds:
            working_folder = os.path.join(self.iteration_working_dir,str(seed))
            working_folder = working_folder + "/"
            self.working_dirs.append(working_folder)

        # Optionally save the paths to a pickle file
        if save_pkl:
            with open(pkl_filename, 'wb') as f:
                pickle.dump(self.working_dirs, f)

        return self.working_dirs
    
    def create_working_dirs_for_corrections(self):
        file_path = self.path_to_current_trajectories
        with open(file_path, "rb") as f:
            loaded_trajectories = pickle.load(f)
        for i in range(self.number_corrections):
            current_traj = loaded_trajectories[i]
            env_state = current_traj.environment_states[self.frame_correction_indices[i]]
            env = instantiate_environment(seed=current_traj.env_seed,starting_env_state=env_state)
            #obs,_ = env.reset(seed=current_traj.env_seed)

            frames, extrinsics, working_folder = generate_camera_view(env=env,**self.camera_view_config,save_with_seed_name=False,correction_index=i)
            self.working_dirs.append(working_folder)

    def save_instance(self, filename="config_instance.pkl",print_path=True):
        save_folder = self.camera_view_config.get("save_folder", "./")
        os.makedirs(save_folder, exist_ok=True)
        full_path = os.path.join(save_folder, filename)
        self.camera_view_config['task'] = None

        with open(full_path, 'wb') as f:
            pickle.dump(self, f)

        if print_path:
            print(f"Instance saved successfully to '{full_path}'")