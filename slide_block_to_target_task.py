from utils.RLBenchFunctions.get_configuration_settings import Configuration
from rlbench.tasks import WipeDesk,SlideBlockToTarget
from utils.RLBenchFunctions.get_configuration_settings import Configuration
from rlbench.tasks import PutItemInDrawer
import pickle
from utils.RLBenchFunctions.extract_density import query_user, build_overview_grids, render_trajectories, generate_waypoint_gaussians
from utils.RLBenchFunctions.extract_density import run_generate_densities_for_all_images,create_combined_object
from utils.RLBenchFunctions.trajectory_generator import fit_NeRF_model
from utils.RLBenchFunctions.plottingFunctions.plot_generated_trajectories import visualize_trajectories, run_visualize_trajectories
from utils.RLBenchFunctions.generate_IVK_trajectory import execute_trajectory, run_ivk_for_corrections, generate_IVK_trajectories, apply_shifts_to_trajs
from utils.RLBenchFunctions.evaluate_policy import evaluate_policy, run_evaluate_policy
from stable_baselines3 import SAC,PPO
import os
from utils.RLBenchFunctions.custom_envs import instantiate_environment
import numpy as np
import matplotlib.pyplot as plt
from utils.RLBenchFunctions.generate_IVK_trajectory import plot_trajectory_grid
from utils.RLBenchFunctions.project_trajectory_onto_image import project_ray_traced_mean,overlay_sampled_trajectories
from utils.RLBenchFunctions.generate_IVK_trajectory import generate_interpolated_trajectories
from utils.RLBenchFunctions.create_frames_from_video import save_video_frames
from utils.RLBenchFunctions.extract_density import combine_segments

#Specify key paths
configuration_path = "slide_block_to_target_initial_demos.json"
path_to_config_pickle = "run_results/slide_block_to_target/demos/config_instance.pkl"
task = SlideBlockToTarget
results_path = "run_results/slide_block_to_target/demos/"


#Boolean values specifying what do do right now
#CONFIG
#CONFIG
create_config_file = True
generate_camera_views = False
load_config = True
use_intrinsics = True
real_demo = False
negative_camera_z = True
trajectory_indices = None

#SUCCESS SEEDS: [1,2,3,4,5,6,7,9,10,11,14,15,16,17,18,19,20]
#FAIL SEEDS: [12,13]


#Limit to correction indices
full_limit_to_correction_indices = [13]

#Easy fix: 18,19, 20




shift_mean = np.array([0,0,0.0]) #slight shift to the mean
min_vals = np.array([-10.0,-10.0,0.78]) #min xyz values in the bounding box. Just dont have waypoitns outside this
open_gripper = 0.3
#[8,9,10,11,12,13,14,15,16,17,18,19,20]


#NERF/TRAJECTORY GENERATION
n_segments = 1                                                  #number of segmetns for this task. If 1 contiuous path, then set this to 1
only_IVK = True

if only_IVK==False:
    query_user_for_sketches = False
    combine_segments_bool = False
    generate_combined_object = True # set this to true if you didnt run query_user_for_sketches and get the combined object
    create_continuous_path = False #If true, then we inteproalte between pixels to generate a continous path
    render_trajectories = False
    build_combined_figures = False
    generate_all_densities = True
    generate_gaussian_densities = False
    gaussian_noise_std = 5.0
    fit_NeRF_bool = True
    project_ray_tracing = True
    sample_trajectories_from_NeRF = True
    create_trajectory_scatter = False
else:
    query_user_for_sketches = False
    combine_segments_bool = False
    generate_combined_object = False  # set this to true if you didnt run query_user_for_sketches and get the combined object
    create_continuous_path = False    # If true, then we inteproalte between pixels to generate a continous path
    render_trajectories = False
    build_combined_figures = False
    generate_all_densities = False
    generate_gaussian_densities = False
    gaussian_noise_std = 5.0
    fit_NeRF_bool = False
    project_ray_tracing = False
    sample_trajectories_from_NeRF = False
    create_trajectory_scatter = False

#Set the other params
use_mean = True
run_IVK_bool = True
use_gripper_orientation_file = True
gripper_action_provided = False
record_ivk_video = True
update_full_ivk_pickle = False
plot_result_from_ivk = False
eval_on_buffer = False
evaluate_policy_bool = False

#Real demo stuff
generate_trajectories_with_orientations = False
apply_shifts = False
include_variance_in_samples = False
variance_scale = 0.01 # if include_variance in samples is set to false, we can still introduce a little bit of variance here


#For SEED 1 and 5 (shift down by half a centimeter)
#shift_mean = np.array([0,0,-0.005]) #shift the mean up by 0.02 meters



#shift_mean = None
mean_shift_indices = []


#Evaluatlion params
max_steps = 500
action_steps = 1
render_mode = "human"
num_episodes = 100
record_video = False
deterministic = False
print_info = False
model_idx = 50000
model_path = None
buffer_path = None
n_trajs_to_keep_for_buffer_analysis = 5


if create_config_file is True:
    # Load in the config for initial demos. Create the working directories 
    configuration_path = configuration_path
    config = Configuration(json_path=configuration_path,task=task)
    working_dirs = config.create_working_dirs(generate_camera_views=generate_camera_views)
    config.save_instance()

if load_config is True:
    with open(path_to_config_pickle, 'rb') as file:
        config = pickle.load(file)


#OTEHR ARGS
config.ivk_generation_kwargs["open_gripper"] = open_gripper
config.execute_trajectory_kwargs["offset_by"][1] = [0,0,-10]
first_pos = [0.3,0.0,1.45]

#Set the number of samples from the RBF kernel
distance_between_points = None #set to none to use the mean
NUM_POINTS_MEAN = config.ray_tracing_params["n_times"]
NUM_POINTS_VAR = 0
NUM_SPEEDS = 1
NUM_TRAJECTORIES = 1



for corr_idx in full_limit_to_correction_indices:
    limit_to_correction_indices = [corr_idx]
    for segment_idx in range(n_segments):
        if query_user_for_sketches is True:
            #Query the user to draw on the trajectories (make corrections)
            query_user(config,limit_to_correction_indices=limit_to_correction_indices)

        if generate_combined_object is True:
            create_combined_object(config,segment_idx=segment_idx,limit_to_correction_indices=limit_to_correction_indices,render_trajectories=render_trajectories,create_continuous_path=create_continuous_path)

        if build_combined_figures is True:
            build_overview_grids(config,segment_idx=segment_idx)

        if generate_all_densities is True:
            #Query the user to draw on the trajectories (make corrections)
            run_generate_densities_for_all_images(config,limit_to_correction_indices=limit_to_correction_indices,segment_idx=segment_idx)

        if generate_gaussian_densities is True:
            generate_waypoint_gaussians(config, limit_to_correction_indices=limit_to_correction_indices,noise_std=gaussian_noise_std)

        if fit_NeRF_bool is True:
            # Fit the NeRF model for each demonstartion
            fit_NeRF_model(config,limit_to_correction_indices=limit_to_correction_indices,mean_shift_indices=[],negative_z=negative_camera_z,segment_idx=segment_idx)

        if project_ray_tracing is True:
            project_ray_traced_mean(config,segment_idx=segment_idx,limit_to_correction_indices=limit_to_correction_indices,real_demo=real_demo,use_intrinsics=use_intrinsics)

    if n_segments>0 and combine_segments_bool == True:
        combine_segments(config,limit_to_correction_indices=limit_to_correction_indices,n_segments=n_segments)

    if project_ray_tracing is True:
        project_ray_traced_mean(config,limit_to_correction_indices=limit_to_correction_indices,real_demo=real_demo,use_intrinsics=use_intrinsics)

    if sample_trajectories_from_NeRF is True:
        run_visualize_trajectories(config,start_pos=first_pos,demos=real_demo,variance_scale=variance_scale,include_variance_in_samples=include_variance_in_samples,distance_between_points=distance_between_points,limit_to_correction_indices=limit_to_correction_indices,task=task,show_plot=False,num_trajectories=NUM_TRAJECTORIES,num_points=NUM_POINTS_MEAN,num_points_var=NUM_POINTS_VAR,num_speeds=NUM_SPEEDS)

    if create_trajectory_scatter ==True:
        overlay_sampled_trajectories(config,limit_to_correction_indices=limit_to_correction_indices)

    if run_IVK_bool is True:
        generate_IVK_trajectories(config,min_vals=min_vals,use_mean=use_mean,shift_mean=shift_mean,use_gripper_orientation_file=use_gripper_orientation_file,gripper_provided=gripper_action_provided,task=task,limit_to_correction_indices=limit_to_correction_indices,trajectory_indices=trajectory_indices)
        #Plot the demo data
        #run_ivk_for_corrections(config,task=task,limit_to_correction_indices=trajectory_indices)

if generate_trajectories_with_orientations is True:
    generate_interpolated_trajectories(config, init_orientation,target_orientations,target_indices,seed_idx=seed_idx_demo)

if apply_shifts is True:
    apply_shifts_to_trajs(config,seed_idx_demo=seed_idx_demo,cartesian_shift_amount=cartesian_shift_amount,indices_shift=indices_shift)

if update_full_ivk_pickle is True:
    merge_full_trajectories(config, results_path)

if plot_result_from_ivk is True:
    #Plot the demo data
    plot_trajectory_grid(results_path)

if eval_on_buffer is True:
    vec_env = instantiate_environment(config,action_repeat=action_steps,task=task,render_mode=render_mode,max_steps=max_steps,absolute_joints=False)
    test_policy_on_buffer(config,results_path=results_path,model_path=model_path,env=vec_env,n_trajs_to_keep=n_trajs_to_keep_for_buffer_analysis)


if evaluate_policy_bool is True:
    # Instantaite the env
    vec_env = instantiate_environment(config,action_repeat=action_steps,task=task,render_mode=render_mode,max_steps=max_steps,absolute_joints=False)
    path_to_model = model_path
    model = SAC.load(path_to_model,env=vec_env)
    model.policy.eval()
    print("Evaluating on Model at path: "+path_to_model)
    config.evaluate_policy_kwargs["record_video"] = record_video
    config.evaluate_policy_kwargs["deterministic"] = deterministic
    config.evaluate_policy_kwargs["num_episodes"] = num_episodes
    config.evaluate_policy_kwargs["print_info"] = print_info
    config.evaluate_policy_kwargs["max_steps"] =  max_steps
    success_rate,trajectories = run_evaluate_policy(config,env=vec_env,model=model)
    print("Success Rate:")
    print(success_rate)