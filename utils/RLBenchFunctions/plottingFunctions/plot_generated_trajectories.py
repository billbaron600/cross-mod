import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from rlbench.gym import RLBenchEnv
from matplotlib.cm import get_cmap
from utils.RLBenchFunctions.trajectory_generator import get_true_first_position
from rlbench.tasks import SlideBlockToTarget
import secrets
import os
import math

def split_evenly(total, num_parts):
    base = total // num_parts
    rem = total % num_parts
    # distribute the remainder to the last 'rem' parts
    return [base + 1 if i >= num_parts - rem else base for i in range(num_parts)]


def sample_clipped_int(mean, std):

    # draw a scalar sample from N(mean, std)
    #OLD
    #x = np.random.normal(loc=mean, scale=std)

    #NEW
    # 1) get a cryptographically‐random 32‐bit seed
    seed = secrets.randbits(32)
    
    # 2) use that to create a new NumPy Generator
    rng = np.random.default_rng(seed)
    
    # 3) draw one sample from N(mean, std)
    x = rng.normal(loc=mean, scale=std)

    # clip to [mean - std, mean + std]
    low, high = mean - std, mean + std
    x_clipped = np.clip(x, low, high)
    # round to nearest integer and convert to Python int
    return int(np.round(x_clipped))


def stack_with_nan_padding(trajs_generated_list):
    """
    trajs_generated_list: list of tensors shaped [k_i, L_i, 3]
                          (k_i trajectories, L_i time-steps)

    Returns
    --------
    stacked : torch.Tensor  # shape [sum(k_i), max(L_i), 3]
              Trajectories are stacked on dim 0; shorter ones
              are nan-padded on dim 1.
    """
    # total number of individual trajectories
    total_k = sum(t.shape[0] for t in trajs_generated_list)
    # longest trajectory length
    max_len = max(t.shape[1] for t in trajs_generated_list)

    # allocate output filled with nan
    first = trajs_generated_list[0]
    out = torch.full(
        (total_k, max_len, 3),
        float("nan"),
        dtype=first.dtype,
        device=first.device,
    )

    # copy blocks in order
    start = 0
    for block in trajs_generated_list:
        k, L, _ = block.shape
        out[start : start + k, :L, :] = block
        start += k

    return out


def save_sample_trajectories(base_path: str,
                             trajectories: torch.Tensor,units = 1e3):
    """
    Create  <base_path>/sample_trajectories  and save every trajectory as
    trajectory_0.npy, trajectory_1.npy, …  (float32).  Any NaN-padded rows
    are removed before saving.

    Parameters
    ----------
    base_path     : str
        Directory that already contains the run's outputs (same root the other
        helper uses).
    trajectories  : torch.Tensor  (M, N, 3)
        Stack of M trajectories with possible NaN padding in the row dimension.
    """
    out_dir = os.path.join(base_path, "sample_trajectories")
    os.makedirs(out_dir, exist_ok=True)

    for idx, traj in enumerate(trajectories):
        # strip NaNs, detach from autograd, move to CPU, cast to float32
        clean = traj.detach()[~torch.isnan(traj[:, 0])].cpu().numpy().astype(np.float32)
        clean[:,:3] = clean[:,:3] * units #put in mm
        np.save(os.path.join(out_dir, f"trajectory_{idx}.npy"), clean)
        print(f"saved  {out_dir}/trajectory_{idx}.npy")


def visualize_trajectories(start_p=None,task=None,variance_scale=None,include_variance_in_samples=False,working_dir=None, show_plot=False,num_trajectories=10,num_points=1000,num_points_var=50,num_speeds=1,object_position=None):
    """
    Load trajectory distributions, generate and visualize trajectories in RLBench.
    
    Parameters:
        working_dir (str): Path to the working directory.
        show_plot (bool): Whether to display the plots.
    """
    # Load trajectory distribution
    with open(f'{working_dir}/trajectory_dist.pkl', 'rb') as f:
        trajectory_dist = pickle.load(f)
    
    device = "cuda"
    # Initialize RLBench environment
    if start_p is None:
        env = RLBenchEnv(task, observation_mode='state', render_mode='rgb_array')
        env.rlbench_task_env._shaped_rewards = True
        env.reset(seed=0)
        
        # Get end effector position
        
        tip_dummy = env.rlbench_task_env._robot.arm.get_tip()
        tip_pos = tip_dummy.get_position()

        start_p = torch.tensor(tip_pos,device=device)
        env.close()
    else:
        start_p = torch.tensor(start_p, device=device)
    #start_p = torch.tensor(tip_pos).to(device)

    
    #ORIGINAL
    #trajs_generated = trajectory_dist.condition_start(start_p, num_trajectories, num_points)

    #NEW
    trajs_generated_list = []
    num_per_speed = split_evenly(num_trajectories, num_speeds)

    for num_for_this_speed in num_per_speed:
        #Sample the distribution dfeind by num_points (mean) and num_points_var to get the desired number of opints in this trajectroy
        number_trajectory_points = sample_clipped_int(num_points, num_points_var)

        #This returns a torch tensor of size [num_for_this_speed,num_trajectory_points,3]
        n_trajs_generated = trajectory_dist.condition_start(start_p,num_for_this_speed,number_trajectory_points,include_variance_in_samples=include_variance_in_samples,variance_scale=variance_scale)

        #Append to trajs_generates_list. creates a torch tensor of size [num_trajectories,N,3]. Here, N 
        # should be the max middle dimensino of all the tensors in the trajs_generated_list. For the
        #  tensors that are smalelr in this demisnosn, fill them out with nan (so everything can be properly concatenated.)
        trajs_generated_list.append(n_trajs_generated)

    #Cocatenate: 
    trajs_generated = stack_with_nan_padding(trajs_generated_list)
    

    # END NEW
    # Visualization - 3D Trajectory Plot
    fig_3d = plt.figure(figsize=(10, 10))
    ax = fig_3d.add_subplot(projection='3d')
    cmap = get_cmap("viridis")

    for i in range(len(trajs_generated)):
        cur_pos_tor = trajs_generated[i]
        color = cmap(i / len(trajs_generated))
        ax.scatter(cur_pos_tor[:, 0].cpu().detach(),
                    cur_pos_tor[:, 1].cpu().detach(),
                    cur_pos_tor[:, 2].cpu().detach(),
                    s=22, alpha=1, color=color, label=f"Traj {i}")

    ax.view_init(45, 45)
    ax.set_xlabel("X", fontsize=14)
    ax.set_ylabel("Y", fontsize=14)
    ax.set_zlabel("Z", fontsize=14)
    ax.legend()
    if object_position is not None:
        ax.scatter(
            object_position[0], 
            object_position[1], 
            object_position[2], 
            color='red', s=500, label='Object Position', edgecolors='red', alpha=1.0
        )
        ax.legend()
    plt.savefig(f'{working_dir}/trajectory_3d_plot.png')
    if show_plot:
        plt.show()
    plt.close(fig_3d)

    # Time-Series Plots with Gradient Coloring and External Legend
    num_trajs = trajs_generated.shape[0]
    num_time_steps = trajs_generated.shape[1]
    time_array = np.linspace(0, 1, num_time_steps)

    fig_ts, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig_ts.subplots_adjust(right=0.8)  # Add padding to the right side for the legend

    handles = []
    labels = []

    for i in range(num_trajs):
        cur_traj = trajs_generated[i].cpu().detach().numpy()
        color = cmap(i / num_trajs)
        handle, = axs[0].plot(time_array, cur_traj[:, 0], label=f"Traj {i}", color=color)
        axs[1].plot(time_array, cur_traj[:, 1], label=f"Traj {i}", color=color)
        axs[2].plot(time_array, cur_traj[:, 2], label=f"Traj {i}", color=color)
        handles.append(handle)
        labels.append(f"Traj {i}")

    axs[0].set_ylabel("X", fontsize=12)
    axs[1].set_ylabel("Y", fontsize=12)
    axs[2].set_ylabel("Z", fontsize=12)
    axs[2].set_xlabel("Time", fontsize=12)

    fig_ts.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))  # Legend outside

    if object_position is not None:
        # Plot horizontal reference lines for X, Y, and Z
        axs[0].axhline(object_position[0], color='red', linestyle='--', linewidth=2, label='Object X')
        axs[1].axhline(object_position[1], color='red', linestyle='--', linewidth=2, label='Object Y')
        axs[2].axhline(object_position[2], color='red', linestyle='--', linewidth=2, label='Object Z')

        # Update the legend with object position
        handles.append(plt.Line2D([], [], color='red', linestyle='--', linewidth=2))
        labels.append('Object Position')

    fig_ts.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(f'{working_dir}/trajectory_time_series.png', bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(fig_ts)
    
    

    return trajs_generated

def run_visualize_trajectories(config,start_pos=None,task=SlideBlockToTarget,variance_scale=None,include_variance_in_samples=True,distance_between_points=None,demos=False,show_plot=False,num_trajectories=30,num_points=100,num_points_var = 50,num_speeds=1,limit_to_correction_indices=None):
    from utils.RLBenchFunctions.trajectory_generator import get_true_first_position
    from utils.RLBenchFunctions.project_trajectory_onto_image import project_ray_traced_trajectories
    working_dirs = config.working_dirs
    
    if start_pos is None:
        if limit_to_correction_indices is None or demos==True:
            first_positions = None
        else:
            first_positions = get_true_first_position(config)

    if limit_to_correction_indices is None:
        limit_to_correction_indices = list(range(len(working_dirs)))
    else:
        first_positions = [start_pos]

    for list_idx in range(0,len(limit_to_correction_indices)):
        idx = limit_to_correction_indices[list_idx]
        working_dir = working_dirs[idx]
        #specify key variables

        # Define the path to the pickle file
        working_folder = working_dir

        pickle_path = os.path.join(working_folder, "grand_traj_tor_r.pkl")
        with open(pickle_path, "rb") as f:
            grand_traj_tor_r = pickle.load(f)
        
        
        
        #USE THE SAME NUMBER OF POITNS
        num_points = grand_traj_tor_r[0][0].shape[1] - 1


        if first_positions is not None and start_pos is None:
            start_pos = first_positions[list_idx][:3]
        #else:
        #    start_pos = None
    
        if distance_between_points is not None:
            temp_traj_generation = visualize_trajectories(start_p=start_pos,task=task,working_dir=working_dir,show_plot=False,num_trajectories=1,num_points=300,num_points_var = 0,num_speeds=1)
            template_traj = temp_traj_generation[0]
            pos = template_traj
            total_displacement = diff   = pos[1:] - pos[:-1]      # (N-1, 3)   vector difference
            dists  = diff.norm(dim=1)        # (N-1,)     Euclidean distance
            L      = dists.sum().item()                  # total path length
            h      = distance_between_points                                # desired spacing (example)

            n_intervals = math.ceil(L / h)
            num_points    = n_intervals + 1
            print("Changed Number of Points to: " + str(num_points))
            

        trajs_generated = visualize_trajectories(start_p=start_pos,variance_scale=variance_scale,include_variance_in_samples=include_variance_in_samples,task=task,working_dir=working_dir,show_plot=show_plot,num_trajectories=num_trajectories,num_points=num_points,num_points_var = num_points_var,num_speeds=num_speeds)
        project_ray_traced_trajectories(config,generate_trajectories=trajs_generated,limit_to_correction_indices=limit_to_correction_indices,real_demo=True,use_intrinsics=True)

        base_path = os.path.join(config.iteration_working_dir, str(idx))
        save_sample_trajectories(base_path, trajs_generated)
        #save the genreated traejctoreis
        # Save
        with open(working_dir+'generated_trajectories.pkl', 'wb') as f:
            pickle.dump(trajs_generated, f)

        #convert to numpy and save that way
        trajs_generated_numpy = trajs_generated.detach().cpu().numpy()
        with open(working_dir+'generated_trajectories_numpy.pkl', 'wb') as f:
            pickle.dump(trajs_generated_numpy, f)