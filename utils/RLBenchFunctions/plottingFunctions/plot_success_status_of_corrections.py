import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pickle
import os

def plot_correction_results(config,show_plots=False,save_figures=True):
    plot_success_status_of_corrections(config,show_plot=show_plots,save_figure=save_figures)
    plot_correction_length(config,show_plot=show_plots,save_figure=save_figures)


def plot_success_status_of_corrections(config,show_plot=False,save_figure=True):
    #load in the correction object
    with open(os.path.join(config.iteration_working_dir,"all_corrections.pkl"), 'rb') as file:
        all_corrections = pickle.load(file)

    # OLD
    #data = np.array([correction.correction_success_status for correction in all_corrections])

    #NEW
    # 2) Extract the list of boolean‐lists
    data_lists = [corr.correction_success_status for corr in all_corrections]

    # 3) Find the maximum length
    max_len = max(len(lst) for lst in data_lists) if data_lists else 0

    # 4) Allocate a (N × max_len) boolean array, default False
    N = len(data_lists)
    data = np.zeros((N, max_len), dtype=bool)

    # 5) Copy each list into the leftmost columns, leaving the rest False
    for i, lst in enumerate(data_lists):
        data[i, : len(lst)] = lst

    #END NEW


    # Create a color map: green for True, red for False
    color_map = np.where(data, 'green', 'red')

    # Plotting
    fig, ax = plt.subplots(figsize=(len(data[0]), len(data)))

    # Create a colored grid with black borders
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            rect = plt.Rectangle((j, i), 1, 1, 
                                facecolor=color_map[i, j], 
                                edgecolor='black', 
                                linewidth=1)
            ax.add_patch(rect)

    # Set axis limits and labels
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(0, data.shape[0])
    ax.set_xticks(np.arange(data.shape[1]) + 0.5)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5)
    ax.set_xticklabels([f"Trajectory Sample {j}" for j in range(data.shape[1])], rotation=90)
    ax.set_yticklabels([f"User Correction {i}" for i in range(data.shape[0])])
    ax.invert_yaxis()  # Optional: Match matrix indexing
    ax.set_aspect('equal')
    ax.tick_params(top=False, bottom=False, left=False, right=False, labeltop=False)

    # Add title
    plt.title("Successful Task Completion Status for Samples from User Drawings", fontsize=14, pad=20)

    plt.tight_layout()
    if show_plot is True:
        plt.show()
    
    if save_figure:
        file_path = os.path.join(config.iteration_working_dir,"correction_trajectory_success_status.png")
        plt.savefig(file_path,dpi=300,bbox_inches='tight')

def plot_correction_length(config,show_plot=False,save_figure=True):
    #load in the correction object
    with open(os.path.join(config.iteration_working_dir,"all_corrections.pkl"), 'rb') as file:
        all_corrections = pickle.load(file)

    # Build the matrix of action lengths
    #OLD
    #data = np.array([
    #    [len(correction.actions) for correction in user_correction.corrections]
    #    for user_correction in all_corrections
    #])

    #NEW
    # 2) Build a list‐of‐lists of action‐lengths
    data_lists = [
        [len(correction.actions) for correction in user_corr.corrections]
        for user_corr in all_corrections
    ]

    # 3) Find the maximum inner length
    max_len = max(len(lst) for lst in data_lists) if data_lists else 0

    # 4) Allocate a (num_users × max_len) array of zeros (or any pad value)
    N = len(data_lists)
    data = np.zeros((N, max_len), dtype=int)

    # 5) Copy each sublist into the left‐most columns of its row
    for i, lengths in enumerate(data_lists):
        data[i, : len(lengths)] = lengths
    #END NEW

    # Normalize the values for color mapping
    norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))
    cmap = cm.get_cmap('RdYlGn')  # Red (low) -> Yellow -> Green (high)

    # Plotting
    fig, ax = plt.subplots(figsize=(len(data[0]), len(data)))

    # Create a colored grid based on action list lengths
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = cmap(norm(value))
            rect = plt.Rectangle((j, i), 1, 1, 
                                facecolor=color, 
                                edgecolor='black', 
                                linewidth=1)
            ax.add_patch(rect)
            # Optional: annotate with the value
            ax.text(j + 0.5, i + 0.5, str(value), va='center', ha='center', fontsize=8, color='black')

    # Set axis limits and labels
    ax.set_xlim(0, data.shape[1])
    ax.set_ylim(0, data.shape[0])
    ax.set_xticks(np.arange(data.shape[1]) + 0.5)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5)
    ax.set_xticklabels([f"Trajectory Sample {j}" for j in range(data.shape[1])], rotation=90)
    ax.set_yticklabels([f"User Correction {i}" for i in range(data.shape[0])])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.tick_params(top=False, bottom=False, left=False, right=False, labeltop=False)

    # Add colorbar for reference
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Trajectory Length (Number of Actions)', rotation=270, labelpad=15)

    # Add title
    plt.title("Trajectory Lengths from User Corrections", fontsize=14, pad=20)

    plt.tight_layout()
    if show_plot is True:
        plt.show()
    
    if save_figure:
        file_path = os.path.join(config.iteration_working_dir,"correction_trajectory_lengths.png")
        plt.savefig(file_path,dpi=300,bbox_inches='tight')