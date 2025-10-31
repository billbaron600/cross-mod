import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

def plot_and_save(mean_tor: torch.Tensor, var_tor: torch.Tensor, save_dir: str, show_plots: bool = False):
    """
    Generates and saves 3D scatter plots and index vs. coordinate plots for the given mean and variance tensors.
    
    Parameters:
    - mean_tor (torch.Tensor): Nx3 tensor of mean xyz coordinates.
    - var_tor (torch.Tensor): Nx3 tensor of variance xyz coordinates.
    - save_dir (str): Directory to save the plots.
    - show_plots (bool): If True, display the plots. Default is False.
    """
    os.makedirs(save_dir, exist_ok=True)

    def save_or_show(fig, filename):
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_xyz_vs_index(data: torch.Tensor, name: str):
        """Helper function to plot x, y, z coordinates versus index and save the figures."""
        valid_mask = ~torch.isnan(data).any(dim=1)
        filtered_data = data[valid_mask]
        coords_np = filtered_data.cpu().detach().numpy()
        indices = np.arange(coords_np.shape[0])

        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        axs[0].plot(indices, coords_np[:, 0], marker='o', label=f"{name} X")
        axs[1].plot(indices, coords_np[:, 1], marker='o', label=f"{name} Y")
        axs[2].plot(indices, coords_np[:, 2], marker='o', label=f"{name} Z")

        axs[0].set_ylabel("X", fontsize=12)
        axs[1].set_ylabel("Y", fontsize=12)
        axs[2].set_ylabel("Z", fontsize=12)
        axs[2].set_xlabel("Index", fontsize=12)

        for ax in axs:
            ax.legend()

        plt.tight_layout()
        save_or_show(fig, f"{name}_xyz_vs_index.png")

    def plot_3d_scatter(data: torch.Tensor, name: str):
        """Helper function to create a 3D scatter plot and save the figure."""
        valid_mask = ~torch.isnan(data).any(dim=1)
        filtered_data = data[valid_mask]
        coords_np = filtered_data.cpu().detach().numpy()

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], label=f"{name} Distribution", c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        save_or_show(fig, f"{name}_3d_scatter.png")

    # Plot and save for mean_tor
    plot_xyz_vs_index(mean_tor, "mean")
    plot_3d_scatter(mean_tor, "mean")

    # Plot and save for var_tor
    plot_xyz_vs_index(var_tor, "var")
    plot_3d_scatter(var_tor, "var")