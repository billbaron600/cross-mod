import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F

class PreferenceModel(nn.Module):
    def __init__(self, input_dim=51, hidden_dims=[64, 64], ema_alpha=0.01, eps=1e-8,reward_offset=-1.0):
        """
        ema_alpha: EMA update weight. Smaller = smoother but slower updates.
        eps: For numerical stability.
        """
        super().__init__()
        self.ema_alpha = ema_alpha
        self.eps = eps
        self.reward_offset = reward_offset

        # MLP layers
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        # Running statistics (buffers so they move with .to(device))
        self.register_buffer("running_mean", torch.tensor(0.0))
        self.register_buffer("running_var", torch.tensor(1.0))
        self.register_buffer("num_updates", torch.tensor(0))

    def forward(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """
        x: shape (batch, input_dim)
        Returns: standardized output of shape (batch, 1)
        """
        raw = self.net(x)  # (batch, 1)
        
        if self.training and update_stats:
            # Update running mean/variance using EMA
            mean = raw.mean()
            var = raw.var(unbiased=False)

            self.running_mean = (1 - self.ema_alpha) * self.running_mean + self.ema_alpha * mean.detach()
            self.running_var = (1 - self.ema_alpha) * self.running_var + self.ema_alpha * var.detach()
            self.num_updates += 1

        # Standardize using running statistics
        std = torch.sqrt(self.running_var + self.eps)
        normalized = (raw - self.running_mean) / std
        normalized = normalized + self.reward_offset
        return normalized

    def score_trajectory(self, traj: np.ndarray, device='cpu') -> torch.Tensor:
        """
        traj: np.ndarray of shape (T, input_dim)
        Returns: standardized scalar score as torch.Tensor with shape (1,)
        """
        traj_tensor = torch.tensor(traj, dtype=torch.float32, device=device)
        return self.forward(traj_tensor, update_stats=True).sum()



class PreferenceModel_DEPRECATED(nn.Module):
    """A simple MLP that maps a single (obs+action) vector to a scalar.
    We will sum over a trajectory's timesteps to get the trajectory score."""
    def __init__(self, input_dim=51, hidden_dims=[64, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))  # final scalar output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is shape (batch, input_dim) for a batch of single timesteps
        # Output is shape (batch, 1)
        return self.net(x)
    
    def score_trajectory(self, traj: np.ndarray, device='cpu') -> torch.Tensor:
        """
        Convert a 2D NumPy array of shape (T, input_dim) to a torch.Tensor,
        run it through the network per-timestep, and sum all outputs.
        
        Returns a scalar torch.Tensor with shape (1,).
        """
        # Convert to torch on the desired device
        traj_tensor = torch.tensor(traj, dtype=torch.float32, device=device)
        # Forward pass => shape (T, 1), then sum => shape (1,)
        return self.forward(traj_tensor).sum()