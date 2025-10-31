import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class PreferenceModel(nn.Module):
    """A simple MLP that maps a single (obs+action) vector to a scalar.
    We will sum over a trajectory's timesteps to get the trajectory score."""
    def __init__(self, input_dim, hidden_dims=[64, 64]):
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

def bradley_terry_loss(score1: torch.Tensor, score2: torch.Tensor, label: int) -> torch.Tensor:
    """
    Bradley–Terry loss for a single pair (score1, score2, label).
      - If label=0 => we want score1 > score2, so loss = -log(sigmoid(score1 - score2))
      - If label=1 => we want score2 > score1, so loss = -log(sigmoid(score2 - score1))
    """
    if label == 0:
        # prefer score1 > score2
        return -torch.log(torch.sigmoid(score1 - score2) + 1e-8)
    else:
        # prefer score2 > score1
        return -torch.log(torch.sigmoid(score2 - score1) + 1e-8)

def train_preference_model(
    pairwise_data,
    input_dim,
    hidden_dims=[64, 64],
    lr=1e-3,
    epochs=10,
    batch_size=16,
    device=None,
):
    """
    Trains a preference model using Bradley–Terry style loss.

    Args:
        pairwise_data: List of (trajA_array, trajB_array, label),
                       each traj is shape (T, input_dim).
        input_dim: Dimension of each row in the trajectory arrays (obs+act).
        hidden_dims: Sizes of hidden layers in the MLP.
        lr: Learning rate for Adam.
        epochs: How many passes over the dataset.
        batch_size: How many pairs per batch.
        device: 'cpu' or 'cuda' or None. If None, auto-detect.
    
    Returns:
        A trained PreferenceModel instance.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PreferenceModel(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # For convenience, we'll store the entire dataset in memory
    # pairwise_data = [(arr1, arr2, label), ...]
    data_size = len(pairwise_data)

    for epoch in range(epochs):
        # Shuffle the data each epoch
        random.shuffle(pairwise_data)

        # Mini-batch loop
        total_loss = 0.0
        num_batches = 0
        
        for start_idx in range(0, data_size, batch_size):
            batch = pairwise_data[start_idx : start_idx + batch_size]
            if not batch:
                continue
            
            # Clear gradients
            optimizer.zero_grad()
            
            batch_loss = 0.0
            for (trajA, trajB, label) in batch:
                scoreA = model.score_trajectory(trajA, device=device)
                scoreB = model.score_trajectory(trajB, device=device)
                loss_ij = bradley_terry_loss(scoreA, scoreB, label)
                batch_loss += loss_ij
            
            # Average loss over the batch
            batch_loss /= len(batch)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return model


