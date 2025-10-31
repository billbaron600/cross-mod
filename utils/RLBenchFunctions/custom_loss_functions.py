import torch.nn as nn
import torch.nn.functional as F
import torch

class BoundedMSE(nn.Module):
    def __init__(self, a_max=[0.8]*8, penalty_scale=10.0,device="cuda"):
        super().__init__()
        self.a_max = torch.tensor(a_max).to(device)
        self.lambda_ = penalty_scale       # λ above

    def forward(self, pred, target):
        mse = 0.5 * (pred - target).pow(2).mean()

        # penalty only where |pred| > a_max
        overflow = (pred.abs() - self.a_max).clamp_min(0.0)
        penalty = self.lambda_ * overflow.pow(2).mean()

        return mse + penalty
