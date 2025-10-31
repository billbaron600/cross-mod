import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import math
from datetime import datetime
from pathlib import Path

class PreferenceModel(nn.Module):
    """
    Encoder-only transformer reward model compatible with PreferenceModel.
    """

    def __init__(
        self,
        input_dim: int,          # 43 in your case
        d_model: int   = 128,    # Kim: 128, 4 heads, 4 layers
        hidden_dims = [64,64], #NOT USED
        n_layers: int  = 4, # changed from 4
        n_heads: int   = 4,
        ema_alpha: float = 0.0,
        target_mean: float = 0.0,
        dropout: float = 0.1,
        normalise_outputs: bool = True,
        eps: float = 1e-8,
        feature_dropout: float = 0.5,
        drop_idxs: list = [3,4,5,6]
    ):
        super().__init__()

        # 1) embed each (flattened) observation
        self.embed = nn.Linear(input_dim, d_model)

        # 2) learned positional encodings (Kim uses sinusoidal; either is fine)
        self.pos = nn.Parameter(torch.zeros(1, 1000, d_model))   # max T = 1000

        # 3) transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # 4) per-token value head → scalar
        self.value_head = nn.Linear(d_model, 1)

        # 5) running-stats block (identical to PreferenceModel)
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var",  torch.ones(1))
        self.ema_alpha = ema_alpha
        self.eps       = eps
        self.target_mean       = target_mean
        self.normalise_outputs = normalise_outputs

        #Sepcify featuer dropout
        self.feature_dropout_p = feature_dropout
        self.drop_idxs = torch.tensor(drop_idxs,dtype=torch.long)

    def _feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zero-out the columns listed in self.drop_idxs with prob p.
        Works for x of shape (B, T, D) or (1, T, D).
        """
        if not self.training or self.feature_dropout_p == 0.0:
            return x

        B, T, D = x.shape
        p = self.feature_dropout_p

        # Build a keep-mask of shape (1, 1, D) = broadcast over B and T
        keep = torch.ones(1, 1, D, device=x.device, dtype=x.dtype)

        # Bernoulli(1-p) for the selected features ONLY
        rand_keep = torch.bernoulli(
            torch.full((1, 1, len(self.drop_idxs)), 1.0 - p, device=x.device)
        )
        keep[..., self.drop_idxs] = rand_keep / (1.0 - p)    # scale like nn.Dropout

        return x * keep

    # -------------------------------------------------------
    def _update_stats(self, raw):
        m, v = raw.mean(), raw.var(unbiased=False)
        a    = self.ema_alpha
        self.running_mean.mul_(1-a).add_(a*m)
        self.running_var .mul_(1-a).add_(a*v)

    def forward(self, x: torch.Tensor, update_stats: bool = True):
        """
        x : (B, D)  or  (T, D)   • returns (B, 1) or (T, 1)
        """
        single_traj = False
        if x.dim() == 2:                  # caller gave (T, D)
            x = x.unsqueeze(0)            # → (1, T, D) for the encoder
            single_traj = True

        #Do feature level droppout to prevent overfitting to one feature
        x = self._feature_dropout(x)

        z = self.embed(x) + self.pos[:, :x.size(1)]
        h = self.encoder(z)
        raw = self.value_head(h).squeeze(-1)   # (B, T)

        if self.normalise_outputs and self.training and update_stats:
            self._update_stats(raw)

        if self.normalise_outputs:
            std = (self.running_var + self.eps).sqrt()
            raw = (raw - self.running_mean) / std + self.target_mean

        out = raw.unsqueeze(-1)           # (B, T, 1)

        if single_traj:                   # drop the batch axis → (T, 1)
            out = out.squeeze(0)

        return out
    # -------------------------------------------------------
    def forward_DEPRECATED(self, x: torch.Tensor, update_stats: bool=True):
        """
        x : (B, D)   or   (T, D) when called by score_trajectory
        Returns (B,1) or (T,1) – just like PreferenceModel.
        """
        single_traj = False
        if x.dim() == 2:                     # (T,D) –– treat T as batch
            x = x.unsqueeze(0)               # → (1,T,D)
            single_traj = True

        z = self.embed(x) + self.pos[:, :x.size(1)]
        h = self.encoder(z)
        raw = self.value_head(h).squeeze(-1)     # (B,T) or (1,T)

        if self.normalise_outputs:
            if self.training and update_stats:
                self._update_stats(raw)
            std = (self.running_var + self.eps).sqrt()
            centred = (raw - self.running_mean) / std
            shifted = centred + self.target_mean
            return shifted.unsqueeze(-1)
        else:
            return raw.unsqueeze(-1)

    # -------------------------------------------------------
    def score_trajectory(self, traj: torch.Tensor, device="cuda"):
        traj = traj.to(device)              # (T,D)
        per_step = self.forward(traj, update_stats=False)  # (T,1)
        return per_step.mean()              # identical contract




class PreferenceEnsemble(nn.Module):
    def __init__(
        self,
        num_members: int,
        input_dim: int,
        hidden_dims = [64, 64],
        ema_alpha: float    = 0.01,
        eps: float          = 1e-8,
        target_mean: float  = 0.0,
        normalise_outputs: bool = True,
    ):
        super().__init__()
        self.members = nn.ModuleList([
            PreferenceModel(
                input_dim          = input_dim,
                hidden_dims        = hidden_dims,
                ema_alpha          = ema_alpha,
                eps                = eps,
                target_mean        = target_mean,
                normalise_outputs  = normalise_outputs
            )
            for _ in range(num_members)
        ])

    def forward(self, x, update_stats=True):
        # returns list of (B,1) tensors, one per member
        outs = []
        for m in self.members:
            outs.append(m(x, update_stats=update_stats))
        return outs

    def score_trajectory(self, traj, device='cuda'):
        # returns tensor of shape (num_members,)
        scores = []
        for m in self.members:
            scores.append(m.score_trajectory(traj, device=device))
        # stack into [num_members]
        return torch.stack(scores, dim=0)


def bradley_terry_loss(score1: torch.Tensor, score2: torch.Tensor, label: int) -> torch.Tensor:
    """
    Bradley-Terry loss for a single pair (score1, score2, label).
      - If label=0 => we want score1 > score2, so loss = -log(sigmoid(score1 - score2))
      - If label=1 => we want score2 > score1, so loss = -log(sigmoid(score2 - score1))
    """
    p_right = torch.sigmoid(score2 - score1)          # model prob(right wins)
    # BCE with a small ε for numerical stability
    eps = 1e-8
    return - (label   * torch.log(p_right + eps) +
              (1.0-label) * torch.log(1.0 - p_right + eps))


def temporal_smoothness_loss(model, traj: np.ndarray, device, λ: float):
    """
    Penalise large step-to-step changes in the per-step reward.

    Parameters
    ----------
    model : PreferenceModel   (one head of the ensemble)
    traj  : np.ndarray        shape (T, D)
    λ     : float             weighting coefficient
    """
    # (T, D) → torch tensor
    t = torch.as_tensor(traj, dtype=torch.float32, device=device)

    # Per-step reward predictions, shape (T, 1) → squeeze to (T,)
    r = model(t, update_stats=False).squeeze(-1)

    # Finite-difference smoothness penalty
    diffs = r[1:] - r[:-1]               # (T-1,)
    return λ * diffs.pow(2).mean()


def train_preference_ensemble(
    pairwise_data,
    model=None,                  # can be a path to a saved ensemble or None
    config=None,
    num_members=3,
    input_dim=51,
    hidden_dims=[64, 64],
    lr=1e-3,
    epochs=10,
    weight_decay=0.0,
    batch_size=128,
    device=None,
    smooth_coefficient=0,
    validation_ratio=None,
    plot_loss=False,
    use_scheduler=False,
    save_every_updates = 100
):
    import torch, random
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import matplotlib.pyplot as plt

    #smooth_coefficient = 0

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Instantiate or load ensemble ---
    if model is None:
        ensemble = PreferenceEnsemble(
            num_members=num_members,
            input_dim=input_dim,
            hidden_dims=hidden_dims
        ).to(device)
    else:
        # load saved ensemble (path or object)
        if isinstance(model, str):
            ensemble = torch.load(model, map_location=device)
        else:
            ensemble = model
        ensemble.to(device)
        #Save this version to the lder models path
        # build the save path  …/previous_reward_models/reward_modelYYYY-MM-DD_HH-MM-SS.pt
        timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir    = Path(config.iteration_working_dir) / "previous_reward_models"
        save_dir.mkdir(parents=True, exist_ok=True)        # create if missing, no overwrite

        save_path   = save_dir / f"reward_model_{timestamp}.pt"
        torch.save(ensemble, save_path)

        print(f"[checkpoint] reward model saved to: {save_path}")

    ensemble.train()

    

    # --- Split train/val ---
    if validation_ratio:
        split = int(len(pairwise_data)*(1-validation_ratio))
        random.shuffle(pairwise_data)
        train_data = pairwise_data[:split]
        val_data   = pairwise_data[split:]
    else:
        train_data = pairwise_data
        val_data   = None

    train_losses, val_losses = [], []

    
    if use_scheduler:
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, verbose=True)
        base_lr   = lr # was 1e-3
        warm_up   = 10                 # steps (batches) of linear warm-up
        bs = batch_size or len(train_data)
        total_it  = epochs * math.ceil(len(train_data)/bs)
        optimizer = torch.optim.Adam(ensemble.parameters(), lr=base_lr)
        # 1) linear warm-up from 0 → base_lr
        scheduler_warm = LinearLR(optimizer, start_factor=1e-7, end_factor=1.0,
                                total_iters=warm_up)

        # 2) cosine decay from base_lr → 0.05·base_lr
        scheduler_cos  = CosineAnnealingLR(optimizer,
                                        T_max=total_it - warm_up,
                                        eta_min=base_lr*0.05)

        # chain them
        scheduler = SequentialLR(optimizer,
                                schedulers=[scheduler_warm, scheduler_cos],
                                milestones=[warm_up])

        print("Scheduler Being Used")
    else:
        optimizer = Adam(ensemble.parameters(), lr=lr, weight_decay=weight_decay)
        print("Scheduler Not Being Used")

    
    #Keep track of update index
    update_idx = 0 #we save every number of updates, so this is used for that

    for epoch in range(epochs):
        random.shuffle(train_data)
        total_loss, batches = 0.0, 0

        #for i in range(0, len(train_data), batch_size):
        bs = batch_size or len(train_data)      # fallback to full batch
        for i in range(0, len(train_data), bs):
            batch = train_data[i:i+bs]
            if not batch:
                continue

            optimizer.zero_grad()
            batch_loss = 0.0

            for trajA, trajB, label in batch:
                scoresA = ensemble.score_trajectory(trajA, device)
                scoresB = ensemble.score_trajectory(trajB, device)

                # average per-member BT losses
                losses = [bradley_terry_loss(sa, sb, label)
                          for sa, sb in zip(scoresA, scoresB)]
                member_loss = torch.stack(losses).mean()

                if smooth_coefficient > 0.0:
                    # use the first member to compute smoothness term
                    member_loss = (
                        member_loss
                        + temporal_smoothness_loss(ensemble.members[0], trajA.numpy(), device, smooth_coefficient)
                        + temporal_smoothness_loss(ensemble.members[0], trajB.numpy(), device, smooth_coefficient)
                    )
                batch_loss += member_loss

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()
            scheduler.step()

            #NEW
            update_idx += 1
            if (save_every_updates is not None and
                update_idx % save_every_updates == 0 and
                config is not None):
                save_dir = (Path(config.iteration_working_dir) /
                            "update_reward_models")
                save_dir.mkdir(parents=True, exist_ok=True)
                ckpt_name = f"reward_model_upd_{update_idx:07d}.pt"
                torch.save(ensemble.state_dict(), save_dir / ckpt_name)
                print(f"[ckpt] saved {ckpt_name} after {update_idx} updates")
                avg_train_temp = total_loss/max(batches,1)
                print(f" ▶ Train: {avg_train_temp:.4f}")
            #END NEW

            total_loss += batch_loss.item()
            batches += 1

        avg_train = total_loss / max(batches,1)
        train_losses.append(avg_train)

        if val_data is not None:
            ensemble.eval()
            with torch.no_grad():
                vloss = 0.0
                for trajA, trajB, label in val_data:
                    scoresA = ensemble.score_trajectory(trajA, device)
                    scoresB = ensemble.score_trajectory(trajB, device)
                    losses = [bradley_terry_loss(sa, sb, label)
                              for sa, sb in zip(scoresA, scoresB)]
                    vloss += torch.stack(losses).mean().item()
                avg_val = vloss / len(val_data)
                val_losses.append(avg_val)
            print(f"Epoch {epoch+1}/{epochs} ▶ Train: {avg_train:.4f}, Val: {avg_val:.4f}")
            ensemble.train()
        else:
            print(f"Epoch {epoch+1}/{epochs} ▶ Train: {avg_train:.4f}")

        # ── NEW: save a copy of the whole ensemble ──────────────────
        if config is not None:                          # need the path root
            save_dir = Path(config.iteration_working_dir) / "epoch_reward_models"
            save_dir.mkdir(parents=True, exist_ok=True)  # create once, no errors
            ckpt_name = f"reward_model_epoch_{epoch+1:03d}.pt"
            torch.save(ensemble, save_dir / ckpt_name)
            print(f"[checkpoint] saved {ckpt_name} in {save_dir}")

        

    if plot_loss:
        plt.plot(train_losses, label="Train")
        if val_data:
            plt.plot(val_losses, label="Val")
        plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.show()

    return ensemble






def run_train_preference_ensemble(
    config,
    num_members: int = 3,
    database_index = None,
    segment_length=100,
    compare_all_segments = False,
    limit_pairs_to_n_most_recent=None,
    first_and_last_segment = False,
    start_from_traj_end = False,
    duplicate_by_n=1,
    random_starting_points=True,
    limit_to_segments=True,
    save_model=True,
    reward_model_filename="reward_model_ensemble.pt",
    smooth_coefficient=0,
    pad_trajectories = True
):
    import torch, os, pickle

    # 1) Load the updated preference database
    #if database_index is None:
    #    db_path = os.path.join(config.iteration_working_dir, "preference_database.pkl")
    #else:
    quickpath = "preference_database_"+str(database_index)+".pkl"
    db_path = os.path.join(config.iteration_working_dir, quickpath)

    with open(db_path, 'rb') as f:
        updated_database = pickle.load(f)

    # 2) Optionally restrict to the latest N pairs and/or duplicate
    #if limit_pairs_to_n_most_recent is not None:
    #    updated_database.pairwise_comparisons = (
    #        updated_database.pairwise_comparisons[-limit_pairs_to_n_most_recent:]
    #        * duplicate_last_n_by_factor
    #    )

    updated_database.pairwise_comparisons = duplicate_by_n * updated_database.pairwise_comparisons

    # 3) Build your tensorized (trajA, trajB, label) list
    pairwise_comparisons = updated_database.generate_segment_pairs(
        segment_length=segment_length,
        random_starting_points=random_starting_points,
        limit_to_segments=limit_to_segments,
        start_from_traj_end = start_from_traj_end,
        compare_all_segments    = compare_all_segments,
        pad_trajectories = pad_trajectories
    )

    if first_and_last_segment is True:
        flipped_start = not start_from_traj_end
        pairwise_comparisons2 = updated_database.generate_segment_pairs(
            segment_length=segment_length,
            random_starting_points=random_starting_points,
            limit_to_segments=limit_to_segments,
            start_from_traj_end = flipped_start,
            compare_all_segments    = compare_all_segments)
        
        pairwise_comparisons = pairwise_comparisons + pairwise_comparisons2
        
    if limit_pairs_to_n_most_recent is not None:
        pairwise_comparisons = pairwise_comparisons[-limit_pairs_to_n_most_recent:]
    print("Number Compairsons:")
    print(len(pairwise_comparisons))
    # 4) Pull ensemble-training kwargs from config
    ensemble_kwargs = config.preference_model_kwargs.copy()
    input_dim   = ensemble_kwargs.pop("input_dim", 51)
    hidden_dims = ensemble_kwargs.pop("hidden_dims", [64,64])
    lr          = ensemble_kwargs.pop("lr", 1e-3)
    epochs      = ensemble_kwargs.pop("epochs", 10)
    batch_size  = ensemble_kwargs.pop("batch_size", None)
    device      = ensemble_kwargs.pop("device", None)

    # 5) Train the ensemble
    reward_ensemble = train_preference_ensemble(
        config        = config,
        pairwise_data = pairwise_comparisons,
        num_members   = num_members,
        input_dim     = input_dim,
        hidden_dims   = hidden_dims,
        lr            = lr,
        epochs        = epochs,
        batch_size    = batch_size,
        device        = device,
        smooth_coefficient = smooth_coefficient,
        **ensemble_kwargs
    )

    # 6) Save it
    if save_model:
        save_path = os.path.join(config.iteration_working_dir, reward_model_filename)
        torch.save(reward_ensemble, save_path)
        print(f"Ensemble reward model saved to: {save_path}")

    return reward_ensemble


    """

    class PreferenceModel_MLP(nn.Module):
    
    #MLP that maps (obs+action) → scalar reward and normalises it online
    #with an exponential moving average (EMA) of mean/variance, then
    #adds a constant shift so the *expected* reward is `target_mean`.
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims      = [64, 64],
        ema_alpha: float = 0.01,     # EMA smoothing factor
        eps: float       = 1e-8,     # numerical stability
        target_mean: float = 0.0,   # For SAC, we want the mean to be arouond 0.5
        normalise_outputs: bool = True,
    ):
        super().__init__()

        # ---------- backbone MLP ----------
        prev = input_dim
        layers = []
        for hd in hidden_dims:
            layers += [nn.Linear(prev, hd), nn.GELU()]
            prev = hd
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # ---------- running stats ----------
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var",  torch.ones(1))
        self.ema_alpha = ema_alpha
        self.eps = eps

        # ---------- extra settings ----------
        self.target_mean = target_mean
        self.normalise_outputs = normalise_outputs

    # ------------------------------------------------------------
    def _update_running_stats(self, raw: torch.Tensor):
        with torch.no_grad():
            m = raw.mean()
            v = raw.var(unbiased=False)
            a = self.ema_alpha
            self.running_mean.mul_(1.0 - a).add_(a * m)
            self.running_var .mul_(1.0 - a).add_(a * v)

    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        raw = self.net(x).squeeze(-1)          # (B,)

        if self.normalise_outputs:
            if self.training and update_stats:
                self._update_running_stats(raw)

            std = (self.running_var + self.eps).sqrt()
            centred = (raw - self.running_mean) / std
            shifted = centred + self.target_mean      # <── shift here
            return shifted.unsqueeze(-1)              # (B,1)
        else:
            return raw.unsqueeze(-1)

    # ------------------------------------------------------------
    def score_trajectory(self, traj: torch.Tensor, update_stats=True,device='cuda') -> torch.Tensor:
        # ensure shape is (T, D), not (D,)
        #if traj.dim() == 1:
        #    traj = traj.unsqueeze(0)

        traj = traj.to(device)
        per_step = self.forward(traj, update_stats=update_stats)
        T = traj.shape[0]                  # true # of time-steps
        total = per_step.sum()             # sum over steps
        #print("Dividing by: " + str(max(1, T)))
        return total / max(1, T)           # mean reward per step

    def run_train_preference_model(
    config,
    segment_length=100,
    limit_pairs_to_n_most_recent=None,
    duplicate_by_n=1,
    start_from_traj_end = False,
    random_starting_points=True,
    limit_to_segments=True,
    save_model=True,
    reward_model_filename="reward_model.pt",
    use_scheduler=False,
    smooth_coefficient=0
    
):

    # Load the updated preference database
    path_to_load = os.path.join(config.iteration_working_dir, "preference_database.pkl")
    with open(path_to_load, 'rb') as f:
        updated_database = pickle.load(f)

    # Pull keyword arguments for the reward model (from config)
    preference_model_kwargs = config.preference_model_kwargs

    #limit to only the n mosot recene pairs if the user sets that
    #if limit_pairs_to_n_most_recent is not None:
    #    updated_database.pairwise_comparisons = updated_database.pairwise_comparisons[-limit_pairs_to_n_most_recent:] * duplicate_last_n_by_factor

    #Duplicate the comparisons by factor "duplicate_by_n"
    updated_database.pairwise_comparisons = duplicate_by_n * updated_database.pairwise_comparisons

    # Generate pairwise comparisons in tensor format
    pairwise_comparisons = updated_database.generate_segment_pairs(
        segment_length=segment_length,
        random_starting_points=random_starting_points,
        limit_to_segments=limit_to_segments,
        start_from_traj_end = start_from_traj_end
    )

    # Train the reward model
    reward_model = train_preference_model(pairwise_comparisons, use_scheduler=False,smooth_coefficient=smooth_coefficient,**preference_model_kwargs)

    # Optionally save the trained model
    if save_model:
        path_to_save = os.path.join(config.iteration_working_dir, reward_model_filename)
        torch.save(reward_model, path_to_save)
        print(f"Reward model saved to: {path_to_save}")

    return reward_model



def train_preference_model(
    pairwise_data,
    model=None,
    input_dim=51,
    hidden_dims=[64, 64],
    lr=1e-3,
    epochs=10,
    weight_decay=0.0,
    batch_size=16,
    device=None,
    smooth_coefficient=0,
    validation_ratio=None,
    plot_loss=False,
    use_scheduler=False
):
    import torch
    import torch.optim as optim
    import random
    import matplotlib.pyplot as plt

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model is None:
        model = PreferenceModel(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    else:
        model = torch.load(model)
        model.train()
        model.to(device)

    if use_scheduler is False:
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
        scheduler = None
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
        #scheduler = ReduceLROnPlateau(
        #optimizer,
        #mode='min',         # we want to minimize validation loss
        #factor=0.5,         # reduce LR by half
        #patience=3,         # wait 3 epochs of no improvement
        #verbose=True)

    if validation_ratio is not None:
        split_idx = int(len(pairwise_data) * (1 - validation_ratio))
        random.shuffle(pairwise_data)
        train_data = pairwise_data[:split_idx]
        val_data = pairwise_data[split_idx:]
    else:
        train_data = pairwise_data
        val_data = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        random.shuffle(train_data)
        total_loss = 0.0
        num_batches = 0

        for start_idx in range(0, len(train_data), batch_size):
            batch = train_data[start_idx : start_idx + batch_size]
            if not batch:
                continue

            optimizer.zero_grad()
            batch_loss = 0.0

            for trajA, trajB, label in batch:
                scoreA = model.score_trajectory(trajA, device=device)
                scoreB = model.score_trajectory(trajB, device=device)
                loss = bradley_terry_loss(scoreA, scoreB, label)
                if smooth_coefficient>0:
                    if smooth_coefficient > 0:
                        loss  = loss \
                            + temporal_smoothness_loss(model, trajA, device, smooth_coefficient) \
                            + temporal_smoothness_loss(model, trajB, device, smooth_coefficient)
                batch_loss += loss

            batch_loss /= len(batch)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for trajA, trajB, label in val_data:
                    scoreA = model.score_trajectory(trajA, device=device)
                    scoreB = model.score_trajectory(trajB, device=device)
                    loss = bradley_terry_loss(scoreA, scoreB, label)
                    val_loss += loss.item()
                avg_val_loss = val_loss / len(val_data)
                val_losses.append(avg_val_loss)
            model.train()
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        if scheduler is not None:
            scheduler.step(avg_train_loss)

    if plot_loss:
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        if val_data is not None:
            plt.plot(val_losses, label='Validation Loss')
        else:
            print("Warning: plot_loss is True but no validation data provided.")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Curve")
        plt.grid(True)
        plt.show()

    
    return model
    """