from rlbench.tasks import PlayJenga
import pickle
from stable_baselines3 import SAC
import os
from utils.RLBenchFunctions.custom_envs import instantiate_environment
from utils.Classes.custom_replay_buffer import ScaledRewardBuffer
from utils.RLBenchFunctions.fill_buffer import preload_from_config
from utils.RLBenchFunctions.plot_reward_classes import plot_reward_distributions
from utils.RLBenchFunctions.train_policy import train_policy, run_train_policy
from imitation.algorithms.bc import BC
from imitation.data.types import DictObs, Transitions
import numpy as np
from numpy.random import default_rng
from types import MethodType
import torch
from utils.RLBenchFunctions.behavior_cloning import make_demo_loader
import math
import matplotlib.pyplot as plt        # ADDED
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box, Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
import gymnasium as gym
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch.nn as nn
import torch.nn.functional as F
from utils.RLBenchFunctions.generate_IVK_trajectory import plot_trajectory_grid
from utils.RLBenchFunctions.custom_loss_functions import BoundedMSE
from utils.Classes.custom_policies import SACQuatSafe



#Specify key paths/variables
path_to_config_pickle = "run_results/play_jenga/demos/config_instance.pkl"
log_dir="tensorboardlogs/play_jenga"
load_config = True
task = PlayJenga

#BEHAVIOR CLONIGN PARAMS
include_second_term = False
EPOCHS      = 100
batch_size = 128
percent_noise = 0.0
n_percent_shorts = 1.0 #percent of expert ttrajectoreis to keep per seed (keep the shortes 60%)
learning_rate = 1e-4
weight_decay = 0.0
skip_idx = (0,1e6)
use_scheduler = True
#save_to = "run_results/play_jenga/demos/policy_models/bc_model"


#Boolean values
fill_replay_buffer_with_trajs = True
results_path = "run_results/play_jenga/demos/"
#results_path = None
sample_evenly = False
include_failed_demos = True
plot_filled_buffer = False

#Plot the demo data
#plot_trajectory_grid(results_path)

#Paths to Models
model_path = None
save_to = "run_results/play_jenga/demos/RL_models/behavior_cloned_model_TD3"
buffer_path = None

#RL Params
normalize_actions = True
use_mean_instead_of_median = False
max_expert_len = 1000 #GOOD RESULTS AT 80
max_steps = 1000
action_repeat = 1
demo_fraction = 1.0
render_mode = "rgb_array"
include_replay_buffer = True
device = "cuda"
print_shapes = False
plot_success_from_demos=False
print("Max Steps: "+str(max_steps))

#Load in the config
with open(path_to_config_pickle, 'rb') as file:
    config = pickle.load(file)

# Instantaite the env
vec_env = instantiate_environment(config,action_repeat=action_repeat,task=task,render_mode=render_mode,max_steps=max_steps)

#Load in the model, or create one
if model_path is None:
    #policy_kwargs = dict(features_extractor_class=JointPosSinCosExtractor)
    # features_extractor_kwargs={"angle_indices": [0, 1, 2, 3, 4, 5, 6]}
    policy_kwargs = dict(net_arch=[512,512],log_std_init=-8.0)
    #policy_kwargs = dict(log_std_init=-4.0)           # σ ≈ 0.018
    from stable_baselines3.common.noise import NormalActionNoise
    #noise_std = 1e-20
    #n_actions = 8
    #action_noise = NormalActionNoise(
    #    mean=np.zeros(n_actions),               # centred on 0
    #    sigma=noise_std * np.ones(n_actions)    # ← this NumPy vector is what you asked for
    #)
    

    model = SACQuatSafe(
        "MultiInputPolicy",
        vec_env,
        replay_buffer_class = ScaledRewardBuffer,
        #action_noise=action_noise,
        verbose            = False,
        tensorboard_log=log_dir,
        ent_coef=0.002,
        #policy_kwargs=policy_kwargs,
        device="cuda")
else:
    model = SAC.load(model_path,env=vec_env)


#If a buffer path is specified, load it in
if buffer_path is not None:
    model.load_replay_buffer(buffer_path)

#If fill buffer is true, add trajectoreis from this config to the buffer
if fill_replay_buffer_with_trajs is True:
    model = preload_from_config(model,config,n_percent_shorts=n_percent_shorts,use_mean_instead_of_median=use_mean_instead_of_median,results_path=results_path,sample_evenly=sample_evenly,skip_idx=skip_idx,plot_success_from_demos=plot_success_from_demos,normalize_actions=normalize_actions,max_length=max_expert_len,include_non_expert=include_failed_demos)
    if plot_filled_buffer is True:
        reward_fig = plot_reward_distributions(model,show=True)





#Set policy stuff
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = model.policy            # SB3 SACPolicy
actor  = policy.actor.to(device) # convenience alias
deterministic = True 


# replay-buffer → DataLoader
demo_loader,val_demo_loader = make_demo_loader(model,
                               batch_size=batch_size,
                               device=device,
                               percent_noise=percent_noise,
                               shuffle=True)
print(f"{len(demo_loader.dataset)} transitions in demo_loader")

# ---------------------------------------------------------------------------
# constants & helpers
# ---------------------------------------------------------------------------
act_dim     = model.action_space.shape[0]


from stable_baselines3.common.noise import NormalActionNoise
# σ = 0.01 per dimension  → very small exploration
n_actions = 8
action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                 sigma=0.025 * np.ones(n_actions))

#Specify devices
from stable_baselines3 import TD3


policy_kwargs = dict(
    net_arch=[512, 512]                           # shared MLP after concat
)
# ------------------------------------------------------------
# 3.  Instantiate TD3 with the multi-input policy
# ------------------------------------------------------------
model = TD3(
    policy="MultiInputPolicy",
    env=vec_env,
    train_freq=(1,"episode"),
    policy_kwargs=policy_kwargs,
    action_noise=action_noise,
    tensorboard_log=log_dir,
    verbose=1
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = model.policy            # SB3 SACPolicy
actor  = policy.actor.to(device) # convenience alias
deterministic = True 


#instantaite the optimizer
optimizer   = torch.optim.Adam(actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_steps = EPOCHS * len(demo_loader)
eta_min     = learning_rate * 0.01           # final LR = 0.1 % of initial (tune!)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
                 optimizer,
                 T_max=total_steps,
                 eta_min=eta_min)
# MSE Errro
critertion = nn.MSELoss()
#critertion = BoundedMSE(a_max=[0.1,0.1,0.1,0.3,0.1,0.5,0.1,2.0])


"""
# grab the filled part of the buffer and drop the dummy dim
acts = model.replay_buffer.actions[:model.replay_buffer.pos]   # (N,1,8) or (N,8)
if acts.ndim == 3 and acts.shape[1] == 1:
    acts = acts.squeeze(1)                                     # → (N, 8)

# compute per-joint stats
mins  = acts.min(axis=0)
maxs  = acts.max(axis=0)
means = acts.mean(axis=0)
stds  = acts.std(axis=0)

#Print the statisitic in the buffer over each joint
print("\nAction statistics per joint")
print("joint |    min     max    mean     std")
for j in range(acts.shape[1]):
    print(f"{j:5d} | {mins[j]:7.4f} {maxs[j]:7.4f} {means[j]:7.4f} {stds[j]:7.4f}")
"""

# logging containers
pred_means_log     = []    # list of (act_dim,) tensors
true_means_log     = []    # same shape
per_dim_error_log  = []    # same shape
loss_log           = []    # list of scalars


# helper that returns a readable shape summary
def fmt_shape(x):
    if isinstance(x, dict):
        return {k: v.shape for k, v in x.items()}
    return x.shape




# ---------------------------------------------------------------------------
# behaviour-cloning loop
# ---------------------------------------------------------------------------
# ───────────────────────── TRAIN / VALIDATION LOOP ───────────────────────
val_loss_points   = []          # (x,y) pairs so we can plot on batch axis
cumulative_batches = 0          # running counter of train batches processed


# ---------------------------------------------------------------------------
# behaviour-cloning loop
# ---------------------------------------------------------------------------
# ───────────────────────── TRAIN / VALIDATION LOOP ───────────────────────
val_loss_points   = []          # (x,y) pairs so we can plot on batch axis
cumulative_batches = 0          # running counter of train batches processed
policy.train()

train_losses, val_losses = [], []                 # ➊ new

#weights[0] = 5
#weights[3] = 5


for epoch in range(EPOCHS):
    dim_loss_sum, samples_seen = torch.zeros(act_dim, device=device), 0
    policy.train()
    for batch_idx, (obs_batch, act_batch) in enumerate(demo_loader):
        optimizer.zero_grad(set_to_none=True)
        if act_batch.dim() == 3 and act_batch.shape[1] == 1:
            act_batch = act_batch.squeeze(1)
        # ---------------------------------------------------------------
        # 1. get μ, log σ from the actor
        # ---------------------------------------------------------------
        action_prediction = policy(obs_batch)
        loss = critertion(action_prediction,act_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())          # ➋ log batch loss

    # ── VALIDATE ──
    policy.eval()
    with torch.no_grad():
        vloss = torch.mean(torch.stack([                 # ➌ one-liner val mean
            critertion(policy(o), a.squeeze(1) if a.dim()==3 and a.shape[1]==1 else a)
            for o, a in val_demo_loader
        ])).item()

        # per-feature error as % of full action range (0.2 = 0.1 − (-0.1))
        per_feat_range_err = torch.cat([
            (policy(o) - (a.squeeze(1) if a.dim()==3 and a.shape[1]==1 else a)
            ).abs() / 0.2                      # ← constant range instead of |a|
            for o, a in val_demo_loader
        ]).mean(0).mul(100)                   # (% of range)

        #print("Per-feature range % error:",
        #    ", ".join(f"{v:.2f}%" for v in per_feat_range_err.tolist()))
        print("Validation Loss: " + str(vloss))
    
        """
        # ── per-feature %-error, same “stack→mean” pattern as vloss ──
        per_feat_pct_err = torch.mean(torch.stack([
            (                                                      # (B…,A) → (A,)
                (policy(o) - tgt).abs() / tgt.abs().clamp_min(1e-6)
            ).reshape(-1, tgt.shape[-1]).mean(0)                   # flatten B… dims
            for o, a in val_demo_loader
            for tgt in [(a.squeeze(1) if a.dim()==3 and a.shape[1]==1 else a)]
        ]), dim=0).mul(100)                                        # (%)
        print("Per-feature % error:", ", ".join(f"{v:.2g}%" for v in per_feat_pct_err.tolist()))
        """
    
    val_losses.append(vloss)


model.save(save_to)

# ── PLOT ────────────────────────────────────────────────────────────────
x_val = np.linspace(0, len(train_losses)-1, len(val_losses))   # align epochs
plt.plot(train_losses, label='train-batch')
plt.plot(x_val, val_losses, 'o-', label='val-epoch')
plt.xlabel('batch index'); plt.ylabel('MSE'); plt.legend(); plt.tight_layout()
plt.show()
