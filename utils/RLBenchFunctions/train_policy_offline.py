from rlbench.gym import RLBenchEnv
from rlbench.tasks import SlideBlockToTarget
from stable_baselines3 import PPO, SAC
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper,JointPositionActionMode, ActionMode
from rlbench.action_modes.arm_action_modes import JointVelocity,JointPosition,ArmActionMode
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import SlideBlockToTarget
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.backend.scene import Scene
#from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
import random
from gymnasium import Wrapper, ObservationWrapper
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import torch
from stable_baselines3 import SAC
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch
from torch.optim import Adam
import torch as th
from stable_baselines3 import SAC
import torch as th
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.RLBenchFunctions.custom_action_modes import MoveArmThenGripperWithBounds
from utils.RLBenchFunctions.custom_envs import instantiate_environment
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy

def plot_loss(actor_losses,critic_losses,return_figure=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Actor loss on the left
    ax1.plot(actor_losses)
    ax1.set_title("Actor Loss")
    ax1.set_xlabel("Gradient Step")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Critic loss on the right
    ax2.plot(critic_losses)
    ax2.set_title("Critic Loss")
    ax2.set_xlabel("Gradient Step")
    ax2.set_ylabel("Loss")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_loss_terms(actor_losses, critic_losses, loss_terms, return_figure=False):
    """
    One comprehensive figure:
    ─ row 0:  actor & critic losses side-by-side
    ─ rows 1-9: diagnostic curves stacked vertically
    """
    # ── set up grid: 10 rows × 2 cols  ───────────────────────────────────
    fig = plt.figure(figsize=(14, 40))
    gs  = gridspec.GridSpec(nrows=10, ncols=2, height_ratios=[1] + [1]*9)

    # Row 0 (two columns) ────────────────────────────────────────────────
    ax_actor  = fig.add_subplot(gs[0, 0])
    ax_critic = fig.add_subplot(gs[0, 1])

    ax_actor.plot(actor_losses)
    ax_actor.set_title("Actor Loss")
    ax_actor.set_xlabel("Gradient Step")
    ax_actor.set_ylabel("Loss")
    ax_actor.grid(True)

    ax_critic.plot(critic_losses)
    ax_critic.set_title("Critic Loss")
    ax_critic.set_xlabel("Gradient Step")
    ax_critic.set_ylabel("Loss")
    ax_critic.grid(True)

    # Helper to add a full-width axis spanning both columns
    def add_fullrow(row_idx):
        return fig.add_subplot(gs[row_idx, :])

    # Row 1: Q_plus
    ax = add_fullrow(1)
    ax.plot(loss_terms["Q_plus_1"], label="Q_plus_1 (online 1)")
    ax.plot(loss_terms["Q_plus_2"], "--", label="Q_plus_2 (online 2)")
    ax.set_title("Online Critics (Q_plus)")
    ax.set_ylabel("Value"); ax.legend(); ax.grid(True)

    # Row 2: Q_minus
    ax = add_fullrow(2)
    ax.plot(loss_terms["Q_minus"], color="purple",
            label="Q_minus (target critic)")
    ax.set_title("Target Critic (Q_minus)")
    ax.set_ylabel("Value"); ax.legend(); ax.grid(True)

    # Row 3: y_t
    ax = add_fullrow(3)
    ax.plot(loss_terms["y_t"], color="brown", label="Soft Bellman Target (y_t)")
    ax.set_title("Soft Bellman Target (y_t)")
    ax.set_ylabel("Value"); ax.legend(); ax.grid(True)

    # Row 4: r_t
    ax = add_fullrow(4)
    ax.plot(loss_terms["r_t"], color="orange", label="Rewards (r_t)")
    ax.set_title("Rewards (r_t)")
    ax.set_ylabel("Value"); ax.legend(); ax.grid(True)

    # Row 5: entropy coef
    ax = add_fullrow(5)
    ax.plot(loss_terms["ent_coef"], color="green", label="Entropy Coefficient")
    ax.set_title("Entropy Coefficient")
    ax.set_ylabel("α"); ax.legend(); ax.grid(True)

    # Row 6: log-prob
    ax = add_fullrow(6)
    ax.plot(loss_terms["actor_log_prob"], color="red", label="Actor Log Prob")
    ax.set_title("Actor Log Probability")
    ax.set_ylabel("log π"); ax.legend(); ax.grid(True)

    # Row 7: Q_plus – y_t
    ax = add_fullrow(7)
    diff_q1 = [q - y for q, y in zip(loss_terms["Q_plus_1"], loss_terms["y_t"])]
    diff_q2 = [q - y for q, y in zip(loss_terms["Q_plus_2"], loss_terms["y_t"])]
    ax.plot(diff_q1, label="Q_plus_1 − y_t")
    ax.plot(diff_q2, "--", label="Q_plus_2 − y_t")
    ax.set_title("Diff (Q_plus − y_t)")
    ax.set_ylabel("Δ"); ax.legend(); ax.grid(True)

    # Row 8: Q_minus − α log π
    ax = add_fullrow(8)
    diff_qm = [q - e*a for q, e, a in
               zip(loss_terms["Q_minus"], loss_terms["ent_coef"],
                   loss_terms["actor_log_prob"])]
    ax.plot(diff_qm, color="magenta",
            label="Q_minus − α·log π")
    ax.set_title("Q_minus − α log π")
    ax.set_ylabel("Value"); ax.legend(); ax.grid(True)

    # Row 9: α log π
    ax = add_fullrow(9)
    prod = [e*a for e, a in zip(loss_terms["ent_coef"],
                                loss_terms["actor_log_prob"])]
    ax.plot(prod, color="blue", label="α·log π")
    ax.set_title("Product: α log π")
    ax.set_ylabel("Value")
    ax.set_xlabel("Gradient Step")
    ax.legend(); ax.grid(True)

    plt.tight_layout()

    if return_figure:
        return fig
    else:
        plt.show()
        return None

def freeze_entropy_parameters(model,value=None):
    # 1) stash the old optimizer
    if value is None:
        print("Entropy Value Frozen but not set explicitly, return out of freeze entropy parameters function.")
        print("Entropy Coefficeint Tensor current Value: " + str(model.ent_coef_tensor.item()))
        return

    model._saved_ent_coef_optimizer = copy.deepcopy(model.ent_coef_optimizer)
    #model._saved_log_ent_coef = copy.deepcopy(model.log_ent_coef)
    model.ent_coef_optimizer = None


    # 2) freeze the log parameter (optional but safer)
    model.log_ent_coef.requires_grad_(False)

    # 3) build a *detached* tensor for the current alpha
    #    here I use model.log_ent_coef directly rather than digging through param_groups
    if value==None:
        #model.ent_coef_tensor = torch.exp(model.log_ent_coef.detach())
        pass
    else:
        model.ent_coef_tensor = torch.tensor(value)
        model.log_ent_coef = torch.log(torch.tensor(value))
    #    detach() severs the graph so no backward will try to reuse it

    # 4) override the float value so SB3’s 'ent_coef' property sees a number
    model.ent_coef = model.ent_coef_tensor.item()

    # 5) make sure no gradient is tracked on your frozen tensor
    model.ent_coef_tensor.requires_grad_(False)


def unfreeze_entropy(model):
    # restore optimizer
    model.ent_coef_optimizer = model._saved_ent_coef_optimizer
    del model._saved_ent_coef_optimizer

    # tell SB3 to go back to auto-tuning
    model.ent_coef = "auto"

    # drop your override so next train() recomputes from log_ent_coef
    #delattr(model, "ent_coef_tensor")

    # re-enable grads on the log parameter
    model.log_ent_coef.requires_grad_(True)

"""
def freeze_entropy_parameters(model):
    #model.log_ent_coef.requires_grad = False
    model._saved_ent_coef_optimizer = copy.deepcopy(model.ent_coef_optimizer)
    model.ent_coef_tensor = torch.exp(model.ent_coef_optimizer.param_groups[0]['params'][0])
    #model.ent_coef = model.ent_coef_tensor.item()
    model.ent_coef = model.ent_coef_tensor
    model.ent_coef_tensor.required_grad = False
    model.ent_coef_optimizer = None


def unfreeze_entropy(model):
    #model.log_ent_coef.requires_grad = True
    model.ent_coef_optimizer = model._saved_ent_coef_optimizer
    model.ent_coef = "auto"
    model.ent_coef_tensor.required_grad = True
"""

def train_policy_offline(model, change_ent_coef_and_lr_back=True,target_entropy=None,use_lr_scheduler=False,env=None,close_env=True,actor_weight_decay=None,include_failed_demos=None,demo_fraction=None,ent_coef=None,unfreeze_actor_at=None,freeze_entropy=False,plot_every=None,gamma=None,log_observability=False,critic_betas=None,task=SlideBlockToTarget,grad_steps=1000,critic_lr = None,actor_lr=None,critic_weight_decay=None,freeze_actor=True,freeze_critic=False,hard_sync=True,tau_offline=None,gradient_steps=100000,log_every=1,batch_size=512,plot_outputs=True,print_outputs=False):
    
    # Method we are calling: train(self, gradient_steps: int, batch_size: int = 64)
    actor_losses = []
    critic_losses = []
    ent_coefs = []

    #Learn for 1 step just to get the logger setup
    if env is None:
        env = instantiate_environment(task=task)
        model.set_env(env)

    #Freeze the actor and train the critic on the updated reward
    model.learn(total_timesteps=1,log_interval=-1)
    
    if close_env is True:
        env.close()
    

    if ent_coef is not None and freeze_entropy is False:
        new_alpha = ent_coef
        device = model.policy.device          # cpu / cuda

        with torch.no_grad():                 # don’t track this in autograd
            model.log_ent_coef.copy_(torch.log(torch.tensor(new_alpha, device=device)))

        

    if freeze_entropy is True:
        freeze_entropy_parameters(model,value=ent_coef)
        """
        if ent_coef is not None:
            new_alpha = ent_coef
            device = model.policy.device          # cpu / cuda

            with torch.no_grad():                 # don’t track this in autograd
                model.log_ent_coef.copy_(torch.log(torch.tensor(new_alpha, device=device)))
            model.ent_coef = new_alpha
        """

    if gamma is not None:
        model.gamma = gamma

    if hard_sync is True:
        model.critic_target.load_state_dict(model.critic.state_dict())

    #Set the models tau
    original_tau = model.tau #will change back after training
    if tau_offline is not None:
        model.tau = tau_offline
    
    #Freeze the actors state if desired by user
    if freeze_actor is True:
        # freeze policy parameters
        for p in model.actor.parameters():
            p.requires_grad_(False)

    if actor_lr is not None:
        for p in model.actor.optimizer.param_groups:
            p["lr"] = actor_lr
    
    if freeze_critic:
        # Leave requires_grad=True
        for pg in model.critic.optimizer.param_groups:
            original_critic_lr = pg["lr"]
            pg["lr"] = 0.0           # no weight update
            #pg["weight_decay"] = 0.0 # (optional) switch off WD as well
    else:
        original_critic_lr = None
    
    #If the user set the critic LR explicitly, change it
    if critic_lr is not None:
        # Lower the critics learning rate to 1e-4
        new_critic_lr = critic_lr
        for pg in model.critic.optimizer.param_groups:
            pg['lr'] = new_critic_lr

    if critic_betas is not None:
        for pg in model.critic.optimizer.param_groups:
            pg['betas'] = critic_betas

    #If the critic weight decay has been explicitly set, change it to that value
    if critic_weight_decay is not None:
        # Lower the critics learning rate to 1e-4
        for pg in model.critic.optimizer.param_groups:
            pg['weight_decay'] = critic_weight_decay

    if actor_weight_decay is not None:
        for pg in model.actor.optimizer.param_groups:
            pg['weight_decay'] = actor_weight_decay

    if target_entropy is not None:
        model.target_entropy = target_entropy

    #if demo_fraction is not None:
    model.replay_buffer.demo_fraction = demo_fraction

    if use_lr_scheduler:
        # current LRs (already set above)
        actor_base_lr  = model.actor.optimizer.param_groups[0]["lr"]
        critic_base_lr = model.critic.optimizer.param_groups[0]["lr"]

        #  Simple cosine annealing: decays to 10 % of start by the last loop
        outer_iterations = int(grad_steps / log_every)    # already computed later
        actor_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                           model.actor.optimizer,  T_max=outer_iterations,
                           eta_min=actor_base_lr * 0.1)
        critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                           model.critic.optimizer, T_max=outer_iterations,
                           eta_min=critic_base_lr * 0.1)
    else:
        actor_sched = critic_sched = None

    #Create a dictionray to keep track of the key terms in the actor/critic loss equations
    # Initialize dictionary explicitly
    loss_terms = {
        "Q_plus_1": [],   # online critic 1
        "Q_plus_2": [],   # online critic 2
        "Q_minus": [],    # target critic prediction (Qθ⁻)
        "y_t": [],        # soft Bellman target (yt)
        "r_t": [],        # rewards (rt)
        "ent_coef": [],
        "actor_log_prob": []
    }

    outer_iterations = int(grad_steps/log_every)

    for i in range(outer_iterations):
        model.train(batch_size=batch_size, gradient_steps=log_every)
        if use_lr_scheduler:
            actor_sched.step()
            critic_sched.step()
        
        key_vals = model.logger.name_to_value
        actor_losses.append(key_vals['train/actor_loss'])
        critic_losses.append(key_vals['train/critic_loss'])
        ent_coefs.append(key_vals['train/ent_coef'])

        if (freeze_actor is True) and (unfreeze_actor_at is not None):
            if unfreeze_actor_at>i*log_every:
                # unfreeze the actor and turn tau back to 0.01
                for p in model.actor.parameters():
                    p.requires_grad_(True)


        if print_outputs==True:
            print("Step: " + str(i*log_every))

        if log_observability is True:
            with torch.no_grad():
                # Get the ent coef
                ent_coef = torch.exp(model.log_ent_coef.detach())

                #Get sample from replay buffer
                replay_data = model.replay_buffer.sample(batch_size)

                #Get next actions and next log prob
                next_actions,next_log_prob = model.actor.action_log_prob(replay_data.next_observations)

                # Compute the next Q values: min over all critics targets
                next_q_values = torch.cat(model.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                #Add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * model.gamma * next_q_values

                #Get current Q values (online critic values)
                current_q_values = model.critic(replay_data.observations,replay_data.actions)

                #Log them
                # Append computed values (means) to loss_terms dictionary
                # Append computed values (means) to loss_terms dictionary
                loss_terms["Q_plus_1"].append(current_q_values[0].cpu().numpy().mean())
                loss_terms["Q_plus_2"].append(current_q_values[1].cpu().numpy().mean())
                loss_terms["Q_minus"].append(next_q_values.cpu().numpy().mean())
                loss_terms["y_t"].append(target_q_values.cpu().numpy().mean())
                loss_terms["r_t"].append(replay_data.rewards.cpu().numpy().mean())
                loss_terms["ent_coef"].append(ent_coef.item())
                loss_terms["actor_log_prob"].append(next_log_prob.cpu().numpy().mean())

        if plot_every is not None and i % plot_every == 0 and i>0:
            if log_observability is not None:
                plot_loss(actor_losses,critic_losses)
                plot_loss_terms(loss_terms)


    
    # unfreeze the actor and turn tau back to 0.01
    for p in model.actor.parameters():
        p.requires_grad_(True)
    
    if change_ent_coef_and_lr_back is True:
        if original_critic_lr is not None:
            # Leave requires_grad=True
            for pg in model.critic.optimizer.param_groups:
                pg["lr"] = original_critic_lr

        if freeze_entropy is True:
            unfreeze_entropy(model)


    #Set tau back to its original value
    model.tau = original_tau

    # Plot if requested: two subplots side by side
    if plot_outputs:
        plot_loss(actor_losses,critic_losses)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Actor loss on the left
        ax1.plot(actor_losses)
        ax1.set_title("Actor Loss")
        ax1.set_xlabel("Gradient Step")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        # Critic loss on the right
        ax2.plot(critic_losses)
        ax2.set_title("Critic Loss")
        ax2.set_xlabel("Gradient Step")
        ax2.set_ylabel("Loss")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
        """

    return model,actor_losses,critic_losses, ent_coefs,loss_terms




