import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Union, Sequence
from stable_baselines3 import SAC
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Union, Sequence, Mapping
from stable_baselines3 import SAC

from typing import Union, Dict, Any
from stable_baselines3 import SAC
import torch
import inspect
import textwrap


def plot_buffer_actions_vs_index(
    rb,
    *,
    only_filled: bool = True,
    joint_labels = None,
):
    """
    Plot each action dimension (column) against timestep index for every
    trajectory stored in a Stable-Baselines 3 replay buffer.

    Parameters
    ----------
    rb : stable_baselines3.common.buffers.ReplayBuffer
        The replay buffer whose `actions` and `dones` fields are inspected.
    only_filled : bool, default True
        If True, consider rows up to `rb.pos` only (recommended for circular
        buffers).  Set False to scan the entire buffer array.
    joint_labels : list[str], optional
        Custom labels for action dimensions.  Defaults to "joint 0 …".
    """
    # ----------- pull arrays from the buffer ---------------------------------
    end = rb.pos if only_filled else None
    actions = rb.actions[:end]          # (N, act_dim)
    dones   = rb.dones[:end]            # (N, 1) or (N,)

    # convert to NumPy if needed
    if isinstance(actions, np.ndarray) is False:
        actions = actions.detach().cpu().numpy()
    if isinstance(dones, np.ndarray) is False:
        dones = dones.detach().cpu().numpy()

    actions = np.asarray(actions)
    dones   = np.asarray(dones).flatten().astype(bool)

    if actions.size == 0:
        raise ValueError("Replay buffer is empty")

    n_actions = actions.shape[1]
    if joint_labels is None:
        joint_labels = [f"joint {j}" for j in range(n_actions)]
    else:
        joint_labels = list(joint_labels)[:n_actions]

    # ----------- split into trajectories -------------------------------------
    segments = []
    start = 0
    for idx, term in enumerate(dones):
        if term:
            segments.append(actions[start : idx + 1])
            start = idx + 1
    if start < len(actions):            # leftover if last traj didn’t end with done
        segments.append(actions[start:])

    # ----------- plotting -----------------------------------------------------
    for t_idx, seg in enumerate(segments):
        timesteps = np.arange(seg.shape[0])

        plt.figure(figsize=(10, 4))
        for j in range(n_actions):
            plt.plot(timesteps, seg[:, j], label=joint_labels[j])

        plt.title(f"Trajectory {t_idx} – actions vs. timestep")
        plt.xlabel("Step index")
        plt.ylabel("Action value")
        plt.legend(ncol=4, fontsize=8, frameon=False)
        plt.tight_layout()

    plt.show()

def _module_summary(module: torch.nn.Module) -> str:
    """
    Pretty-print the layer-by-layer structure of a torch.nn.Module.
    Returns a multi-line string instead of printing directly so the caller
    can decide what to do with it.
    """
    lines = []
    for name, layer in module.named_children():
        layer_str = repr(layer)
        # Indent nested modules for readability
        layer_str = textwrap.indent(layer_str, "    ")
        lines.append(f"{name}:\n{layer_str}")
    if not lines:  # e.g. Sequential with layers in ._modules
        for i, layer in enumerate(module.children()):
            layer_str = textwrap.indent(repr(layer), "    ")
            lines.append(f"{i}:\n{layer_str}")
    return "\n".join(lines)

def print_sac_architecture_info(model: SAC, *, return_dict: bool = False):
    """
    Inspect an SB3 SAC model and print (or return) the most relevant architectural
    details for both actor and critic networks.

    Args:
        model (SAC):  A stable_baselines3.SAC instance.
        return_dict (bool): If True, return a dictionary instead of printing.

    Returns:
        None or Dict[str, Any]:  Depending on `return_dict`.
    """
    policy = model.policy
    data: Dict[str, Any] = {}

    # --- High-level policy info ------------------------------------------------
    data["policy_class"]   = policy.__class__.__name__
    data["observation_space"] = model.observation_space
    data["action_space"]      = model.action_space
    data["features_extractor"] = {
        "class": policy.features_extractor.__class__.__name__,
        "output_dim": getattr(policy, "features_dim", "unknown")
    }

    # --- Actor -----------------------------------------------------------------
    actor = getattr(policy, "actor", None) or getattr(policy, "pi", None)
    if actor is not None:
        data["actor"] = {
            "summary": _module_summary(actor),
            "n_params": sum(p.numel() for p in actor.parameters())
        }
    else:
        data["actor"] = "Could not locate actor network (attribute name changed?)."

    # --- Critics ---------------------------------------------------------------
    # Handle both two-Q and single-Q implementations
    qf1 = getattr(policy, "critic", None) \
          or getattr(policy, "qf1", None) \
          or getattr(policy, "q_net", None)
    qf2 = getattr(policy, "qf2", None) \
          or getattr(policy, "critic_target", None) \
          or getattr(policy, "q_net_target", None)

    critics_info = {}
    if qf1 is not None:
        critics_info["Q-network-1"] = {
            "summary": _module_summary(qf1),
            "n_params": sum(p.numel() for p in qf1.parameters())
        }
    if qf2 is not None:
        critics_info["Q-network-2"] = {
            "summary": _module_summary(qf2),
            "n_params": sum(p.numel() for p in qf2.parameters())
        }
    if not critics_info:
        critics_info = "Could not locate critic networks (attribute names changed?)."
    data["critics"] = critics_info

    # --- Total parameter counts -----------------------------------------------
    data["total_parameters"] = sum(p.numel() for p in policy.parameters())

    # --------------------------------------------------------------------------
    if return_dict:
        return data

    # Otherwise pretty-print
    print("# === SAC Architecture Information ===")
    print(f"Policy class     : {data['policy_class']}")
    print(f"Observation space: {data['observation_space']}")
    print(f"Action space     : {data['action_space']}\n")

    fe = data['features_extractor']
    print("Feature extractor")
    print(f"  Class          : {fe['class']}")
    print(f"  Output dim     : {fe['output_dim']}\n")

    print("Actor network")
    if isinstance(data['actor'], dict):
        print(data['actor']['summary'])
        print(f"  # parameters   : {data['actor']['n_params']}\n")
    else:
        print("  " + data['actor'])

    print("Critic network(s)")
    if isinstance(data['critics'], dict):
        for name, info in data['critics'].items():
            print(f"{name}")
            print(info['summary'])
            print(f"  # parameters   : {info['n_params']}\n")
    else:
        print("  " + data['critics'])

    print(f"Total parameters (policy): {data['total_parameters']}")


def print_replay_observation_ranges(
    model: SAC,
    *,
    only_filled: bool = True,        # ignore the “un-written” tail of a circular buffer
    include_next: bool = False,      # also scan replay_buffer.next_observations
    plot: bool = True,               # draw IQR + hist/PDF figures as well
    joint_labels: Sequence[str] = tuple(f"joint {i}" for i in range(8)),
    histogram_bins: int = 40,
) -> None:
    """
    Summarise min / max values for each observation key stored in the
    model's replay buffer and (optionally) plot their distributions.

    Parameters
    ----------
    model : stable_baselines3.SAC
        The SAC model whose `model.replay_buffer` uses Dict observations.
    only_filled : bool, default True
        If True, consider only the valid part of the circular buffer
        (`buffer[:replay_buffer.pos]`); otherwise scan the whole array.
    include_next : bool, default False
        If True, also show ranges for `next_observations`.
    plot : bool, default True
        Whether to draw box-plots + histograms/PDFs.
    joint_labels : iterable of str, default ('joint 0', … 'joint 7')
        Labels used for the *columns* of each Dict observation (e.g. joints).
    histogram_bins : int, default 40
        Number of bins in each per-joint histogram.
    """
    rb = model.replay_buffer
    valid_slice = slice(None, rb.pos) if only_filled else slice(None)

    def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        return x.detach().cpu().numpy() if torch.is_tensor(x) else x

    def _scan(obs_dict: Mapping[str, Union[np.ndarray, torch.Tensor]], label: str):
        print(f"\n=== {label} ===")
        for key, arr in obs_dict.items():
            data = _to_numpy(arr)[valid_slice]

            # 2-D (T, D) array where D is number of joints/features
            flat = data.reshape(-1, data.shape[-1])

            kmin = flat.min(axis=0)
            kmax = flat.max(axis=0)

            # ---- textual report -------------------------------------------------
            if kmin.size == 1:
                print(f"{key:<25}  min {kmin.item(): .5f}   max {kmax.item(): .5f}")
            else:
                rng = [f"[{lo: .5f}, {hi: .5f}]" for lo, hi in zip(kmin, kmax)]
                print(f"{key:<25}  " + ", ".join(rng))

            # ---- plotting -------------------------------------------------------
            if plot:
                _plot_distributions(
                    flat,
                    title_prefix=f"{label} → {key}",
                    joint_labels=joint_labels[: flat.shape[1]],
                    bins=histogram_bins,
                )

    _scan(rb.observations, "observations")
    if include_next and hasattr(rb, "next_observations"):
        _scan(rb.next_observations, "next_observations")


# --------------------------------------------------------------------------- #
#                           plotting utilities                                #
# --------------------------------------------------------------------------- #
def _plot_distributions(
    flat_array: np.ndarray,
    *,
    title_prefix: str,
    joint_labels: Sequence[str],
    bins: int,
) -> None:
    """
    Draw a box-plot (IQR) and per-joint histogram+PDF figure for one
    observation Dict entry.

    Parameters
    ----------
    flat_array : np.ndarray, shape (N, D)
        N samples × D joints/features.
    title_prefix : str
        Figure titles will start with this string.
    joint_labels : sequence of str
        Labels for the D columns.
    bins : int
        Number of histogram bins.
    """
    assert flat_array.ndim == 2, "input must be (N, D)"

    n_joints = flat_array.shape[1]
    labels = list(joint_labels)

    # -- 1) Box / IQR figure -------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, n_joints * 0.6), 4))
    ax.boxplot(
        flat_array,
        labels=labels,
        showfliers=False,
        whis=[25, 75],   # IQR whiskers
    )
    ax.set_title(f"{title_prefix} – IQR / box-plot")
    ax.set_ylabel("value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # -- 2) Histograms + fitted normal PDFs ----------------------------------
    cols = 2
    rows = int(np.ceil(n_joints / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)

    for i in range(n_joints):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        data = flat_array[:, i]
        μ, σ = data.mean(), data.std(ddof=0)

        # histogram (density=True normalises to ∫ PDF = 1)
        ax.hist(data, bins=bins, density=True, alpha=0.6)
        # overlay fitted Gaussian PDF
        x = np.linspace(data.min(), data.max(), 200)
        pdf = (1 / (σ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - μ) / σ) ** 2)
        ax.plot(x, pdf, linewidth=2)

        ax.set_title(f"{labels[i]}  (μ={μ:.2e}, σ={σ:.2e})", fontsize=9)
        ax.set_xlabel("value")
        ax.set_ylabel("density")

    # hide unsed axes if n_joints is odd
    for j in range(n_joints, rows * cols):
        fig.delaxes(axes.flatten()[j])

    fig.suptitle(f"{title_prefix} – per-joint histogram & fitted N(μ, σ²)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])




def print_replay_action_ranges(
    model: SAC,
    *,
    only_filled: bool = True,          # ignore the “un-written” tail of the circular buffer
    plot: bool = True,                 # draw IQR + hist/PDF figures
    joint_labels: Sequence = None,
    histogram_bins: int = 40,
) -> None:
    """
    Summarise min/max values of each *action* dimension stored in
    `model.replay_buffer.actions` and (optionally) plot their distributions.

    Parameters
    ----------
    model : stable_baselines3.SAC
        The SAC model whose replay buffer we want to inspect.
    only_filled : bool, default True
        If True, consider only the valid part of the circular buffer
        (`buffer[:replay_buffer.pos]`); otherwise scan the whole array.
    plot : bool, default True
        Whether to draw box-plots and histograms.
    joint_labels : sequence of str, optional
        Custom labels for each action dimension (defaults to “joint 0 …”).
    histogram_bins : int, default 40
        Number of bins for each histogram.
    """
    rb = model.replay_buffer
    valid_slice = slice(None, rb.pos) if only_filled else slice(None)

    # -- grab actions as a NumPy array --------------------------------------
    actions = rb.actions                       # (buffer_size, act_dim) ndarray / tensor
    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    else:
        actions = np.asarray(actions)

    actions = np.squeeze(actions[valid_slice],axis=1)             # (N, act_dim)

    n_samples,act_dim = actions.shape

    if joint_labels is None:
        joint_labels = tuple(f"joint {i}" for i in range(act_dim))
    joint_labels = joint_labels[:act_dim]

    # -- textual min / max summary ------------------------------------------
    print("\n=== actions ===")
    for d in range(act_dim):
        col = actions[:, d]
        print(f"{joint_labels[d]:<15}  min {col.min(): .5f}   max {col.max(): .5f}")

    # -- plotting -----------------------------------------------------------
    if plot:
        _plot_distributions(
            actions,
            title_prefix="actions",
            joint_labels=joint_labels,
            bins=histogram_bins,
        )


# --------------------------------------------------------------------------- #
#                 generic plotting helper (unchanged)                         #
# --------------------------------------------------------------------------- #
def _plot_distributions(
    flat_array: np.ndarray,
    *,
    title_prefix: str,
    joint_labels: Sequence,
    bins: int,
) -> None:
    """
    Draw a box-plot (IQR) and per-joint histogram+PDF figure for a 2-D array.

    flat_array : shape (N, D)
    """
    assert flat_array.ndim == 2, "input must be (N, D)"
    n_joints = flat_array.shape[1]
    labels = list(joint_labels)

    # -- 1) Box / IQR -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, n_joints * 0.6), 4))
    ax.boxplot(
        flat_array,
        labels=labels,
        showfliers=False,
        whis=[25, 75],
    )
    ax.set_title(f"{title_prefix} – IQR / box-plot")
    ax.set_ylabel("value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # -- 2) Histograms + fitted normal PDFs --------------------------------
    cols = 2
    rows = int(np.ceil(n_joints / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)

    for i in range(n_joints):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        data = flat_array[:, i]
        μ, σ = data.mean(), data.std(ddof=0)

        ax.hist(data, bins=bins, density=True, alpha=0.6)
        x = np.linspace(data.min(), data.max(), 200)
        pdf = (1 / (σ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - μ) / σ) ** 2)
        ax.plot(x, pdf, linewidth=2)

        ax.set_title(f"{labels[i]}  (μ={μ:.2e}, σ={σ:.2e})", fontsize=9)
        ax.set_xlabel("value")
        ax.set_ylabel("density")

    # remove unused axes (for odd D)
    for j in range(n_joints, rows * cols):
        fig.delaxes(axes.flatten()[j])

    fig.suptitle(f"{title_prefix} – per-joint histogram & fitted N(μ,σ²)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])