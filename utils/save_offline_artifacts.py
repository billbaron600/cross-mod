import os
import json
from datetime import datetime
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt  # only for type hints; not needed elsewhere
from stable_baselines3.common.base_class import BaseAlgorithm


def _make_offline_dir(base_dir: str) -> str:
    """
    Create <base_dir>/offline_models/<YYYY-MM-DD_HH-MM-SS>/ and return the path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    offline_dir = os.path.join(base_dir, "offline_models", timestamp)
    os.makedirs(offline_dir, exist_ok=True)
    return offline_dir


def _save_txt(path: str, d: Dict):
    """Write a small dict to an easily-readable *.txt* file."""
    with open(path, "w") as f:
        for k, v in d.items():
            f.write(f"{k}: {v}\n")


def save_offline_artifacts(
    model: BaseAlgorithm,
    model_idx: int,
    buffer_idx: int,
    output_fig: Optional[plt.Figure],
    reward_fig: Optional[plt.Figure],
    train_kwargs: Dict,
    # --- anything you want to log from the buffer ---------------------------
    # ------------------------------------------------------------------------
    base_config_dir: str,
    add_datetime_folder=True
):
    """
    Save model, replay buffer, training kwargs, buffer metadata, and figure.

    Parameters
    ----------
    model            : trained SB3 algorithm (e.g. SAC)
    model_idx        : newest model step index (→ filename)
    buffer_idx       : replay-buffer step index    (→ filename)
    output_fig       : matplotlib Figure (can be None)
    train_kwargs     : kwargs dict used for training
    raw_clip_limit   : tuple passed to recompute_stats
    clip_limit       : tuple passed to recompute_stats
    base_config_dir  : config.iteration_working_dir
    """
    if add_datetime_folder is True:
        save_dir = _make_offline_dir(base_config_dir)
    else:
        save_dir = base_config_dir

    # 1) kwargs ----------------------------------------------------------------
    _save_txt(os.path.join(save_dir, "train_kwargs.txt"), train_kwargs)

    # 2) model -----------------------------------------------------------------
    model_path = os.path.join(save_dir, f"model_{model_idx}.zip")
    #model.save(model_path)

    # 3) replay buffer + metadata ---------------------------------------------
    buf_path = os.path.join(save_dir, f"buffer_{buffer_idx}.pkl")
    #model.save_replay_buffer(buf_path)

    buf_meta = {
        "model_idx": model_idx,
        "buffer_idx": buffer_idx,
        "recent_window": getattr(model.replay_buffer, "recent_window", None),
        "std_scale_factor": getattr(model.replay_buffer, "std_scale_factor", None),
    }
    with open(os.path.join(save_dir, "replay_metadata.json"), "w") as f:
        json.dump(buf_meta, f, indent=2)

    # 4) figure ----------------------------------------------------------------
    if output_fig is not None:
        fig_path = os.path.join(save_dir, "loss_terms.png")
        output_fig.savefig(fig_path, dpi=300, bbox_inches="tight")

    if reward_fig is not None:
        fig_path = os.path.join(save_dir, "reward_distributions.png")
        reward_fig.savefig(fig_path, dpi=300, bbox_inches="tight")

    print(f"[save_offline_artifacts] All artifacts saved to:\n  {save_dir}")
    #return save_dir