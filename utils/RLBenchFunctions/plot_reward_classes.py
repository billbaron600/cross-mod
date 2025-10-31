import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from stable_baselines3.common.base_class import BaseAlgorithm


def plot_reward_distributions(
    model: BaseAlgorithm,
    bins: int = 50,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Visualise raw & normalised rewards stored in a SB3 replay buffer.

    Rows:
      0: raw_rewards_unclipped
      1: raw_rewards  (clipped)
      2: rewards      (normalised / std-scaled)

    Columns:
      0 : index → value line plot
      1 : full-buffer histogram
      2 : histogram of *valid* indices
          (where raw_rewards_unclipped >= min_reward_threshold)
    """
    buf = model.replay_buffer
    pos = buf.pos

    # --------- flatten reward arrays -----------------------------------------
    raw_unclipped = buf.raw_rewards_unclipped[:pos].flatten()
    raw_clipped   = buf.raw_rewards[:pos].flatten()
    rewards_norm  = buf.rewards[:pos].flatten()

    data_rows = [
        ("Raw Unclipped Reward", raw_unclipped),
        ("Raw Clipped Reward",   raw_clipped),
        ("Normalised Reward",    rewards_norm),
    ]

    # --------- mask of "valid" indices ---------------------------------------
    thr = getattr(buf, "min_reward_threshold", None)
    if thr is not None:
        valid_mask = raw_unclipped >= thr
    else:
        # if the attribute is missing, keep every index
        valid_mask = np.ones_like(raw_unclipped, dtype=bool)

    # --------- figure & axes --------------------------------------------------
    fig, axs = plt.subplots(
        nrows=3, ncols=3, figsize=(18, 12), sharex=False, sharey=False
    )

    col_labels = ["Time-series", "Histogram (all)", "Histogram (valid mask)"]
    for c, lbl in enumerate(col_labels):
        axs[0, c].set_title(lbl, fontsize=11, pad=12)

    for r, (label, arr) in enumerate(data_rows):
        # 0) time-series -------------------------------------------------------
        ax_time = axs[r, 0]
        ax_time.plot(arr, lw=0.8)
        ax_time.set_ylabel("Value")
        ax_time.set_xlabel("Index")
        ax_time.grid(True)

        # 1) histogram (full) --------------------------------------------------
        ax_hist_all = axs[r, 1]
        ax_hist_all.hist(arr, bins=bins, edgecolor="black", alpha=0.75)
        ax_hist_all.set_xlabel("Value")
        ax_hist_all.set_ylabel("Count")
        ax_hist_all.grid(True)

        # 2) histogram (valid indices) ----------------------------------------
        ax_hist_valid = axs[r, 2]
        ax_hist_valid.hist(arr[valid_mask], bins=bins, edgecolor="black", alpha=0.75)
        ax_hist_valid.set_xlabel("Value")
        ax_hist_valid.set_ylabel("Count")
        ax_hist_valid.grid(True)

        # row-label on the leftmost subplot
        ax_time.set_title(label, loc="left", fontsize=10, pad=12)

    # --------- figure-level title & layout -----------------------------------
    buf_attrs = {
        "raw_clip_limit":   getattr(buf, "raw_clip_limit", None),
        "clip_limit":       getattr(buf, "clip_limit", None),
        "min_reward_thr":   getattr(buf, "min_reward_threshold", None),
        "std_scale_factor": getattr(buf, "std_scale_factor", None),
    }
    fig.suptitle(
        "Replay Buffer Reward Distributions\n"
        + ", ".join(f"{k}={v}" for k, v in buf_attrs.items()),
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # --------- info printout --------------------------------------------------
    if buf.clip_limit is not None:
        low, high = buf.clip_limit
        in_window = ((buf.rewards[:pos].flatten() > low)
                     & (buf.rewards[:pos].flatten() < high)).sum()
        print(f"{in_window} out of {pos} rewards are within ({low}, {high}).")
    if thr is not None:
        print(f"{valid_mask.sum()} indices pass the min_reward_threshold "
              f"({thr}).")

    if show:
        plt.show()

    return fig