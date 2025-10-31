import random
import torch
import numpy as np
from typing import List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox
import cv2, copy, itertools, threading, time
from PIL import Image, ImageTk
from typing import List, Tuple, Literal, Optional
from utils.Classes.policy_trajectory_class import PolicyTrajectory
import os, sys, threading, pickle, copy, traceback
import types

import os, sys, threading, time, random, copy, traceback, cv2, tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from typing import List, Tuple, Optional

# --- Python 3.8 compatibility for Literal ------------------------------------
try:
    from typing import Literal          # 3.9+
except ImportError:                     # 3.8
    from typing_extensions import Literal


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def run_preference_elicitation(
        db: "PreferenceDatabase",
        pairs: Optional[List[Tuple["PolicyTrajectory", "PolicyTrajectory"]]] = None,
        *,
        write_to_db: bool = False,
        quicksave_path: Optional[str] = None,
        save_every: Optional[int] = 10,
    ) -> List[Tuple["PolicyTrajectory", "PolicyTrajectory", Optional[float]]]:
    """
    Launch a GUI, show each pair of videos side-by-side, and let the user pick
    among: A, B, BOTH, NEITHER.

    Returns
    -------
    labels : list of (trajA, trajB, pref_value)
        pref_value = 0     → prefer B
                     0.5   → prefer BOTH
                     1     → prefer A
                     None  → NEITHER / skipped
    """
    # ──────────────────────────────────────────────────────────────────
    # Select pairs to show
    # ──────────────────────────────────────────────────────────────────
    if pairs is None:
        pairs = [(a, b) for (a, b, lbl) in db.BALD_pairs if lbl is None]
    if not pairs:
        print("[GUI] No unlabeled pairs supplied.", file=sys.stderr)
        return []

    # Shuffle display order so the user doesn’t always see the same order
    random.shuffle(pairs)
    pairs_iter = iter(pairs)
    labels: List[Tuple["PolicyTrajectory", "PolicyTrajectory", Optional[float]]] = []

    # ──────────────────────────────────────────────────────────────────
    # Lightweight video-panel widget
    # ──────────────────────────────────────────────────────────────────
    class VideoPanel(tk.Frame):
        def __init__(self, master, traj, *a, **kw):
            super().__init__(master, width=600, height=600, *a, **kw)
            self.pack_propagate(False)

            # Segment metadata (if present)
            try:
                start  = getattr(traj, "segment_meta", {}).get("start", 0)
                length = getattr(traj, "segment_meta", {}).get("length", None)
            except AttributeError:
                start = 0
                length = None
            self.start_frame = start
            self.end_frame   = None if length is None else start + length

            # Capture
            self.cap = cv2.VideoCapture(traj.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {traj.video_path}")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            self.fps      = 60.0
            self.playing  = False
            self.thread   = None

            self.label = tk.Label(self)
            self.label.pack()
            self._show_first()

        def _show_first(self):
            ok, frame = self.cap.read()
            if ok:
                frame = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_AREA)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.label.configure(image=img)
                self.label.image = img
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        def _loop(self):
            dt = 1.0 / self.fps
            while self.playing:
                pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if self.end_frame is not None and pos >= self.end_frame:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

                ok, frame = self.cap.read()
                if not ok:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                    continue

                frame = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_AREA)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = ImageTk.PhotoImage(Image.fromarray(rgb))
                try:
                    self.label.configure(image=img)
                    self.label.image = img   # keep reference
                except tk.TclError:         # window closed
                    break
                time.sleep(dt)

        def toggle(self):
            if self.playing:
                self.playing = False
            else:
                self.playing = True
                self.thread  = threading.Thread(target=self._loop, daemon=True)
                self.thread.start()
        def restart(self):
            """Jump back to the first frame and show it."""
            self.playing = False               # stop if it was running
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self._show_first()

        def destroy(self):
            self.playing = False
            if self.cap:
                self.cap.release()
            super().destroy()

    # ──────────────────────────────────────────────────────────────────
    # Build main window
    # ──────────────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("Preference Elicitation")
    root.geometry("1280x720")

    panel_left  = tk.Frame(root); panel_left.pack(side="left",  expand=True, fill="both")
    panel_right = tk.Frame(root); panel_right.pack(side="right", expand=True, fill="both")

    current_pair: Optional[Tuple["PolicyTrajectory", "PolicyTrajectory"]] = None

    def load_next_pair():
        nonlocal current_pair
        try:
            current_pair = next(pairs_iter)
        except StopIteration:
            messagebox.showinfo("Done", "No more pairs.")
            root.destroy()
            return

        # Clear old panels
        for p in (panel_left, panel_right):
            for child in p.winfo_children():
                child.destroy()

        # New video panels
        left_vid  = VideoPanel(panel_left,  current_pair[0]); left_vid.pack(expand=True, fill="both")
        right_vid = VideoPanel(panel_right, current_pair[1]); right_vid.pack(expand=True, fill="both")

        panel_left.video  = left_vid
        panel_right.video = right_vid

    # ──────────────────────────────────────────────────────────────────
    # Button actions
    # ──────────────────────────────────────────────────────────────────
    LabelType = Literal["A", "B", "BOTH", "NEITHER"]

    def record(choice: LabelType):
        if current_pair is None:
            return

        pref_value = {"A": 0.0, "B": 1.0, "BOTH": 0.5, "NEITHER": None}[choice]
        labels.append((*current_pair, pref_value))

        # Update BALD bookkeeping
        for idx, (a, b, _) in enumerate(db.BALD_pairs):
            if (a is current_pair[0] and b is current_pair[1]) or \
               (a is current_pair[1] and b is current_pair[0]):
                db.BALD_pairs.pop(idx)
                db.BALD_pairs_labelled.append((a, b, pref_value))
                break

        # Immediate DB write-back (optional)
        if write_to_db and pref_value is not None:
            if   pref_value == 0.0: db._add_pref_unique(current_pair[0], current_pair[1], 0)   # A > B
            elif pref_value == 1.0: db._add_pref_unique(current_pair[1], current_pair[0], 0)   # B > A
            else:                   db._add_pref_unique(current_pair[0], current_pair[1], 0.5) # BOTH

        # Periodic quick-save
        if save_every and len(labels) % save_every == 0:
            for btn in btn_row.winfo_children():
                btn.configure(state="disabled")
                root.update_idletasks()
            db.quicksave(quicksave_path)
            for btn in btn_row.winfo_children():
                btn.configure(state="normal")

        load_next_pair()

    # ──────────────────────────────────────────────────────────────────
    # Controls
    # ──────────────────────────────────────────────────────────────────
    btn_row = ttk.Frame(root); btn_row.pack(side="bottom", fill="x", pady=10)
    pad = dict(side="left", padx=12, pady=8)

    ttk.Button(btn_row, text="▶/⏸", width=6,
               command=lambda: (panel_left.video.toggle(), panel_right.video.toggle())
               ).pack(**pad)
    
    ttk.Button(btn_row, text="⏮ Restart", width=8,
               command=lambda: (panel_left.video.restart(),
                                panel_right.video.restart())).pack(**pad)
    ttk.Button(btn_row, text="Prefer A",    width=10, command=lambda: record("A")).pack(**pad)
    ttk.Button(btn_row, text="Prefer B",    width=10, command=lambda: record("B")).pack(**pad)
    ttk.Button(btn_row, text="Prefer BOTH", width=12, command=lambda: record("BOTH")).pack(**pad)
    ttk.Button(btn_row, text="Prefer NEITHER", width=12,
               command=lambda: record("NEITHER")).pack(**pad)
    ttk.Button(btn_row, text="DONE", width=8, command=root.destroy
               ).pack(side="right", padx=12, pady=8)

    # Kick off first pair
    load_next_pair()
    root.mainloop()

    # Final sync to DB + save
    if write_to_db:
        for tA, tB, pref in labels:
            if pref is None:
                continue
            if   pref == 0.0: db._add_pref_unique(tA, tB, 0)
            elif pref == 1.0: db._add_pref_unique(tB, tA, 0)
            else:             db._add_pref_unique(tA, tB, 0.5)

    if save_every:
        db.quicksave(quicksave_path)

    return labels

class Correction:
    def __init__(self, *, current_policy_trajectory=None, corrections=None,correction_success_status=None):
        self.current_policy_trajectory = current_policy_trajectory
        self.corrections = corrections or []
        self.correction_success_status = correction_success_status or []

    def append_corrections_list(self, correction,success_status=None):
        self.corrections.append(correction)
        self.correction_success_status.append(success_status)

class PreferenceDatabase:
    def __init__(self, expert_demos=None, non_expert_demos=None, noise_trajectories=None, corrections=None, feat_stats = None):
        self.expert_demos = expert_demos if expert_demos is not None else []
        self.non_expert_demos = non_expert_demos if non_expert_demos is not None else []
        self.noise_trajectories = noise_trajectories if noise_trajectories is not None else []
        self.corrections = corrections if corrections is not None else []
        self.pairwise_comparisons = []
        # — new — list[tuple(trajA, trajB, label)] where label starts None
        self.BALD_pairs = []
        self.BALD_pairs_labelled = []
        self.policy_samples = []

        #Normalization limits for features (dictionary)
        self.feat_stats = feat_stats

        # OPTIONAL – give trajectories a default rating the first time you use Elo
        self.default_elo = 1_000


        # optional default path for quicksave()
        self._default_save_path = "preference_database.pkl"
        self.save_round = 0

    def is_valid_traj(self,traj):
        """
        Return True only if the trajectory carries a non-empty observation (or
        action) array.  Test whatever field you rely on when you later build the
        tensor.
        """
        return hasattr(traj, "observations") and len(traj.observations) > 0
    
    def add_comparisons(self, experts, non_experts, include_noise=False, triangular=False, noise_fraction=1.0):
        """
        Add pairwise comparisons for provided experts and non-experts.
        If include_noise is True, also compare these trajectories against noise trajectories.
        - triangular=True enforces equal counts of expert↔non-expert, expert↔noise, non-expert↔noise.
        - noise_fraction scales the number of noise comparisons (0.0 to 1.0) when triangular=False.
        """
        # Always add expert vs non-expert
        for expert in experts:
            for non_expert in non_experts:
                self.pairwise_comparisons.append((expert, non_expert, 0))

        if not include_noise or not self.noise_trajectories:
            return

        if triangular:
            # Sample equal number of noise pairs as expert-non_expert pairs
            num_pairs = len(experts) * len(non_experts)
            # expert ↔ noise
            for _ in range(num_pairs):
                expert = random.choice(experts)
                noise = random.choice(self.noise_trajectories)
                self.pairwise_comparisons.append((expert, noise, 0))
            # non-expert ↔ noise
            for _ in range(num_pairs):
                non_expert = random.choice(non_experts)
                noise = random.choice(self.noise_trajectories)
                self.pairwise_comparisons.append((non_expert, noise, 0))
        else:
            # Full cross-product, but possibly subsampled
            # Determine how many noise trajectories to sample
            sample_count = int(len(self.noise_trajectories) * noise_fraction)
            sampled_noise = random.sample(self.noise_trajectories, sample_count)
            for traj in experts + non_experts:
                for noise in sampled_noise:
                    self.pairwise_comparisons.append((traj, noise, 0))

        self._register_trajectories(experts, non_experts)
    
    def generate_initial_comparisons_DEPRECATED_DONT_USE(self):
        """Generate pairwise comparisons dataset."""
        self.pairwise_comparisons = []

        # Expert preferred over non-expert
        for expert in self.expert_demos:
            for non_expert in self.non_expert_demos:
                self.pairwise_comparisons.append((expert, non_expert, 0))

        # Expert and non-expert both preferred over noise
        for better_traj in self.expert_demos + self.non_expert_demos:
            for noise in self.noise_trajectories:
                self.pairwise_comparisons.append((better_traj, noise, 0))
            

    def add_corrections_to_database(self, current_policy_trajectory, corrections):
        """
        Add pairwise comparisons where each correction trajectory is preferred over the current policy trajectory.
        """
        #correction_instance = Correction(current_policy_trajectory=current_policy_trajectory, corrections=corrections)
        correction_instance = Correction(  #fixed
            current_policy_trajectory=current_policy_trajectory,
            corrections=corrections
        )
        self.corrections.append(correction_instance)
        for correction in corrections:
            self.pairwise_comparisons.append((correction, current_policy_trajectory, 0))
        self._register_trajectories([], corrections)

    def generate_segment_pairs(
        self,
        segment_length: int = 75,
        random_starting_points: bool = False,
        limit_to_segments: bool = False,
        start_from_traj_end: bool = False,
        compare_all_segments: bool    = False,
        pad_trajectories: bool = True
    ):
        """
        Return a list of (tensor_traj1, tensor_traj2, label).

        • If *limit_to_segments* is False
            – pad the shorter trajectory so both tensors have equal length.

        • If *limit_to_segments* is True
            – extract a segment of *segment_length* timesteps from each traj:
                • random_starting_points = True   → random window
                • random_starting_points = False  → front window  [0 : L)
                • start_from_traj_end    = True & random_starting_points = False
                                                    → back window  [T-L : T)
        (If the trajectory is shorter than *segment_length*, pad by repeating
        the final row to reach the desired length.)
        """

        segment_pairs = []

        for traj1, traj2, label in self.pairwise_comparisons:
            tensor_traj1 = traj1.generate_tensor_from_trajectory(feat_stats = self.feat_stats)
            tensor_traj2 = traj2.generate_tensor_from_trajectory(feat_stats = self.feat_stats)

            if limit_to_segments:
                tensor_traj1 = self._slice_or_pad(
                    tensor_traj1, segment_length,
                    random_starting_points, start_from_traj_end
                )
                tensor_traj2 = self._slice_or_pad(
                    tensor_traj2, segment_length,
                    random_starting_points, start_from_traj_end
                )
            else:
                if pad_trajectories is True:
                    max_len = max(tensor_traj1.shape[0], tensor_traj2.shape[0])
                    tensor_traj1 = self._pad_to_length(tensor_traj1, max_len)
                    tensor_traj2 = self._pad_to_length(tensor_traj2, max_len)

            segment_pairs.append((tensor_traj1, tensor_traj2, label))

        return segment_pairs

    def _slice_or_pad(
        self,
        tensor,
        L: int,
        random_start: bool,
        start_from_end: bool
    ):
        """Return a view of length *L*, padding if tensor is too short."""
        T = tensor.shape[0]

        if T >= L:
            if random_start:
                import numpy as np
                s = np.random.randint(0, T - L + 1)
            elif start_from_end:
                s = T - L
            else:
                s = 0
            segment = tensor[s : s + L]
        else:
            segment = self._pad_to_length(tensor, L)

        return segment

    
    def _get_segment_with_padding(self, traj_tensor, segment_length, random_starting_points):
        traj_len = traj_tensor.shape[0]

        if traj_len >= segment_length:          # same as before
            start_idx = random.randint(0, traj_len - segment_length) if random_starting_points else 0
            return traj_tensor[start_idx : start_idx + segment_length]

        # --- NEW: zero‑pad instead of repeating last row -------------
        pad_len = segment_length - traj_len
        pad     = torch.zeros(
            pad_len,
            traj_tensor.shape[1],
            dtype = traj_tensor.dtype,
            device= traj_tensor.device
        )
        return torch.cat([traj_tensor, pad], dim=0)
    def _get_segment_with_padding_DEPRECATED(self, traj_tensor, segment_length, random_starting_points):
        traj_len = traj_tensor.shape[0]

        if traj_len >= segment_length:
            start_idx = random.randint(0, traj_len - segment_length) if random_starting_points else 0
            segment = traj_tensor[start_idx:start_idx + segment_length]
        else:
            padding = traj_tensor[-1].unsqueeze(0).repeat(segment_length - traj_len, 1)
            segment = torch.cat([traj_tensor, padding], dim=0)

        return segment

    def _pad_to_length(self, traj_tensor, desired_length, pad_val=0.0):
        """
        Pad *traj_tensor* (T, D) up to *desired_length* with a sentinel vector.
        Using all-zeros is safe if your features are already standardised
        (≈ 0 ± 1).  Nothing else in the pipeline needs to change.
        """
        traj_len, feat_dim = traj_tensor.shape
        if traj_len < desired_length:
            pad = torch.full(
                (desired_length - traj_len, feat_dim),
                pad_val,
                dtype=traj_tensor.dtype,
                device=traj_tensor.device
            )
            traj_tensor = torch.cat([traj_tensor, pad], dim=0)
        return traj_tensor

    def _pad_to_length_DEPRECATED(self, traj_tensor, desired_length):
        traj_len = traj_tensor.shape[0]
        if traj_len < desired_length:
            padding = traj_tensor[-1].unsqueeze(0).repeat(desired_length - traj_len, 1)
            traj_tensor = torch.cat([traj_tensor, padding], dim=0)
        return traj_tensor

    def _extend_unique(self, lst, traj):
        if traj not in lst:          # identity check – preserves dedup the way you ok’d
            lst.append(traj)

    def _register_trajectories(self, experts, non_experts):
        for t in experts:     self._extend_unique(self.expert_demos,     t)
        for t in non_experts: self._extend_unique(self.non_expert_demos, t)

    def sample_pairs_BALD(
        self,
        ensemble: Optional["PreferenceEnsemble"] = None,
        top_k_percent: float = 0.10,
        limit_to_segment: Optional[int] = None,
        random_start: bool = True,
        n_duplicate: int = 1,
        rng: Optional[random.Random] = None):
        """
        Return a list of trajectory pairs ranked by ensemble disagreement.

        • If *ensemble* is None → returns an empty list. Call *sample_pairs_random*
          instead when you want random selection.
        • Disagreement score = variance of (reward_A – reward_B) across models.
        • Pairs considered:
              - every (expert_i, expert_j), i < j
              - every (expert, non_expert)
          across *all* demos / corrections.
        • If *limit_to_segment* is given, each trajectory is cloned into a
          lightweight *segment* view of that length.  The clone is tagged with:
              traj.is_segment = True
              traj.segment_meta = {'start': s, 'length': L}
        """

        if ensemble is None:
            return []          # explicit; user can fall back to random sampler

        # ----------------------------------------------------
        # 1) build the candidate list
        # ----------------------------------------------------
        experts      = self.expert_demos
        non_experts  = self.non_expert_demos

        candidates: List[Tuple[PolicyTrajectory, PolicyTrajectory]] = []

        # expert–expert (unique unordered pairs)
        for i in range(len(experts)):
            for j in range(i + 1, len(experts)):
                candidates.append((experts[i], experts[j]))

        # expert–non-expert
        for ea in experts:
            for nb in non_experts:
                candidates.append((ea, nb))

        # 1a) filter out any “junk” trajectories
        # ----------------------------------------------------
        candidates = [
            (a, b) for (a, b) in candidates
            if self.is_valid_traj(a) and self.is_valid_traj(b)
        ]

        #dupclaite the canidate list (so we can get multiple segments from the same pairwise trajctoreis)
        candidates = candidates * n_duplicate

        
        if not candidates:
            return []

        # ----------------------------------------------------
        # 2) optionally create segment clones (shallow copy)
        # ----------------------------------------------------
        if limit_to_segment is not None:
            seg_pairs = []
            for traj_a, traj_b in candidates:
                seg_pairs.append((
                    self._make_segment_clone(traj_a,
                                             limit_to_segment,
                                             random_start),
                    self._make_segment_clone(traj_b,
                                             limit_to_segment,
                                             random_start),
                ))
            candidates = seg_pairs

        # ----------------------------------------------------
        # 3) compute disagreement score
        # ----------------------------------------------------
        scores = []
        for tA, tB in candidates:
            # ensemble.score_trajectory() → ndarray shape (n_models,)
            sA = ensemble.score_trajectory(tA.generate_tensor_from_trajectory(feat_stats = self.feat_stats))
            sB = ensemble.score_trajectory(tB.generate_tensor_from_trajectory(feat_stats = self.feat_stats))
            delta = sA - sB                       # per-model preference logit
            #var   = np.var(delta)                 # scalar disagreement proxy
            var = torch.var(delta, unbiased=False).item()
            scores.append(var)

        # ----------------------------------------------------
        # 4) rank and truncate
        # ----------------------------------------------------
        k = max(1, int(len(candidates) * top_k_percent))
        idx_sorted = np.argsort(scores)[::-1]     # high variance first
        best_pairs = [candidates[i] for i in idx_sorted[:k]]

        #append the attribute
        self.BALD_pairs = [(a, b, None) for (a, b) in best_pairs]

        return best_pairs

    # ---------- helper for segment clones ----------
    def _make_segment_clone(
        self,
        traj: PolicyTrajectory,
        L: int,
        random_start: bool
    ):
        """
        Return a shallow clone of *traj* whose trajectory-level tensors are
        trimmed to length *L*.  Does NOT modify the original.
        """
        import numpy as np

        # pick a start idx (never pad)
        T = len(traj.observations)          # assumes obs length == trajectory length
        if T <= L:
            start = 0                       # no trim necessary
        else:
            if random_start:
                start = np.random.randint(0, T - L + 1)
            else:
                start = 0

        end = start + min(L, T)

        seg = copy.copy(traj)               # shallow copy keeps video_path pointer
        # slice every timestep-dependent field if they exist
        for attr in (
            "observations",
            "actions",
            "rewards",
            "ee_positions",
            "trajectory_tensor",
        ):
            if hasattr(traj, attr) and getattr(traj, attr) is not None:
                setattr(seg, attr, getattr(traj, attr)[start:end])

        seg.is_segment   = True
        seg.segment_meta = {"start": int(start), "length": int(end - start)}
        return seg

    def sample_pairs_random(
        self,
        percentage: float = 0.10,
        rng: Optional[random.Random] = None,
    ):
        """
        Uniformly sample the requested percentage of candidate pairs
        (same definition of candidates as in *sample_pairs_BALD*).
        """
        rng = rng or random
        experts      = self.expert_demos
        non_experts  = self.non_expert_demos

        candidates: List[Tuple[PolicyTrajectory, PolicyTrajectory]] = []
        for i in range(len(experts)):
            for j in range(i + 1, len(experts)):
                candidates.append((experts[i], experts[j]))

        for ea in experts:
            for nb in non_experts:
                candidates.append((ea, nb))

        # 1a) filter out any “junk” trajectories
        # ----------------------------------------------------
        candidates = [
            (a, b) for (a, b) in candidates
            if self.is_valid_traj(a) and self.is_valid_traj(b)
        ]
        if not candidates:
            return []

        if not candidates:
            return []

        rng.shuffle(candidates)
        k = max(1, int(len(candidates) * percentage))
        return candidates[:k]

    def quicksave_DEPRECATED(self, path: Optional[str] = None):
        """Pickle the entire PreferenceDatabase to *path* (default set in __init__)."""
        #import pickle, os
        #path = path or self._default_save_path
        #path = "preference_database_" + str(self.save_round)+".pkl"
        #self.save_round+=1
        #os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        #with open(path, "wb") as f:
        #    pickle.dump(self, f)
        path = "preference_database_" + str(self.save_round)+".pkl"
        self.save_round+=1
        #snap = copy.deepcopy(self)          # <-- optional: avoid mutating while saving
        #def _job(p):
        #    with open(p, "wb") as f:
        #        pickle.dump(snap, f)
        #threading.Thread(target=_job, args=(path or self._default_save_path,), daemon=True).start()
        print("Saved to Path: " + path)
    def quicksave(self, path: Optional[str] = None):
        """Pickle the entire PreferenceDatabase to *path* (defaulting to 
        preference_database_{self.save_round}.pkl)."""
        import os, pickle

        # decide on filename
        #save_path = path if path is not None else f"preference_database_{self.save_round}.pkl"
        save_path = f"preference_database_{self.save_round}.pkl"
        self.save_round += 1

        # ensure directory exists
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # write it out
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

        print(f"Saved to Path: {save_path}")
    


    @classmethod
    def quickload(cls, path: str):
        """Re-load a PreferenceDatabase that was saved with quicksave()."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------
    # helper: insert once
    # ------------------------------------------------------------
    def _add_pref_unique(self, a, b, pref_value):
        tup = (a, b, pref_value)
        if tup not in self.pairwise_comparisons:
            self.pairwise_comparisons.append(tup)

    # ============================================================
    #  Simple viewer for existing comparisons (read-only)
    # ============================================================
    def view_comparisons_tk(self):
        """
        Pop up a Tk window to browse the current pairwise_comparisons list.

        • Starts from the most-recent comparison (end of the list) and walks
          backwards.
        • No labels are recorded; you just inspect what’s already stored.
        """

        import tkinter as tk
        from tkinter import ttk
        import cv2, threading, time
        from PIL import Image, ImageTk

        if not self.pairwise_comparisons:
            print("No comparisons to display.")
            return

        # ---------- internal widget ----------
        class VideoPanel(tk.Frame):
            def __init__(self, master, traj, *a, **kw):
                super().__init__(master, width=600, height=600, *a, **kw)
                self.pack_propagate(False)
                self.cap  = cv2.VideoCapture(traj.video_path)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.fps  = 60.0                        # constant playback
                self.playing = False
                self.label = tk.Label(self)
                self.label.pack()
                self.thread = None
                self._show_first()

            def _show_first(self):
                ok, frame = self.cap.read()
                if ok:
                    frame = cv2.resize(frame, (600, 600), cv2.INTER_AREA)
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img   = ImageTk.PhotoImage(Image.fromarray(rgb))
                    self.label.configure(image=img)
                    self.label.image = img
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            def _loop(self):
                dt = 1.0 / self.fps
                while self.playing:
                    ok, frame = self.cap.read()
                    if not ok:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    frame = cv2.resize(frame, (600, 600), cv2.INTER_AREA)
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img   = ImageTk.PhotoImage(Image.fromarray(rgb))
                    try:
                        self.label.configure(image=img)
                        self.label.image = img
                    except tk.TclError:     # widget destroyed
                        break
                    time.sleep(dt)

            def toggle(self):
                if self.playing:
                    self.playing = False
                else:
                    self.playing = True
                    self.thread = threading.Thread(target=self._loop, daemon=True)
                    self.thread.start()

            def destroy(self):
                self.playing = False
                if self.cap:
                    self.cap.release()
                super().destroy()

        # ---------- main window ----------
        root = tk.Tk()
        root.title("Comparison Viewer")
        root.geometry("1280x760")

        idx  = len(self.pairwise_comparisons) - 1   # start at newest

        panel_left  = tk.Frame(root); panel_left.pack(side="left",  expand=True, fill="both")
        panel_right = tk.Frame(root); panel_right.pack(side="right", expand=True, fill="both")
        status_lbl  = ttk.Label(root, font=("TkDefaultFont", 12, "bold"))
        status_lbl.pack(side="top", pady=6)

        def show_current():
            # clear old
            for w in (panel_left, panel_right):
                for c in w.winfo_children(): c.destroy()

            trajA, trajB, pref = self.pairwise_comparisons[idx]
            lp = VideoPanel(panel_left,  trajA); lp.pack(expand=True, fill="both")
            rp = VideoPanel(panel_right, trajB); rp.pack(expand=True, fill="both")
            panel_left.video  = lp
            panel_right.video = rp

            if   pref == 0:   txt = "Preferred: LEFT"
            elif pref == 1:   txt = "Preferred: RIGHT"
            elif pref == 0.5: txt = "Preferred: BOTH"
            else:             txt = "Preferred: NEITHER"
            status_lbl.config(text=f"[{idx+1}/{len(self.pairwise_comparisons)}]  {txt}")

        def next_pair(step):
            nonlocal idx
            idx = (idx + step) % len(self.pairwise_comparisons)
            show_current()

        # controls
        btnrow = ttk.Frame(root); btnrow.pack(side="bottom", pady=10)
        ttk.Button(btnrow, text="⟵ Prev", width=10,
                   command=lambda: next_pair(-1)).pack(side="left", padx=8)
        ttk.Button(btnrow, text="Play/⏸", width=10,
                   command=lambda: (panel_left.video.toggle(),
                                    panel_right.video.toggle())).pack(side="left", padx=8)
        ttk.Button(btnrow, text="Next ⟶", width=10,
                   command=lambda: next_pair(+1)).pack(side="left", padx=8)
        ttk.Button(btnrow, text="Close", width=8,
                   command=root.destroy).pack(side="right", padx=8)

        show_current()
        root.mainloop()

    # ============================================================
    #  BALD on *external* trajectory list
    # ============================================================
    def sample_policy_pairs_BALD(
        self,
        traj_list = [],
        self_compare_frac = 0.30,          # ← new arg
        compare_to_expert = True,
        empty_current_pairs = False,
        ensemble = None,
        top_k_percent = 0.10,
        limit_to_segment = None,
        random_start = True,
        rng = None):
        """
        Rank comparisons of *traj_list* against expert/non-expert demos,
        **plus** a self-comparision subset of the policy samples themselves.

        * self_compare_frac : fraction (0-1) of total candidate pairs that
        should be policy-vs-policy.  The remainder are the usual
        demo-vs-policy pairs.
        """

        if empty_current_pairs is True:
            #Empty current BALD pairs
            self.BALD_pairs = []

        if ensemble is None or not traj_list:
            return []

        self.policy_samples.append(list(traj_list))          # bookkeeping

        candidates = []

        # ── 1a. demo ↔ policy candidates (optional) ────────────────────────
        if compare_to_expert:
            experts, non_experts = self.expert_demos, self.non_expert_demos
            for p in traj_list:
                for e in experts:      candidates.append((p, e))
                for n in non_experts:  candidates.append((p, n))

        # ── 1b. policy ↔ policy candidates (sampled) ───────────────────────
        if self_compare_frac > 0.0 and len(traj_list) > 1:
            import itertools, numpy as np
            all_self = list(itertools.combinations(traj_list, 2))

            if compare_to_expert:
                n_demo = len(candidates)
                n_self = int(round(n_demo * self_compare_frac / (1.0 - self_compare_frac)))
            else:
                # no demo pairs ⇒ take *all* self pairs, subsample if necessary
                n_self = int(round(len(all_self) * self_compare_frac))

            rng = rng or np.random
            idx = rng.choice(len(all_self),
                            size=min(n_self, len(all_self)),
                            replace=len(all_self) < n_self)
            candidates.extend(all_self[i] for i in idx)

        # ── 3. keep only valid trajectories ───────────────────────────────
        candidates = [ (a,b) for a,b in candidates
                    if self.is_valid_traj(a) and self.is_valid_traj(b) ]
        if not candidates:
            return []

        # ── 4. optional segment clones ────────────────────────────────────
        if limit_to_segment is not None:
            candidates = [
                ( self._make_segment_clone(a, limit_to_segment, random_start),
                self._make_segment_clone(b, limit_to_segment, random_start) )
                for a, b in candidates
            ]

        # ── 5. BALD score (variance of preference logits) ─────────────────
        scores = []
        for a, b in candidates:
            sA  = ensemble.score_trajectory(a.generate_tensor_from_trajectory(feat_stats = self.feat_stats))
            sB  = ensemble.score_trajectory(b.generate_tensor_from_trajectory(feat_stats = self.feat_stats))
            scores.append(torch.var(sA - sB, unbiased=False).item())

        # ── 6. rank + take top-k % ─────────────────────────────────────────
        k          = max(1, int(len(candidates) * top_k_percent))
        top_idx    = np.argsort(scores)[::-1][:k]
        best_pairs = [candidates[i] for i in top_idx]

        # ── 7. overwrite BALD list and return ─────────────────────────────
        self.BALD_pairs = [ (a, b, None) for a, b in best_pairs ]
        return best_pairs

    # -------------------------------------------------------------------------
    # 57.  Convert “too-long” comparisons into segment pairs ready for the GUI
    # -------------------------------------------------------------------------
    def segmentify_long_comparisons(
        self,
        *,
        segment_length: int = 75,
        segments_per_pair: int = 1,
        random_start: bool = True,
        empty_current_pairs: bool = True
    ):
        """
        Convert overly long pairwise comparisons into random segment-pairs,
        *except* when a comparison involves a noise trajectory or a trajectory
        whose video file no longer exists.

        Parameters
        ----------
        segment_length     : int   - target length for each segment clone
        segments_per_pair  : int   - # random segments to draw *per* trajectory
        random_start       : bool  - passed straight to `_make_segment_clone`
        empty_current_pairs: bool  - wipe `self.BALD_pairs` before refilling

        Side-effects
        ------------
        • `self.BALD_pairs` is overwritten with the new unlabeled segment pairs.
        • Original comparisons are moved to `self.archived_comparisons`.
        • Trajectories whose `video_path` is invalid are collected in
        `self.videoless_trajectories`.
        """

        # --------------------------------------------------------------
        # (0) housekeeping containers
        # --------------------------------------------------------------
        if empty_current_pairs:
            self.BALD_pairs = []

        if not hasattr(self, "archived_comparisons"):
            self.archived_comparisons = []

        if not hasattr(self, "videoless_trajectories"):
            self.videoless_trajectories = []

        noise_set = set(getattr(self, "noise_trajectories", []))

        new_pairs  = []
        to_remove  = []

        # --------------------------------------------------------------
        # (1) iterate over a *copy* so we can modify the original list
        # --------------------------------------------------------------
        for trajA, trajB, lbl in list(self.pairwise_comparisons):

            # ---------- rule-out ❶: noise trajectories -----------------
            noiseA = (trajA in noise_set) or (trajA.video_path is None)
            noiseB = (trajB in noise_set) or (trajB.video_path is None)
            if noiseA or noiseB:
                to_remove.append((trajA, trajB, lbl))
                self.archived_comparisons.append((trajA, trajB, lbl))
                continue

            # ---------- rule-out ❷: missing video files ----------------
            badA = trajA.video_path and not os.path.isfile(trajA.video_path)
            badB = trajB.video_path and not os.path.isfile(trajB.video_path)
            if badA or badB:
                to_remove.append((trajA, trajB, lbl))
                self.archived_comparisons.append((trajA, trajB, lbl))
                if badA:
                    self.videoless_trajectories.append(trajA)
                if badB:
                    self.videoless_trajectories.append(trajB)
                continue

            # ---------- segmentation logic (unchanged) -----------------
            longA = len(trajA.observations) > segment_length
            longB = len(trajB.observations) > segment_length
            if not (longA or longB):
                # keep short-enough comparisons in the DB; skip segmentation
                continue
            
            if len(trajA.observations)>200:
                segments_per_pair_A = segments_per_pair
            else:
                segments_per_pair_A = 1
            
            if len(trajB.observations)>200:
                segments_per_pair_B = segments_per_pair
            else:
                segments_per_pair_B = 1


            segsA = [
                self._make_segment_clone(trajA, segment_length, random_start)
                if longA else trajA
                for _ in range(segments_per_pair_A)
            ]
            segsB = [
                self._make_segment_clone(trajB, segment_length, random_start)
                if longB else trajB
                for _ in range(segments_per_pair_B)
            ]

            for sA in segsA:
                for sB in segsB:
                    new_pairs.append((sA, sB, None))

            to_remove.append((trajA, trajB, lbl))
            self.archived_comparisons.append((trajA, trajB, lbl))

        # --------------------------------------------------------------
        # (2) excise everything we archived
        # --------------------------------------------------------------
        for item in to_remove:
            self.pairwise_comparisons.remove(item)

        # --------------------------------------------------------------
        # (3) activate the freshly built BALD list
        # --------------------------------------------------------------
        self.BALD_pairs = new_pairs
        return new_pairs

    def sample_pairs_ELO(
        self,
        traj_list=None,
        include_expert=True,
        include_non_expert=True,
        limit_to_segment = None,
        random_start = True,
        elo_K=32,
        init_rating=1000):
            """
            1. Build the *complete* set of pairwise indices
            among `traj_list` (and optionally expert / non-expert DB members).

            2. Store them in `self.ELO_pairs` **and** `self.BALD_pairs`
            so downstream code keeps working.

            3. Initialise / update an `elo_rating` field for every trajectory.
            """
            # ----- build candidate pool --------------------------------
            pool = list(traj_list)
            if include_expert:
                pool += list(self.expert_demos)        # existing attributes
            if include_non_expert:
                pool += list(self.non_expert_demos)

            # give every trajectory a unique ID and an Elo rating
            for t in pool:
                if not hasattr(t, "elo_rating"):
                    t.elo_rating = init_rating

            # 2) optional segment clones -----------------------------------------
            if limit_to_segment is not None:
                pool = [
                    self._make_segment_clone(t, limit_to_segment, random_start)
                    for t in pool
                ]

            # ----- all-vs-all pair indices -----------------------------
            pairs = [
                (i, j) for i in range(len(pool))
                    for j in range(i + 1, len(pool))
            ]

            

            # reuse the familiar attribute so gui code stays unchanged
            self.BALD_pairs = [(pool[i], pool[j], None) for i, j in pairs]
            self.ELO_pairs  = self.BALD_pairs          # alias

            # optional: return the list if the caller wants it
            return self.BALD_pairs
    # ------------------------------------------------------------------
    #  NEW: user-label entry-point
    # ------------------------------------------------------------------
    def record_user_choice(self, traj_a, traj_b, choice):
        """
        choice: 'a', 'b', or 'neither'
        """
        if choice == 'neither':
            # user rejected both → stop here
            return

        # store the comparison exactly as before
        label = 0 if choice == 'a' else 1
        self.pairwise_comparisons.append((traj_a, traj_b, label))

        # make sure both trajectories carry an Elo field
        for t in (traj_a, traj_b):
            if not hasattr(t, 'elo_rating'):
                t.elo_rating = self.default_elo

        # update Elo ratings in-place
        winner, loser = (traj_a, traj_b) if choice == 'a' else (traj_b, traj_a)
        self._elo_update(winner, loser)

    # ------------------------------------------------------------------
    #  NEW: static Elo helper
    # ------------------------------------------------------------------
    @staticmethod
    def _elo_update(winner, loser, K=32):
        Ra, Rb = winner.elo_rating, loser.elo_rating
        Ea     = 1.0 / (1.0 + 10 ** ((Rb - Ra) / 400))
        Eb     = 1.0 - Ea
        winner.elo_rating += K * (1 - Ea)   # S=1 for winner
        loser.elo_rating  += K * (0 - Eb)   # S=0 for loser
