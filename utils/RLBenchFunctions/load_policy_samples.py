from pathlib import Path
import pickle
from typing import List, Union

def load_recent_policy_samples(root: Union[str, Path],
                               n_most_recent: int = 10):
    """
    Collect policy samples from the *n_most_recent* sub-folders.

    Parameters
    ----------
    root : str | Path
        Directory whose immediate sub-folders will be scanned.
    n_most_recent : int, default=10
        Number of newest folders (by creation/ctime) to look at.

    Returns
    -------
    List
        Concatenated list of policy samples.  Folders lacking
        ``policy_samples.pkl`` (or unreadable files) are skipped.
    """
    root = Path(root).expanduser()

    # 1–3) grab immediate dirs and sort by creation time (newest first)
    dirs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_ctime,      # st_ctime ≈ “creation” on Windows, inode-change on *nix
        reverse=True
    )

    combined: List = []
    for folder in dirs[:n_most_recent]:       # 4–5) iterate over N newest
        pkl = folder / "policy_samples.pkl"   # 6) expected file location
        if not pkl.is_file():                 # 8) skip if missing
            continue
        try:
            with pkl.open("rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):        # 7) extend the master list
                combined.extend(data)
        except Exception as err:              # 8–9) skip unreadable / corrupted files
            print(f"[warn] {pkl}: {err!s}")

    return combined