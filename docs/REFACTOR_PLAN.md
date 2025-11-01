CrossMod Refactor Plan (initial scaffold)

This commit introduces a non-breaking package scaffold so we can gradually migrate code without disrupting existing scripts. Nothing is deleted; new modules act as thin wrappers around the current scripts.

New package layout:
- crossmod/cli.py  : unified CLI (argparse) with subcommands
- crossmod/config/schema.py  : typed config models (dataclasses)
- crossmod/config/loader.py  : load/merge JSON or YAML
- crossmod/perception/rendering_helpers.py : wrapper, re-export existing helpers
- crossmod/molmo/generate_points.py : wrapper around generate_molmo_points.py
- crossmod/planning/ik.py : wrapper around generate_IVK_trajectory.py
- crossmod/rlbench/runner.py : wrapper around utils/RLBenchFunctions
- crossmod/utils/io.py : small IO helpers

Migration strategy:
1) Keep task scripts working as-is; begin importing from crossmod.* gradually.
2) Split utils/RLBenchFunctions into focused modules (ray_casting, kinematics, traj, viz).
3) Replace ad-hoc JSON config usage with crossmod.config.loader (JSON/YAML).
4) Expose entry point: python -m crossmod.cli run-task --task close_drawer ...

Notes:
- Never resize images (keep original resolution, e.g., 1200x1200).
- No files were removed; wrappers import the original modules to avoid breakage.
