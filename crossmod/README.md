# CrossMod package

This package provides a cleaned module boundary for the project:

- `perception/` — rendering and image utilities
- `geometry/` — ray casting, projections, transforms
- `kinematics/` — IK helpers / solvers
- `planning/` — trajectory generation and time parameterization
- `io/` — configuration I/O and merge
- `agents/` — wrappers around external agents (e.g., Molmo)
- `compat/` — import-compat utilities during the refactor

During the transition period, `crossmod.compat.rewriter.enable()` lets
legacy imports keep working while files are moved.
