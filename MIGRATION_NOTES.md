# Migration notes for `refactored-changes`

This introduces a **non-breaking** package skeleton (`crossmod/`) and a
compat layer so legacy imports keep working while we move code in stages.

## New layout (initial)
```
crossmod/
  agents/               # Molmo wrappers (will wrap generate_molmo_points.py)
  compat/               # import shim to keep old module paths working
  geometry/             # ray casting, projections, frames
  io/                   # configs, persistence
  kinematics/           # IK helpers / solvers
  perception/           # rendering + image helpers
  planning/             # trajectories, waypoints, time-param
```

## Backwards compatibility
Add this near the top of entry scripts during the transition:
```python
try:
    from crossmod.compat.rewriter import enable as _enable_crossmod_import_shims
    _enable_crossmod_import_shims()
except Exception:
    pass
```

## Next steps
- Move `rendering_helpers.py` → `crossmod/perception/` (keep a tiny stub at the old path that re-exports from the new module).
- Split `generate_IVK_trajectory.py` into `crossmod/kinematics/inverse_kinematics.py` and `crossmod/planning/trajectory.py`.
- Extract ray-casting utils from `utils/RLBenchFunctions.py` → `crossmod/geometry/ray_casting.py`.
- Replace ad-hoc JSON config usage with `crossmod/io/config.py`.
- Update task scripts to import from `crossmod.*` directly (remove the shim).
