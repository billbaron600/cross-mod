"""Wrapper around the legacy `generate_molmo_points.py`.
Preserves current behavior while we migrate into a package.
"""
from __future__ import annotations
try:
    from generate_molmo_points import *  # type: ignore
except Exception as e:
    raise ImportError("Legacy generate_molmo_points.py not found; port in progress.") from e
