"""Compatibility shim for `rendering_helpers` during the refactor.
Once the legacy file is moved here, replace this shim with the actual code.
"""
from __future__ import annotations
try:
    from rendering_helpers import *  # type: ignore
except Exception as e:
    raise ImportError("Legacy rendering_helpers.py not found on PYTHONPATH; port in progress.") from e
