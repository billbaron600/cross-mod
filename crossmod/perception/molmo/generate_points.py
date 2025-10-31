"""Re-export Molmo point generation from the legacy implementation.
This makes functionality available under crossmod without duplicating code.
"""
try:
    from generate_molmo_points import *  # type: ignore
except Exception as e:
    raise ImportError('Legacy generate_molmo_points.py not found at repo root.') from e
