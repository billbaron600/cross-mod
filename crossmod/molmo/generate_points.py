# Wrapper around your existing Molmo/Malmo scripts.
# Tries common names so the wrapper is robust to typos/renames.

try:
    from generate_molmo_points import *  # type: ignore
except Exception:
    try:
        from generate_malmo_points import *  # type: ignore
    except Exception:
        try:
            # Support a package-style path if it exists
            from molmo.generate_points import *  # type: ignore
        except Exception:  # pragma: no cover
            pass
