# Re-export execution/correction entry points from the legacy monolith.
try:
    from utils.RLBenchFunctions.generate_IVK_trajectory import (
        execute_trajectory,
        run_ivk_for_corrections,
    )
except Exception as e:
    raise ImportError('Missing utils.RLBenchFunctions.generate_IVK_trajectory') from e
__all__ = ['execute_trajectory','run_ivk_for_corrections']
