"""Import rewriter to maintain backward compatibility during refactor.
Usage: call `enable()` once at program startup.
"""
from __future__ import annotations
import importlib, sys
from types import ModuleType

_MAP = {
    'utils.RLBenchFunctions': 'crossmod.geometry.ray_casting',
    'generate_IVK_trajectory': 'crossmod.kinematics.inverse_kinematics',
    'rendering_helpers': 'crossmod.perception.rendering_helpers',
}

class _Proxy(ModuleType):
    def __init__(self, target: str):
        super().__init__(target)
        self._target = target
        self._mod = None
    def _load(self):
        if self._mod is None:
            self._mod = importlib.import_module(self._target)
        return self._mod
    def __getattr__(self, name):
        return getattr(self._load(), name)

def enable() -> None:
    for legacy, new in _MAP.items():
        sys.modules.setdefault(legacy, _Proxy(new))
