"""Config utilities: JSON + argparse merge with simple dot-key overrides.

Example:
    cfg = Config.from_json('slide_block_to_target_initial_demos.json')
    # or:
    cfg = Config.from_args(argv=['--config','file.json','--override','seed=0','--override','task.name=slide'])
"""
from __future__ import annotations
import json, argparse, pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

def _set_deep(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split('.')
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value

@dataclass
class Config:
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "Config":
        with open(path, 'r', encoding='utf-8') as f:
            return cls(json.load(f))

    @classmethod
    def from_args(cls, argv: List[str] | None = None) -> "Config":
        p = argparse.ArgumentParser(description='CrossMod run config')
        p.add_argument('--config', type=str, help='Path to base JSON config')
        p.add_argument('--override', type=str, action='append', default=[],
                       help='Repeatable key=value overrides (dot.keys allowed)')
        ns = p.parse_args(argv)
        base: Dict[str, Any] = {}
        if ns.config:
            base = cls.from_json(ns.config).data
        for kv in ns.override:
            if '=' not in kv:
                continue
            k, v = kv.split('=', 1)
            try:
                vv = json.loads(v)
            except Exception:
                vv = v
            _set_deep(base, k, vv)
        return cls(base)

    def get(self, key: str, default: Any=None) -> Any:
        return self.data.get(key, default)
