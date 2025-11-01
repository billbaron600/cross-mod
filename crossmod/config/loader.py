import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from .schema import DEFAULTS


def _parse_overrides(items: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in items:
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        try:
            out[k] = json.loads(v)
        except Exception:
            out[k] = v
    return out


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str], overrides: Iterable[str] = ()):
    cfg: Dict[str, Any] = dict(DEFAULTS)
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Config not found: {p}')
        if p.suffix.lower() in {'.yml', '.yaml'}:
            if yaml is None:
                raise RuntimeError('PyYAML not installed but YAML config provided')
            loaded = yaml.safe_load(p.read_text()) or {}
        else:
            loaded = json.loads(p.read_text())
        cfg = deep_merge(cfg, loaded)

    ov = _parse_overrides(overrides)
    cfg = deep_merge(cfg, ov)
    return cfg
