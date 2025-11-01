from pathlib import Path


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text(path: str | Path, content: str) -> None:
    Path(path).write_text(content)
