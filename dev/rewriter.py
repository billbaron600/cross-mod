#!/usr/bin/env python3

# Simple import codemod to migrate to the new crossmod layout.
# Dry run:  python dev/rewriter.py --dry-run
# Apply:    python dev/rewriter.py

import argparse, os, re
from pathlib import Path

MAPS = [
    # module-wide replacement
    (re.compile(r'\bfrom\s+rendering_helpers\s+import\s+'), 'from crossmod.perception.rendering_helpers import '),
    (re.compile(r'\bimport\s+rendering_helpers\b'), 'import crossmod.perception.rendering_helpers as rendering_helpers'),

    # monolith → package facade
    (re.compile(r'\bfrom\s+utils\.RLBenchFunctions\.generate_IVK_trajectory\s+import\s+'), 'from crossmod.planning import '),
    (re.compile(r'\bimport\s+utils\.RLBenchFunctions\.generate_IVK_trajectory\b'), 'import crossmod.planning as generate_IVK_trajectory'),
]

EXCLUDE_DIRS = {'.git', '.venv', 'venv', 'run_results', 'tensorboardlogs', '__pycache__'}

def should_skip_dir(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & EXCLUDE_DIRS)

def rewrite_text(text: str) -> str:
    out = text
    for pattern, repl in MAPS:
        out = pattern.sub(repl, out)
    return out

def process_file(p: Path, dry: bool) -> bool:
    try:
        src = p.read_text(encoding='utf-8')
    except Exception:
        return False
    out = rewrite_text(src)
    if out != src and not dry:
        p.write_text(out, encoding='utf-8')
    return out != src

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    root = Path(args.root)
    changed = 0
    for dp, _, files in os.walk(root):
        dpp = Path(dp)
        if should_skip_dir(dpp):
            continue
        for fn in files:
            if fn.endswith('.py'):
                if process_file(dpp / fn, args.dry_run):
                    changed += 1
    print(f'Updated {changed} files' + (' (dry run)' if args.dry_run else ''))

if __name__ == '__main__':
    main()
