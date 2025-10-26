"""Utilities for resolving project-relative paths robustly."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


SENTINELS = ("pyproject.toml", ".git", "config.yaml")


def find_project_root(start: Optional[Path] = None) -> Path:
    """Locate the project root by walking up from a start path.

    Recognition: presence of one of SENTINELS.
    Honors CURRENCY_ASSISTANT_ROOT if set.
    """
    env_root = os.getenv("CURRENCY_ASSISTANT_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return p

    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    # Current working directory
    candidates.append(Path.cwd())
    # This file location
    candidates.append(Path(__file__).resolve())

    seen = set()
    for c in candidates:
        for p in [c] + list(c.parents):
            if p in seen:
                continue
            seen.add(p)
            for s in SENTINELS:
                if (p / s).exists():
                    return p
    # Fallback
    return Path.cwd()


def resolve_project_path(path_str: str, root: Optional[Path] = None) -> Path:
    """Resolve a possibly relative path with respect to project root.

    If path_str is absolute, returns as-is. Otherwise joins with root
    (detected if not provided) and returns absolute.
    """
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    base = root or find_project_root()
    return (base / p).resolve()

