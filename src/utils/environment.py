"""Environment loading shared by every application entry point."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.utils.paths import find_project_root


def get_project_env_path() -> Path:
    """Return the configured env file or the project-root ``.env`` file."""
    configured_path = os.getenv("CURRENCY_ASSISTANT_ENV_FILE")
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = find_project_root() / path
        return path.resolve()

    return (find_project_root() / ".env").resolve()


def load_project_environment(env_path: Optional[Path] = None) -> bool:
    """Load project secrets without replacing variables set by the shell.

    Loading from an explicit, project-relative location avoids depending on
    the process working directory or on which configuration module happens to
    initialize first.
    """
    path = Path(env_path).expanduser().resolve() if env_path else get_project_env_path()
    return load_dotenv(dotenv_path=path, override=False)
