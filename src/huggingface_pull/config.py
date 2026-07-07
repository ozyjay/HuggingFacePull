from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "HuggingFacePull"
DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_MAX_WORKERS = 1
DEFAULT_STALL_TIMEOUT_SECONDS = 180.0


def default_library_dir() -> Path:
    configured = os.environ.get("HUGGINGFACE_PULL_LIBRARY")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "huggingfacepull" / "library"


def default_log_file() -> Path:
    configured = os.environ.get("HUGGINGFACE_PULL_LOG_FILE")
    if configured:
        return Path(configured).expanduser()
    if sys.platform.startswith("linux"):
        state_home = os.environ.get("XDG_STATE_HOME")
        base_dir = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
        return base_dir / APP_NAME / "app.log"
    return Path.home() / "Library" / "Logs" / APP_NAME / "app.log"


def env_int(name: str, default: int) -> int:
    configured = os.environ.get(name)
    if configured is None or not configured.strip():
        return default
    try:
        value = int(configured)
    except ValueError:
        return default
    return value if value > 0 else default


def env_float(name: str, default: float) -> float:
    configured = os.environ.get(name)
    if configured is None or not configured.strip():
        return default
    try:
        value = float(configured)
    except ValueError:
        return default
    return value if value > 0 else default


def default_max_workers() -> int:
    return env_int("HUGGINGFACE_PULL_MAX_WORKERS", DEFAULT_MAX_WORKERS)


def default_stall_timeout_seconds() -> float:
    return env_float("HUGGINGFACE_PULL_STALL_TIMEOUT_SECONDS", DEFAULT_STALL_TIMEOUT_SECONDS)


def safe_repo_dir_name(repo_id: str) -> str:
    return repo_id.strip().replace("/", "--")
