from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "HuggingFacePull"
DEFAULT_ENDPOINT = "https://huggingface.co"


def default_library_dir() -> Path:
    configured = os.environ.get("HUGGINGFACE_PULL_LIBRARY")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "huggingfacepull" / "library"


def default_log_file() -> Path:
    configured = os.environ.get("HUGGINGFACE_PULL_LOG_FILE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Library" / "Logs" / APP_NAME / "app.log"


def safe_repo_dir_name(repo_id: str) -> str:
    return repo_id.strip().replace("/", "--")
