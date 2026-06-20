# HuggingFacePull FastAPI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `HuggingFacePull`, a local FastAPI tool for searching, queueing, downloading, resuming, listing, and cleaning Hugging Face Hub model snapshots, with a capability shape similar to the local `OllamaPull` project.

**Architecture:** Keep the proven `OllamaPull` boundaries: a small core download layer, a single-item worker queue, a local web API, and a static browser UI. Use `huggingface_hub` for Hub API calls and snapshot downloads rather than reimplementing Hub auth, redirects, Xet, LFS, or cache semantics.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, Pydantic, `huggingface_hub`, `pytest`, `httpx`, static HTML/CSS/JS.

---

## Reference Behaviour From `OllamaPull`

Mirror these capabilities:

- Local-only server bound to `127.0.0.1`.
- Queue one download at a time.
- Search remote models, while still allowing direct repo IDs.
- Show state, queue rows, messages, current file, overall progress, speed, ETA, and installed models.
- Start, pause after current item, and stop after current file.
- Retry failed items and remove non-running items.
- Reuse existing verified downloads and resume interrupted work.
- Delete installed model records without blindly deleting shared cached files.
- Scan and delete stale or orphaned local files conservatively.
- Provide CLI entry points for direct pull, cleanup, and web UI launch.

Important difference: Hugging Face snapshots are not Ollama manifests. Treat each installed model as a repo snapshot rooted under a configurable library directory, with metadata written by this tool.

## Planned File Structure

- `pyproject.toml`: package metadata, dependencies, entry points, pytest config.
- `README.md`: user-facing setup, CLI, web UI, safety notes.
- `.gitignore`: Python, virtualenv, build, cache, local model library.
- `src/huggingface_pull/__init__.py`: public exports.
- `src/huggingface_pull/__main__.py`: `python3 -m huggingface_pull` entry.
- `src/huggingface_pull/app_logging.py`: local log writer.
- `src/huggingface_pull/config.py`: default paths and environment variables.
- `src/huggingface_pull/hub.py`: Hugging Face search, metadata, pull, installed list, cleanup.
- `src/huggingface_pull/queue.py`: thread-safe serial download queue.
- `src/huggingface_pull/api.py`: FastAPI app factory and API routes.
- `src/huggingface_pull/cli.py`: direct pull, cleanup, and web launch CLIs.
- `src/huggingface_pull/web/index.html`: local UI shell.
- `src/huggingface_pull/web/app.js`: browser API client and rendering.
- `src/huggingface_pull/web/styles.css`: restrained local-tool UI.
- `tests/test_hub.py`: parsing, metadata, cleanup, dry-run behaviour.
- `tests/test_queue.py`: serial execution, dedupe, pause, retry, stop-after-file.
- `tests/test_api.py`: FastAPI route contracts and static file safety.
- `tests/test_cli.py`: argument parsing and CLI behaviour.

## API Shape

Keep API names close to `OllamaPull`, but use Hugging Face language:

- `GET /api/state`
- `GET /api/search?q=...`
- `GET /api/models/{repo_id:path}/files?revision=main`
- `POST /api/queue`
- `POST /api/start`
- `POST /api/pause`
- `POST /api/stop-after-file`
- `POST /api/retry/{item_id}`
- `POST /api/remove/{item_id}`
- `POST /api/installed/remove`
- `POST /api/cleanup/scan`
- `POST /api/cleanup/delete`

Queue item body:

```json
{
  "repo_id": "Qwen/Qwen3-Embedding-0.6B",
  "revision": "main",
  "repo_type": "model",
  "allow_patterns": ["*.json", "*.safetensors", "*.model"],
  "ignore_patterns": ["*.msgpack", "*.h5"],
  "local_dir": null
}
```

## Task 1: Scaffold Python Project

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/huggingface_pull/__init__.py`
- Create: `src/huggingface_pull/__main__.py`
- Create: `src/huggingface_pull/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Create packaging metadata**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "huggingfacepull"
version = "0.1.0"
description = "Local FastAPI tool for searching and downloading Hugging Face Hub model snapshots."
readme = "README.md"
requires-python = ">=3.10"
authors = [{ name = "cpjjh" }]
license = { text = "MIT" }
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "huggingface_hub>=0.32",
  "pydantic>=2.7",
]

[project.optional-dependencies]
dev = [
  "httpx>=0.27",
  "pytest>=8.0",
]

[project.scripts]
hfpull = "huggingface_pull.cli:main"
hfpull-web = "huggingface_pull.cli:run_web"

[tool.setuptools]
package-dir = { "" = "src" }

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
huggingface_pull = ["web/*"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create initial CLI test**

```python
from huggingface_pull.cli import build_parser


def test_parser_accepts_repo_revision_and_filters():
    args = build_parser().parse_args(
        [
            "Qwen/Qwen3-Embedding-0.6B",
            "--revision",
            "main",
            "--allow",
            "*.safetensors",
            "--ignore",
            "*.bin",
            "--repo-type",
            "model",
        ]
    )

    assert args.repo_id == "Qwen/Qwen3-Embedding-0.6B"
    assert args.revision == "main"
    assert args.allow == ["*.safetensors"]
    assert args.ignore == ["*.bin"]
    assert args.repo_type == "model"
```

- [ ] **Step 3: Implement minimal CLI parser**

```python
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face Hub model snapshots into a local library."
    )
    parser.add_argument("repo_id", help="Hub repo ID, for example Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--allow", action="append", default=[], help="Glob pattern to include.")
    parser.add_argument("--ignore", action="append", default=[], help="Glob pattern to exclude.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0


def run_web(argv: list[str] | None = None) -> int:
    return main(argv)
```

- [ ] **Step 4: Add module entry point**

```python
from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run scaffold tests**

Run: `python3 -m pytest tests/test_cli.py -v`

Expected: one passing parser test.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md src tests
git commit -m "chore: scaffold HuggingFacePull package"
```

## Task 2: Add Config, Logging, and Data Models

**Files:**
- Create: `src/huggingface_pull/config.py`
- Create: `src/huggingface_pull/app_logging.py`
- Create: `src/huggingface_pull/models.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write config tests**

```python
from pathlib import Path

from huggingface_pull.config import default_library_dir, safe_repo_dir_name


def test_default_library_dir_uses_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_PULL_LIBRARY", str(tmp_path / "models"))

    assert default_library_dir() == tmp_path / "models"


def test_safe_repo_dir_name_preserves_repo_identity_without_slashes():
    assert safe_repo_dir_name("Qwen/Qwen3-Embedding-0.6B") == "Qwen--Qwen3-Embedding-0.6B"
```

- [ ] **Step 2: Implement config helpers**

```python
from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "HuggingFacePull"
DEFAULT_ENDPOINT = "https://huggingface.co"


def default_library_dir() -> Path:
    configured = os.environ.get("HUGGINGFACE_PULL_LIBRARY")
    if configured:
        return Path(configured).expanduser()
    return (Path.home() / ".cache" / "huggingfacepull" / "library").expanduser()


def default_log_file() -> Path:
    configured = os.environ.get("HUGGINGFACE_PULL_LOG_FILE")
    if configured:
        return Path(configured).expanduser()
    return (Path.home() / "Library" / "Logs" / APP_NAME / "app.log").expanduser()


def safe_repo_dir_name(repo_id: str) -> str:
    return repo_id.strip().replace("/", "--")
```

- [ ] **Step 3: Implement JSON-lines logging**

```python
from __future__ import annotations

import json
import time
from typing import Any

from .config import default_log_file


def write_log(message: str, **fields: Any) -> None:
    path = default_log_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.time(), "message": message, **{k: str(v) for k, v in fields.items()}}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
```

- [ ] **Step 4: Add Pydantic models**

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RepoType = Literal["model", "dataset", "space"]


class QueueRequest(BaseModel):
    repo_id: str = Field(min_length=1)
    revision: str = "main"
    repo_type: RepoType = "model"
    allow_patterns: list[str] = Field(default_factory=list)
    ignore_patterns: list[str] = Field(default_factory=list)
    local_dir: str | None = None


class InstalledRemoveRequest(BaseModel):
    repo_id: str = Field(min_length=1)
    revision: str = "main"
    repo_type: RepoType = "model"


class CleanupRequest(BaseModel):
    include_partials: bool = False
    older_than_days: int = Field(default=7, ge=0)
```

- [ ] **Step 5: Run config tests**

Run: `python3 -m pytest tests/test_config.py -v`

Expected: config tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/huggingface_pull/config.py src/huggingface_pull/app_logging.py src/huggingface_pull/models.py tests/test_config.py
git commit -m "chore: add config logging and request models"
```

## Task 3: Implement Hugging Face Hub Core

**Files:**
- Create: `src/huggingface_pull/hub.py`
- Modify: `src/huggingface_pull/__init__.py`
- Test: `tests/test_hub.py`

- [ ] **Step 1: Write core tests using mocks**

```python
from pathlib import Path
from types import SimpleNamespace

import huggingface_pull.hub as hub


def test_canonical_ref_normalises_revision_repo_type_and_filters():
    ref = hub.HubRef(
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        revision="main",
        repo_type="model",
        allow_patterns=["*.safetensors"],
        ignore_patterns=["*.bin"],
    )

    assert hub.canonical_ref(ref) == "model:Qwen/Qwen3-Embedding-0.6B@main?allow=*.safetensors&ignore=*.bin"


def test_installed_models_reads_metadata(tmp_path):
    metadata = tmp_path / "Qwen--Qwen3-Embedding-0.6B" / "main" / ".huggingfacepull.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        '{"repo_id":"Qwen/Qwen3-Embedding-0.6B","revision":"main","repo_type":"model","snapshot_path":"/tmp/snap","size":12}\n'
    )

    assert hub.installed_models(tmp_path) == [
        {
            "repo_id": "Qwen/Qwen3-Embedding-0.6B",
            "revision": "main",
            "repo_type": "model",
            "snapshot_path": "/tmp/snap",
            "size": 12,
        }
    ]


def test_search_models_maps_hf_model_info(monkeypatch):
    class FakeApi:
        def list_models(self, search, limit, sort, direction, token):
            assert search == "qwen"
            assert limit == 20
            assert token is None
            return [
                SimpleNamespace(
                    modelId="Qwen/Qwen3",
                    pipeline_tag="text-generation",
                    tags=["transformers", "safetensors"],
                    downloads=123,
                    likes=45,
                )
            ]

    monkeypatch.setattr(hub, "HfApi", lambda endpoint=None: FakeApi())

    result = hub.search_models("qwen")

    assert result["available"] is True
    assert result["results"][0]["repo_id"] == "Qwen/Qwen3"
    assert result["results"][0]["tags"] == ["transformers", "safetensors"]
```

- [ ] **Step 2: Implement HubRef, search, installed metadata, and cleanup skeleton**

```python
from __future__ import annotations

import dataclasses
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import HfApi, snapshot_download

from .config import DEFAULT_ENDPOINT, safe_repo_dir_name


ProgressCallback = Callable[[dict[str, Any]], None]
StopAfterFileCallback = Callable[[], bool]


class DownloadStoppedAfterFile(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class HubRef:
    repo_id: str
    revision: str = "main"
    repo_type: str = "model"
    allow_patterns: tuple[str, ...] | list[str] = dataclasses.field(default_factory=tuple)
    ignore_patterns: tuple[str, ...] | list[str] = dataclasses.field(default_factory=tuple)


def canonical_ref(ref: HubRef) -> str:
    allow = ",".join(sorted(ref.allow_patterns))
    ignore = ",".join(sorted(ref.ignore_patterns))
    return f"{ref.repo_type}:{ref.repo_id}@{ref.revision}?allow={allow}&ignore={ignore}"


def metadata_path(library_dir: Path, ref: HubRef) -> Path:
    return Path(library_dir) / safe_repo_dir_name(ref.repo_id) / ref.revision / ".huggingfacepull.json"


def search_models(query: str, *, endpoint: str = DEFAULT_ENDPOINT, token: str | None = None) -> dict[str, Any]:
    if not query.strip():
        return {"available": True, "results": [], "error": None}
    try:
        api = HfApi(endpoint=endpoint)
        models = api.list_models(search=query, limit=20, sort="downloads", direction=-1, token=token)
        results = [
            {
                "repo_id": model.modelId,
                "name": model.modelId,
                "pipeline_tag": getattr(model, "pipeline_tag", None),
                "tags": list(getattr(model, "tags", []) or []),
                "downloads": getattr(model, "downloads", None),
                "likes": getattr(model, "likes", None),
            }
            for model in models
        ]
        return {"available": True, "results": results, "error": None}
    except Exception as error:
        return {"available": False, "results": [], "error": str(error)}


def repo_files(ref: HubRef, *, endpoint: str = DEFAULT_ENDPOINT, token: str | None = None) -> dict[str, Any]:
    api = HfApi(endpoint=endpoint)
    info = api.model_info(
        ref.repo_id,
        revision=ref.revision,
        files_metadata=True,
        token=token,
    )
    files = [
        {
            "path": sibling.rfilename,
            "size": sibling.size,
            "blob_id": sibling.blob_id,
        }
        for sibling in info.siblings
    ]
    return {"repo_id": ref.repo_id, "revision": ref.revision, "files": files}


def installed_models(library_dir: Path) -> list[dict[str, Any]]:
    installed: list[dict[str, Any]] = []
    for marker in Path(library_dir).glob("*/*/.huggingfacepull.json"):
        try:
            installed.append(json.loads(marker.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(installed, key=lambda item: (item["repo_id"].lower(), item["revision"]))


def remove_installed_model(library_dir: Path, ref: HubRef) -> None:
    root = metadata_path(library_dir, ref).parent
    if not root.exists():
        raise KeyError(ref.repo_id)
    shutil.rmtree(root)


def cleanup_library(library_dir: Path, *, delete: bool = False, include_partials: bool = False, older_than_days: int = 7) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    cutoff = time.time() - older_than_days * 24 * 60 * 60
    root = Path(library_dir)
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if ".incomplete" in path.name or ".tmp" in path.name:
                stat = path.stat()
                if include_partials and stat.st_mtime <= cutoff:
                    candidates.append({"path": str(path), "name": path.name, "size": stat.st_size, "modified_at": stat.st_mtime})
    deleted: list[str] = []
    if delete:
        for item in candidates:
            path = Path(item["path"])
            try:
                path.unlink()
                deleted.append(str(path))
            except FileNotFoundError:
                pass
    return {"dry_run": not delete, "stale_partial_count": len(candidates), "stale_partials": candidates, "deleted": deleted}
```

- [ ] **Step 3: Add actual pull function**

Append this to `hub.py`:

```python
def pull_snapshot(
    ref: HubRef,
    *,
    library_dir: Path,
    endpoint: str = DEFAULT_ENDPOINT,
    token: str | None = None,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
    stop_after_file: StopAfterFileCallback | None = None,
) -> Path:
    target = metadata_path(library_dir, ref).parent
    if progress:
        progress({"type": "manifest-fetch", "repo_id": ref.repo_id, "revision": ref.revision})
    if dry_run:
        if progress:
            progress({"type": "model-complete", "repo_id": ref.repo_id, "dry_run": True})
        return target
    snapshot_path = snapshot_download(
        repo_id=ref.repo_id,
        revision=ref.revision,
        repo_type=None if ref.repo_type == "model" else ref.repo_type,
        local_dir=target,
        allow_patterns=list(ref.allow_patterns) or None,
        ignore_patterns=list(ref.ignore_patterns) or None,
        endpoint=endpoint,
        token=token,
    )
    marker = metadata_path(library_dir, ref)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "repo_id": ref.repo_id,
                "revision": ref.revision,
                "repo_type": ref.repo_type,
                "snapshot_path": str(snapshot_path),
                "size": directory_size(marker.parent),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if progress:
        progress({"type": "model-complete", "repo_id": ref.repo_id, "snapshot_path": str(snapshot_path)})
    if stop_after_file is not None and stop_after_file():
        raise DownloadStoppedAfterFile
    return Path(snapshot_path)


def directory_size(path: Path) -> int:
    return sum(candidate.stat().st_size for candidate in Path(path).rglob("*") if candidate.is_file())
```

- [ ] **Step 4: Run Hub tests**

Run: `python3 -m pytest tests/test_hub.py -v`

Expected: Hub tests pass without network access.

- [ ] **Step 5: Commit**

```bash
git add src/huggingface_pull/hub.py src/huggingface_pull/__init__.py tests/test_hub.py
git commit -m "feat: add hugging face hub core"
```

## Task 4: Port the Serial Queue

**Files:**
- Create: `src/huggingface_pull/queue.py`
- Test: `tests/test_queue.py`

- [ ] **Step 1: Write queue tests adapted from `OllamaPull`**

```python
import tempfile
import threading
import time
from pathlib import Path

from huggingface_pull.hub import DownloadStoppedAfterFile
from huggingface_pull.queue import DownloadQueue


def test_add_creates_waiting_item():
    with tempfile.TemporaryDirectory() as tmp:
        queue = DownloadQueue(library_dir=Path(tmp), pull_func=lambda *args, **kwargs: None)

        item = queue.add({"repo_id": "Qwen/Qwen3", "revision": "main", "repo_type": "model"})

        assert item["repo_id"] == "Qwen/Qwen3"
        assert item["status"] == "waiting"
        assert item["progress"]["phase"] == "waiting"


def test_worker_runs_one_item_at_a_time():
    calls = []

    def fake_pull(ref, **kwargs):
        calls.append(ref.repo_id)
        kwargs["progress"]({"type": "model-complete", "repo_id": ref.repo_id})

    with tempfile.TemporaryDirectory() as tmp:
        queue = DownloadQueue(library_dir=Path(tmp), pull_func=fake_pull)
        queue.add({"repo_id": "first", "revision": "main", "repo_type": "model"})
        queue.add({"repo_id": "second", "revision": "main", "repo_type": "model"})

        queue.start()
        assert queue.wait_until_idle(2)

    assert calls == ["first", "second"]


def test_stop_after_file_returns_running_item_to_waiting():
    first_done = threading.Event()
    release = threading.Event()

    def fake_pull(ref, **kwargs):
        kwargs["progress"]({"type": "file-complete", "path": "weights.safetensors", "downloaded": 1, "total": 1})
        first_done.set()
        release.wait(2)
        if kwargs["stop_after_file"]():
            raise DownloadStoppedAfterFile

    with tempfile.TemporaryDirectory() as tmp:
        queue = DownloadQueue(library_dir=Path(tmp), pull_func=fake_pull)
        item = queue.add({"repo_id": "Qwen/Qwen3", "revision": "main", "repo_type": "model"})

        queue.start()
        assert first_done.wait(2)
        queue.stop_after_current_file()
        release.set()
        assert queue.wait_until_idle(2)
        snapshot = queue.snapshot()

    assert snapshot["items"][0]["id"] == item["id"]
    assert snapshot["items"][0]["status"] == "waiting"
```

- [ ] **Step 2: Implement queue from the `OllamaPull` pattern**

Implement these methods with the same locking discipline as `OllamaPull`:

```python
class DownloadQueue:
    def __init__(self, *, library_dir: Path, endpoint: str = DEFAULT_ENDPOINT, token: str | None = None, pull_func: PullFunc = pull_snapshot) -> None: ...
    def add(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def start(self) -> None: ...
    def pause_after_current(self) -> None: ...
    def stop_after_current_file(self) -> dict[str, Any]: ...
    def retry(self, item_id: str) -> dict[str, Any]: ...
    def remove(self, item_id: str) -> dict[str, Any]: ...
    def snapshot(self) -> dict[str, Any]: ...
    def wait_until_idle(self, timeout: float | None = None) -> bool: ...
```

The item shape must be:

```python
{
    "id": "1",
    "repo_id": "Qwen/Qwen3",
    "revision": "main",
    "repo_type": "model",
    "canonical_ref": "model:Qwen/Qwen3@main?allow=&ignore=",
    "status": "waiting",
    "error": None,
    "messages": [],
    "progress": {"phase": "waiting", "overall": {"downloaded": 0, "total": None, "percent": None}, "current_file": None},
    "created_at": now,
    "updated_at": now,
}
```

- [ ] **Step 3: Run queue tests**

Run: `python3 -m pytest tests/test_queue.py -v`

Expected: queue tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/huggingface_pull/queue.py tests/test_queue.py
git commit -m "feat: add serial download queue"
```

## Task 5: Build FastAPI App

**Files:**
- Create: `src/huggingface_pull/api.py`
- Modify: `src/huggingface_pull/cli.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write API tests with `TestClient`**

```python
from pathlib import Path

from fastapi.testclient import TestClient

from huggingface_pull.api import create_app


def test_state_endpoint_returns_snapshot(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path))

    response = client.get("/api/state")

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["library_dir"] == str(tmp_path)


def test_queue_endpoint_validates_and_queues(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path))

    response = client.post("/api/queue", json={"repo_id": "Qwen/Qwen3"})

    assert response.status_code == 200
    assert response.json()["repo_id"] == "Qwen/Qwen3"
    assert response.json()["status"] == "waiting"


def test_bad_queue_body_returns_422(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path))

    response = client.post("/api/queue", json={"repo_id": ""})

    assert response.status_code == 422
```

- [ ] **Step 2: Implement FastAPI routes**

```python
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_ENDPOINT, default_library_dir
from .hub import HubRef, cleanup_library, remove_installed_model, repo_files, search_models
from .models import CleanupRequest, InstalledRemoveRequest, QueueRequest
from .queue import DownloadQueue


WEB_DIR = Path(__file__).with_name("web")


def create_app(
    *,
    library_dir: Path | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    token: str | None = None,
) -> FastAPI:
    queue = DownloadQueue(library_dir=library_dir or default_library_dir(), endpoint=endpoint, token=token)
    app = FastAPI(title="HuggingFacePull")
    app.state.queue = queue

    @app.get("/api/state")
    def state() -> dict:
        return queue.snapshot()

    @app.get("/api/search")
    def search(q: str = "") -> dict:
        return search_models(q, endpoint=endpoint, token=token)

    @app.get("/api/models/{repo_id:path}/files")
    def files(repo_id: str, revision: str = "main", repo_type: str = "model") -> dict:
        if repo_type != "model":
            raise HTTPException(status_code=400, detail="File listing currently supports model repos only")
        return repo_files(HubRef(repo_id=repo_id, revision=revision, repo_type=repo_type), endpoint=endpoint, token=token)

    @app.post("/api/queue")
    def add(payload: QueueRequest) -> dict:
        return queue.add(payload.model_dump())

    @app.post("/api/start")
    def start() -> dict:
        queue.start()
        return queue.snapshot()

    @app.post("/api/pause")
    def pause() -> dict:
        queue.pause_after_current()
        return queue.snapshot()

    @app.post("/api/stop-after-file")
    def stop_after_file() -> dict:
        return queue.stop_after_current_file()

    @app.post("/api/retry/{item_id}")
    def retry(item_id: str) -> dict:
        try:
            return queue.retry(item_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/remove/{item_id}")
    def remove(item_id: str) -> dict:
        try:
            queue.remove(item_id)
            return {"ok": True}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/installed/remove")
    def remove_installed(payload: InstalledRemoveRequest) -> dict:
        try:
            remove_installed_model(queue.library_dir, HubRef(**payload.model_dump()))
            return {"ok": True}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/cleanup/scan")
    def cleanup_scan(payload: CleanupRequest = CleanupRequest()) -> dict:
        return cleanup_library(queue.library_dir, delete=False, **payload.model_dump())

    @app.post("/api/cleanup/delete")
    def cleanup_delete(payload: CleanupRequest = CleanupRequest()) -> dict:
        return cleanup_library(queue.library_dir, delete=True, **payload.model_dump())

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
```

- [ ] **Step 3: Wire `hfpull-web` to Uvicorn**

```python
def run_web(argv: list[str] | None = None) -> int:
    import argparse
    import webbrowser
    import uvicorn

    from .api import create_app
    from .config import default_library_dir

    parser = argparse.ArgumentParser(description="Run the HuggingFacePull local web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--library-dir", type=Path, default=default_library_dir())
    args = parser.parse_args(argv)

    app = create_app(library_dir=args.library_dir.expanduser())
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    webbrowser.open(f"http://{args.host}:{args.port}/")
    server.run()
    return 0
```

- [ ] **Step 4: Run API tests**

Run: `python3 -m pytest tests/test_api.py -v`

Expected: API route tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/huggingface_pull/api.py src/huggingface_pull/cli.py tests/test_api.py
git commit -m "feat: add fastapi local server"
```

## Task 6: Add Browser UI

**Files:**
- Create: `src/huggingface_pull/web/index.html`
- Create: `src/huggingface_pull/web/app.js`
- Create: `src/huggingface_pull/web/styles.css`
- Test: `tests/test_api.py`

- [ ] **Step 1: Add static smoke test**

```python
def test_static_index_served(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path))

    response = client.get("/")

    assert response.status_code == 200
    assert "HuggingFacePull" in response.text
```

- [ ] **Step 2: Build UI by adapting `OllamaPull`**

Use the existing `OllamaPull` web concepts:

- Search form and direct repo ID add.
- Installed snapshots list.
- Queue controls: start, pause, stop after file.
- Queue rows with status badges and progress.
- Detail panel with messages and current file.
- Cleanup scan/delete controls.

Text substitutions:

- "model" becomes "repo" where the input is a Hugging Face repo ID.
- "blob" becomes "file".
- "models directory" becomes "library directory".
- "source registry" becomes "Hub endpoint".

- [ ] **Step 3: Poll state and call API**

In `app.js`, implement:

```javascript
async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { Accept: "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function refresh() {
  state.snapshot = await api("/api/state");
  render();
}

setInterval(refresh, 1000);
refresh();
```

- [ ] **Step 4: Run static and API tests**

Run: `python3 -m pytest tests/test_api.py -v`

Expected: static index and API tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/huggingface_pull/web tests/test_api.py
git commit -m "feat: add local web interface"
```

## Task 7: Finish CLI Pull and Cleanup

**Files:**
- Modify: `src/huggingface_pull/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add CLI tests for direct pull and cleanup dispatch**

```python
from pathlib import Path

import huggingface_pull.cli as cli


def test_main_calls_pull_snapshot(monkeypatch, tmp_path):
    calls = []

    def fake_pull(ref, **kwargs):
        calls.append((ref.repo_id, kwargs["library_dir"]))
        return tmp_path

    monkeypatch.setattr(cli, "pull_snapshot", fake_pull)

    assert cli.main(["Qwen/Qwen3", "--library-dir", str(tmp_path)]) == 0
    assert calls == [("Qwen/Qwen3", tmp_path)]


def test_gc_calls_cleanup(monkeypatch, tmp_path):
    calls = []

    def fake_cleanup(path, **kwargs):
        calls.append((path, kwargs["delete"]))
        return {"dry_run": True, "stale_partial_count": 0, "deleted": []}

    monkeypatch.setattr(cli, "cleanup_library", fake_cleanup)

    assert cli.main(["gc", "--library-dir", str(tmp_path)]) == 0
    assert calls == [(tmp_path, False)]
```

- [ ] **Step 2: Implement CLI dispatch**

`main()` should:

- Dispatch `gc` first when `argv[:1] == ["gc"]`.
- Build `HubRef` from args.
- Call `pull_snapshot(..., dry_run=args.dry_run)`.
- Print the snapshot path on success.
- Return `1` and print to stderr on exceptions.

- [ ] **Step 3: Run CLI tests**

Run: `python3 -m pytest tests/test_cli.py -v`

Expected: parser, direct pull, and cleanup tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/huggingface_pull/cli.py tests/test_cli.py
git commit -m "feat: complete cli pull and cleanup"
```

## Task 8: End-to-End Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Install editable package with active pyenv Python**

Run: `python3 -m pip install -e ".[dev]"`

Expected: package installs into the active pyenv environment.

- [ ] **Step 2: Run all tests**

Run: `python3 -m pytest -v`

Expected: all tests pass.

- [ ] **Step 3: Compile Python**

Run: `python3 -m py_compile src/huggingface_pull/*.py tests/*.py`

Expected: no syntax errors.

- [ ] **Step 4: Smoke test FastAPI app without network**

Run: `python3 -m uvicorn "huggingface_pull.api:create_app" --factory --host 127.0.0.1 --port 8019`

Expected: server starts and `GET http://127.0.0.1:8019/api/state` returns JSON.

- [ ] **Step 5: Smoke test real dry run**

Run: `hfpull openai-community/gpt2 --allow config.json --dry-run`

Expected: exits `0` without downloading model weights.

- [ ] **Step 6: Update README with exact commands**

README must include:

```markdown
# HuggingFacePull

Local FastAPI tool for searching and downloading Hugging Face Hub model snapshots.

## Setup

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"

## Run the Web UI

hfpull-web

## Pull a Repo

hfpull Qwen/Qwen3-Embedding-0.6B --allow "*.json" --allow "*.safetensors"

## Cleanup

hfpull gc
hfpull gc --delete --include-partials --older-than-days 7
```

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs: add setup and usage guide"
```

## Open Design Decisions

- Folder name: the current directory is `HugginFacePull`; keep the Python package correctly named `huggingface_pull` and consider renaming the folder to `HuggingFacePull` before first commit.
- Progress fidelity: `huggingface_hub.snapshot_download()` handles concurrent file downloads and cache metadata, but does not expose the same simple per-blob callback as the Ollama registry downloader. First version should report phases accurately, then improve per-file progress only if needed.
- Storage model: use a tool-owned library under `~/.cache/huggingfacepull/library` by default, while allowing users to use the standard Hugging Face cache via `HF_HOME` independently.
- Auth: rely on locally saved Hugging Face tokens or `HF_TOKEN`; add an explicit token input only if there is a real workflow need.

## Self-Review

- Spec coverage: the plan covers an empty project scaffold, FastAPI server, queue, search, download, resume-by-Hub-cache, installed listing, conservative cleanup, CLI, web UI, and tests.
- Placeholder scan: the plan avoids "TBD" and gives explicit paths, commands, API routes, and expected test results.
- Type consistency: queue payloads use `repo_id`, `revision`, `repo_type`, `allow_patterns`, and `ignore_patterns` consistently across API, CLI, Hub core, and UI.
