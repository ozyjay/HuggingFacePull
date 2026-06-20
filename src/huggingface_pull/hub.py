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
    """Raised when a caller requests a stop after a completed snapshot pull."""


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


def safe_revision_dir_name(revision: str) -> str:
    stripped = revision.strip()
    if not stripped:
        return "main"
    parts = [part for part in stripped.replace("\\", "/").split("/") if part]
    safe = "--".join(parts)
    if safe in {".", ".."}:
        return safe.replace(".", "_")
    return safe


def metadata_path(library_dir: Path, ref: HubRef) -> Path:
    return (
        Path(library_dir)
        / safe_repo_dir_name(ref.repo_id)
        / safe_revision_dir_name(ref.revision)
        / ".huggingfacepull.json"
    )


def search_models(
    query: str,
    endpoint: str = DEFAULT_ENDPOINT,
    token: str | None = None,
) -> dict[str, Any]:
    if not query.strip():
        return {"available": True, "results": [], "error": None}

    try:
        api = HfApi(endpoint=endpoint)
        models = api.list_models(
            search=query,
            limit=20,
            sort="downloads",
            direction=-1,
            token=token,
        )
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


def repo_files(
    ref: HubRef,
    endpoint: str = DEFAULT_ENDPOINT,
    token: str | None = None,
) -> dict[str, Any]:
    if ref.repo_type != "model":
        raise NotImplementedError("Only model repository files are supported for now.")

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
            metadata = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        if "repo_id" not in metadata or "revision" not in metadata:
            continue
        installed.append(metadata)
    return sorted(
        installed,
        key=lambda item: (str(item["repo_id"]).lower(), str(item["revision"])),
    )


def remove_installed_model(library_dir: Path, ref: HubRef) -> None:
    library_root = Path(library_dir).resolve()
    marker = metadata_path(library_dir, ref)
    if not marker.exists():
        raise KeyError(ref.repo_id)

    root = metadata_path(library_dir, ref).parent
    try:
        root.resolve().relative_to(library_root)
    except ValueError as error:
        raise KeyError(ref.repo_id) from error
    if not root.exists():
        raise KeyError(ref.repo_id)
    shutil.rmtree(root)


def cleanup_library(
    library_dir: Path,
    delete: bool = False,
    include_partials: bool = False,
    older_than_days: int = 7,
) -> dict[str, Any]:
    stale_partials: list[dict[str, Any]] = []
    cutoff = time.time() - older_than_days * 24 * 60 * 60
    root = Path(library_dir)

    if include_partials and root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if ".incomplete" not in path.name and ".tmp" not in path.name:
                continue
            stat = path.stat()
            if stat.st_mtime > cutoff:
                continue
            stale_partials.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "size": stat.st_size,
                    "modified_at": stat.st_mtime,
                }
            )

    deleted: list[str] = []
    if delete:
        for item in stale_partials:
            path = Path(item["path"])
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            deleted.append(str(path))

    return {
        "dry_run": not delete,
        "stale_partial_count": len(stale_partials),
        "stale_partials": stale_partials,
        "deleted": deleted,
    }


def pull_snapshot(
    ref: HubRef,
    library_dir: Path,
    endpoint: str = DEFAULT_ENDPOINT,
    token: str | None = None,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
    stop_after_file: StopAfterFileCallback | None = None,
) -> Path:
    target = metadata_path(library_dir, ref).parent
    if progress is not None:
        progress({"type": "manifest-fetch", "repo_id": ref.repo_id, "revision": ref.revision})

    if dry_run:
        if progress is not None:
            progress({"type": "model-complete", "repo_id": ref.repo_id, "dry_run": True})
        return target

    snapshot_path = Path(
        snapshot_download(
            repo_id=ref.repo_id,
            revision=ref.revision,
            repo_type=None if ref.repo_type == "model" else ref.repo_type,
            local_dir=target,
            allow_patterns=list(ref.allow_patterns) or None,
            ignore_patterns=list(ref.ignore_patterns) or None,
            endpoint=endpoint,
            token=token,
        )
    )
    marker = metadata_path(library_dir, ref)
    marker.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "repo_id": ref.repo_id,
        "revision": ref.revision,
        "repo_type": ref.repo_type,
        "snapshot_path": str(snapshot_path),
        "size": directory_size(marker.parent),
    }
    marker.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")

    if progress is not None:
        progress(
            {
                "type": "model-complete",
                "repo_id": ref.repo_id,
                "snapshot_path": str(snapshot_path),
            }
        )
    if stop_after_file is not None and stop_after_file():
        raise DownloadStoppedAfterFile
    return snapshot_path


def directory_size(path: Path) -> int:
    return sum(candidate.stat().st_size for candidate in Path(path).rglob("*") if candidate.is_file())
