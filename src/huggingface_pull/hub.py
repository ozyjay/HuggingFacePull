from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils import tqdm as hf_tqdm

from .app_logging import write_log
from .config import DEFAULT_ENDPOINT, safe_repo_dir_name


ProgressCallback = Callable[[dict[str, Any]], None]
StopAfterFileCallback = Callable[[], bool]
PROGRESS_EMIT_INTERVAL_SECONDS = 0.5
_LOGGED_SKIPPED_CACHE_SNAPSHOTS: set[tuple[str, str, str, str]] = set()


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


def cached_hub_models(cache_dir: Path | str | None = None) -> list[dict[str, Any]]:
    root = Path(cache_dir or HF_HUB_CACHE)
    cached: list[dict[str, Any]] = []
    if not root.exists():
        return cached

    for repo_dir in sorted(root.glob("models--*")):
        if not repo_dir.is_dir():
            continue
        repo_id = _repo_id_from_cache_dir(repo_dir.name, "models")
        if repo_id is None:
            continue
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        refs = _cache_refs(repo_dir)
        if refs:
            for revision, commit in refs.items():
                snapshot = snapshots / commit
                skip_reason = _cache_snapshot_skip_reason(snapshot)
                if skip_reason is None:
                    cached.append(
                        {
                            "repo_id": repo_id,
                            "revision": revision,
                            "repo_type": "model",
                            "snapshot_path": str(snapshot),
                            "source": "huggingface_cache",
                        }
                    )
                else:
                    _log_cache_snapshot_skipped(repo_id, revision, snapshot, skip_reason)
            continue
        for snapshot in sorted(snapshots.iterdir()):
            skip_reason = _cache_snapshot_skip_reason(snapshot)
            if skip_reason is None:
                cached.append(
                    {
                        "repo_id": repo_id,
                        "revision": snapshot.name,
                        "repo_type": "model",
                        "snapshot_path": str(snapshot),
                        "source": "huggingface_cache",
                    }
                )
            else:
                _log_cache_snapshot_skipped(repo_id, snapshot.name, snapshot, skip_reason)
    return cached


def _cache_snapshot_skip_reason(snapshot: Path) -> dict[str, Any] | None:
    if not snapshot.is_dir():
        return {"reason": "missing"}

    found_file = False
    for path in snapshot.rglob("*"):
        if path.is_dir():
            continue
        if _is_partial_file(path):
            return {"reason": "partial_file", "path": path}
        if not path.exists():
            return {
                "reason": "broken_symlink" if path.is_symlink() else "missing_file",
                "path": path,
            }
        if path.is_file():
            found_file = True
    if not found_file:
        return {"reason": "empty"}
    return None


def _log_cache_snapshot_skipped(
    repo_id: str,
    revision: str,
    snapshot: Path,
    skip_reason: dict[str, Any],
) -> None:
    reason = str(skip_reason.get("reason", "unknown"))
    key = (repo_id, revision, str(snapshot), reason)
    if key in _LOGGED_SKIPPED_CACHE_SNAPSHOTS:
        return
    _LOGGED_SKIPPED_CACHE_SNAPSHOTS.add(key)
    _log(
        "cache snapshot skipped",
        repo_id=repo_id,
        revision=revision,
        snapshot_path=snapshot,
        **skip_reason,
    )


def _repo_id_from_cache_dir(name: str, prefix: str) -> str | None:
    prefix_text = f"{prefix}--"
    if not name.startswith(prefix_text):
        return None
    encoded = name[len(prefix_text):]
    if "--" not in encoded:
        return None
    return encoded.replace("--", "/")


def _cache_refs(repo_dir: Path) -> dict[str, str]:
    refs_dir = repo_dir / "refs"
    refs: dict[str, str] = {}
    if not refs_dir.is_dir():
        return refs
    for ref in sorted(refs_dir.iterdir()):
        if not ref.is_file():
            continue
        try:
            commit = ref.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if commit:
            refs[ref.name] = commit
    return refs


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


def _is_partial_file(path: Path) -> bool:
    return ".incomplete" in path.name or ".tmp" in path.name


def _partial_scan_roots(library_dir: Path) -> list[tuple[Path, str]]:
    roots: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for root, source in (
        (Path(library_dir), "library"),
        (Path(HF_HUB_CACHE), "huggingface_cache"),
    ):
        if not root.exists():
            continue
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append((root, source))
    return roots


def _collect_stale_partials(root: Path, source: str, cutoff: float) -> list[dict[str, Any]]:
    stale_partials: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_partial_file(path):
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if stat.st_mtime > cutoff:
            continue
        stale_partials.append(
            {
                "path": str(path),
                "name": path.name,
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
                "source": source,
            }
        )
    return stale_partials


def _log(message: str, /, **fields: Any) -> None:
    try:
        write_log(message, **fields)
    except Exception:
        pass


def cleanup_library(
    library_dir: Path,
    delete: bool = False,
    include_partials: bool = False,
    older_than_days: int = 7,
) -> dict[str, Any]:
    stale_partials: list[dict[str, Any]] = []
    cutoff = time.time() - older_than_days * 24 * 60 * 60

    if include_partials:
        seen_paths: set[Path] = set()
        for root, source in _partial_scan_roots(library_dir):
            for item in _collect_stale_partials(root, source, cutoff):
                resolved_path = Path(item["path"]).resolve()
                if resolved_path in seen_paths:
                    continue
                seen_paths.add(resolved_path)
                stale_partials.append(item)

    _log(
        "cleanup scanned",
        library_dir=library_dir,
        include_partials=include_partials,
        older_than_days=older_than_days,
        stale_partial_count=len(stale_partials),
        delete=delete,
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
            _log("cleanup partial deleted", path=path, source=item["source"])

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

    previous_disable_xet = os.environ.get("HF_HUB_DISABLE_XET")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
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
                tqdm_class=_progress_tqdm_class(ref.repo_id, progress, stop_after_file)
                if progress is not None
                else None,
            )
        )
    finally:
        if previous_disable_xet is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = previous_disable_xet
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


def _progress_tqdm_class(
    repo_id: str,
    progress: ProgressCallback,
    stop_after_file: StopAfterFileCallback | None = None,
) -> type[hf_tqdm]:
    class ProgressTqdm(hf_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._hfp_ready = False
            self._hfp_last_emit_at: float | None = None
            self._hfp_last_signature: tuple[int | float | None, int | float | None] | None = None
            super().__init__(*args, **kwargs)
            self._hfp_ready = True
            self._emit_download_progress(force=True)

        def update(self, n: int | float | None = 1) -> Any:
            result = super().update(n)
            self._emit_download_progress()
            self._raise_if_stop_requested()
            return result

        def refresh(self, *args: Any, **kwargs: Any) -> Any:
            result = super().refresh(*args, **kwargs)
            if getattr(self, "_hfp_ready", False):
                self._emit_download_progress(force=True)
                self._raise_if_stop_requested()
            return result

        def close(self) -> None:
            if getattr(self, "_hfp_ready", False):
                self._emit_download_progress(force=True)
            super().close()

        def _raise_if_stop_requested(self) -> None:
            if stop_after_file is not None and stop_after_file():
                raise DownloadStoppedAfterFile

        def _emit_download_progress(self, *, force: bool = False) -> None:
            if getattr(self, "unit", None) != "B":
                return
            downloaded = _numeric_progress_value(getattr(self, "n", None))
            total = _numeric_progress_value(getattr(self, "total", None))
            if downloaded is None and total is None:
                return
            signature = (downloaded, total)
            if signature == getattr(self, "_hfp_last_signature", None):
                return
            now = time.monotonic()
            last_emit_at = getattr(self, "_hfp_last_emit_at", None)
            if (
                not force
                and last_emit_at is not None
                and now - last_emit_at < PROGRESS_EMIT_INTERVAL_SECONDS
            ):
                return
            rate = self.format_dict.get("rate")
            speed = float(rate) if isinstance(rate, (int, float)) and rate > 0 else None
            eta = None
            if speed and total is not None and downloaded is not None:
                eta = int(max(total - downloaded, 0) / speed)
            percent = (
                downloaded / total * 100
                if downloaded is not None and total is not None and total > 0
                else None
            )
            progress(
                {
                    "type": "download-progress",
                    "repo_id": repo_id,
                    "downloaded": downloaded,
                    "total": total,
                    "percent": percent,
                    "bytes_per_second": speed,
                    "eta_seconds": eta,
                }
            )
            self._hfp_last_emit_at = now
            self._hfp_last_signature = signature

    return ProgressTqdm


def _numeric_progress_value(value: Any) -> int | float | None:
    if isinstance(value, (int, float)):
        return value
    return None


def directory_size(path: Path) -> int:
    return sum(candidate.stat().st_size for candidate in Path(path).rglob("*") if candidate.is_file())
