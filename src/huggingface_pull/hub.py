from __future__ import annotations

import dataclasses
import fnmatch
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from .app_logging import write_log
from .config import DEFAULT_ENDPOINT, default_max_workers, safe_repo_dir_name


ProgressCallback = Callable[[dict[str, Any]], None]
StopAfterFileCallback = Callable[[], bool]
PROGRESS_EMIT_INTERVAL_SECONDS = 0.5
_LOGGED_SKIPPED_CACHE_SNAPSHOTS: set[tuple[str, str, str, str]] = set()
HF_HUB_CACHE = os.environ.get(
    "HF_HUB_CACHE",
    str(Path.home() / ".cache" / "huggingface" / "hub"),
)
HfApi: Any | None = None
snapshot_download: Any | None = None
hf_tqdm: Any | None = None


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
        api = _hf_api_class()(endpoint=endpoint)
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

    api = _hf_api_class()(endpoint=endpoint)
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


def _is_download_metadata_file(path: Path) -> bool:
    return path.name.endswith(".metadata") or path.name.endswith(".lock")


def _collect_incomplete_library_snapshots(library_dir: Path) -> list[dict[str, Any]]:
    root = Path(library_dir)
    if not root.exists():
        return []

    incomplete: list[dict[str, Any]] = []
    for repo_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for revision_dir in sorted(path for path in repo_dir.iterdir() if path.is_dir()):
            if (revision_dir / ".huggingfacepull.json").exists():
                continue
            download_dir = revision_dir / ".cache" / "huggingface" / "download"
            if not download_dir.is_dir():
                continue
            evidence = [
                str(path)
                for path in sorted(download_dir.rglob("*"))
                if path.is_file() and (_is_download_metadata_file(path) or _is_partial_file(path))
            ]
            if not evidence:
                continue
            incomplete.append(
                {
                    "path": str(revision_dir),
                    "repo_dir": repo_dir.name,
                    "revision": revision_dir.name,
                    "size": directory_size(revision_dir),
                    "reason": "missing_metadata_marker",
                    "evidence": evidence,
                    "source": "library",
                }
            )
    return incomplete


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
    incomplete_snapshots: list[dict[str, Any]] = []
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
        incomplete_snapshots = _collect_incomplete_library_snapshots(library_dir)

    _log(
        "cleanup scanned",
        library_dir=library_dir,
        include_partials=include_partials,
        older_than_days=older_than_days,
        stale_partial_count=len(stale_partials),
        incomplete_snapshot_count=len(incomplete_snapshots),
        delete=delete,
    )

    deleted: list[str] = []
    deleted_snapshots: list[str] = []
    if delete:
        for item in stale_partials:
            path = Path(item["path"])
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            deleted.append(str(path))
            _log("cleanup partial deleted", path=path, source=item["source"])
        library_root = Path(library_dir).resolve()
        for item in incomplete_snapshots:
            path = Path(item["path"])
            try:
                path.resolve().relative_to(library_root)
            except ValueError:
                continue
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                continue
            deleted_snapshots.append(str(path))
            _log(
                "cleanup incomplete snapshot deleted",
                path=path,
                source=item["source"],
                reason=item["reason"],
            )

    return {
        "dry_run": not delete,
        "stale_partial_count": len(stale_partials),
        "stale_partials": stale_partials,
        "incomplete_snapshot_count": len(incomplete_snapshots),
        "incomplete_snapshots": incomplete_snapshots,
        "deleted": deleted,
        "deleted_snapshots": deleted_snapshots,
    }


def pull_snapshot(
    ref: HubRef,
    library_dir: Path,
    endpoint: str = DEFAULT_ENDPOINT,
    token: str | None = None,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
    stop_after_file: StopAfterFileCallback | None = None,
    max_workers: int | None = None,
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
        api = _hf_api_class()(endpoint=endpoint)
        info = api.model_info(
            ref.repo_id,
            revision=ref.revision,
            files_metadata=True,
            token=token,
        )
        files = _filter_repo_files(
            [
                {
                    "path": sibling.rfilename,
                    "size": sibling.size,
                    "blob_id": getattr(sibling, "blob_id", None),
                }
                for sibling in info.siblings
            ],
            allow_patterns=ref.allow_patterns,
            ignore_patterns=ref.ignore_patterns,
        )
        if progress is not None:
            progress(
                {
                    "type": "model-plan",
                    "repo_id": ref.repo_id,
                    "revision": ref.revision,
                    "total_bytes": _sum_file_sizes(files),
                    "files": files,
                }
            )
        target.mkdir(parents=True, exist_ok=True)
        snapshot_path = Path(
            _snapshot_download_func()(
                repo_id=ref.repo_id,
                revision=ref.revision,
                repo_type=None if ref.repo_type == "model" else ref.repo_type,
                local_dir=target,
                endpoint=endpoint,
                token=token,
                allow_patterns=list(ref.allow_patterns) or None,
                ignore_patterns=list(ref.ignore_patterns) or None,
                max_workers=max_workers if max_workers is not None else default_max_workers(),
                tqdm_class=_progress_tqdm_class(ref.repo_id, progress, stop_after_file)
                if progress is not None
                else None,
            )
        )
        if stop_after_file is not None and stop_after_file():
            raise DownloadStoppedAfterFile
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
    return snapshot_path


def _filter_repo_files(
    files: list[dict[str, Any]],
    *,
    allow_patterns: tuple[str, ...] | list[str],
    ignore_patterns: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    filtered = []
    for file in files:
        path = str(file["path"])
        if allow_patterns and not _matches_any(path, allow_patterns):
            continue
        if ignore_patterns and _matches_any(path, ignore_patterns):
            continue
        filtered.append(file)
    return filtered


def _matches_any(path: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _sum_file_sizes(files: list[dict[str, Any]]) -> int | None:
    sizes = [file.get("size") for file in files]
    if not all(isinstance(size, int) for size in sizes):
        return None
    return sum(sizes)


def _local_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _hf_api_class() -> Any:
    if HfApi is not None:
        return HfApi
    from huggingface_hub import HfApi as imported_hf_api

    return imported_hf_api


def _snapshot_download_func() -> Any:
    if snapshot_download is not None:
        return snapshot_download
    from huggingface_hub import snapshot_download as imported_snapshot_download

    return imported_snapshot_download


def _hf_tqdm_class() -> Any:
    global hf_tqdm
    if hf_tqdm is None:
        from huggingface_hub.utils import tqdm as imported_hf_tqdm

        hf_tqdm = imported_hf_tqdm
    return hf_tqdm


def _file_progress_tqdm_class(
    repo_id: str,
    file: dict[str, Any],
    progress: ProgressCallback,
) -> Any:
    path = str(file["path"])
    blob_id = file.get("blob_id")
    size = file.get("size")
    base_class = _progress_tqdm_class(repo_id, progress)

    class FileProgressTqdm(base_class):
        def _emit_download_progress(self, *, force: bool = False) -> None:
            if getattr(self, "unit", None) != "B":
                return
            downloaded = _numeric_progress_value(getattr(self, "n", None))
            total = _numeric_progress_value(getattr(self, "total", None))
            if total is None and isinstance(size, int):
                total = size
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
                    "type": "file-progress",
                    "repo_id": repo_id,
                    "path": path,
                    "downloaded": downloaded,
                    "total": total,
                    "percent": percent,
                    "bytes_per_second": speed,
                    "eta_seconds": eta,
                    "blob_id": blob_id,
                }
            )
            self._hfp_last_emit_at = now
            self._hfp_last_signature = signature

    return FileProgressTqdm


def _progress_tqdm_class(
    repo_id: str,
    progress: ProgressCallback,
    stop_after_file: StopAfterFileCallback | None = None,
) -> Any:
    class ProgressTqdm(_hf_tqdm_class()):
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
