from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .app_logging import write_log
from .config import (
    DEFAULT_ENDPOINT,
    default_hf_hub_cache,
    default_max_workers,
    safe_repo_dir_name,
)


ProgressCallback = Callable[[dict[str, Any]], None]
StopAfterFileCallback = Callable[[], bool]
PROGRESS_EMIT_INTERVAL_SECONDS = 0.5
MODEL_PAYLOAD_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".msgpack",
)
SHARDED_PAYLOAD_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})(?P<suffix>\.[^.]+)$"
)
COMMIT_SHA_RE = re.compile(r"^[a-f0-9]{40}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
COMPLETION_FORMAT = "huggingfacepull-completion"
COMPLETION_VERSION = 2
COMPLETION_VERIFICATIONS = {"sha256", "size_only"}
_LOGGED_SKIPPED_CACHE_SNAPSHOTS: set[tuple[str, str, str, str]] = set()
HF_HUB_CACHE = str(default_hf_hub_cache())
HfApi: Any | None = None
snapshot_download: Any | None = None
hf_tqdm: Any | None = None


def force_disable_xet() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    os.environ.pop("HF_XET_CHUNK_CACHE_SIZE_BYTES", None)
    os.environ.pop("HF_XET_SHARD_CACHE_SIZE_LIMIT", None)


def enable_xet() -> None:
    os.environ.pop("HF_HUB_DISABLE_XET", None)


def configure_xet(enabled: bool) -> None:
    if enabled:
        enable_xet()
    else:
        force_disable_xet()


force_disable_xet()


class DownloadStoppedAfterFile(Exception):
    """Raised when a caller requests a stop after a completed snapshot pull."""


@dataclasses.dataclass(frozen=True)
class HubRef:
    repo_id: str
    revision: str = "main"
    repo_type: str = "model"
    expected_commit: str | None = None
    allow_patterns: tuple[str, ...] | list[str] = dataclasses.field(default_factory=tuple)
    ignore_patterns: tuple[str, ...] | list[str] = dataclasses.field(default_factory=tuple)
    xet_enabled: bool = False

    def __post_init__(self) -> None:
        _validate_repo_id(self.repo_id)
        if self.repo_type not in {"model", "dataset", "space", "kernel"}:
            raise ValueError(f"Unsupported Hugging Face repository type: {self.repo_type}")
        if self.expected_commit is not None and COMMIT_SHA_RE.fullmatch(self.expected_commit) is None:
            raise ValueError("Expected commit must be a 40-character lowercase hexadecimal SHA")


def canonical_ref(ref: HubRef) -> str:
    allow = ",".join(sorted(ref.allow_patterns))
    ignore = ",".join(sorted(ref.ignore_patterns))
    xet = "1" if ref.xet_enabled else "0"
    expected = f"expected={ref.expected_commit}&" if ref.expected_commit else ""
    return f"{ref.repo_type}:{ref.repo_id}@{ref.revision}?{expected}allow={allow}&ignore={ignore}&xet={xet}"


def safe_revision_dir_name(revision: str) -> str:
    stripped = revision.strip()
    if not stripped:
        return "main"
    parts = [part for part in stripped.replace("\\", "/").split("/") if part]
    safe = "--".join(parts)
    if safe in {".", ".."}:
        return safe.replace(".", "_")
    return safe


def _validate_repo_id(repo_id: str) -> None:
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("Repository ID must be a non-empty string")
    if "\x00" in repo_id or "\\" in repo_id:
        raise ValueError("Repository ID must not contain path separators or NUL bytes")
    parts = repo_id.strip().split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Repository ID must not contain traversal components")


def metadata_path(library_dir: Path, ref: HubRef) -> Path:
    return (
        Path(library_dir)
        / safe_repo_dir_name(ref.repo_id)
        / safe_revision_dir_name(ref.revision)
        / ".huggingfacepull.json"
    )


def read_completion_marker(marker: Path) -> dict[str, Any]:
    """Read a legacy marker or validate a versioned completion marker offline."""
    try:
        metadata = json.loads(Path(marker).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read completion marker {marker}") from error
    if not isinstance(metadata, dict):
        raise ValueError("Completion marker must contain a JSON object")
    if "format" not in metadata and "version" not in metadata:
        if not isinstance(metadata.get("repo_id"), str) or not isinstance(
            metadata.get("revision"), str
        ):
            raise ValueError("Legacy completion marker is missing repo_id or revision")
        return metadata
    return _validate_versioned_marker(metadata)


def _validate_versioned_marker(metadata: dict[str, Any]) -> dict[str, Any]:
    required = {
        "format",
        "version",
        "repo_id",
        "repo_type",
        "requested_revision",
        "revision",
        "expected_commit",
        "resolved_revision",
        "snapshot_path",
        "xet_enabled",
        "size",
        "files",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Completion marker is missing required fields: {', '.join(missing)}")
    if metadata["format"] != COMPLETION_FORMAT:
        raise ValueError(f"Unsupported completion marker format: {metadata['format']!r}")
    if metadata["version"] != COMPLETION_VERSION:
        raise ValueError(f"Unsupported completion marker version: {metadata['version']!r}")
    _validate_repo_id(metadata["repo_id"])
    if metadata["repo_type"] not in {"model", "dataset", "space", "kernel"}:
        raise ValueError("Completion marker contains an unsupported repository type")
    if not isinstance(metadata["requested_revision"], str) or not metadata["requested_revision"].strip():
        raise ValueError("Completion marker requested_revision must be a non-empty string")
    if metadata["revision"] != metadata["requested_revision"]:
        raise ValueError("Completion marker revision must equal requested_revision")
    expected_commit = metadata["expected_commit"]
    if expected_commit is not None and (
        not isinstance(expected_commit, str) or COMMIT_SHA_RE.fullmatch(expected_commit) is None
    ):
        raise ValueError("Completion marker expected_commit must be a 40-character SHA or null")
    resolved_revision = metadata["resolved_revision"]
    if not isinstance(resolved_revision, str) or COMMIT_SHA_RE.fullmatch(resolved_revision) is None:
        raise ValueError("Completion marker resolved_revision must be a 40-character SHA")
    if not isinstance(metadata["snapshot_path"], str) or not metadata["snapshot_path"].strip():
        raise ValueError("Completion marker snapshot_path must be a non-empty string")
    if not isinstance(metadata["xet_enabled"], bool):
        raise ValueError("Completion marker xet_enabled must be a boolean")
    if not _is_non_negative_int(metadata["size"]):
        raise ValueError("Completion marker size must be a non-negative integer")
    files = metadata["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Completion marker files must be a non-empty array")
    normalised_files = [_normalise_file_record(file, require_verification=True) for file in files]
    paths = [file["path"] for file in normalised_files]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Completion marker files must be unique and sorted by path")
    if metadata["size"] != sum(file["size"] for file in normalised_files):
        raise ValueError("Completion marker size must equal the selected-file size total")
    normalised = dict(metadata)
    normalised["files"] = normalised_files
    return normalised


def _normalise_file_record(file: Any, *, require_verification: bool) -> dict[str, Any]:
    if not isinstance(file, dict):
        raise ValueError("Completion marker file entries must be objects")
    required = {"path", "size", "blob_id", "lfs_sha256", "lfs_size", "xet_hash"}
    if require_verification:
        required.add("verification")
    missing = sorted(required - file.keys())
    if missing:
        raise ValueError(f"Completion marker file entry is missing fields: {', '.join(missing)}")
    path = _validate_relative_file_path(file["path"])
    if not _is_non_negative_int(file["size"]):
        raise ValueError(f"Completion marker file {path} has an invalid size")
    blob_id = file["blob_id"]
    if blob_id is not None and (not isinstance(blob_id, str) or not blob_id):
        raise ValueError(f"Completion marker file {path} has an invalid blob_id")
    lfs_sha256 = file["lfs_sha256"]
    lfs_size = file["lfs_size"]
    if lfs_sha256 is None:
        if lfs_size is not None:
            raise ValueError(f"Completion marker file {path} has lfs_size without lfs_sha256")
    elif not isinstance(lfs_sha256, str) or SHA256_RE.fullmatch(lfs_sha256) is None:
        raise ValueError(f"Completion marker file {path} has an invalid lfs_sha256")
    elif not _is_non_negative_int(lfs_size) or lfs_size != file["size"]:
        raise ValueError(f"Completion marker file {path} has inconsistent LFS size metadata")
    xet_hash = file["xet_hash"]
    if xet_hash is not None and (
        not isinstance(xet_hash, str) or SHA256_RE.fullmatch(xet_hash) is None
    ):
        raise ValueError(f"Completion marker file {path} has an invalid xet_hash")
    normalised = {
        "path": path,
        "size": file["size"],
        "blob_id": blob_id,
        "lfs_sha256": lfs_sha256,
        "lfs_size": lfs_size,
        "xet_hash": xet_hash,
    }
    if require_verification:
        verification = file["verification"]
        if verification not in COMPLETION_VERIFICATIONS:
            raise ValueError(f"Completion marker file {path} has an invalid verification status")
        if verification == "sha256" and lfs_sha256 is None:
            raise ValueError(f"Completion marker file {path} cannot be SHA-256 verified without LFS metadata")
        normalised["verification"] = verification
    return normalised


def _validate_relative_file_path(path: Any) -> str:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError("Repository file path must be a non-empty POSIX relative path")
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute() or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError(f"Repository file path escapes the snapshot: {path!r}")
    return str(pure_path)


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _write_marker_atomic(marker: Path, metadata: dict[str, Any]) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{marker.name}.", suffix=".tmp", dir=marker.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
        try:
            directory_fd = os.open(marker.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _safe_snapshot_file_path(snapshot: Path, relative_path: str) -> Path:
    parts = PurePosixPath(_validate_relative_file_path(relative_path)).parts
    snapshot_root = snapshot.resolve()
    candidate = snapshot.joinpath(*parts)
    try:
        candidate.parent.resolve().relative_to(snapshot_root)
    except ValueError as error:
        raise ValueError(f"Repository file path escapes the snapshot: {relative_path!r}") from error
    if candidate.is_symlink():
        target = candidate.resolve()
        blob_root = snapshot.parent.parent.resolve() / "blobs"
        try:
            target.relative_to(blob_root)
        except ValueError as error:
            raise ValueError(f"Repository file symlink escapes the cache: {relative_path!r}") from error
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def upgrade_legacy_markers(library_dir: Path) -> dict[str, list[str]]:
    """Upgrade legacy markers using local cache metadata only; never contacts the Hub."""
    upgraded: list[str] = []
    skipped: list[str] = []
    for marker in Path(library_dir).glob("*/*/.huggingfacepull.json"):
        try:
            raw = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped.append(str(marker))
            continue
        if not isinstance(raw, dict):
            skipped.append(str(marker))
            continue
        if raw.get("format") == COMPLETION_FORMAT and raw.get("version") == COMPLETION_VERSION:
            continue
        try:
            metadata = _upgrade_legacy_marker(raw)
            _write_marker_atomic(marker, metadata)
        except (OSError, ValueError):
            skipped.append(str(marker))
            continue
        upgraded.append(str(marker))
    return {"upgraded": upgraded, "skipped": skipped}


def _upgrade_legacy_marker(raw: dict[str, Any]) -> dict[str, Any]:
    repo_id = raw.get("repo_id")
    revision = raw.get("revision")
    repo_type = raw.get("repo_type", "model")
    snapshot_path = raw.get("snapshot_path")
    expected_commit = raw.get("expected_commit")
    if not isinstance(revision, str) or not isinstance(snapshot_path, str):
        raise ValueError("Legacy marker lacks revision or snapshot_path")
    ref = HubRef(
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        expected_commit=expected_commit,
        xet_enabled=bool(raw.get("xet_enabled", False)),
    )
    resolved_revision = raw.get("resolved_revision")
    if not isinstance(resolved_revision, str) or COMMIT_SHA_RE.fullmatch(resolved_revision) is None:
        candidate = Path(snapshot_path).name
        if COMMIT_SHA_RE.fullmatch(candidate) is None:
            raise ValueError("Legacy marker has no immutable resolved revision")
        resolved_revision = candidate
    snapshot = Path(snapshot_path)
    if snapshot.resolve() != _expected_snapshot_path(ref, resolved_revision).resolve():
        raise ValueError("Legacy marker snapshot is outside the derived Hugging Face snapshot")
    legacy_files = raw.get("files")
    if not isinstance(legacy_files, list) or not legacy_files:
        raise ValueError("Legacy marker has no selected files")
    tree_entries = _cached_tree_entries(ref, resolved_revision)
    upgraded_files = []
    for legacy_file in legacy_files:
        if not isinstance(legacy_file, dict):
            raise ValueError("Legacy marker contains malformed file metadata")
        path = _validate_relative_file_path(legacy_file.get("path"))
        tree_entry = tree_entries.get(path, {})
        size = legacy_file.get("size", tree_entry.get("size"))
        blob_id = legacy_file.get("blob_id", tree_entry.get("blob_id"))
        if tree_entry and (
            (size != tree_entry.get("size"))
            or (blob_id is not None and blob_id != tree_entry.get("blob_id"))
        ):
            raise ValueError("Legacy marker disagrees with cached tree metadata")
        upgraded_files.append(
            _normalise_file_record(
                {
                    "path": path,
                    "size": size,
                    "blob_id": blob_id,
                    "lfs_sha256": tree_entry.get("lfs_sha256"),
                    "lfs_size": tree_entry.get("lfs_size"),
                    "xet_hash": tree_entry.get("xet_hash"),
                    "verification": "size_only",
                },
                require_verification=True,
            )
        )
    upgraded_files.sort(key=lambda file: file["path"])
    if len({file["path"] for file in upgraded_files}) != len(upgraded_files):
        raise ValueError("Legacy marker contains duplicate file paths")
    skip_reason = _snapshot_integrity_skip_reason(snapshot, upgraded_files)
    if skip_reason is not None:
        raise ValueError(f"Legacy snapshot is incomplete: {skip_reason['reason']}")
    return {
        "format": COMPLETION_FORMAT,
        "version": COMPLETION_VERSION,
        "repo_id": ref.repo_id,
        "requested_revision": ref.revision,
        "revision": ref.revision,
        "repo_type": ref.repo_type,
        "expected_commit": ref.expected_commit,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot),
        "xet_enabled": ref.xet_enabled,
        "size": sum(file["size"] for file in upgraded_files),
        "files": upgraded_files,
    }


def _cached_tree_entries(ref: HubRef, resolved_revision: str) -> dict[str, dict[str, Any]]:
    snapshot_root = _expected_snapshot_path(ref, resolved_revision).parent.parent
    tree_path = snapshot_root / "trees" / f"{resolved_revision}.json"
    try:
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    files = tree.get("files") if isinstance(tree, dict) else None
    if not isinstance(tree, dict) or tree.get("format_version") != 1 or not isinstance(files, dict):
        return {}
    return {path: value for path, value in files.items() if isinstance(path, str) and isinstance(value, dict)}


def search_models(
    query: str,
    endpoint: str = DEFAULT_ENDPOINT,
    token: str | None = None,
) -> dict[str, Any]:
    if not query.strip():
        return {"available": True, "results": [], "error": None}

    try:
        api = _hf_api_class()(endpoint=endpoint)
        try:
            models = _search_model_list(api, query, token)
        except Exception as error:
            if token is not None or not _is_unauthorised_hub_error(error):
                raise
            write_log(
                "hub search retrying anonymously",
                reason="cached_token_unauthorised",
            )
            models = _search_model_list(api, query, False)
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


def _search_model_list(api: Any, query: str, token: str | bool | None) -> list[Any]:
    return list(
        api.list_models(
            search=query,
            limit=20,
            sort="downloads",
            token=token,
        )
    )


def _is_unauthorised_hub_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 401


def repo_files(
    ref: HubRef,
    endpoint: str = DEFAULT_ENDPOINT,
    token: str | None = None,
) -> dict[str, Any]:
    api = _hf_api_class()(endpoint=endpoint)
    if ref.repo_type == "model":
        info = api.model_info(
            ref.repo_id,
            revision=ref.revision,
            files_metadata=True,
            token=token,
        )
    else:
        info = api.repo_info(
            ref.repo_id,
            revision=ref.revision,
            repo_type=ref.repo_type,
            files_metadata=True,
            token=token,
        )
    files = _repo_file_records(api, info, ref=ref, token=token)
    return {"repo_id": ref.repo_id, "revision": ref.revision, "files": files}


def _repo_file_records(api: Any, info: Any, *, ref: HubRef, token: str | None) -> list[dict[str, Any]]:
    siblings = getattr(info, "siblings", None)
    if siblings is None:
        siblings = api.list_repo_tree(
            ref.repo_id,
            recursive=True,
            revision=ref.revision,
            repo_type=ref.repo_type,
            token=token,
        )
    elif hasattr(api, "list_repo_tree") and any(
        getattr(sibling, "xet_hash", None) is None for sibling in siblings
    ):
        # model_info() does not consistently expose Xet metadata across supported
        # huggingface_hub versions. Enrich from the tree when the client can provide it.
        try:
            tree = api.list_repo_tree(
                ref.repo_id,
                recursive=True,
                revision=ref.revision,
                repo_type=ref.repo_type,
                token=token,
            )
        except Exception:
            tree = []
        tree_by_path = {
            getattr(item, "rfilename", None) or getattr(item, "path", None): item
            for item in tree
        }
        siblings = [
            tree_by_path.get(getattr(item, "rfilename", None) or getattr(item, "path", None), item)
            for item in siblings
        ]
    files = []
    for sibling in siblings:
        path = getattr(sibling, "rfilename", None) or getattr(sibling, "path", None)
        if path is None or not hasattr(sibling, "size"):
            continue
        lfs = getattr(sibling, "lfs", None)
        if isinstance(lfs, dict):
            lfs_sha256 = lfs.get("sha256")
            lfs_size = lfs.get("size")
        else:
            lfs_sha256 = getattr(lfs, "sha256", None)
            lfs_size = getattr(lfs, "size", None)
        files.append(
            {
                "path": path,
                "size": getattr(sibling, "size", None),
                "blob_id": getattr(sibling, "blob_id", None),
                "lfs_sha256": lfs_sha256,
                "lfs_size": lfs_size,
                "xet_hash": getattr(sibling, "xet_hash", None),
            }
        )
    return files


def installed_models(library_dir: Path) -> list[dict[str, Any]]:
    installed: list[dict[str, Any]] = []
    for marker in Path(library_dir).glob("*/*/.huggingfacepull.json"):
        try:
            metadata = read_completion_marker(marker)
        except ValueError:
            continue
        metadata = _metadata_with_current_cache_path(metadata)
        skip_reason = _installed_metadata_skip_reason(metadata)
        if skip_reason is not None:
            _log(
                "installed metadata skipped",
                marker=marker,
                repo_id=metadata.get("repo_id"),
                revision=metadata.get("revision"),
                **skip_reason,
            )
            continue
        installed.append(metadata)
    return sorted(
        installed,
        key=lambda item: (str(item["repo_id"]).lower(), str(item["revision"])),
    )


def _metadata_with_current_cache_path(metadata: dict[str, Any]) -> dict[str, Any]:
    configured_path = metadata.get("snapshot_path")
    if not isinstance(configured_path, str) or not configured_path:
        return metadata

    snapshot_path = Path(configured_path)
    if snapshot_path.exists() or snapshot_path.name in {"", ".", ".."}:
        return metadata

    repo_id = metadata.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id.strip():
        return metadata
    try:
        _validate_repo_id(repo_id)
    except ValueError:
        return metadata
    prefix = {
        "dataset": "datasets",
        "space": "spaces",
        "kernel": "kernels",
    }.get(metadata.get("repo_type", "model"), "models")
    snapshots_dir = (
        Path(HF_HUB_CACHE)
        / f"{prefix}--{safe_repo_dir_name(repo_id)}"
        / "snapshots"
    )
    candidate = snapshots_dir / snapshot_path.name
    try:
        candidate.resolve().relative_to(snapshots_dir.resolve())
    except (OSError, ValueError):
        return metadata
    if not candidate.is_dir():
        return metadata

    relocated = dict(metadata)
    relocated["snapshot_path"] = str(candidate)
    return relocated


def cached_hub_models(cache_dir: Path | str | None = None) -> list[dict[str, Any]]:
    root = Path(cache_dir or HF_HUB_CACHE)
    cached: list[dict[str, Any]] = []
    if not root.exists():
        return cached

    for repo_type, prefix in (("model", "models"), ("kernel", "kernels")):
        for repo_dir in sorted(root.glob(f"{prefix}--*")):
            if not repo_dir.is_dir():
                continue
            repo_id = _repo_id_from_cache_dir(repo_dir.name, prefix)
            if repo_id is None:
                continue
            snapshots = repo_dir / "snapshots"
            if not snapshots.is_dir():
                continue
            refs = _cache_refs(repo_dir)
            if refs:
                for revision, commit in refs.items():
                    snapshot = snapshots / commit
                    skip_reason = _cache_snapshot_skip_reason(
                        snapshot, require_model_payload=repo_type == "model"
                    )
                    if skip_reason is None:
                        cached.append(
                            {
                                "repo_id": repo_id,
                                "revision": revision,
                                "repo_type": repo_type,
                                "snapshot_path": str(snapshot),
                                "source": "huggingface_cache",
                            }
                        )
                    else:
                        _log_cache_snapshot_skipped(repo_id, revision, snapshot, skip_reason)
                continue
            for snapshot in sorted(snapshots.iterdir()):
                skip_reason = _cache_snapshot_skip_reason(
                    snapshot, require_model_payload=repo_type == "model"
                )
                if skip_reason is None:
                    cached.append(
                        {
                            "repo_id": repo_id,
                            "revision": snapshot.name,
                            "repo_type": repo_type,
                            "snapshot_path": str(snapshot),
                            "source": "huggingface_cache",
                        }
                    )
                else:
                    _log_cache_snapshot_skipped(repo_id, snapshot.name, snapshot, skip_reason)
    return cached


def partial_cached_hub_models(cache_dir: Path | str | None = None) -> list[dict[str, Any]]:
    root = Path(cache_dir or HF_HUB_CACHE)
    partial: list[dict[str, Any]] = []
    if not root.exists():
        return partial

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
                skip_reason = _cache_snapshot_skip_reason(snapshot, require_model_payload=True)
                entry = _partial_cache_entry(repo_id, revision, snapshot, skip_reason)
                if entry is not None:
                    partial.append(entry)
            continue
        for snapshot in sorted(snapshots.iterdir()):
            skip_reason = _cache_snapshot_skip_reason(snapshot, require_model_payload=True)
            entry = _partial_cache_entry(repo_id, snapshot.name, snapshot, skip_reason)
            if entry is not None:
                partial.append(entry)
    return partial


def _partial_cache_entry(
    repo_id: str,
    revision: str,
    snapshot: Path,
    skip_reason: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if skip_reason is None:
        return None
    reason = skip_reason.get("reason")
    if reason not in {"partial_file", "incomplete_sharded_payload"}:
        return None
    return {
        "repo_id": repo_id,
        "revision": revision,
        "repo_type": "model",
        "snapshot_path": str(snapshot),
        "source": "huggingface_cache",
        "cache_status": "partial",
        "reason": reason,
    }


def _cache_snapshot_skip_reason(
    snapshot: Path,
    *,
    require_model_payload: bool = False,
) -> dict[str, Any] | None:
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
    sharded_reason = _sharded_payload_skip_reason(snapshot)
    if sharded_reason is not None:
        return sharded_reason
    if require_model_payload and not _has_model_payload(snapshot):
        return {"reason": "missing_model_payload"}
    return None


def _sharded_payload_skip_reason(snapshot: Path) -> dict[str, Any] | None:
    shard_groups: dict[tuple[str, int, str], set[int]] = {}
    for path in snapshot.rglob("*"):
        if path.is_dir() or not path.exists():
            continue
        match = SHARDED_PAYLOAD_RE.match(path.name)
        if match is None:
            continue
        suffix = match.group("suffix")
        if suffix not in MODEL_PAYLOAD_SUFFIXES:
            continue
        total = int(match.group("total"))
        index = int(match.group("index"))
        if total <= 1:
            continue
        shard_groups.setdefault(
            (match.group("prefix"), total, suffix),
            set(),
        ).add(index)

    for (prefix, total, suffix), found in sorted(shard_groups.items()):
        expected = set(range(1, total + 1))
        missing = sorted(expected - found)
        if missing:
            return {
                "reason": "incomplete_sharded_payload",
                "path": snapshot,
                "prefix": prefix,
                "suffix": suffix,
                "expected_shards": total,
                "found_shards": len(found),
                "missing_shards": missing,
            }
    return None


def _has_model_payload(snapshot: Path) -> bool:
    for path in snapshot.rglob("*"):
        if path.is_dir() or not path.exists():
            continue
        if path.name.endswith(MODEL_PAYLOAD_SUFFIXES):
            return True
    return False


def _installed_metadata_skip_reason(metadata: dict[str, Any]) -> dict[str, Any] | None:
    snapshot_path = metadata.get("snapshot_path")
    if not isinstance(snapshot_path, str) or not snapshot_path.strip():
        return None
    if metadata.get("format") == COMPLETION_FORMAT:
        try:
            expected_snapshot = _expected_snapshot_path(
                HubRef(
                    repo_id=metadata["repo_id"],
                    revision=metadata["requested_revision"],
                    repo_type=metadata["repo_type"],
                    expected_commit=metadata["expected_commit"],
                    xet_enabled=metadata["xet_enabled"],
                ),
                metadata["resolved_revision"],
            )
            if Path(snapshot_path).resolve() != expected_snapshot.resolve():
                return {"reason": "outside_derived_snapshot", "path": Path(snapshot_path)}
        except (KeyError, TypeError, ValueError):
            return {"reason": "invalid_completion_marker"}
    return _snapshot_integrity_skip_reason(Path(snapshot_path), metadata.get("files"))


def _snapshot_integrity_skip_reason(
    snapshot: Path,
    expected_files: Any = None,
) -> dict[str, Any] | None:
    basic_reason = _cache_snapshot_skip_reason(snapshot)
    if basic_reason is not None:
        return basic_reason
    if not isinstance(expected_files, list):
        return None

    for file in expected_files:
        if not isinstance(file, dict):
            continue
        relative_path = file.get("path")
        if not isinstance(relative_path, str) or not relative_path.strip():
            continue
        try:
            path = _safe_snapshot_file_path(snapshot, relative_path)
        except ValueError:
            return {"reason": "path_traversal", "path": snapshot / relative_path}
        if not path.exists():
            return {
                "reason": "broken_symlink" if path.is_symlink() else "missing_file",
                "path": path,
            }
        if _is_partial_file(path):
            return {"reason": "partial_file", "path": path}
        expected_size = file.get("size")
        if isinstance(expected_size, int):
            try:
                actual_size = path.stat().st_size
            except FileNotFoundError:
                return {
                    "reason": "broken_symlink" if path.is_symlink() else "missing_file",
                    "path": path,
                }
            if actual_size != expected_size:
                return {
                    "reason": "size_mismatch",
                    "path": path,
                    "expected_size": expected_size,
                    "actual_size": actual_size,
                }
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


def delete_installed_model(library_dir: Path, ref: HubRef) -> int:
    """Delete a cached snapshot and its HuggingFacePull metadata record."""
    from huggingface_hub import scan_cache_dir

    marker = metadata_path(library_dir, ref)
    snapshot_path: Path | None = None
    if marker.is_file():
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, dict) and isinstance(metadata.get("snapshot_path"), str):
            snapshot_path = Path(metadata["snapshot_path"])

    cache_root = Path(HF_HUB_CACHE)
    if not cache_root.is_dir():
        raise KeyError(ref.repo_id)
    cache_info = scan_cache_dir(cache_root)
    cached_repo = next(
        (
            repo
            for repo in cache_info.repos
            if repo.repo_id == ref.repo_id and repo.repo_type == ref.repo_type
        ),
        None,
    )
    if cached_repo is None:
        raise KeyError(ref.repo_id)

    revision = next(
        (
            cached_revision
            for cached_revision in cached_repo.revisions
            if ref.revision == cached_revision.commit_hash
            or ref.revision in cached_revision.refs
            or (
                snapshot_path is not None
                and snapshot_path.name == cached_revision.commit_hash
            )
        ),
        None,
    )
    if revision is None:
        raise KeyError(ref.repo_id)

    strategy = cache_info.delete_revisions(revision.commit_hash)
    freed_size = strategy.expected_freed_size
    strategy.execute()

    if marker.exists():
        remove_installed_model(library_dir, ref)
    return freed_size


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
    marker = metadata_path(library_dir, ref)
    metadata_dir = marker.parent
    if progress is not None:
        progress({"type": "manifest-fetch", "repo_id": ref.repo_id, "revision": ref.revision})

    if dry_run:
        if progress is not None:
            progress({"type": "model-complete", "repo_id": ref.repo_id, "dry_run": True})
        return metadata_dir

    configure_xet(ref.xet_enabled)
    api = _hf_api_class()(endpoint=endpoint)
    if ref.repo_type == "model":
        info = api.model_info(
            ref.repo_id,
            revision=ref.revision,
            files_metadata=True,
            token=token,
        )
    else:
        info = api.repo_info(
            ref.repo_id,
            revision=ref.revision,
            repo_type=ref.repo_type,
            files_metadata=True,
            token=token,
        )
    resolved_revision = str(getattr(info, "sha", "") or "")
    if COMMIT_SHA_RE.fullmatch(resolved_revision) is None:
        raise RuntimeError("Hub did not return a valid immutable resolved commit")
    if ref.expected_commit is not None and resolved_revision != ref.expected_commit:
        raise RuntimeError(
            f"Resolved commit {resolved_revision or 'unknown'} does not match expected commit "
            f"{ref.expected_commit}"
        )
    resolved_ref = dataclasses.replace(ref, revision=resolved_revision)
    files = _filter_repo_files(
        _repo_file_records(api, info, ref=resolved_ref, token=token),
        allow_patterns=ref.allow_patterns,
        ignore_patterns=ref.ignore_patterns,
    )
    try:
        files = [_normalise_file_record(file, require_verification=False) for file in files]
    except ValueError as error:
        raise RuntimeError(f"Invalid selected file metadata: {error}") from error
    files.sort(key=lambda file: file["path"])
    if not files:
        raise RuntimeError("No repository files matched the selected download patterns")
    if len({file["path"] for file in files}) != len(files):
        raise RuntimeError("Selected repository file metadata contains duplicate paths")
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
    snapshot_path = Path(
        _snapshot_download_func()(
            repo_id=ref.repo_id,
            revision=resolved_revision,
            repo_type=None if ref.repo_type == "model" else ref.repo_type,
            cache_dir=Path(HF_HUB_CACHE),
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

    if ref.xet_enabled:
        if os.environ.get("HF_HUB_DISABLE_XET") == "1":
            raise RuntimeError("HF_HUB_DISABLE_XET must be unset when Xet is enabled")
    elif os.environ.get("HF_HUB_DISABLE_XET") != "1":
        raise RuntimeError("HF_HUB_DISABLE_XET must remain set to 1 when Xet is disabled")

    expected_snapshot = _expected_snapshot_path(ref, resolved_revision)
    if snapshot_path.resolve() != expected_snapshot.resolve():
        raise RuntimeError(
            f"Downloaded snapshot path {snapshot_path} is outside the derived Hugging Face snapshot"
        )

    skip_reason = _snapshot_integrity_skip_reason(snapshot_path, files)
    if skip_reason is not None:
        _log(
            "downloaded snapshot incomplete",
            repo_id=ref.repo_id,
            revision=ref.revision,
            snapshot_path=snapshot_path,
            **skip_reason,
        )
        reason = skip_reason.get("reason", "incomplete")
        detail = f" at {skip_reason['path']}" if "path" in skip_reason else ""
        raise RuntimeError(f"Downloaded snapshot is incomplete: {reason}{detail}")

    completed_files = []
    for file in files:
        completed = dict(file)
        if file["lfs_sha256"] is None:
            completed["verification"] = "size_only"
        else:
            path = _safe_snapshot_file_path(snapshot_path, file["path"])
            actual_sha256 = _sha256_file(path)
            if actual_sha256 != file["lfs_sha256"]:
                raise RuntimeError(
                    f"Downloaded snapshot checksum mismatch at {path}: expected "
                    f"{file['lfs_sha256']}, got {actual_sha256}"
                )
            completed["verification"] = "sha256"
        completed_files.append(completed)

    metadata = {
        "format": COMPLETION_FORMAT,
        "version": COMPLETION_VERSION,
        "repo_id": ref.repo_id,
        "requested_revision": ref.revision,
        "revision": ref.revision,
        "repo_type": ref.repo_type,
        "expected_commit": ref.expected_commit,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot_path),
        "size": sum(file["size"] for file in completed_files),
        "files": completed_files,
        "xet_enabled": ref.xet_enabled,
    }
    _write_marker_atomic(marker, metadata)

    if progress is not None:
        progress(
            {
                "type": "model-complete",
                "repo_id": ref.repo_id,
                "snapshot_path": str(snapshot_path),
            }
        )
    return snapshot_path


def _expected_snapshot_path(ref: HubRef, resolved_revision: str) -> Path:
    if COMMIT_SHA_RE.fullmatch(resolved_revision) is None:
        raise ValueError("Resolved revision must be a 40-character lowercase hexadecimal SHA")
    prefix = {
        "dataset": "datasets",
        "space": "spaces",
        "kernel": "kernels",
    }.get(ref.repo_type, "models")
    return (
        Path(HF_HUB_CACHE)
        / f"{prefix}--{safe_repo_dir_name(ref.repo_id)}"
        / "snapshots"
        / resolved_revision
    )


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
            downloaded = _numeric_progress_value(getattr(self, "n", None))
            total = _numeric_progress_value(getattr(self, "total", None))
            if downloaded is None and total is None:
                return
            unit = getattr(self, "unit", None)
            desc = getattr(self, "desc", None)
            signature = (unit, desc, downloaded, total)
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
            if unit == "B":
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
            else:
                progress(
                    {
                        "type": "fetch-progress",
                        "repo_id": repo_id,
                        "downloaded": downloaded,
                        "total": total,
                        "percent": percent,
                        "unit": unit,
                        "description": desc,
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
