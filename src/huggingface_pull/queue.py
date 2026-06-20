from __future__ import annotations

import itertools
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import hub
from .app_logging import write_log
from .config import DEFAULT_ENDPOINT
from .hub import DownloadStoppedAfterFile, HubRef, canonical_ref, pull_snapshot


PullFunc = Callable[..., Any]


def _empty_progress() -> dict[str, Any]:
    return {
        "phase": "waiting",
        "overall": {"downloaded": 0, "total": None, "percent": None},
        "current_file": None,
    }


class DownloadQueue:
    def __init__(
        self,
        *,
        library_dir: Path,
        endpoint: str = DEFAULT_ENDPOINT,
        token: str | None = None,
        pull_func: PullFunc = pull_snapshot,
    ) -> None:
        self.library_dir = Path(library_dir)
        self.endpoint = endpoint
        self.token = token
        self.pull_func = pull_func
        self._items: list[dict[str, Any]] = []
        self._ids = itertools.count(1)
        self._condition = threading.Condition()
        self._worker: threading.Thread | None = None
        self._pause_requested = False
        self._stop_after_file_requested = False

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        ref = self._hub_ref_from_payload(payload)
        canonical = canonical_ref(ref)
        now = time.time()
        with self._condition:
            for existing in self._items:
                if existing["canonical_ref"] != canonical:
                    continue
                if existing["status"] in {"waiting", "running", "completed", "failed"}:
                    copied = self._copy_item(existing)
                    copied["deduplicated"] = True
                    return copied

            item = {
                "id": str(next(self._ids)),
                "repo_id": ref.repo_id,
                "revision": ref.revision,
                "repo_type": ref.repo_type,
                "allow_patterns": list(ref.allow_patterns),
                "ignore_patterns": list(ref.ignore_patterns),
                "canonical_ref": canonical,
                "deduplicated": False,
                "status": "waiting",
                "error": None,
                "messages": [],
                "progress": _empty_progress(),
                "_ref": ref,
                "_planned_files": {},
                "_planned_file_order": [],
                "_completed_files": {},
                "created_at": now,
                "updated_at": now,
            }
            self._items.append(item)
            self._condition.notify_all()
            return self._copy_item(item)

    def start(self) -> None:
        with self._condition:
            if self._worker is not None:
                return
            self._pause_requested = False
            self._stop_after_file_requested = False
            worker = threading.Thread(target=self._run_worker, daemon=True)
            self._worker = worker
        try:
            worker.start()
        except Exception:
            with self._condition:
                if self._worker is worker:
                    self._worker = None
                self._condition.notify_all()
            raise

    def pause_after_current(self) -> None:
        with self._condition:
            self._pause_requested = True
            self._condition.notify_all()

    def stop_after_current_file(self) -> dict[str, Any]:
        with self._condition:
            self._pause_requested = True
            self._stop_after_file_requested = True
            self._condition.notify_all()
            return self.snapshot()

    def retry(self, item_id: str) -> dict[str, Any]:
        with self._condition:
            item = self._find_item(item_id)
            if item["status"] != "failed":
                raise ValueError("Only failed items can be retried")
            item["status"] = "waiting"
            item["error"] = None
            item["messages"] = []
            item["progress"] = _empty_progress()
            item["_planned_files"] = {}
            item["_planned_file_order"] = []
            item["_completed_files"] = {}
            item["updated_at"] = time.time()
            self._condition.notify_all()
            return self._copy_item(item)

    def remove(self, item_id: str) -> dict[str, Any]:
        with self._condition:
            for index, item in enumerate(self._items):
                if item["id"] != item_id:
                    continue
                if item["status"] == "running":
                    raise ValueError("Running items cannot be removed")
                removed = self._items.pop(index)
                self._condition.notify_all()
                return self._copy_item(removed)
        raise KeyError(item_id)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "running": self._worker is not None
                or any(item["status"] == "running" for item in self._items),
                "pause_requested": self._pause_requested,
                "stop_after_file_requested": self._stop_after_file_requested,
                "library_dir": str(self.library_dir),
                "endpoint": self.endpoint,
                "installed_models": hub.installed_models(self.library_dir),
                "items": [self._copy_item(item) for item in self._items],
            }

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                worker_reserved = self._worker is not None
                running_item = any(item["status"] == "running" for item in self._items)
                if not worker_reserved and not running_item:
                    return True
                if deadline is None:
                    remaining = None
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                self._condition.wait(remaining)

    def _run_worker(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._pause_requested:
                        self._clear_worker_locked()
                        return
                    item = self._next_waiting_item()
                    if item is None:
                        self._clear_worker_locked()
                        return
                    item["status"] = "running"
                    item["updated_at"] = time.time()
                    self._condition.notify_all()

                try:
                    self.pull_func(
                        item["_ref"],
                        library_dir=self.library_dir,
                        endpoint=self.endpoint,
                        token=self.token,
                        dry_run=False,
                        progress=lambda event, item_id=item["id"]: self._record_progress(
                            item_id, event
                        ),
                        stop_after_file=self._stop_after_file_requested_locked,
                    )
                except DownloadStoppedAfterFile:
                    with self._condition:
                        if item["progress"]["phase"] == "completed":
                            item["status"] = "completed"
                        else:
                            item["status"] = "waiting"
                            item["progress"]["phase"] = "waiting"
                            self._append_message_locked(item, "stopped after current snapshot")
                        self._stop_after_file_requested = False
                        item["updated_at"] = time.time()
                        self._condition.notify_all()
                except Exception as error:
                    with self._condition:
                        item["status"] = "failed"
                        item["error"] = str(error)
                        item["progress"]["phase"] = "failed"
                        self._append_message_locked(item, f"failed: {error}")
                        item["updated_at"] = time.time()
                        self._condition.notify_all()
                    try:
                        write_log("download failed", repo_id=item["repo_id"], error=error)
                    except Exception:
                        pass
                else:
                    with self._condition:
                        item["status"] = "completed"
                        self._complete_progress_locked(item)
                        item["updated_at"] = time.time()
                        self._condition.notify_all()
        finally:
            with self._condition:
                self._clear_worker_locked()
                self._condition.notify_all()

    def _record_progress(self, item_id: str, event: dict[str, Any]) -> None:
        with self._condition:
            item = self._find_item(item_id)
            event_type = event.get("type")
            if event_type is not None:
                self._append_message_locked(item, str(event_type))
            self._update_progress_locked(item, event)
            item["updated_at"] = time.time()
            self._condition.notify_all()

    def _update_progress_locked(self, item: dict[str, Any], event: dict[str, Any]) -> None:
        event_type = event.get("type")
        progress = item["progress"]
        if event_type == "manifest-fetch":
            progress["phase"] = "fetching"
            return
        if event_type == "model-plan":
            planned = self._planned_files_from_event(event)
            item["_planned_files"] = planned
            item["_planned_file_order"] = list(planned)
            progress["overall"] = {
                "downloaded": 0,
                "total": event.get("total_bytes") or event.get("total") or self._sum_sizes(planned),
                "percent": None,
            }
            return
        if event_type in {"file-start", "blob-start"}:
            identifier = self._event_identifier(event)
            progress["phase"] = "downloading"
            progress["current_file"] = {
                **self._file_identity_fields(event),
                **self._file_position_fields(item, identifier),
                "downloaded": event.get("resume_at", event.get("downloaded", 0)),
                "total": self._planned_total(item, identifier, event),
                "percent": event.get("percent"),
                "bytes_per_second": self._event_speed(event),
                "eta_seconds": event.get("eta_seconds"),
                "line": event.get("line"),
            }
            self._refresh_overall_locked(item)
            return
        if event_type in {"file-progress", "blob-progress"}:
            progress["phase"] = "downloading"
            progress["current_file"] = self._file_progress_from_event(item, event)
            self._refresh_overall_locked(item)
            return
        if event_type in {"file-complete", "blob-complete"}:
            identifier = self._event_identifier(event)
            if identifier is not None:
                item["_completed_files"][identifier] = event.get("total") or event.get("downloaded")
            progress["current_file"] = self._file_progress_from_event(item, event)
            self._refresh_overall_locked(item)
            return
        if event_type in {"failure", "failed", "error"}:
            progress["phase"] = "failed"
            item["error"] = str(event.get("error") or event.get("message") or event_type)
            return
        if event_type == "model-complete":
            self._complete_progress_locked(item)

    def _hub_ref_from_payload(self, payload: dict[str, Any]) -> HubRef:
        repo_id = str(payload["repo_id"])
        revision = str(payload.get("revision") or "main")
        repo_type = str(payload.get("repo_type") or "model")
        return HubRef(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            allow_patterns=list(payload.get("allow_patterns") or []),
            ignore_patterns=list(payload.get("ignore_patterns") or []),
        )

    def _planned_files_from_event(self, event: dict[str, Any]) -> dict[str, Any]:
        planned = {}
        for file in event.get("files", []) or []:
            identifier = (
                file.get("path")
                or file.get("name")
                or file.get("digest")
                or file.get("blob_id")
            )
            if identifier is None:
                continue
            planned[str(identifier)] = file.get("size") or file.get("total")
        return planned

    def _file_progress_from_event(
        self, item: dict[str, Any], event: dict[str, Any]
    ) -> dict[str, Any]:
        identifier = self._event_identifier(event)
        return {
            **self._file_identity_fields(event),
            **self._file_position_fields(item, identifier),
            "downloaded": event.get("downloaded"),
            "total": self._planned_total(item, identifier, event),
            "percent": event.get("percent"),
            "bytes_per_second": self._event_speed(event),
            "eta_seconds": event.get("eta_seconds"),
            "line": event.get("line"),
        }

    def _planned_total(
        self, item: dict[str, Any], identifier: str | None, event: dict[str, Any]
    ) -> Any:
        if event.get("total") is not None:
            return event.get("total")
        if identifier is not None:
            return item["_planned_files"].get(identifier)
        return None

    def _event_identifier(self, event: dict[str, Any]) -> str | None:
        if str(event.get("type", "")).startswith("blob"):
            identifier = event.get("digest") or event.get("blob_id") or event.get("path")
        else:
            identifier = event.get("path") or event.get("name") or event.get("digest") or event.get("blob_id")
        return str(identifier) if identifier is not None else None

    def _file_identity_fields(self, event: dict[str, Any]) -> dict[str, Any]:
        fields = {}
        for key in ("path", "name", "digest", "blob_id"):
            value = event.get(key)
            if value is not None:
                fields[key] = value
        return fields

    def _event_speed(self, event: dict[str, Any]) -> Any:
        return event.get("bytes_per_second", event.get("bytes_per_s"))

    def _file_position_fields(
        self, item: dict[str, Any], identifier: str | None
    ) -> dict[str, int]:
        if identifier is None:
            return {}
        order = item.get("_planned_file_order", [])
        try:
            index = order.index(identifier) + 1
        except ValueError:
            return {}
        return {"index": index, "total_files": len(order)}

    def _refresh_overall_locked(self, item: dict[str, Any]) -> None:
        progress = item["progress"]
        current = progress.get("current_file") or {}
        completed = sum(
            size for size in item["_completed_files"].values() if isinstance(size, int)
        )
        current_identifier = self._current_identifier(current)
        current_downloaded = current.get("downloaded")
        if current_identifier in item["_completed_files"]:
            current_downloaded = 0
        elif not isinstance(current_downloaded, int):
            current_downloaded = 0
        downloaded = completed + current_downloaded
        total = progress["overall"].get("total")
        percent = downloaded / total * 100 if total and total > 0 else None
        overall = {
            "downloaded": downloaded,
            "total": total,
            "percent": percent,
        }
        speed = current.get("bytes_per_second")
        if isinstance(speed, (int, float)) and speed > 0:
            overall["bytes_per_second"] = float(speed)
            overall["eta_seconds"] = int(max((total or downloaded) - downloaded, 0) / speed) if total else None
        progress["overall"] = overall

    def _current_identifier(self, current: dict[str, Any]) -> str | None:
        identifier = (
            current.get("path")
            or current.get("name")
            or current.get("digest")
            or current.get("blob_id")
        )
        return str(identifier) if identifier is not None else None

    def _complete_progress_locked(self, item: dict[str, Any]) -> None:
        progress = item["progress"]
        total = progress["overall"].get("total")
        if total and total > 0:
            progress["overall"] = {"downloaded": total, "total": total, "percent": 100.0}
        progress["phase"] = "completed"

    def _sum_sizes(self, planned: dict[str, Any]) -> int | None:
        sizes = [size for size in planned.values() if isinstance(size, int)]
        return sum(sizes) if sizes else None

    def _append_message_locked(self, item: dict[str, Any], text: str) -> None:
        item["messages"].append({"timestamp": time.time(), "text": text})

    def _next_waiting_item(self) -> dict[str, Any] | None:
        for item in self._items:
            if item["status"] == "waiting":
                return item
        return None

    def _find_item(self, item_id: str) -> dict[str, Any]:
        for item in self._items:
            if item["id"] == item_id:
                return item
        raise KeyError(item_id)

    def _stop_after_file_requested_locked(self) -> bool:
        with self._condition:
            return self._stop_after_file_requested

    def _clear_worker_locked(self) -> None:
        if self._worker is threading.current_thread():
            self._worker = None
        if not any(item["status"] == "running" for item in self._items):
            self._stop_after_file_requested = False
        self._condition.notify_all()

    def _copy_item(self, item: dict[str, Any]) -> dict[str, Any]:
        copied = {key: value for key, value in item.items() if not key.startswith("_")}
        copied["allow_patterns"] = list(item["allow_patterns"])
        copied["ignore_patterns"] = list(item["ignore_patterns"])
        copied["messages"] = [dict(message) for message in item["messages"]]
        copied["progress"] = {
            "phase": item["progress"]["phase"],
            "overall": dict(item["progress"]["overall"]),
            "current_file": dict(item["progress"]["current_file"])
            if item["progress"]["current_file"] is not None
            else None,
        }
        return copied
