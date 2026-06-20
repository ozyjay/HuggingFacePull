import threading
import time
from unittest import mock

import pytest

import huggingface_pull.queue as queue_module
from huggingface_pull import hub
from huggingface_pull.queue import DownloadQueue


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def test_add_creates_waiting_item(tmp_path):
    queue = DownloadQueue(library_dir=tmp_path, pull_func=lambda *args, **kwargs: None)

    item = queue.add(
        {
            "repo_id": "Qwen/Qwen3-Embedding-0.6B",
            "allow_patterns": ["*.safetensors", "*.json"],
            "ignore_patterns": ["*.bin"],
        }
    )

    assert item["repo_id"] == "Qwen/Qwen3-Embedding-0.6B"
    assert item["revision"] == "main"
    assert item["repo_type"] == "model"
    assert item["allow_patterns"] == ["*.safetensors", "*.json"]
    assert item["ignore_patterns"] == ["*.bin"]
    assert (
        item["canonical_ref"]
        == "model:Qwen/Qwen3-Embedding-0.6B@main?allow=*.json,*.safetensors&ignore=*.bin"
    )
    assert item["deduplicated"] is False
    assert item["status"] == "waiting"
    assert item["error"] is None
    assert item["messages"] == []
    assert item["progress"] == {
        "phase": "waiting",
        "overall": {"downloaded": 0, "total": None, "percent": None},
        "current_file": None,
    }
    assert isinstance(item["id"], str)
    assert isinstance(item["created_at"], float)
    assert isinstance(item["updated_at"], float)


def test_snapshot_includes_installed_models(monkeypatch, tmp_path):
    installed = [{"repo_id": "Qwen/Qwen3", "revision": "main", "size": 12}]
    monkeypatch.setattr(queue_module.hub, "installed_models", lambda library_dir: installed)
    queue = DownloadQueue(
        library_dir=tmp_path,
        endpoint="https://hf.example",
        pull_func=lambda *args, **kwargs: None,
    )
    queue.add({"repo_id": "Qwen/Qwen3"})

    snapshot = queue.snapshot()

    assert snapshot["running"] is False
    assert snapshot["pause_requested"] is False
    assert snapshot["stop_after_file_requested"] is False
    assert snapshot["library_dir"] == str(tmp_path)
    assert snapshot["endpoint"] == "https://hf.example"
    assert snapshot["installed_models"] == installed
    assert len(snapshot["items"]) == 1


def test_add_deduplicates_implicit_same_canonical_payload(tmp_path):
    queue = DownloadQueue(library_dir=tmp_path, pull_func=lambda *args, **kwargs: None)

    first = queue.add({"repo_id": "Qwen/Qwen3"})
    duplicate = queue.add({"repo_id": "Qwen/Qwen3", "revision": "main", "repo_type": "model"})

    assert duplicate["id"] == first["id"]
    assert duplicate["canonical_ref"] == first["canonical_ref"]
    assert duplicate["deduplicated"] is True
    assert len(queue.snapshot()["items"]) == 1


def test_worker_runs_one_item_at_a_time(tmp_path):
    entered = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_pull(ref, **kwargs):
        nonlocal active, max_active
        progress = kwargs["progress"]
        with lock:
            active += 1
            max_active = max(max_active, active)
            entered.append(ref.repo_id)
        progress({"type": "file-progress", "path": f"{ref.repo_id}.bin", "downloaded": 1, "total": 1})
        time.sleep(0.02)
        with lock:
            active -= 1

    queue = DownloadQueue(library_dir=tmp_path, pull_func=fake_pull)
    first = queue.add({"repo_id": "first/model"})
    second = queue.add({"repo_id": "second/model"})

    queue.start()

    assert queue.wait_until_idle(2)
    snapshot = queue.snapshot()
    assert entered == ["first/model", "second/model"]
    assert max_active == 1
    assert [item["status"] for item in snapshot["items"]] == ["completed", "completed"]
    assert snapshot["items"][0]["messages"][0]["text"] == "file-progress"
    assert isinstance(snapshot["items"][0]["messages"][0]["timestamp"], float)
    assert first["status"] == "waiting"
    assert second["status"] == "waiting"


def test_stop_after_file_returns_running_item_to_waiting_for_pre_complete_cooperative_stop(
    tmp_path,
):
    stop_seen = threading.Event()

    def fake_pull(ref, **kwargs):
        progress = kwargs["progress"]
        progress({"type": "manifest-fetch", "repo_id": ref.repo_id})
        assert stop_seen.wait(2)
        if kwargs["stop_after_file"]():
            raise hub.DownloadStoppedAfterFile

    queue = DownloadQueue(library_dir=tmp_path, pull_func=fake_pull)
    item = queue.add({"repo_id": "Qwen/Qwen3"})
    queue.start()

    assert wait_for(lambda: queue.snapshot()["items"][0]["status"] == "running")
    stopped_snapshot = queue.stop_after_current_file()
    stop_seen.set()

    assert stopped_snapshot["pause_requested"] is True
    assert stopped_snapshot["stop_after_file_requested"] is True
    assert queue.wait_until_idle(2)
    [stopped_item] = queue.snapshot()["items"]
    assert stopped_item["id"] == item["id"]
    assert stopped_item["status"] == "waiting"
    assert stopped_item["progress"]["phase"] == "waiting"
    assert stopped_item["messages"][-1]["text"] == "stopped after current snapshot"


def test_stop_after_file_keeps_model_complete_item_completed_for_current_hub_semantics(
    tmp_path,
):
    stop_seen = threading.Event()

    def fake_pull(ref, **kwargs):
        progress = kwargs["progress"]
        progress({"type": "manifest-fetch", "repo_id": ref.repo_id})
        assert stop_seen.wait(2)
        progress({"type": "model-complete", "repo_id": ref.repo_id})
        if kwargs["stop_after_file"]():
            raise hub.DownloadStoppedAfterFile

    queue = DownloadQueue(library_dir=tmp_path, pull_func=fake_pull)
    queue.add({"repo_id": "Qwen/Qwen3"})
    queue.start()

    assert wait_for(lambda: queue.snapshot()["items"][0]["status"] == "running")
    queue.stop_after_current_file()
    stop_seen.set()

    assert queue.wait_until_idle(2)
    snapshot = queue.snapshot()
    [completed_item] = snapshot["items"]
    assert completed_item["status"] == "completed"
    assert completed_item["progress"]["phase"] == "completed"
    assert snapshot["stop_after_file_requested"] is False


def test_failure_marks_item_failed_and_retry_resets_to_waiting(monkeypatch, tmp_path):
    attempts = []
    log_events = []
    monkeypatch.setattr(queue_module, "write_log", lambda message, **fields: log_events.append((message, fields)))

    def fake_pull(ref, **kwargs):
        attempts.append(ref.repo_id)
        kwargs["progress"]({"type": "file-progress", "path": "weights.bin", "downloaded": 3, "total": 10})
        raise RuntimeError("download broke")

    queue = DownloadQueue(library_dir=tmp_path, pull_func=fake_pull)
    item = queue.add({"repo_id": "broken/model"})

    queue.start()

    assert queue.wait_until_idle(2)
    failed = queue.snapshot()["items"][0]
    retried = queue.retry(item["id"])

    assert attempts == ["broken/model"]
    assert failed["status"] == "failed"
    assert failed["error"] == "download broke"
    assert failed["progress"]["phase"] == "failed"
    assert failed["messages"][-1]["text"] == "failed: download broke"
    assert log_events == [("download failed", {"repo_id": "broken/model", "error": mock.ANY})]
    assert retried["status"] == "waiting"
    assert retried["error"] is None
    assert retried["messages"] == []
    assert retried["progress"] == {
        "phase": "waiting",
        "overall": {"downloaded": 0, "total": None, "percent": None},
        "current_file": None,
    }


def test_logging_failure_does_not_leave_download_failure_running(monkeypatch, tmp_path):
    monkeypatch.setattr(
        queue_module,
        "write_log",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("log disk full")),
    )

    def fake_pull(ref, **kwargs):
        raise RuntimeError("download broke")

    queue = DownloadQueue(library_dir=tmp_path, pull_func=fake_pull)
    queue.add({"repo_id": "broken/model"})

    queue.start()

    assert queue.wait_until_idle(2)
    [failed] = queue.snapshot()["items"]
    assert failed["status"] == "failed"
    assert failed["progress"]["phase"] == "failed"
    assert "download broke" in failed["error"]


def test_remove_rejects_running_item_and_missing_item(tmp_path):
    release_download = threading.Event()

    def fake_pull(ref, **kwargs):
        release_download.wait(2)

    queue = DownloadQueue(library_dir=tmp_path, pull_func=fake_pull)
    item = queue.add({"repo_id": "Qwen/Qwen3"})
    queue.start()

    assert wait_for(lambda: queue.snapshot()["items"][0]["status"] == "running")

    with pytest.raises(ValueError, match="Running items cannot be removed"):
        queue.remove(item["id"])
    with pytest.raises(KeyError):
        queue.remove("missing")

    release_download.set()
    assert queue.wait_until_idle(2)


def test_pause_after_current_pauses_before_next_waiting_item(tmp_path):
    release_first = threading.Event()
    entered = []

    def fake_pull(ref, **kwargs):
        entered.append(ref.repo_id)
        if ref.repo_id == "first/model":
            release_first.wait(2)

    queue = DownloadQueue(library_dir=tmp_path, pull_func=fake_pull)
    queue.add({"repo_id": "first/model"})
    queue.add({"repo_id": "second/model"})
    queue.start()

    assert wait_for(lambda: queue.snapshot()["items"][0]["status"] == "running")
    queue.pause_after_current()
    release_first.set()

    assert queue.wait_until_idle(2)
    snapshot = queue.snapshot()
    assert entered == ["first/model"]
    assert [item["status"] for item in snapshot["items"]] == ["completed", "waiting"]
    assert snapshot["pause_requested"] is True


def test_progress_tracks_manifest_model_plan_file_progress_and_model_complete(tmp_path):
    queue = DownloadQueue(library_dir=tmp_path, pull_func=lambda *args, **kwargs: None)
    item = queue.add({"repo_id": "Qwen/Qwen3"})

    queue._record_progress(item["id"], {"type": "manifest-fetch", "repo_id": "Qwen/Qwen3"})
    queue._record_progress(
        item["id"],
        {
            "type": "model-plan",
            "total_bytes": 10,
            "files": [
                {"path": "config.json", "size": 4},
                {"path": "weights.safetensors", "size": 6},
            ],
        },
    )
    queue._record_progress(
        item["id"],
        {
            "type": "file-progress",
            "path": "config.json",
            "downloaded": 2,
            "total": 4,
            "percent": 50.0,
            "bytes_per_second": 8.0,
            "eta_seconds": 1,
            "line": "50.0% 2B/4B 8B/s eta 0m01s",
        },
    )

    snapshot = queue.snapshot()
    progress = snapshot["items"][0]["progress"]
    assert progress["phase"] == "downloading"
    assert progress["overall"] == {
        "downloaded": 2,
        "total": 10,
        "percent": 20.0,
        "bytes_per_second": 8.0,
        "eta_seconds": 1,
    }
    assert progress["current_file"] == {
        "path": "config.json",
        "index": 1,
        "total_files": 2,
        "downloaded": 2,
        "total": 4,
        "percent": 50.0,
        "bytes_per_second": 8.0,
        "eta_seconds": 1,
        "line": "50.0% 2B/4B 8B/s eta 0m01s",
    }

    queue._record_progress(
        item["id"],
        {
            "type": "file-complete",
            "path": "config.json",
            "downloaded": 4,
            "total": 4,
            "percent": 100.0,
        },
    )
    queue._record_progress(item["id"], {"type": "model-complete", "repo_id": "Qwen/Qwen3"})

    completed = queue.snapshot()["items"][0]
    assert completed["progress"]["phase"] == "completed"
    assert completed["progress"]["overall"] == {"downloaded": 10, "total": 10, "percent": 100.0}
    assert [message["text"] for message in completed["messages"]] == [
        "manifest-fetch",
        "model-plan",
        "file-progress",
        "file-complete",
        "model-complete",
    ]


def test_concurrent_start_calls_reserve_single_worker_slot(tmp_path):
    active = 0
    max_active = 0
    lock = threading.Lock()
    first_started = threading.Event()
    release_downloads = threading.Event()

    def fake_pull(ref, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        first_started.set()
        release_downloads.wait(2)
        with lock:
            active -= 1

    original_thread = queue_module.threading.Thread

    queue = DownloadQueue(library_dir=tmp_path, pull_func=fake_pull)
    queue.add({"repo_id": "first/model"})
    queue.add({"repo_id": "second/model"})

    starters = [original_thread(target=queue.start) for _ in range(8)]
    for starter in starters:
        starter.start()
    for starter in starters:
        starter.join()

    assert first_started.wait(2)
    release_downloads.set()
    assert queue.wait_until_idle(2)
    assert max_active == 1


def test_add_and_start_during_worker_shutdown_gap_is_not_lost(tmp_path):
    calls = []
    shutdown_gap_open = threading.Event()
    release_shutdown = threading.Event()

    def fake_pull(ref, **kwargs):
        calls.append(ref.repo_id)

    queue = DownloadQueue(library_dir=tmp_path, pull_func=fake_pull)
    original_condition = queue._condition
    original_next_waiting_item = queue._next_waiting_item

    class PausingCondition:
        def __enter__(self):
            return original_condition.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            result = original_condition.__exit__(exc_type, exc_value, traceback)
            if getattr(self, "pause_on_worker_exit", False):
                self.pause_on_worker_exit = False
                shutdown_gap_open.set()
                release_shutdown.wait(2)
            return result

        def notify_all(self):
            original_condition.notify_all()

        def wait(self, timeout=None):
            return original_condition.wait(timeout)

    pausing_condition = PausingCondition()

    def next_waiting_item_with_shutdown_pause():
        item = original_next_waiting_item()
        if item is None:
            pausing_condition.pause_on_worker_exit = True
        return item

    queue._condition = pausing_condition
    queue._next_waiting_item = next_waiting_item_with_shutdown_pause
    queue.add({"repo_id": "first/model"})

    queue.start()
    assert shutdown_gap_open.wait(2)
    queue.add({"repo_id": "second/model"})
    queue.start()
    release_shutdown.set()

    assert queue.wait_until_idle(2)
    assert calls == ["first/model", "second/model"]
