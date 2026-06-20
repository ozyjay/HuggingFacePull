import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import huggingface_pull
import huggingface_pull.hub as hub


def install_fake_file_download(monkeypatch, files, download_func=None, endpoint="https://huggingface.co"):
    class FakeApi:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def model_info(self, repo_id, revision, files_metadata, token):
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(
                        rfilename=file["path"],
                        size=file.get("size"),
                        blob_id=file.get("blob_id"),
                    )
                    for file in files
                ]
            )

    def default_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        path = local_dir / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * int(next(file.get("size") or 0 for file in files if file["path"] == kwargs["filename"])))
        return str(path)

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "hf_hub_download", download_func or default_download)


def test_canonical_ref_normalises_revision_repo_type_and_filters():
    ref = hub.HubRef(
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        revision="main",
        repo_type="model",
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.bin", "*.h5"],
    )

    assert (
        hub.canonical_ref(ref)
        == "model:Qwen/Qwen3-Embedding-0.6B@main?allow=*.json,*.safetensors&ignore=*.bin,*.h5"
    )


def test_metadata_path_uses_safe_repo_directory_and_revision(tmp_path):
    ref = hub.HubRef(repo_id="Qwen/Qwen3", revision="refs-pr-1")

    assert (
        hub.metadata_path(tmp_path, ref)
        == tmp_path / "Qwen--Qwen3" / "refs-pr-1" / ".huggingfacepull.json"
    )


def test_metadata_path_encodes_slash_revision_as_single_safe_directory(tmp_path):
    ref = hub.HubRef(repo_id="Qwen/Qwen3", revision="refs/pr/1")

    marker = hub.metadata_path(tmp_path, ref)

    assert marker == tmp_path / "Qwen--Qwen3" / "refs--pr--1" / ".huggingfacepull.json"
    assert marker.parent.parent == tmp_path / "Qwen--Qwen3"
    assert marker.resolve().relative_to(tmp_path.resolve())


def test_metadata_path_traversal_revision_cannot_escape_library(tmp_path):
    ref = hub.HubRef(repo_id="Qwen/Qwen3", revision="../outside")

    marker = hub.metadata_path(tmp_path, ref)

    assert marker == tmp_path / "Qwen--Qwen3" / "..--outside" / ".huggingfacepull.json"
    assert marker.resolve().relative_to(tmp_path.resolve())


def test_package_exports_callback_types():
    assert huggingface_pull.ProgressCallback is hub.ProgressCallback
    assert huggingface_pull.StopAfterFileCallback is hub.StopAfterFileCallback


def test_installed_models_reads_metadata_skips_malformed_and_sorts(tmp_path):
    first = tmp_path / "zeta--Repo" / "dev" / ".huggingfacepull.json"
    second = tmp_path / "Alpha--Repo" / "main" / ".huggingfacepull.json"
    malformed = tmp_path / "bad--Repo" / "main" / ".huggingfacepull.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    malformed.parent.mkdir(parents=True)
    first.write_text(
        json.dumps(
            {
                "repo_id": "zeta/Repo",
                "revision": "dev",
                "repo_type": "model",
                "snapshot_path": "/tmp/zeta",
                "size": 12,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "repo_id": "Alpha/Repo",
                "revision": "main",
                "repo_type": "model",
                "snapshot_path": "/tmp/alpha",
                "size": 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    malformed.write_text("{not json", encoding="utf-8")

    assert hub.installed_models(tmp_path) == [
        {
            "repo_id": "Alpha/Repo",
            "revision": "main",
            "repo_type": "model",
            "snapshot_path": "/tmp/alpha",
            "size": 8,
        },
        {
            "repo_id": "zeta/Repo",
            "revision": "dev",
            "repo_type": "model",
            "snapshot_path": "/tmp/zeta",
            "size": 12,
        },
    ]


def test_cached_hub_models_reads_huggingface_cache_repos(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    model = cache / "models--Qwen--Qwen2.5-0.5B"
    snapshot = model / "snapshots" / "abc123"
    ref = model / "refs" / "main"
    snapshot.mkdir(parents=True)
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (cache / "datasets--user--data" / "snapshots" / "def456").mkdir(parents=True)
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))

    assert hub.cached_hub_models() == [
        {
            "repo_id": "Qwen/Qwen2.5-0.5B",
            "revision": "main",
            "repo_type": "model",
            "snapshot_path": str(snapshot),
            "source": "huggingface_cache",
        }
    ]


def test_cached_hub_models_skips_empty_snapshot_directories(tmp_path, monkeypatch):
    log_events = []
    cache = tmp_path / "hub"
    model = cache / "models--Qwen--Qwen2.5-1.5B-Instruct"
    snapshot = model / "snapshots" / "abc123"
    ref = model / "refs" / "main"
    snapshot.mkdir(parents=True)
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123", encoding="utf-8")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(
        hub,
        "write_log",
        lambda message, **fields: log_events.append((message, fields)),
        raising=False,
    )

    assert hub.cached_hub_models() == []
    assert log_events == [
        (
            "cache snapshot skipped",
            {
                "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "revision": "main",
                "snapshot_path": snapshot,
                "reason": "empty",
            },
        )
    ]


def test_cached_hub_models_logs_repeated_skipped_snapshot_once(tmp_path, monkeypatch):
    log_events = []
    cache = tmp_path / "hub"
    model = cache / "models--Qwen--Qwen2.5-1.5B-Instruct"
    snapshot = model / "snapshots" / "abc123"
    ref = model / "refs" / "main"
    snapshot.mkdir(parents=True)
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123", encoding="utf-8")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(
        hub,
        "write_log",
        lambda message, **fields: log_events.append((message, fields)),
        raising=False,
    )
    monkeypatch.setattr(hub, "_LOGGED_SKIPPED_CACHE_SNAPSHOTS", set())

    assert hub.cached_hub_models() == []
    assert hub.cached_hub_models() == []
    assert [message for message, _ in log_events] == ["cache snapshot skipped"]


def test_cached_hub_models_skips_snapshots_with_broken_blob_links(tmp_path, monkeypatch):
    log_events = []
    cache = tmp_path / "hub"
    model = cache / "models--Qwen--Qwen2.5-1.5B-Instruct"
    snapshot = model / "snapshots" / "abc123"
    ref = model / "refs" / "main"
    snapshot.mkdir(parents=True)
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to("../../blobs/missing")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(
        hub,
        "write_log",
        lambda message, **fields: log_events.append((message, fields)),
        raising=False,
    )

    assert hub.cached_hub_models() == []
    assert log_events == [
        (
            "cache snapshot skipped",
            {
                "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "revision": "main",
                "snapshot_path": snapshot,
                "reason": "broken_symlink",
                "path": snapshot / "model.safetensors",
            },
        )
    ]


def test_cached_hub_models_skips_snapshots_with_valid_metadata_but_missing_weights(
    tmp_path, monkeypatch
):
    cache = tmp_path / "hub"
    model = cache / "models--Qwen--Qwen2.5-1.5B-Instruct"
    snapshot = model / "snapshots" / "abc123"
    ref = model / "refs" / "main"
    snapshot.mkdir(parents=True)
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to("../../blobs/missing")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))

    assert hub.cached_hub_models() == []


def test_search_models_empty_query_returns_available_without_api(monkeypatch):
    monkeypatch.setattr(
        hub,
        "HfApi",
        lambda endpoint=None: pytest.fail("empty search should not create an API client"),
    )

    assert hub.search_models("   ") == {"available": True, "results": [], "error": None}


def test_search_models_maps_hf_model_info(monkeypatch):
    class FakeApi:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def list_models(self, search, limit, sort, token):
            assert self.endpoint == "https://hf.example"
            assert search == "qwen"
            assert limit == 20
            assert sort == "downloads"
            assert token == "secret"
            return [
                SimpleNamespace(
                    modelId="Qwen/Qwen3",
                    pipeline_tag="text-generation",
                    tags=["transformers", "safetensors"],
                    downloads=123,
                    likes=45,
                )
            ]

    monkeypatch.setattr(hub, "HfApi", FakeApi)

    result = hub.search_models("qwen", endpoint="https://hf.example", token="secret")

    assert result == {
        "available": True,
        "results": [
            {
                "repo_id": "Qwen/Qwen3",
                "name": "Qwen/Qwen3",
                "pipeline_tag": "text-generation",
                "tags": ["transformers", "safetensors"],
                "downloads": 123,
                "likes": 45,
            }
        ],
        "error": None,
    }


def test_search_models_uses_list_models_signature_without_direction(monkeypatch):
    class FakeApi:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def list_models(self, *, search, limit, sort, token):
            assert self.endpoint == "https://hf.example"
            assert search == "Qwen2.5-1.5B-Instruct"
            assert limit == 20
            assert sort == "downloads"
            assert token == "secret"
            return [SimpleNamespace(modelId="Qwen/Qwen2.5-1.5B-Instruct")]

    monkeypatch.setattr(hub, "HfApi", FakeApi)

    result = hub.search_models(
        "Qwen2.5-1.5B-Instruct",
        endpoint="https://hf.example",
        token="secret",
    )

    assert result["available"] is True
    assert result["results"][0]["repo_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert result["error"] is None


def test_search_models_returns_unavailable_on_api_error(monkeypatch):
    class FakeApi:
        def __init__(self, endpoint):
            pass

        def list_models(self, **kwargs):
            raise RuntimeError("hub unavailable")

    monkeypatch.setattr(hub, "HfApi", FakeApi)

    result = hub.search_models("qwen")

    assert result["available"] is False
    assert result["results"] == []
    assert "hub unavailable" in result["error"]


def test_repo_files_maps_model_siblings(monkeypatch):
    class FakeApi:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def model_info(self, repo_id, revision, files_metadata, token):
            assert self.endpoint == "https://hf.example"
            assert repo_id == "Qwen/Qwen3"
            assert revision == "main"
            assert files_metadata is True
            assert token == "secret"
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=20, blob_id="abc"),
                    SimpleNamespace(rfilename="model.safetensors", size=120, blob_id="def"),
                ]
            )

    monkeypatch.setattr(hub, "HfApi", FakeApi)

    assert hub.repo_files(
        hub.HubRef(repo_id="Qwen/Qwen3"),
        endpoint="https://hf.example",
        token="secret",
    ) == {
        "repo_id": "Qwen/Qwen3",
        "revision": "main",
        "files": [
            {"path": "config.json", "size": 20, "blob_id": "abc"},
            {"path": "model.safetensors", "size": 120, "blob_id": "def"},
        ],
    }


def test_pull_snapshot_dry_run_emits_progress_and_skips_download(monkeypatch, tmp_path):
    monkeypatch.setattr(
        hub,
        "hf_hub_download",
        lambda **kwargs: pytest.fail("dry run should not download"),
    )
    events = []
    ref = hub.HubRef(repo_id="Qwen/Qwen3")

    target = hub.pull_snapshot(ref, library_dir=tmp_path, dry_run=True, progress=events.append)

    assert target == tmp_path / "Qwen--Qwen3" / "main"
    assert events == [
        {"type": "manifest-fetch", "repo_id": "Qwen/Qwen3", "revision": "main"},
        {"type": "model-complete", "repo_id": "Qwen/Qwen3", "dry_run": True},
    ]
    assert not hub.metadata_path(tmp_path, ref).exists()


def test_pull_snapshot_downloads_and_writes_metadata_without_network(monkeypatch, tmp_path):
    calls = []

    class FakeApi:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def model_info(self, repo_id, revision, files_metadata, token):
            assert self.endpoint == "https://hf.example"
            assert repo_id == "Qwen/Qwen3"
            assert revision == "v1"
            assert files_metadata is True
            assert token == "secret"
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=2, blob_id="cfg"),
                    SimpleNamespace(rfilename="weights.bin", size=7, blob_id="weights"),
                    SimpleNamespace(rfilename="weights.safetensors", size=9, blob_id="safe"),
                ]
            )

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / kwargs["filename"]).write_bytes(b"1234567")
        return str(local_dir / kwargs["filename"])

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "hf_hub_download", fake_hf_hub_download)
    events = []
    ref = hub.HubRef(
        repo_id="Qwen/Qwen3",
        revision="v1",
        allow_patterns=["*.bin"],
    )

    snapshot_path = hub.pull_snapshot(
        ref,
        library_dir=tmp_path,
        endpoint="https://hf.example",
        token="secret",
        progress=events.append,
    )

    target = tmp_path / "Qwen--Qwen3" / "v1"
    assert snapshot_path == target
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3",
            "filename": "weights.bin",
            "revision": "v1",
            "repo_type": None,
            "local_dir": target,
            "endpoint": "https://hf.example",
            "token": "secret",
            "tqdm_class": calls[0]["tqdm_class"],
        }
    ]
    assert issubclass(calls[0]["tqdm_class"], hub.hf_tqdm)
    metadata = json.loads(hub.metadata_path(tmp_path, ref).read_text(encoding="utf-8"))
    assert metadata == {
        "repo_id": "Qwen/Qwen3",
        "revision": "v1",
        "repo_type": "model",
        "snapshot_path": str(target),
        "size": 7,
    }
    assert events == [
        {"type": "manifest-fetch", "repo_id": "Qwen/Qwen3", "revision": "v1"},
        {
            "type": "model-plan",
            "repo_id": "Qwen/Qwen3",
            "revision": "v1",
            "total_bytes": 7,
            "files": [{"path": "weights.bin", "size": 7, "blob_id": "weights"}],
        },
        {
            "type": "file-start",
            "repo_id": "Qwen/Qwen3",
            "path": "weights.bin",
            "size": 7,
            "blob_id": "weights",
            "resume_at": 0,
        },
        {
            "type": "file-complete",
            "repo_id": "Qwen/Qwen3",
            "path": "weights.bin",
            "downloaded": 7,
            "total": 7,
            "percent": 100.0,
            "blob_id": "weights",
        },
        {"type": "model-complete", "repo_id": "Qwen/Qwen3", "snapshot_path": str(target)},
    ]


def test_pull_snapshot_stops_after_current_file_before_next_download(monkeypatch, tmp_path):
    calls = []
    events = []

    class FakeApi:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def model_info(self, repo_id, revision, files_metadata, token):
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=2, blob_id="cfg"),
                    SimpleNamespace(rfilename="model.safetensors", size=8, blob_id="weights"),
                ]
            )

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs["filename"])
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / kwargs["filename"]).parent.mkdir(parents=True, exist_ok=True)
        (local_dir / kwargs["filename"]).write_bytes(b"x" * (2 if kwargs["filename"] == "config.json" else 8))
        return str(local_dir / kwargs["filename"])

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "hf_hub_download", fake_hf_hub_download)
    ref = hub.HubRef(repo_id="Qwen/Qwen3")

    with pytest.raises(hub.DownloadStoppedAfterFile):
        hub.pull_snapshot(
            ref,
            library_dir=tmp_path,
            progress=events.append,
            stop_after_file=lambda: bool(calls),
        )

    assert calls == ["config.json"]
    assert [event["type"] for event in events] == [
        "manifest-fetch",
        "model-plan",
        "file-start",
        "file-complete",
    ]
    assert not hub.metadata_path(tmp_path, ref).exists()


def test_pull_snapshot_disables_xet_during_download(monkeypatch, tmp_path):
    seen = []

    def fake_hf_hub_download(**kwargs):
        seen.append(os.environ.get("HF_HUB_DISABLE_XET"))
        local_dir = Path(kwargs["local_dir"])
        path = local_dir / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
        return str(path)

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    install_fake_file_download(
        monkeypatch,
        [{"path": "weights.bin", "size": 4, "blob_id": "weights"}],
        fake_hf_hub_download,
    )

    hub.pull_snapshot(hub.HubRef(repo_id="Qwen/Qwen3"), library_dir=tmp_path)

    assert seen == ["1"]
    assert os.environ.get("HF_HUB_DISABLE_XET") is None


def test_pull_snapshot_restores_existing_xet_setting(monkeypatch, tmp_path):
    seen = []

    def fake_hf_hub_download(**kwargs):
        seen.append(os.environ.get("HF_HUB_DISABLE_XET"))
        local_dir = Path(kwargs["local_dir"])
        path = local_dir / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
        return str(path)

    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    install_fake_file_download(
        monkeypatch,
        [{"path": "weights.bin", "size": 4, "blob_id": "weights"}],
        fake_hf_hub_download,
    )

    hub.pull_snapshot(hub.HubRef(repo_id="Qwen/Qwen3"), library_dir=tmp_path)

    assert seen == ["1"]
    assert os.environ.get("HF_HUB_DISABLE_XET") == "0"


def test_pull_snapshot_emits_byte_progress_from_hub_tqdm(monkeypatch, tmp_path):
    def fake_hf_hub_download(**kwargs):
        progress_bar = kwargs["tqdm_class"](
            total=10,
            initial=2,
            unit="B",
            desc="Downloading",
        )
        progress_bar.update(3)
        progress_bar.total += 5
        progress_bar.refresh()
        progress_bar.update(10)
        progress_bar.close()

        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / kwargs["filename"]).write_bytes(b"123")
        return str(local_dir / kwargs["filename"])

    install_fake_file_download(
        monkeypatch,
        [{"path": "weights.bin", "size": 15, "blob_id": "weights"}],
        fake_hf_hub_download,
    )
    events = []

    hub.pull_snapshot(
        hub.HubRef(repo_id="Qwen/Qwen3"),
        library_dir=tmp_path,
        progress=events.append,
    )

    progress_events = [event for event in events if event["type"] == "file-progress"]
    assert progress_events == [
        {
            "type": "file-progress",
            "repo_id": "Qwen/Qwen3",
            "path": "weights.bin",
            "downloaded": 2,
            "total": 10,
            "percent": 20.0,
            "bytes_per_second": None,
            "eta_seconds": None,
            "blob_id": "weights",
        },
        {
            "type": "file-progress",
            "repo_id": "Qwen/Qwen3",
            "path": "weights.bin",
            "downloaded": 5,
            "total": 15,
            "percent": 33.33333333333333,
            "bytes_per_second": None,
            "eta_seconds": None,
            "blob_id": "weights",
        },
        {
            "type": "file-progress",
            "repo_id": "Qwen/Qwen3",
            "path": "weights.bin",
            "downloaded": 15,
            "total": 15,
            "percent": 100.0,
            "bytes_per_second": None,
            "eta_seconds": None,
            "blob_id": "weights",
        },
    ]


def test_pull_snapshot_throttles_rapid_byte_progress(monkeypatch, tmp_path):
    ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(hub.time, "monotonic", lambda: next(ticks, 0.4))

    def fake_hf_hub_download(**kwargs):
        progress_bar = kwargs["tqdm_class"](total=100, initial=0, unit="B")
        for _ in range(4):
            progress_bar.update(10)
        progress_bar.close()

        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / kwargs["filename"]).write_bytes(b"x" * 40)
        return str(local_dir / kwargs["filename"])

    install_fake_file_download(
        monkeypatch,
        [{"path": "weights.bin", "size": 100, "blob_id": "weights"}],
        fake_hf_hub_download,
    )
    events = []

    hub.pull_snapshot(
        hub.HubRef(repo_id="Qwen/Qwen3"),
        library_dir=tmp_path,
        progress=events.append,
    )

    progress_events = [event for event in events if event["type"] == "file-progress"]
    assert progress_events == [
        {
            "type": "file-progress",
            "repo_id": "Qwen/Qwen3",
            "path": "weights.bin",
            "downloaded": 0,
            "total": 100,
            "percent": 0.0,
            "bytes_per_second": None,
            "eta_seconds": None,
            "blob_id": "weights",
        },
        {
            "type": "file-progress",
            "repo_id": "Qwen/Qwen3",
            "path": "weights.bin",
            "downloaded": 40,
            "total": 100,
            "percent": 40.0,
            "bytes_per_second": None,
            "eta_seconds": None,
            "blob_id": "weights",
        },
    ]


def test_pull_snapshot_stops_after_file_even_when_requested_during_progress(monkeypatch, tmp_path):
    stop_requested = False

    def fake_hf_hub_download(**kwargs):
        progress_bar = kwargs["tqdm_class"](total=100, initial=0, unit="B")
        progress_bar.update(10)

        nonlocal stop_requested
        stop_requested = True
        progress_bar.update(10)

        local_dir = Path(kwargs["local_dir"])
        path = local_dir / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 20)
        return str(path)

    install_fake_file_download(
        monkeypatch,
        [{"path": "weights.bin", "size": 100, "blob_id": "weights"}],
        fake_hf_hub_download,
    )
    ref = hub.HubRef(repo_id="Qwen/Qwen3")

    with pytest.raises(hub.DownloadStoppedAfterFile):
        hub.pull_snapshot(
            ref,
            library_dir=tmp_path,
            progress=lambda event: None,
            stop_after_file=lambda: stop_requested,
        )

    assert not hub.metadata_path(tmp_path, ref).exists()


def test_pull_snapshot_raises_stop_after_file_after_metadata(monkeypatch, tmp_path):
    def fake_hf_hub_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        path = local_dir / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
        return str(path)

    install_fake_file_download(
        monkeypatch,
        [{"path": "weights.bin", "size": 4, "blob_id": "weights"}],
        fake_hf_hub_download,
    )
    ref = hub.HubRef(repo_id="Qwen/Qwen3")

    with pytest.raises(hub.DownloadStoppedAfterFile):
        hub.pull_snapshot(ref, library_dir=tmp_path, stop_after_file=lambda: True)

    assert not hub.metadata_path(tmp_path, ref).exists()


def test_remove_installed_model_removes_revision_directory_and_raises_for_missing(tmp_path):
    ref = hub.HubRef(repo_id="Qwen/Qwen3", revision="main")
    marker = hub.metadata_path(tmp_path, ref)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    (marker.parent / "weights.bin").write_bytes(b"data")

    hub.remove_installed_model(tmp_path, ref)

    assert not marker.parent.exists()
    with pytest.raises(KeyError):
        hub.remove_installed_model(tmp_path, ref)


def test_remove_installed_model_requires_metadata_marker(tmp_path):
    ref = hub.HubRef(repo_id="Qwen/Qwen3", revision="main")
    model_dir = hub.metadata_path(tmp_path, ref).parent
    model_dir.mkdir(parents=True)
    payload = model_dir / "weights.bin"
    payload.write_bytes(b"data")

    with pytest.raises(KeyError):
        hub.remove_installed_model(tmp_path, ref)

    assert model_dir.exists()
    assert payload.exists()


def test_remove_installed_model_traversal_revision_cannot_delete_outside_library(tmp_path):
    ref = hub.HubRef(repo_id="Qwen/Qwen3", revision="../outside")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_payload = outside / "keep.bin"
    outside_payload.write_bytes(b"keep")
    marker = hub.metadata_path(tmp_path, ref)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")

    hub.remove_installed_model(tmp_path, ref)

    assert outside.exists()
    assert outside_payload.exists()
    assert not marker.parent.exists()


def test_cleanup_library_reports_and_deletes_stale_partials_only_when_enabled(monkeypatch, tmp_path):
    old_partial = tmp_path / "Qwen--Qwen3" / "main" / "weights.bin.incomplete"
    recent_partial = tmp_path / "Qwen--Qwen3" / "main" / "tokenizer.tmp"
    normal_file = tmp_path / "Qwen--Qwen3" / "main" / "config.json"
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    old_partial.parent.mkdir(parents=True)
    old_partial.write_bytes(b"old")
    recent_partial.write_bytes(b"new")
    normal_file.write_bytes(b"ok")
    old_time = time.time() - 10 * 24 * 60 * 60
    os.utime(old_partial, (old_time, old_time))

    disabled = hub.cleanup_library(tmp_path, include_partials=False, older_than_days=7)
    dry_run = hub.cleanup_library(tmp_path, include_partials=True, older_than_days=7)

    assert disabled["stale_partial_count"] == 0
    assert dry_run["dry_run"] is True
    assert dry_run["stale_partial_count"] == 1
    assert dry_run["stale_partials"][0]["path"] == str(old_partial)
    assert old_partial.exists()

    deleted = hub.cleanup_library(tmp_path, delete=True, include_partials=True, older_than_days=7)

    assert deleted["dry_run"] is False
    assert deleted["deleted"] == [str(old_partial)]
    assert not old_partial.exists()
    assert recent_partial.exists()
    assert normal_file.exists()


def test_cleanup_library_includes_huggingface_cache_partials(monkeypatch, tmp_path):
    log_events = []
    library_root = tmp_path / "library"
    library_partial = library_root / "Qwen--Qwen3" / "main" / "weights.bin.incomplete"
    cache_root = tmp_path / "hf-cache"
    cache_partial = cache_root / "models--Qwen--Qwen3" / "blobs" / "abc.123.incomplete"
    lock_file = cache_root / ".locks" / "models--Qwen--Qwen3" / "abc.lock"
    library_partial.parent.mkdir(parents=True)
    cache_partial.parent.mkdir(parents=True)
    lock_file.parent.mkdir(parents=True)
    library_partial.write_bytes(b"library")
    cache_partial.write_bytes(b"cache")
    lock_file.write_bytes(b"lock")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache_root))
    monkeypatch.setattr(
        hub,
        "write_log",
        lambda message, **fields: log_events.append((message, fields)),
        raising=False,
    )

    dry_run = hub.cleanup_library(library_root, include_partials=True, older_than_days=0)

    assert dry_run["stale_partial_count"] == 2
    assert [item["path"] for item in dry_run["stale_partials"]] == [
        str(library_partial),
        str(cache_partial),
    ]
    assert [item["source"] for item in dry_run["stale_partials"]] == [
        "library",
        "huggingface_cache",
    ]
    assert str(lock_file) not in [item["path"] for item in dry_run["stale_partials"]]

    deleted = hub.cleanup_library(library_root, delete=True, include_partials=True, older_than_days=0)

    assert deleted["deleted"] == [str(library_partial), str(cache_partial)]
    assert not library_partial.exists()
    assert not cache_partial.exists()
    assert lock_file.exists()
    assert log_events == [
        (
            "cleanup scanned",
            {
                "library_dir": library_root,
                "include_partials": True,
                "older_than_days": 0,
                "stale_partial_count": 2,
                "incomplete_snapshot_count": 0,
                "delete": False,
            },
        ),
        (
            "cleanup scanned",
            {
                "library_dir": library_root,
                "include_partials": True,
                "older_than_days": 0,
                "stale_partial_count": 2,
                "incomplete_snapshot_count": 0,
                "delete": True,
            },
        ),
        ("cleanup partial deleted", {"path": library_partial, "source": "library"}),
        ("cleanup partial deleted", {"path": cache_partial, "source": "huggingface_cache"}),
    ]


def test_cleanup_library_reports_and_deletes_incomplete_library_snapshots(
    monkeypatch, tmp_path
):
    log_events = []
    library_root = tmp_path / "library"
    snapshot = library_root / "Qwen--Qwen2.5-1.5B-Instruct" / "main"
    metadata_dir = snapshot / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (metadata_dir / "config.json.metadata").write_text("{}", encoding="utf-8")
    (metadata_dir / "model.safetensors.lock").write_text("", encoding="utf-8")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(tmp_path / "hf-cache"))
    monkeypatch.setattr(
        hub,
        "write_log",
        lambda message, **fields: log_events.append((message, fields)),
        raising=False,
    )

    dry_run = hub.cleanup_library(library_root, include_partials=True, older_than_days=0)

    assert dry_run["dry_run"] is True
    assert dry_run["incomplete_snapshot_count"] == 1
    assert dry_run["incomplete_snapshots"] == [
        {
            "path": str(snapshot),
            "repo_dir": "Qwen--Qwen2.5-1.5B-Instruct",
            "revision": "main",
            "size": hub.directory_size(snapshot),
            "reason": "missing_metadata_marker",
            "evidence": [
                str(metadata_dir / "config.json.metadata"),
                str(metadata_dir / "model.safetensors.lock"),
            ],
            "source": "library",
        }
    ]
    assert snapshot.exists()

    deleted = hub.cleanup_library(library_root, delete=True, include_partials=True, older_than_days=0)

    assert deleted["deleted_snapshots"] == [str(snapshot)]
    assert not snapshot.exists()
    assert log_events == [
        (
            "cleanup scanned",
            {
                "library_dir": library_root,
                "include_partials": True,
                "older_than_days": 0,
                "stale_partial_count": 0,
                "incomplete_snapshot_count": 1,
                "delete": False,
            },
        ),
        (
            "cleanup scanned",
            {
                "library_dir": library_root,
                "include_partials": True,
                "older_than_days": 0,
                "stale_partial_count": 0,
                "incomplete_snapshot_count": 1,
                "delete": True,
            },
        ),
        (
            "cleanup incomplete snapshot deleted",
            {"path": snapshot, "source": "library", "reason": "missing_metadata_marker"},
        ),
    ]


def test_cleanup_library_keeps_completed_snapshots_with_huggingface_metadata(
    monkeypatch, tmp_path
):
    library_root = tmp_path / "library"
    snapshot = library_root / "Qwen--Qwen2.5-1.5B-Instruct" / "main"
    metadata_dir = snapshot / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    (snapshot / ".huggingfacepull.json").write_text("{}", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (metadata_dir / "config.json.metadata").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(tmp_path / "hf-cache"))

    report = hub.cleanup_library(library_root, delete=True, include_partials=True, older_than_days=0)

    assert report["incomplete_snapshot_count"] == 0
    assert report["deleted_snapshots"] == []
    assert snapshot.exists()


def test_directory_size_sums_nested_files(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"123")
    (tmp_path / "nested" / "b.bin").write_bytes(b"45")

    assert hub.directory_size(tmp_path) == 5
