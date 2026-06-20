import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import huggingface_pull
import huggingface_pull.hub as hub


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
        "snapshot_download",
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

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        (local_dir / "weights.bin").write_bytes(b"1234567")
        return local_dir

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
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
            "revision": "v1",
            "repo_type": None,
            "local_dir": target,
            "allow_patterns": ["*.bin"],
            "ignore_patterns": None,
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
        {"type": "model-complete", "repo_id": "Qwen/Qwen3", "snapshot_path": str(target)},
    ]


def test_pull_snapshot_disables_xet_during_download(monkeypatch, tmp_path):
    seen = []

    def fake_snapshot_download(**kwargs):
        seen.append(os.environ.get("HF_HUB_DISABLE_XET"))
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        return local_dir

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)

    hub.pull_snapshot(hub.HubRef(repo_id="Qwen/Qwen3"), library_dir=tmp_path)

    assert seen == ["1"]
    assert os.environ.get("HF_HUB_DISABLE_XET") is None


def test_pull_snapshot_restores_existing_xet_setting(monkeypatch, tmp_path):
    seen = []

    def fake_snapshot_download(**kwargs):
        seen.append(os.environ.get("HF_HUB_DISABLE_XET"))
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        return local_dir

    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)

    hub.pull_snapshot(hub.HubRef(repo_id="Qwen/Qwen3"), library_dir=tmp_path)

    assert seen == ["1"]
    assert os.environ.get("HF_HUB_DISABLE_XET") == "0"


def test_pull_snapshot_emits_byte_progress_from_hub_tqdm(monkeypatch, tmp_path):
    def fake_snapshot_download(**kwargs):
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
        local_dir.mkdir(parents=True)
        (local_dir / "weights.bin").write_bytes(b"123")
        return local_dir

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    events = []

    hub.pull_snapshot(
        hub.HubRef(repo_id="Qwen/Qwen3"),
        library_dir=tmp_path,
        progress=events.append,
    )

    progress_events = [event for event in events if event["type"] == "download-progress"]
    assert progress_events == [
        {
            "type": "download-progress",
            "repo_id": "Qwen/Qwen3",
            "downloaded": 2,
            "total": 10,
            "percent": 20.0,
            "bytes_per_second": None,
            "eta_seconds": None,
        },
        {
            "type": "download-progress",
            "repo_id": "Qwen/Qwen3",
            "downloaded": 5,
            "total": 15,
            "percent": 33.33333333333333,
            "bytes_per_second": None,
            "eta_seconds": None,
        },
        {
            "type": "download-progress",
            "repo_id": "Qwen/Qwen3",
            "downloaded": 15,
            "total": 15,
            "percent": 100.0,
            "bytes_per_second": None,
            "eta_seconds": None,
        },
    ]


def test_pull_snapshot_throttles_rapid_byte_progress(monkeypatch, tmp_path):
    ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(hub.time, "monotonic", lambda: next(ticks, 0.4))

    def fake_snapshot_download(**kwargs):
        progress_bar = kwargs["tqdm_class"](total=100, initial=0, unit="B")
        for _ in range(4):
            progress_bar.update(10)
        progress_bar.close()

        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        return local_dir

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    events = []

    hub.pull_snapshot(
        hub.HubRef(repo_id="Qwen/Qwen3"),
        library_dir=tmp_path,
        progress=events.append,
    )

    progress_events = [event for event in events if event["type"] == "download-progress"]
    assert progress_events == [
        {
            "type": "download-progress",
            "repo_id": "Qwen/Qwen3",
            "downloaded": 0,
            "total": 100,
            "percent": 0.0,
            "bytes_per_second": None,
            "eta_seconds": None,
        },
        {
            "type": "download-progress",
            "repo_id": "Qwen/Qwen3",
            "downloaded": 40,
            "total": 100,
            "percent": 40.0,
            "bytes_per_second": None,
            "eta_seconds": None,
        },
    ]


def test_pull_snapshot_stops_during_hub_progress_when_requested(monkeypatch, tmp_path):
    stop_requested = False

    def fake_snapshot_download(**kwargs):
        progress_bar = kwargs["tqdm_class"](total=100, initial=0, unit="B")
        progress_bar.update(10)

        nonlocal stop_requested
        stop_requested = True
        progress_bar.update(10)

        pytest.fail("download should stop during progress update")

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
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
    def fake_snapshot_download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        return local_dir

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    ref = hub.HubRef(repo_id="Qwen/Qwen3")

    with pytest.raises(hub.DownloadStoppedAfterFile):
        hub.pull_snapshot(ref, library_dir=tmp_path, stop_after_file=lambda: True)

    assert hub.metadata_path(tmp_path, ref).exists()


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


def test_directory_size_sums_nested_files(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"123")
    (tmp_path / "nested" / "b.bin").write_bytes(b"45")

    assert hub.directory_size(tmp_path) == 5
