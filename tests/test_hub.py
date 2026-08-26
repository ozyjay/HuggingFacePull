import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import huggingface_pull.hub as hub

TEST_COMMIT = "a" * 40


def install_fake_hub(monkeypatch, files, snapshot_func=None, endpoint="https://huggingface.co"):
    cache_dir = Path(tempfile.mkdtemp(prefix="hfpull-test-cache-", dir="/tmp"))

    class FakeApi:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def model_info(self, repo_id, revision, files_metadata, token):
            return SimpleNamespace(
                sha=TEST_COMMIT,
                siblings=[
                    SimpleNamespace(
                        rfilename=file["path"],
                        size=file.get("size"),
                        blob_id=file.get("blob_id"),
                        lfs=file.get("lfs"),
                        xet_hash=file.get("xet_hash"),
                    )
                    for file in files
                ]
            )

    def default_snapshot_download(**kwargs):
        local_dir = (
            Path(kwargs["cache_dir"])
            / "models--Qwen--Qwen3"
            / "snapshots"
            / kwargs["revision"]
        )
        for file in files:
            path = local_dir / file["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * int(file.get("size") or 0))
        return str(local_dir)

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "snapshot_download", snapshot_func or default_snapshot_download)
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache_dir))


def fake_snapshot_path(kwargs):
    return (
        Path(kwargs["cache_dir"])
        / "models--Qwen--Qwen3"
        / "snapshots"
        / kwargs["revision"]
    )


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
        == "model:Qwen/Qwen3-Embedding-0.6B@main?allow=*.json,*.safetensors&ignore=*.bin,*.h5&xet=0"
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


def test_installed_models_reads_metadata_skips_malformed_and_sorts(tmp_path):
    alpha_snapshot = tmp_path / "snapshots" / "alpha"
    zeta_snapshot = tmp_path / "snapshots" / "zeta"
    alpha_snapshot.mkdir(parents=True)
    zeta_snapshot.mkdir(parents=True)
    (alpha_snapshot / "config.json").write_text("{}", encoding="utf-8")
    (zeta_snapshot / "config.json").write_text("{}", encoding="utf-8")
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
                "snapshot_path": str(zeta_snapshot),
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
                "snapshot_path": str(alpha_snapshot),
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
            "snapshot_path": str(alpha_snapshot),
            "size": 8,
        },
        {
            "repo_id": "zeta/Repo",
            "revision": "dev",
            "repo_type": "model",
            "snapshot_path": str(zeta_snapshot),
            "size": 12,
        },
    ]


def test_installed_models_resolves_metadata_after_hf_cache_moves(tmp_path, monkeypatch):
    old_snapshot = tmp_path / "old-cache" / "models--Qwen--Qwen3" / "snapshots" / "abc123"
    cache = tmp_path / "new-cache"
    new_snapshot = cache / "models--Qwen--Qwen3" / "snapshots" / "abc123"
    new_snapshot.mkdir(parents=True)
    (new_snapshot / "weights.bin").write_bytes(b"weights")
    marker = tmp_path / "library" / "Qwen--Qwen3" / "main" / ".huggingfacepull.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "repo_id": "Qwen/Qwen3",
                "revision": "main",
                "repo_type": "model",
                "snapshot_path": str(old_snapshot),
                "size": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))

    assert hub.installed_models(tmp_path / "library") == [
        {
            "repo_id": "Qwen/Qwen3",
            "revision": "main",
            "repo_type": "model",
            "snapshot_path": str(new_snapshot),
            "size": 7,
        }
    ]
    assert json.loads(marker.read_text(encoding="utf-8"))["snapshot_path"] == str(old_snapshot)


def test_cached_hub_models_reads_huggingface_cache_repos(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    model = cache / "models--Qwen--Qwen2.5-0.5B"
    snapshot = model / "snapshots" / "abc123"
    ref = model / "refs" / "main"
    snapshot.mkdir(parents=True)
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
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


def test_cached_hub_models_skips_metadata_only_snapshots(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    model = cache / "models--Qwen--Qwen3"
    snapshot = model / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))

    assert hub.cached_hub_models() == []


def test_cached_hub_models_skips_incomplete_sharded_payload(tmp_path, monkeypatch):
    log_events = []
    cache = tmp_path / "hub"
    model = cache / "models--google--diffusiongemma-26B-A4B-it"
    snapshot = model / "snapshots" / "abc123"
    ref = model / "refs" / "main"
    snapshot.mkdir(parents=True)
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00003.safetensors").write_bytes(b"one")
    (snapshot / "model-00002-of-00003.safetensors").write_bytes(b"two")
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
                "repo_id": "google/diffusiongemma-26B-A4B-it",
                "revision": "main",
                "snapshot_path": snapshot,
                "reason": "incomplete_sharded_payload",
                "path": snapshot,
                "prefix": "model",
                "suffix": ".safetensors",
                "expected_shards": 3,
                "found_shards": 2,
                "missing_shards": [3],
            },
        )
    ]


def test_partial_cached_hub_models_reports_incomplete_sharded_payload(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    model = cache / "models--google--diffusiongemma-26B-A4B-it"
    snapshot = model / "snapshots" / "abc123"
    ref = model / "refs" / "main"
    snapshot.mkdir(parents=True)
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00003.safetensors").write_bytes(b"one")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))

    assert hub.partial_cached_hub_models() == [
        {
            "repo_id": "google/diffusiongemma-26B-A4B-it",
            "revision": "main",
            "repo_type": "model",
            "snapshot_path": str(snapshot),
            "source": "huggingface_cache",
            "cache_status": "partial",
            "reason": "incomplete_sharded_payload",
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


def test_search_models_retries_anonymously_after_cached_token_401(monkeypatch):
    calls = []

    class CachedTokenRejected(Exception):
        response = SimpleNamespace(status_code=401)

    class FakeApi:
        def __init__(self, endpoint):
            assert endpoint == "https://hf.example"

        def list_models(self, *, search, limit, sort, token):
            calls.append(token)
            if token is None:
                raise CachedTokenRejected("cached token rejected")
            assert token is False
            return [SimpleNamespace(modelId="google/gemma-3-4b-it")]

    monkeypatch.setattr(hub, "HfApi", FakeApi)

    result = hub.search_models("gemma", endpoint="https://hf.example")

    assert result["available"] is True
    assert result["results"][0]["repo_id"] == "google/gemma-3-4b-it"
    assert result["error"] is None
    assert calls == [None, False]


def test_search_models_does_not_bypass_explicit_token_401(monkeypatch):
    calls = []

    class ExplicitTokenRejected(Exception):
        response = SimpleNamespace(status_code=401)

    class FakeApi:
        def __init__(self, endpoint):
            pass

        def list_models(self, **kwargs):
            calls.append(kwargs["token"])
            raise ExplicitTokenRejected("explicit token rejected")

    monkeypatch.setattr(hub, "HfApi", FakeApi)

    result = hub.search_models("gemma", token="invalid-explicit-token")

    assert result["available"] is False
    assert result["results"] == []
    assert "explicit token rejected" in result["error"]
    assert calls == ["invalid-explicit-token"]


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
                sha=TEST_COMMIT,
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
            {"path": "config.json", "size": 20, "blob_id": "abc", "lfs_sha256": None, "lfs_size": None, "xet_hash": None},
            {"path": "model.safetensors", "size": 120, "blob_id": "def", "lfs_sha256": None, "lfs_size": None, "xet_hash": None},
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


def test_pull_snapshot_uses_hf_cache_and_writes_metadata_without_network(monkeypatch, tmp_path):
    calls = []
    cache_dir = tmp_path / "hf-cache"

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
                sha=TEST_COMMIT,
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=2, blob_id="cfg"),
                    SimpleNamespace(rfilename="weights.bin", size=7, blob_id="weights"),
                    SimpleNamespace(rfilename="weights.safetensors", size=9, blob_id="safe"),
                ]
            )

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        assert "local_dir" not in kwargs
        snapshot_dir = (
            Path(kwargs["cache_dir"])
            / "models--Qwen--Qwen3"
            / "snapshots"
            / kwargs["revision"]
        )
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "weights.bin").write_bytes(b"1234567")
        return str(snapshot_dir)

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache_dir))
    events = []
    ref = hub.HubRef(
        repo_id="Qwen/Qwen3",
        revision="v1",
        allow_patterns=["*.bin"],
        ignore_patterns=["*.safetensors"],
    )

    snapshot_path = hub.pull_snapshot(
        ref,
        library_dir=tmp_path,
        endpoint="https://hf.example",
        token="secret",
        progress=events.append,
    )

    target = cache_dir / "models--Qwen--Qwen3" / "snapshots" / TEST_COMMIT
    marker = hub.metadata_path(tmp_path, ref)
    assert snapshot_path == target
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3",
            "revision": TEST_COMMIT,
            "repo_type": None,
            "cache_dir": cache_dir,
            "endpoint": "https://hf.example",
            "token": "secret",
            "allow_patterns": ["*.bin"],
            "ignore_patterns": ["*.safetensors"],
            "max_workers": 1,
            "tqdm_class": calls[0]["tqdm_class"],
        }
    ]
    assert calls[0]["tqdm_class"] is not None
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    assert metadata == {
        "format": "huggingfacepull-completion",
        "version": 2,
        "repo_id": "Qwen/Qwen3",
        "requested_revision": "v1",
        "revision": "v1",
        "repo_type": "model",
        "expected_commit": None,
        "resolved_revision": TEST_COMMIT,
        "snapshot_path": str(target),
        "size": 7,
        "files": [{
            "path": "weights.bin",
            "size": 7,
            "blob_id": "weights",
            "lfs_sha256": None,
            "lfs_size": None,
            "xet_hash": None,
            "verification": "size_only",
        }],
        "xet_enabled": False,
    }
    assert sorted(path.name for path in marker.parent.iterdir()) == [".huggingfacepull.json"]
    assert events == [
        {"type": "manifest-fetch", "repo_id": "Qwen/Qwen3", "revision": "v1"},
        {
            "type": "model-plan",
            "repo_id": "Qwen/Qwen3",
            "revision": "v1",
            "total_bytes": 7,
            "files": [{"path": "weights.bin", "size": 7, "blob_id": "weights", "lfs_sha256": None, "lfs_size": None, "xet_hash": None}],
        },
        {"type": "model-complete", "repo_id": "Qwen/Qwen3", "snapshot_path": str(target)},
    ]


def test_pull_snapshot_rejects_missing_expected_file_without_metadata(
    monkeypatch, tmp_path
):
    cache_dir = tmp_path / "hf-cache"
    log_events = []

    def fake_snapshot_download(**kwargs):
        snapshot_dir = Path(kwargs["cache_dir"]) / "models--Qwen--Qwen3" / "snapshots" / kwargs["revision"]
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(snapshot_dir)

    install_fake_hub(
        monkeypatch,
        [
            {"path": "config.json", "size": 2, "blob_id": "cfg"},
            {"path": "weights.bin", "size": 7, "blob_id": "weights"},
        ],
        fake_snapshot_download,
    )
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache_dir))
    monkeypatch.setattr(
        hub,
        "write_log",
        lambda message, **fields: log_events.append((message, fields)),
        raising=False,
    )
    ref = hub.HubRef(repo_id="Qwen/Qwen3")

    with pytest.raises(RuntimeError, match="Downloaded snapshot is incomplete"):
        hub.pull_snapshot(ref, library_dir=tmp_path)

    assert not hub.metadata_path(tmp_path, ref).exists()
    assert log_events == [
        (
            "downloaded snapshot incomplete",
            {
                "repo_id": "Qwen/Qwen3",
                "revision": "main",
                "snapshot_path": cache_dir / "models--Qwen--Qwen3" / "snapshots" / TEST_COMMIT,
                "reason": "missing_file",
                "path": cache_dir / "models--Qwen--Qwen3" / "snapshots" / TEST_COMMIT / "weights.bin",
            },
        )
    ]


def test_pull_snapshot_rejects_size_mismatch_without_metadata(monkeypatch, tmp_path):
    cache_dir = tmp_path / "hf-cache"

    def fake_snapshot_download(**kwargs):
        snapshot_dir = Path(kwargs["cache_dir"]) / "models--Qwen--Qwen3" / "snapshots" / kwargs["revision"]
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "weights.bin").write_bytes(b"short")
        return str(snapshot_dir)

    install_fake_hub(
        monkeypatch,
        [{"path": "weights.bin", "size": 7, "blob_id": "weights"}],
        fake_snapshot_download,
    )
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache_dir))
    ref = hub.HubRef(repo_id="Qwen/Qwen3")

    with pytest.raises(RuntimeError, match="size_mismatch"):
        hub.pull_snapshot(ref, library_dir=tmp_path)

    assert not hub.metadata_path(tmp_path, ref).exists()


def test_installed_models_skips_metadata_for_incomplete_snapshot(tmp_path, monkeypatch):
    log_events = []
    snapshot = tmp_path / "hf-cache" / "models--Qwen--Qwen3" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    marker = tmp_path / "library" / "Qwen--Qwen3" / "main" / ".huggingfacepull.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "repo_id": "Qwen/Qwen3",
                "revision": "main",
                "repo_type": "model",
                "snapshot_path": str(snapshot),
                "files": [{"path": "weights.bin", "size": 7, "blob_id": "weights"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hub,
        "write_log",
        lambda message, **fields: log_events.append((message, fields)),
        raising=False,
    )

    assert hub.installed_models(tmp_path / "library") == []
    assert log_events == [
        (
            "installed metadata skipped",
            {
                "marker": marker,
                "repo_id": "Qwen/Qwen3",
                "revision": "main",
                "reason": "empty",
            },
        )
    ]


def test_pull_snapshot_passes_none_for_empty_snapshot_patterns(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        snapshot_dir = Path(kwargs["cache_dir"]) / "models--Qwen--Qwen3" / "snapshots" / kwargs["revision"]
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(snapshot_dir)

    install_fake_hub(
        monkeypatch,
        [{"path": "config.json", "size": 2, "blob_id": "cfg"}],
        fake_snapshot_download,
    )

    hub.pull_snapshot(hub.HubRef(repo_id="Qwen/Qwen3"), library_dir=tmp_path)

    assert calls[0]["allow_patterns"] is None
    assert calls[0]["ignore_patterns"] is None
    assert "local_dir" not in calls[0]
    assert Path(calls[0]["cache_dir"]) == Path(hub.HF_HUB_CACHE)


def test_pull_snapshot_disables_xet_by_default_before_lazy_hub_import(monkeypatch, tmp_path):
    seen = []

    def fake_snapshot_download(**kwargs):
        seen.append(os.environ.get("HF_HUB_DISABLE_XET"))
        local_dir = fake_snapshot_path(kwargs)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "weights.bin").write_bytes(b"data")
        return str(local_dir)

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    install_fake_hub(
        monkeypatch,
        [{"path": "weights.bin", "size": 4, "blob_id": "weights"}],
        fake_snapshot_download,
    )

    hub.pull_snapshot(hub.HubRef(repo_id="Qwen/Qwen3"), library_dir=tmp_path)

    assert seen == ["1"]
    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"


def test_pull_snapshot_overrides_existing_xet_setting_by_default(monkeypatch, tmp_path):
    seen = []

    def fake_snapshot_download(**kwargs):
        seen.append(os.environ.get("HF_HUB_DISABLE_XET"))
        local_dir = fake_snapshot_path(kwargs)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "weights.bin").write_bytes(b"data")
        return str(local_dir)

    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    install_fake_hub(
        monkeypatch,
        [{"path": "weights.bin", "size": 4, "blob_id": "weights"}],
        fake_snapshot_download,
    )

    hub.pull_snapshot(hub.HubRef(repo_id="Qwen/Qwen3"), library_dir=tmp_path)

    assert seen == ["1"]
    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"


def test_pull_snapshot_can_enable_xet(monkeypatch, tmp_path):
    seen = []

    def fake_snapshot_download(**kwargs):
        seen.append(os.environ.get("HF_HUB_DISABLE_XET"))
        local_dir = fake_snapshot_path(kwargs)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "weights.bin").write_bytes(b"data")
        return str(local_dir)

    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    install_fake_hub(
        monkeypatch,
        [{"path": "weights.bin", "size": 4, "blob_id": "weights"}],
        fake_snapshot_download,
    )

    hub.pull_snapshot(hub.HubRef(repo_id="Qwen/Qwen3", xet_enabled=True), library_dir=tmp_path)

    assert seen == [None]
    assert os.environ.get("HF_HUB_DISABLE_XET") is None


def test_pull_snapshot_emits_aggregate_byte_progress_from_snapshot_tqdm(monkeypatch, tmp_path):
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

        local_dir = fake_snapshot_path(kwargs)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "weights.bin").write_bytes(b"123")
        return str(local_dir)

    install_fake_hub(
        monkeypatch,
        [{"path": "weights.bin", "size": 3, "blob_id": "weights"}],
        fake_snapshot_download,
    )
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



def test_pull_snapshot_emits_fetch_progress_from_snapshot_tqdm(monkeypatch, tmp_path):
    def fake_snapshot_download(**kwargs):
        progress_bar = kwargs["tqdm_class"](
            total=37,
            initial=30,
            unit="it",
            desc="Fetching 37 files",
        )
        progress_bar.update(1)
        progress_bar.refresh()
        progress_bar.close()

        local_dir = fake_snapshot_path(kwargs)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "weights.bin").write_bytes(b"123")
        return str(local_dir)

    install_fake_hub(
        monkeypatch,
        [{"path": "weights.bin", "size": 3, "blob_id": "weights"}],
        fake_snapshot_download,
    )
    events = []

    hub.pull_snapshot(
        hub.HubRef(repo_id="Qwen/Qwen3"),
        library_dir=tmp_path,
        progress=events.append,
    )

    fetch_events = [event for event in events if event["type"] == "fetch-progress"]
    assert fetch_events
    assert fetch_events[-1]["repo_id"] == "Qwen/Qwen3"
    assert fetch_events[-1]["downloaded"] == 31
    assert fetch_events[-1]["total"] == 37
    assert fetch_events[-1]["unit"] == "it"
    assert fetch_events[-1]["description"] == "Fetching 37 files"

def test_pull_snapshot_throttles_rapid_byte_progress(monkeypatch, tmp_path):
    ticks = iter([0.0, 0.0, 0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(hub.time, "monotonic", lambda: next(ticks, 0.4))

    def fake_snapshot_download(**kwargs):
        progress_bar = kwargs["tqdm_class"](total=100, initial=0, unit="B")
        for _ in range(4):
            progress_bar.update(10)
        progress_bar.close()

        local_dir = fake_snapshot_path(kwargs)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "weights.bin").write_bytes(b"x" * 40)
        return str(local_dir)

    install_fake_hub(
        monkeypatch,
        [{"path": "weights.bin", "size": 40, "blob_id": "weights"}],
        fake_snapshot_download,
    )
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


def test_pull_snapshot_stops_after_snapshot_progress_when_requested(monkeypatch, tmp_path):
    stop_requested = False

    def fake_snapshot_download(**kwargs):
        progress_bar = kwargs["tqdm_class"](total=100, initial=0, unit="B")
        progress_bar.update(10)

        nonlocal stop_requested
        stop_requested = True
        progress_bar.update(10)

        local_dir = fake_snapshot_path(kwargs)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "weights.bin").write_bytes(b"x" * 20)
        return str(local_dir)

    install_fake_hub(
        monkeypatch,
        [{"path": "weights.bin", "size": 100, "blob_id": "weights"}],
        fake_snapshot_download,
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
    def fake_snapshot_download(**kwargs):
        local_dir = fake_snapshot_path(kwargs)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "weights.bin").write_bytes(b"data")
        return str(local_dir)

    install_fake_hub(
        monkeypatch,
        [{"path": "weights.bin", "size": 4, "blob_id": "weights"}],
        fake_snapshot_download,
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


def test_delete_installed_model_removes_snapshot_record_and_unshared_blobs(
    tmp_path, monkeypatch
):
    cache = tmp_path / "hub"
    repo = cache / "models--Qwen--Qwen3"
    blobs = repo / "blobs"
    main_snapshot = repo / "snapshots" / "abc123"
    dev_snapshot = repo / "snapshots" / "def456"
    refs = repo / "refs"
    blobs.mkdir(parents=True)
    main_snapshot.mkdir(parents=True)
    dev_snapshot.mkdir(parents=True)
    refs.mkdir(parents=True)

    (blobs / "shared").write_bytes(b"shared")
    (blobs / "main-only").write_bytes(b"main")
    (blobs / "dev-only").write_bytes(b"dev")
    (main_snapshot / "config.json").symlink_to("../../blobs/shared")
    (main_snapshot / "model.safetensors").symlink_to("../../blobs/main-only")
    (dev_snapshot / "config.json").symlink_to("../../blobs/shared")
    (dev_snapshot / "model.safetensors").symlink_to("../../blobs/dev-only")
    (refs / "main").write_text("abc123", encoding="utf-8")
    (refs / "dev").write_text("def456", encoding="utf-8")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))

    library = tmp_path / "library"
    ref = hub.HubRef(repo_id="Qwen/Qwen3", revision="main")
    marker = hub.metadata_path(library, ref)
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "repo_id": ref.repo_id,
                "revision": ref.revision,
                "repo_type": ref.repo_type,
                "snapshot_path": str(main_snapshot),
            }
        ),
        encoding="utf-8",
    )

    freed_size = hub.delete_installed_model(library, ref)

    assert freed_size == 4
    assert not main_snapshot.exists()
    assert not (refs / "main").exists()
    assert not (blobs / "main-only").exists()
    assert not marker.parent.exists()
    assert dev_snapshot.exists()
    assert (refs / "dev").exists()
    assert (blobs / "shared").exists()
    assert (blobs / "dev-only").exists()


def test_delete_installed_model_removes_cache_only_repo(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    repo = cache / "models--Qwen--Qwen3"
    blob = repo / "blobs" / "weight"
    snapshot = repo / "snapshots" / "abc123"
    ref_path = repo / "refs" / "main"
    blob.parent.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    ref_path.parent.mkdir(parents=True)
    blob.write_bytes(b"model")
    (snapshot / "model.safetensors").symlink_to("../../blobs/weight")
    ref_path.write_text("abc123", encoding="utf-8")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))

    freed_size = hub.delete_installed_model(
        tmp_path / "library", hub.HubRef(repo_id="Qwen/Qwen3", revision="main")
    )

    assert freed_size == 5
    assert not repo.exists()


def test_delete_installed_model_requires_cached_revision(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))

    with pytest.raises(KeyError):
        hub.delete_installed_model(
            tmp_path / "library", hub.HubRef(repo_id="Qwen/Qwen3", revision="main")
        )


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


def test_pull_snapshot_records_and_verifies_lfs_and_xet_metadata(monkeypatch, tmp_path):
    content = b"verified"
    sha256 = __import__("hashlib").sha256(content).hexdigest()
    cache = tmp_path / "hub"
    calls = []

    class FakeApi:
        def __init__(self, endpoint):
            pass

        def model_info(self, *args, **kwargs):
            return SimpleNamespace(
                sha=TEST_COMMIT,
                siblings=[
                    SimpleNamespace(
                        rfilename="weights.gguf",
                        size=len(content),
                        blob_id="b" * 40,
                        lfs=SimpleNamespace(sha256=sha256, size=len(content)),
                        xet_hash="c" * 64,
                    )
                ],
            )

    def fake_download(**kwargs):
        calls.append(kwargs)
        snapshot = cache / "models--Qwen--Qwen3" / "snapshots" / kwargs["revision"]
        snapshot.mkdir(parents=True)
        (snapshot / "weights.gguf").write_bytes(content)
        return str(snapshot)

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "snapshot_download", fake_download)
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))

    ref = hub.HubRef(repo_id="Qwen/Qwen3", revision="main")
    hub.pull_snapshot(ref, library_dir=tmp_path / "library")

    marker = hub.read_completion_marker(hub.metadata_path(tmp_path / "library", ref))
    assert calls[0]["revision"] == TEST_COMMIT
    assert marker["resolved_revision"] == TEST_COMMIT
    assert marker["files"] == [{
        "path": "weights.gguf",
        "size": len(content),
        "blob_id": "b" * 40,
        "lfs_sha256": sha256,
        "lfs_size": len(content),
        "xet_hash": "c" * 64,
        "verification": "sha256",
    }]


def test_pull_snapshot_rejects_lfs_checksum_mismatch(monkeypatch, tmp_path):
    cache = tmp_path / "hub"
    expected_sha256 = "a" * 64

    class FakeApi:
        def __init__(self, endpoint):
            pass

        def model_info(self, *args, **kwargs):
            return SimpleNamespace(
                sha=TEST_COMMIT,
                siblings=[
                    SimpleNamespace(
                        rfilename="weights.gguf",
                        size=4,
                        blob_id="b" * 40,
                        lfs=SimpleNamespace(sha256=expected_sha256, size=4),
                    )
                ],
            )

    def fake_download(**kwargs):
        snapshot = cache / "models--Qwen--Qwen3" / "snapshots" / kwargs["revision"]
        snapshot.mkdir(parents=True)
        (snapshot / "weights.gguf").write_bytes(b"nope")
        return str(snapshot)

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "snapshot_download", fake_download)
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))
    ref = hub.HubRef(repo_id="Qwen/Qwen3")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        hub.pull_snapshot(ref, library_dir=tmp_path / "library")
    assert not hub.metadata_path(tmp_path / "library", ref).exists()


def test_pull_snapshot_rejects_empty_or_traversal_selection(monkeypatch, tmp_path):
    install_fake_hub(monkeypatch, [{"path": "../outside", "size": 1, "blob_id": "bad"}])
    with pytest.raises(RuntimeError, match="Invalid selected file metadata"):
        hub.pull_snapshot(hub.HubRef(repo_id="Qwen/Qwen3"), library_dir=tmp_path)

    install_fake_hub(monkeypatch, [{"path": "weights.gguf", "size": 1, "blob_id": "good"}])
    with pytest.raises(RuntimeError, match="No repository files matched"):
        hub.pull_snapshot(
            hub.HubRef(repo_id="Qwen/Qwen3", allow_patterns=["*.bin"]),
            library_dir=tmp_path,
        )


def test_pull_snapshot_rejects_malformed_integrity_metadata_before_download(monkeypatch, tmp_path):
    install_fake_hub(
        monkeypatch,
        [{"path": "weights.gguf", "size": 1, "blob_id": "blob", "lfs": {"sha256": "bad", "size": 1}}],
        lambda **kwargs: pytest.fail("malformed metadata must prevent download"),
    )

    with pytest.raises(RuntimeError, match="Invalid selected file metadata"):
        hub.pull_snapshot(hub.HubRef(repo_id="Qwen/Qwen3"), library_dir=tmp_path)


def test_pull_snapshot_records_multiple_selected_files_and_selected_size(monkeypatch, tmp_path):
    install_fake_hub(
        monkeypatch,
        [
            {"path": "z.gguf", "size": 2, "blob_id": "z"},
            {"path": "a.gguf", "size": 1, "blob_id": "a"},
        ],
    )
    ref = hub.HubRef(repo_id="Qwen/Qwen3")

    hub.pull_snapshot(ref, library_dir=tmp_path / "library")

    marker = hub.read_completion_marker(hub.metadata_path(tmp_path / "library", ref))
    assert marker["size"] == 3
    assert [file["path"] for file in marker["files"]] == ["a.gguf", "z.gguf"]


def test_hub_ref_rejects_repository_path_traversal():
    with pytest.raises(ValueError, match="traversal"):
        hub.HubRef(repo_id="..")


def test_read_completion_marker_supports_legacy_fixture_and_rejects_unknown_version(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({
        "expected_commit": TEST_COMMIT,
        "files": [{"path": "weights.gguf", "size": 2, "blob_id": "blob"}],
        "repo_id": "Qwen/Qwen3",
        "resolved_revision": TEST_COMMIT,
        "revision": "main",
        "size": 2,
        "snapshot_path": "/cache/snapshot",
        "xet_enabled": False,
    }), encoding="utf-8")
    assert hub.read_completion_marker(legacy)["expected_commit"] == TEST_COMMIT

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"format": hub.COMPLETION_FORMAT, "version": 999}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        hub.read_completion_marker(unknown)


def test_atomic_marker_write_preserves_previous_marker_when_replace_fails(monkeypatch, tmp_path):
    marker = tmp_path / ".huggingfacepull.json"
    marker.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(hub.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("nope")))

    with pytest.raises(OSError, match="nope"):
        hub._write_marker_atomic(marker, {"value": "new"})
    assert marker.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".huggingfacepull.json.*.tmp")) == []


def test_upgrade_legacy_marker_enriches_cached_tree_without_rehashing(monkeypatch, tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--Qwen--Qwen3" / "snapshots" / TEST_COMMIT
    snapshot.mkdir(parents=True)
    (snapshot / "weights.gguf").write_bytes(b"data")
    tree = cache / "models--Qwen--Qwen3" / "trees" / f"{TEST_COMMIT}.json"
    tree.parent.mkdir(parents=True)
    tree.write_text(json.dumps({
        "format_version": 1,
        "files": {
            "weights.gguf": {
                "size": 4,
                "blob_id": "b" * 40,
                "lfs_sha256": "c" * 64,
                "lfs_size": 4,
                "xet_hash": "d" * 64,
            }
        },
    }), encoding="utf-8")
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))
    ref = hub.HubRef(repo_id="Qwen/Qwen3")
    marker = hub.metadata_path(tmp_path / "library", ref)
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({
        "repo_id": ref.repo_id,
        "revision": ref.revision,
        "repo_type": ref.repo_type,
        "snapshot_path": str(snapshot),
        "files": [{"path": "weights.gguf", "size": 4, "blob_id": "b" * 40}],
        "xet_enabled": False,
    }), encoding="utf-8")

    assert hub.upgrade_legacy_markers(tmp_path / "library") == {"upgraded": [str(marker)], "skipped": []}
    upgraded = hub.read_completion_marker(marker)
    assert upgraded["files"][0]["lfs_sha256"] == "c" * 64
    assert upgraded["files"][0]["verification"] == "size_only"


def test_kernel_ref_requires_a_valid_expected_commit():
    with pytest.raises(ValueError, match="Expected commit"):
        hub.HubRef(
            repo_id="kernels-community/finegrained-fp8",
            repo_type="kernel",
            expected_commit="not-a-commit",
        )


def test_canonical_kernel_ref_includes_expected_commit():
    ref = hub.HubRef(
        repo_id="kernels-community/finegrained-fp8",
        revision="v3",
        repo_type="kernel",
        expected_commit="a" * 40,
        allow_patterns=["build/torch-rocm/*"],
    )

    assert "expected=" + "a" * 40 in hub.canonical_ref(ref)


def test_pull_kernel_requires_resolved_commit_and_records_it(monkeypatch, tmp_path):
    commit = "a" * 40
    cache = tmp_path / "hub"

    class FakeApi:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def repo_info(self, repo_id, revision, repo_type, files_metadata, token):
            assert (repo_id, revision, repo_type) == (
                "kernels-community/finegrained-fp8",
                "v3",
                "kernel",
            )
            return SimpleNamespace(
                sha=commit,
            )

        def list_repo_tree(self, repo_id, recursive, revision, repo_type, token):
            return [SimpleNamespace(path="build/torch-rocm/matmul.py", size=6, blob_id="code")]

    def fake_snapshot_download(**kwargs):
        snapshot = cache / "kernels--kernels-community--finegrained-fp8" / "snapshots" / commit
        source = snapshot / "build" / "torch-rocm" / "matmul.py"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"kernel")
        return str(snapshot)

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(cache))
    ref = hub.HubRef(
        repo_id="kernels-community/finegrained-fp8",
        revision="v3",
        repo_type="kernel",
        expected_commit=commit,
        allow_patterns=["build/torch-rocm/*"],
    )

    hub.pull_snapshot(ref, library_dir=tmp_path / "library")

    metadata = json.loads(hub.metadata_path(tmp_path / "library", ref).read_text())
    assert metadata["expected_commit"] == commit
    assert metadata["resolved_revision"] == commit


def test_pull_kernel_rejects_unexpected_commit_before_download(monkeypatch, tmp_path):
    class FakeApi:
        def __init__(self, endpoint):
            pass

        def repo_info(self, repo_id, **kwargs):
            return SimpleNamespace(sha="b" * 40, siblings=[])

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "snapshot_download", lambda **kwargs: pytest.fail("must not download"))
    ref = hub.HubRef(
        repo_id="kernels-community/finegrained-fp8",
        revision="v3",
        repo_type="kernel",
        expected_commit="a" * 40,
    )

    with pytest.raises(RuntimeError, match="does not match expected commit"):
        hub.pull_snapshot(ref, library_dir=tmp_path)
