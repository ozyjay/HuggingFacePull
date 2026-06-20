from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

import huggingface_pull.cli as cli_module
from huggingface_pull.api import create_app
from huggingface_pull.queue import DownloadQueue


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def run_app_lifespan(app):
    async def _run():
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(_run())


def test_state_endpoint_returns_snapshot(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path))

    response = client.get("/api/state")

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["library_dir"] == str(tmp_path)


def test_queue_endpoint_queues_repo(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path))

    response = client.post("/api/queue", json={"repo_id": "Qwen/Qwen3"})

    assert response.status_code == 200
    assert response.json()["repo_id"] == "Qwen/Qwen3"
    assert response.json()["status"] == "waiting"


def test_bad_queue_body_returns_422(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path))

    response = client.post("/api/queue", json={"repo_id": ""})

    assert response.status_code == 422


def test_search_delegates_url_decoded_query(monkeypatch, tmp_path):
    import huggingface_pull.api as api_module

    calls = []

    def fake_search_models(q, endpoint, token):
        calls.append((q, endpoint, token))
        return {"available": True, "results": [{"repo_id": "Qwen/Qwen3"}], "error": None}

    monkeypatch.setattr(api_module, "search_models", fake_search_models)
    client = TestClient(
        create_app(library_dir=tmp_path, endpoint="https://hf.example", token="secret")
    )

    response = client.get("/api/search?q=Qwen%2FQwen3%20Instruct")

    assert response.status_code == 200
    assert response.json()["results"] == [{"repo_id": "Qwen/Qwen3"}]
    assert calls == [("Qwen/Qwen3 Instruct", "https://hf.example", "secret")]


def test_files_endpoint_delegates_for_model_repo(monkeypatch, tmp_path):
    import huggingface_pull.api as api_module

    calls = []

    def fake_repo_files(ref, endpoint, token):
        calls.append((ref, endpoint, token))
        return {"repo_id": ref.repo_id, "revision": ref.revision, "files": []}

    monkeypatch.setattr(api_module, "repo_files", fake_repo_files)
    client = TestClient(
        create_app(library_dir=tmp_path, endpoint="https://hf.example", token="secret")
    )

    response = client.get("/api/models/Qwen/Qwen3/files?revision=v1&repo_type=model")

    assert response.status_code == 200
    assert response.json() == {"repo_id": "Qwen/Qwen3", "revision": "v1", "files": []}
    [(ref, endpoint, token)] = calls
    assert ref.repo_id == "Qwen/Qwen3"
    assert ref.revision == "v1"
    assert ref.repo_type == "model"
    assert endpoint == "https://hf.example"
    assert token == "secret"


def test_files_endpoint_rejects_non_model_repo(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path))

    response = client.get("/api/models/user/dataset/files?repo_type=dataset")

    assert response.status_code == 400
    assert "model" in response.json()["detail"]


def test_start_pause_and_stop_endpoints_delegate_to_queue(monkeypatch, tmp_path):
    import huggingface_pull.api as api_module

    class FakeQueue:
        def __init__(self, *, library_dir, endpoint, token):
            self.library_dir = library_dir
            self.calls = []

        def snapshot(self):
            return {"library_dir": str(self.library_dir), "calls": list(self.calls), "items": []}

        def start(self):
            self.calls.append("start")

        def pause_after_current(self):
            self.calls.append("pause")

        def stop_after_current_file(self):
            self.calls.append("stop")
            return self.snapshot()

    monkeypatch.setattr(api_module, "DownloadQueue", FakeQueue)
    app = create_app(library_dir=tmp_path)
    client = TestClient(app)

    assert client.post("/api/start").json()["calls"] == ["start"]
    assert client.post("/api/pause").json()["calls"] == ["start", "pause"]
    assert client.post("/api/stop-after-file").json()["calls"] == [
        "start",
        "pause",
        "stop",
    ]
    assert app.state.queue.calls == ["start", "pause", "stop"]


def test_retry_and_remove_return_404_for_missing_items(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path))

    retry_response = client.post("/api/retry/missing")
    remove_response = client.post("/api/remove/missing")

    assert retry_response.status_code == 404
    assert remove_response.status_code == 404


def test_retry_waiting_item_returns_409_conflict(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path), raise_server_exceptions=False)
    item = client.post("/api/queue", json={"repo_id": "Qwen/Qwen3"}).json()

    response = client.post(f"/api/retry/{item['id']}")

    assert response.status_code == 409
    assert response.json()["detail"] == "Only failed items can be retried"


def test_remove_running_item_returns_409_conflict(monkeypatch, tmp_path):
    import huggingface_pull.api as api_module

    release_download = threading.Event()

    def fake_pull(*args, **kwargs):
        release_download.wait(2)

    class TestQueue(DownloadQueue):
        def __init__(self, *, library_dir, endpoint, token):
            super().__init__(
                library_dir=library_dir,
                endpoint=endpoint,
                token=token,
                pull_func=fake_pull,
            )

    monkeypatch.setattr(api_module, "DownloadQueue", TestQueue)
    app = create_app(library_dir=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    item = client.post("/api/queue", json={"repo_id": "Qwen/Qwen3"}).json()
    client.post("/api/start")

    assert wait_for(lambda: app.state.queue.snapshot()["items"][0]["status"] == "running")
    response = client.post(f"/api/remove/{item['id']}")
    release_download.set()

    assert response.status_code == 409
    assert response.json()["detail"] == "Running items cannot be removed"
    assert app.state.queue.wait_until_idle(2)


def test_remove_returns_ok_for_existing_item(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path))
    item = client.post("/api/queue", json={"repo_id": "Qwen/Qwen3"}).json()

    response = client.post(f"/api/remove/{item['id']}")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_installed_remove_404_for_missing_model(tmp_path):
    client = TestClient(create_app(library_dir=tmp_path))

    response = client.post(
        "/api/installed/remove",
        json={"repo_id": "Qwen/Qwen3", "revision": "main", "repo_type": "model"},
    )

    assert response.status_code == 404


def test_cleanup_scan_and_delete_delegate(monkeypatch, tmp_path):
    import huggingface_pull.api as api_module

    calls = []

    def fake_cleanup_library(library_dir, delete, include_partials, older_than_days):
        calls.append((library_dir, delete, include_partials, older_than_days))
        return {"delete": delete}

    monkeypatch.setattr(api_module, "cleanup_library", fake_cleanup_library)
    client = TestClient(create_app(library_dir=tmp_path))

    scan_response = client.post("/api/cleanup/scan", json={})
    delete_response = client.post(
        "/api/cleanup/delete",
        json={"include_partials": True, "older_than_days": 3},
    )

    assert scan_response.status_code == 200
    assert scan_response.json() == {"delete": False}
    assert delete_response.status_code == 200
    assert delete_response.json() == {"delete": True}
    assert calls == [
        (tmp_path, False, False, 7),
        (tmp_path, True, True, 3),
    ]


def test_run_web_constructs_uvicorn_server_and_opens_browser_on_startup(monkeypatch, tmp_path):
    calls = []
    opened = []

    class FakeConfig:
        def __init__(self, app, host, port, log_level):
            self.app = app
            self.host = host
            self.port = port
            self.log_level = log_level
            calls.append(("config", self))

    class FakeServer:
        def __init__(self, config):
            self.config = config
            calls.append(("server", config))

        def run(self):
            calls.append(("run", self.config))
            assert opened == []
            run_app_lifespan(self.config.app)

    class FakeUvicorn:
        Config = FakeConfig
        Server = FakeServer

    monkeypatch.setattr(cli_module, "uvicorn", FakeUvicorn)
    monkeypatch.setattr(cli_module.webbrowser, "open", opened.append)

    result = cli_module.run_web(["--host", "0.0.0.0", "--port", "8123", "--library-dir", str(tmp_path)])

    assert result == 0
    assert opened == ["http://127.0.0.1:8123/"]
    assert [kind for kind, _ in calls] == ["config", "server", "run"]
    config = calls[0][1]
    assert config.host == "0.0.0.0"
    assert config.port == 8123
    assert config.log_level == "warning"
    assert config.app.state.queue.library_dir == tmp_path


def test_run_web_uses_localhost_browser_url_for_ipv6_wildcard(monkeypatch, tmp_path):
    opened = []

    class FakeConfig:
        def __init__(self, app, host, port, log_level):
            self.app = app
            self.host = host
            self.port = port
            self.log_level = log_level

    class FakeServer:
        def __init__(self, config):
            self.config = config

        def run(self):
            run_app_lifespan(self.config.app)

    class FakeUvicorn:
        Config = FakeConfig
        Server = FakeServer

    monkeypatch.setattr(cli_module, "uvicorn", FakeUvicorn)
    monkeypatch.setattr(cli_module.webbrowser, "open", opened.append)

    assert cli_module.run_web(["--host", "::", "--port", "8123", "--library-dir", str(tmp_path)]) == 0
    assert opened == ["http://127.0.0.1:8123/"]


def test_run_web_uses_default_host_port_and_library_dir(monkeypatch, tmp_path):
    calls = []

    class FakeConfig:
        def __init__(self, app, host, port, log_level):
            self.app = app
            calls.append((app, host, port, log_level))

    class FakeServer:
        def __init__(self, config):
            self.config = config

        def run(self):
            run_app_lifespan(self.config.app)

    class FakeUvicorn:
        Config = FakeConfig
        Server = FakeServer

    monkeypatch.setattr(cli_module, "uvicorn", FakeUvicorn)
    monkeypatch.setattr(cli_module.webbrowser, "open", lambda url: None)
    monkeypatch.setattr(cli_module, "default_library_dir", lambda: tmp_path)

    assert cli_module.run_web([]) == 0
    [(app, host, port, log_level)] = calls
    assert host == "127.0.0.1"
    assert port == 8019
    assert log_level == "warning"
    assert app.state.queue.library_dir == tmp_path
