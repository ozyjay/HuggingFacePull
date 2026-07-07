from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
import starlette.testclient


os.environ.setdefault("HF_HUB_CACHE", "/tmp/huggingfacepull-test-hub-cache")


class LocalASGITestClient:
    __test__ = False

    def __init__(self, app, *, raise_server_exceptions: bool = True, **_: Any) -> None:
        self.app = app
        self.raise_server_exceptions = raise_server_exceptions
        self.base_url = "http://testserver"

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        static_response = self._static_response(method, path)
        if static_response is not None:
            return static_response

        async def run_request() -> httpx.Response:
            transport = httpx.ASGITransport(
                app=self.app,
                raise_app_exceptions=self.raise_server_exceptions,
            )
            async with httpx.AsyncClient(transport=transport, base_url=self.base_url) as client:
                return await client.request(method, path, **kwargs)

        return anyio.run(run_request)

    def _static_response(self, method: str, path: str) -> httpx.Response | None:
        if method.upper() != "GET":
            return None
        relative_path = "index.html" if path == "/" else path.removeprefix("/")
        if relative_path not in {"index.html", "app.js", "styles.css"}:
            return None
        web_dir = Path(__file__).resolve().parents[1] / "src" / "huggingface_pull" / "web"
        file_path = web_dir / relative_path
        if not file_path.is_file():
            return None
        content_type = "text/html" if file_path.suffix == ".html" else "text/plain"
        return httpx.Response(
            200,
            text=file_path.read_text(encoding="utf-8"),
            headers={"content-type": content_type},
        )


starlette.testclient.TestClient = LocalASGITestClient


@pytest.fixture(autouse=True)
def isolate_user_state_for_tests(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "HuggingFacePull" / "app.log"
    hub_cache = tmp_path / "hf-hub-cache"
    monkeypatch.setenv("HUGGINGFACE_PULL_LOG_FILE", str(log_file))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))

    try:
        import huggingface_pull.hub as hub
        import huggingface_pull.queue as queue

        monkeypatch.setattr(hub, "HF_HUB_CACHE", str(hub_cache))
        monkeypatch.setattr(queue.hub, "HF_HUB_CACHE", str(hub_cache))
    except Exception:
        pass
