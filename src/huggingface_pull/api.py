from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_ENDPOINT, default_library_dir
from .hub import HubRef, cached_hub_models, cleanup_library, installed_models, remove_installed_model, repo_files, search_models
from .models import CleanupRequest, InstalledRemoveRequest, QueueRequest
from .queue import DownloadQueue


WEB_DIR = Path(__file__).with_name("web")


def create_app(
    library_dir: Path | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    token: str | None = None,
) -> FastAPI:
    queue = DownloadQueue(
        library_dir=library_dir or default_library_dir(),
        endpoint=endpoint,
        token=token,
    )
    app = FastAPI(title="HuggingFacePull")
    app.state.queue = queue

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        snapshot = queue.snapshot()
        snapshot["cached_models"] = cached_hub_models()
        return snapshot

    @app.get("/api/search")
    def search(q: str = "") -> dict[str, Any]:
        return search_models(q, endpoint=endpoint, token=token)

    @app.get("/api/models/{repo_id:path}/files")
    def files(
        repo_id: str,
        revision: str = "main",
        repo_type: str = "model",
    ) -> dict[str, Any]:
        if repo_type != "model":
            raise HTTPException(
                status_code=400,
                detail="File listing currently supports model repos only.",
            )
        return repo_files(
            HubRef(repo_id=repo_id, revision=revision, repo_type=repo_type),
            endpoint=endpoint,
            token=token,
        )

    @app.post("/api/queue")
    def add(payload: QueueRequest) -> dict[str, Any]:
        requested = payload.model_dump()
        for installed in installed_models(queue.library_dir):
            if (
                installed.get("repo_id") == requested["repo_id"]
                and installed.get("revision", "main") == requested["revision"]
                and installed.get("repo_type", "model") == requested["repo_type"]
            ):
                raise HTTPException(status_code=409, detail="Snapshot is already installed")
        for cached in cached_hub_models():
            if (
                cached.get("repo_id") == requested["repo_id"]
                and cached.get("revision", requested["revision"]) == requested["revision"]
                and cached.get("repo_type", "model") == requested["repo_type"]
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Model is already available in the Hugging Face cache",
                )
        return queue.add(payload.model_dump())

    @app.post("/api/start")
    def start() -> dict[str, Any]:
        queue.start()
        return queue.snapshot()

    @app.post("/api/pause")
    def pause() -> dict[str, Any]:
        queue.pause_after_current()
        return queue.snapshot()

    @app.post("/api/stop-after-file")
    def stop_after_file() -> dict[str, Any]:
        return queue.stop_after_current_file()

    @app.post("/api/retry/{item_id}")
    def retry(item_id: str) -> dict[str, Any]:
        try:
            return queue.retry(item_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/remove/{item_id}")
    def remove(item_id: str) -> dict[str, bool]:
        try:
            queue.remove(item_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"ok": True}

    @app.post("/api/installed/remove")
    def remove_installed(payload: InstalledRemoveRequest) -> dict[str, bool]:
        try:
            remove_installed_model(queue.library_dir, HubRef(**payload.model_dump()))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {"ok": True}

    @app.post("/api/cleanup/scan")
    def cleanup_scan(payload: CleanupRequest = CleanupRequest()) -> dict[str, Any]:
        return cleanup_library(queue.library_dir, delete=False, **payload.model_dump())

    @app.post("/api/cleanup/delete")
    def cleanup_delete(payload: CleanupRequest = CleanupRequest()) -> dict[str, Any]:
        return cleanup_library(queue.library_dir, delete=True, **payload.model_dump())

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return app
