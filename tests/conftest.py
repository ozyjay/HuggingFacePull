from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_user_state_for_tests(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "HuggingFacePull" / "app.log"
    hub_cache = tmp_path / "huggingface-hub"
    monkeypatch.setenv("HUGGINGFACE_PULL_LOG_FILE", str(log_file))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))

    import huggingface_pull.hub as hub

    monkeypatch.setattr(hub, "HF_HUB_CACHE", str(hub_cache))
