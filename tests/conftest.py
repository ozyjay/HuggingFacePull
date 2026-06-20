from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def redirect_app_log_for_tests(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "HUGGINGFACE_PULL_LOG_FILE",
        str(tmp_path / "logs" / "HuggingFacePull" / "app.log"),
    )

