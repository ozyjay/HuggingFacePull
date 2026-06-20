import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from huggingface_pull.app_logging import write_log
from huggingface_pull.config import default_library_dir, default_log_file, safe_repo_dir_name
from huggingface_pull.models import CleanupRequest, QueueRequest
from huggingface_pull.models import InstalledRemoveRequest


def test_default_library_dir_uses_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_PULL_LIBRARY", str(tmp_path / "models"))
    assert default_library_dir() == tmp_path / "models"


def test_default_library_dir_expands_user_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HUGGINGFACE_PULL_LIBRARY", "~/hf-models")

    assert default_library_dir() == tmp_path / "hf-models"


def test_default_log_file_expands_user_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HUGGINGFACE_PULL_LOG_FILE", "~/hf.log")

    assert default_log_file() == tmp_path / "hf.log"


def test_safe_repo_dir_name_preserves_repo_identity_without_slashes():
    assert safe_repo_dir_name("Qwen/Qwen3-Embedding-0.6B") == "Qwen--Qwen3-Embedding-0.6B"


def test_safe_repo_dir_name_strips_surrounding_whitespace_before_replacing_slashes():
    assert safe_repo_dir_name(" Qwen/Qwen3 ") == "Qwen--Qwen3"


def test_write_log_writes_json_line_with_stringified_fields(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "app.log"
    monkeypatch.setenv("HUGGINGFACE_PULL_LOG_FILE", str(log_file))

    write_log("queued", repo_id="Qwen/Qwen3", count=2)

    assert log_file.parent.exists()
    event = json.loads(log_file.read_text(encoding="utf-8"))
    assert event["message"] == "queued"
    assert event["fields"] == {"count": "2", "repo_id": "Qwen/Qwen3"}
    assert "timestamp" in event


def test_write_log_keeps_caller_fields_from_overriding_reserved_keys(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "app.log"
    monkeypatch.setenv("HUGGINGFACE_PULL_LOG_FILE", str(log_file))

    write_log("queued", timestamp="caller timestamp", message="caller message")

    event = json.loads(log_file.read_text(encoding="utf-8"))
    assert event["message"] == "queued"
    assert event["timestamp"] != "caller timestamp"
    assert event["fields"] == {
        "message": "caller message",
        "timestamp": "caller timestamp",
    }


def test_queue_request_defaults_are_independent_lists():
    first = QueueRequest(repo_id="Qwen/Qwen3")
    second = QueueRequest(repo_id="Qwen/Qwen3")

    first.allow_patterns.append("*.safetensors")

    assert first.revision == "main"
    assert first.repo_type == "model"
    assert first.local_dir is None
    assert second.allow_patterns == []
    assert second.ignore_patterns == []


def test_queue_request_strips_repo_id_surrounding_whitespace():
    request = QueueRequest(repo_id=" Qwen/Qwen3 ")

    assert request.repo_id == "Qwen/Qwen3"


def test_queue_request_rejects_whitespace_only_repo_id():
    with pytest.raises(ValidationError):
        QueueRequest(repo_id="   ")


def test_installed_remove_request_strips_repo_id_surrounding_whitespace():
    request = InstalledRemoveRequest(repo_id=" Qwen/Qwen3 ")

    assert request.repo_id == "Qwen/Qwen3"


def test_installed_remove_request_rejects_whitespace_only_repo_id():
    with pytest.raises(ValidationError):
        InstalledRemoveRequest(repo_id="   ")


def test_cleanup_request_defaults_and_validates_days():
    request = CleanupRequest()

    assert request.include_partials is False
    assert request.older_than_days == 7


def test_cleanup_request_rejects_negative_days():
    with pytest.raises(ValidationError):
        CleanupRequest(older_than_days=-1)
