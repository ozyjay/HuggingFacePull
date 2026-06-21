import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "list_hf_caches.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("list_hf_caches", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discovers_env_defaults_and_project_local_caches(tmp_path, monkeypatch):
    module = load_script_module()
    home = tmp_path / "home"
    project = tmp_path / "project"
    hf_hub = home / ".cache" / "huggingface" / "hub"
    datasets = home / ".cache" / "huggingface" / "datasets"
    project_cache = project / ".cache" / "huggingface"
    env_cache = tmp_path / "env-cache"
    hf_hub.mkdir(parents=True)
    datasets.mkdir(parents=True)
    project_cache.mkdir(parents=True)
    env_cache.mkdir(parents=True)
    (hf_hub / "config.json").write_text("{}", encoding="utf-8")
    (project_cache / "model.bin").write_bytes(b"abc")
    (env_cache / "weights.bin").write_bytes(b"12345")
    monkeypatch.setenv("HF_HUB_CACHE", str(env_cache))
    monkeypatch.delenv("HF_HOME", raising=False)

    entries = module.discover_caches(home=home, project_root=project)

    by_label = {entry["label"]: entry for entry in entries}
    assert by_label["HF_HUB_CACHE"]["path"] == str(env_cache)
    assert by_label["default: huggingface hub"]["path"] == str(hf_hub)
    assert by_label["default: datasets"]["path"] == str(datasets)
    assert by_label["project: .cache/huggingface"]["path"] == str(project_cache)
    assert "default: huggingfacepull" not in by_label
    assert "project: hf-test" not in by_label
    assert by_label["HF_HUB_CACHE"]["exists"] is True
    assert by_label["HF_HUB_CACHE"]["file_count"] == 1
    assert by_label["HF_HUB_CACHE"]["size_bytes"] == 5


def test_json_output_is_parseable(tmp_path, capsys):
    module = load_script_module()
    home = tmp_path / "home"
    cache = home / ".cache" / "huggingface" / "hub"
    cache.mkdir(parents=True)
    (cache / "model.bin").write_bytes(b"abcd")

    exit_code = module.main(["--json", "--home", str(home), "--project-root", str(tmp_path)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    labels = {entry["label"] for entry in payload}
    assert "default: huggingface hub" in labels
    assert "default: huggingfacepull" not in labels
    assert "project: hf-test" not in labels
    assert any(entry["size_bytes"] == 4 for entry in payload)


def test_table_output_marks_missing_paths(tmp_path, capsys):
    module = load_script_module()

    exit_code = module.main(["--home", str(tmp_path / "home"), "--project-root", str(tmp_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Label" in output
    assert "Path" in output
    assert "missing" in output
