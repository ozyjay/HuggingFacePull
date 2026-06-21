from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_powershell_workflow_scripts_exist():
    scripts = ROOT / "scripts"

    assert (scripts / "setup.ps1").is_file()
    assert (scripts / "test.ps1").is_file()
    assert (scripts / "run.ps1").is_file()
    assert (scripts / "common.ps1").is_file()
    assert (scripts / "list_hf_caches.py").is_file()


def test_powershell_scripts_cover_core_workflows():
    setup = (ROOT / "scripts" / "setup.ps1").read_text(encoding="utf-8")
    test = (ROOT / "scripts" / "test.ps1").read_text(encoding="utf-8")
    run = (ROOT / "scripts" / "run.ps1").read_text(encoding="utf-8")
    cache_lister = (ROOT / "scripts" / "list_hf_caches.py").read_text(encoding="utf-8")

    assert 'Invoke-Checked "python3" "-m" "venv" ".venv"' in setup
    assert '".[dev]"' in setup
    assert "pytest" in test
    assert "py_compile" in test
    assert "hfpull-web" in run
    assert "HF_HUB_CACHE" in cache_lister
    assert "--json" in cache_lister


def test_gitignore_blocks_project_local_model_artifacts():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    for pattern in (
        "hf-test/",
        "models/",
        "library/",
        ".cache/huggingface/",
        "*.safetensors",
        "*.gguf",
        "*.bin",
    ):
        assert pattern in gitignore
