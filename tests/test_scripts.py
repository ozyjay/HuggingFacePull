import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_workflow_scripts_exist():
    scripts = ROOT / "scripts"

    assert (ROOT / "package.json").is_file()
    assert (ROOT / "desktop" / "main.cjs").is_file()
    assert (ROOT / "desktop" / "package.cjs").is_file()
    assert (scripts / "install.sh").is_file()
    assert (scripts / "setup.ps1").is_file()
    assert (scripts / "test.ps1").is_file()
    assert (scripts / "run.ps1").is_file()
    assert (scripts / "common.ps1").is_file()
    assert (scripts / "list_hf_caches.py").is_file()


def test_workflow_scripts_cover_core_workflows():
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    desktop_main = (ROOT / "desktop" / "main.cjs").read_text(encoding="utf-8")
    desktop_package = (ROOT / "desktop" / "package.cjs").read_text(encoding="utf-8")
    install = (ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
    setup = (ROOT / "scripts" / "setup.ps1").read_text(encoding="utf-8")
    test = (ROOT / "scripts" / "test.ps1").read_text(encoding="utf-8")
    run = (ROOT / "scripts" / "run.ps1").read_text(encoding="utf-8")
    cache_lister = (ROOT / "scripts" / "list_hf_caches.py").read_text(encoding="utf-8")

    assert package["main"] == "desktop/main.cjs"
    assert package["scripts"]["desktop:start"] == "electron ."
    assert package["scripts"]["desktop:package"] == "node desktop/package.cjs"
    assert "@electron/packager" in package["devDependencies"]
    assert "executableName: \"huggingfacepull\"" in desktop_package
    assert "/^\\/\\.venv($|\\/)/" in desktop_package
    assert "/^\\/node_modules($|\\/)/" in desktop_package
    assert "BrowserWindow" in desktop_main
    assert "--no-browser" in desktop_main
    assert "DEFAULT_PORT = 8019" in desktop_main
    assert "probeExistingServer" in desktop_main
    assert "HFPULL_DESKTOP_PORT" in desktop_main
    assert "portAcceptsConnections" in desktop_main
    assert "nodeIntegration: false" in desktop_main
    assert "contextIsolation: true" in desktop_main
    assert "Detected platform" in install
    assert "HF_HUB_DISABLE_XET=1" in install
    assert "fedora" in install
    assert "debian" in install
    assert "macos" in install
    assert "python3-venv" in install
    assert 'pip install -e "$install_target"' in install
    assert 'Invoke-Checked "python3" "-m" "venv" ".venv"' in setup
    assert '".[dev]"' in setup
    assert "pytest" in test
    assert "py_compile" in test
    assert "desktop/main.cjs" in test
    assert "desktop/package.cjs" in test
    assert "hfpull-web" in run
    assert "HF_HUB_DISABLE_XET" in run
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
