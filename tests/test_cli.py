import huggingface_pull.cli as cli
from huggingface_pull.cli import build_parser


def test_parser_accepts_repo_revision_and_filters():
    args = build_parser().parse_args(
        [
            "Qwen/Qwen3-Embedding-0.6B",
            "--revision",
            "main",
            "--allow",
            "*.safetensors",
            "--ignore",
            "*.bin",
            "--repo-type",
            "model",
        ]
    )

    assert args.repo_id == "Qwen/Qwen3-Embedding-0.6B"
    assert args.revision == "main"
    assert args.allow == ["*.safetensors"]
    assert args.ignore == ["*.bin"]
    assert args.repo_type == "model"


def test_run_web_without_args_starts_server(monkeypatch):
    opened = []
    ran = []

    class FakeServer:
        def __init__(self, config):
            self.config = config

        def run(self):
            for handler in self.config.app.router.on_startup:
                handler()
            ran.append(True)

    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    monkeypatch.setattr(cli.uvicorn, "Server", FakeServer)

    assert cli.run_web([]) == 0
    assert ran == [True]
    assert opened == ["http://127.0.0.1:8019/"]


def test_main_calls_pull_snapshot(monkeypatch, tmp_path):
    calls = []

    def fake_pull_snapshot(ref, library_dir, **kwargs):
        calls.append((ref.repo_id, library_dir))
        return tmp_path / "Qwen--Qwen3" / "main"

    monkeypatch.setattr(cli, "pull_snapshot", fake_pull_snapshot)

    assert cli.main(["Qwen/Qwen3", "--library-dir", str(tmp_path)]) == 0
    assert calls == [("Qwen/Qwen3", tmp_path)]


def test_gc_calls_cleanup(monkeypatch, tmp_path):
    calls = []

    def fake_cleanup_library(library_dir, **kwargs):
        calls.append((library_dir, kwargs["delete"]))
        return {"stale_partial_count": 0, "deleted": []}

    monkeypatch.setattr(cli, "cleanup_library", fake_cleanup_library)

    assert cli.main(["gc", "--library-dir", str(tmp_path)]) == 0
    assert calls == [(tmp_path, False)]


def test_main_reports_hub_ref_errors(monkeypatch, capsys):
    def bad_hub_ref(**kwargs):
        raise RuntimeError("bad ref")

    monkeypatch.setattr(cli, "HubRef", bad_hub_ref)

    assert cli.main(["Qwen/Qwen3"]) == 1
    assert "Error: bad ref" in capsys.readouterr().err
