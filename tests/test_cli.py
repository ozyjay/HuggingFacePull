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
            "--max-workers",
            "2",
        ]
    )

    assert args.repo_id == "Qwen/Qwen3-Embedding-0.6B"
    assert args.revision == "main"
    assert args.allow == ["*.safetensors"]
    assert args.ignore == ["*.bin"]
    assert args.repo_type == "model"
    assert args.max_workers == 2


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


def test_run_web_logs_pre_launch_hf_diagnostics(monkeypatch, tmp_path):
    log_events = []
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "30")
    monkeypatch.setenv("HUGGINGFACE_PULL_MAX_WORKERS", "3")
    monkeypatch.setattr(cli, "write_log", lambda message, **fields: log_events.append((message, fields)))

    class FakeServer:
        def __init__(self, config):
            self.config = config

        def run(self):
            pass

    monkeypatch.setattr(cli.uvicorn, "Server", FakeServer)

    assert cli.run_web(["--library-dir", str(tmp_path)]) == 0
    assert (
        "pre-launch diagnostics",
        {
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
            "HF_HUB_ETAG_TIMEOUT": "30",
            "HUGGINGFACE_PULL_MAX_WORKERS": "3",
        },
    ) in log_events


def test_main_calls_pull_snapshot(monkeypatch, tmp_path):
    calls = []

    def fake_pull_snapshot(ref, library_dir, **kwargs):
        calls.append((ref.repo_id, library_dir, kwargs["max_workers"]))
        return tmp_path / "Qwen--Qwen3" / "main"

    monkeypatch.setattr(cli, "pull_snapshot", fake_pull_snapshot)

    assert cli.main(["Qwen/Qwen3", "--library-dir", str(tmp_path), "--max-workers", "4"]) == 0
    assert calls == [("Qwen/Qwen3", tmp_path, 4)]


def test_gc_calls_cleanup(monkeypatch, tmp_path):
    calls = []

    def fake_cleanup_library(library_dir, **kwargs):
        calls.append((library_dir, kwargs["delete"]))
        return {"stale_partial_count": 0, "deleted": []}

    monkeypatch.setattr(cli, "cleanup_library", fake_cleanup_library)

    assert cli.main(["gc", "--library-dir", str(tmp_path)]) == 0
    assert calls == [(tmp_path, False)]


def test_gc_reports_incomplete_snapshot_counts(monkeypatch, tmp_path, capsys):
    def fake_cleanup_library(library_dir, **kwargs):
        return {
            "stale_partial_count": 1,
            "incomplete_snapshot_count": 2,
            "deleted": ["partial"],
            "deleted_snapshots": ["first", "second"],
        }

    monkeypatch.setattr(cli, "cleanup_library", fake_cleanup_library)

    assert cli.main(["gc", "--delete", "--include-partials", "--library-dir", str(tmp_path)]) == 0

    assert (
        capsys.readouterr().out.strip()
        == "Stale partials: 1; deleted: 1; incomplete snapshots: 2; snapshots deleted: 2"
    )


def test_main_reports_hub_ref_errors(monkeypatch, capsys):
    def bad_hub_ref(**kwargs):
        raise RuntimeError("bad ref")

    monkeypatch.setattr(cli, "HubRef", bad_hub_ref)

    assert cli.main(["Qwen/Qwen3"]) == 1
    assert "Error: bad ref" in capsys.readouterr().err
