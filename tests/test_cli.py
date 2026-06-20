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
