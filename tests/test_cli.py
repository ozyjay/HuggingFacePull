from huggingface_pull.cli import build_parser
from huggingface_pull.cli import run_web


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


def test_run_web_without_args_returns_non_zero_placeholder():
    assert run_web([]) != 0
