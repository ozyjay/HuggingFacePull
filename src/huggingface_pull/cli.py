from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face Hub model snapshots into a local library."
    )
    parser.add_argument(
        "repo_id",
        help="Hub repo ID, for example Qwen/Qwen3-Embedding-0.6B",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--repo-type",
        choices=["model", "dataset", "space"],
        default="model",
    )
    parser.add_argument("--allow", action="append", default=[], help="Glob pattern to include.")
    parser.add_argument("--ignore", action="append", default=[], help="Glob pattern to exclude.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_web_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Launch the local HuggingFacePull web UI."
    )


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0


def run_web(argv: list[str] | None = None) -> int:
    build_web_parser().parse_args(argv)
    print("The HuggingFacePull web UI is not implemented yet.", file=sys.stderr)
    return 1
