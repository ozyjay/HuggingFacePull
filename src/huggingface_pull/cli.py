from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import uvicorn

from .api import create_app
from .config import default_library_dir


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
    parser = argparse.ArgumentParser(
        description="Run the HuggingFacePull local web app."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--library-dir", type=Path, default=default_library_dir())
    return parser


def _browser_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    return f"http://{browser_host}:{port}/"


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0


def run_web(argv: list[str] | None = None) -> int:
    args = build_web_parser().parse_args(argv)
    app = create_app(library_dir=args.library_dir.expanduser())
    browser_url = _browser_url(args.host, args.port)

    def open_browser() -> None:
        webbrowser.open(browser_url)

    app.router.on_startup.append(open_browser)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()
    return 0
