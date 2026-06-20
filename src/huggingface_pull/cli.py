from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import uvicorn

from .api import create_app
from .app_logging import write_log
from .config import DEFAULT_ENDPOINT, default_library_dir
from .hub import HubRef, cleanup_library, pull_snapshot


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
    parser.add_argument("--library-dir", type=Path, default=default_library_dir())
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_gc_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean up stale HuggingFacePull library files."
    )
    parser.add_argument("--library-dir", type=Path, default=default_library_dir())
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--include-partials", action="store_true")
    parser.add_argument("--older-than-days", type=int, default=7)
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


def _log(message: str, /, **fields: object) -> None:
    try:
        write_log(message, **fields)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["gc"]:
        args = build_gc_parser().parse_args(argv[1:])
        try:
            report = cleanup_library(
                args.library_dir.expanduser(),
                delete=args.delete,
                include_partials=args.include_partials,
                older_than_days=args.older_than_days,
            )
        except Exception as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        stale_count = report.get("stale_partial_count", 0)
        deleted_count = len(report.get("deleted", []))
        print(f"Stale partials: {stale_count}; deleted: {deleted_count}")
        return 0

    args = build_parser().parse_args(argv)
    try:
        ref = HubRef(
            repo_id=args.repo_id,
            revision=args.revision,
            repo_type=args.repo_type,
            allow_patterns=args.allow,
            ignore_patterns=args.ignore,
        )
        snapshot_path = pull_snapshot(
            ref,
            library_dir=args.library_dir.expanduser(),
            endpoint=args.endpoint,
            dry_run=args.dry_run,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(snapshot_path)
    return 0


def run_web(argv: list[str] | None = None) -> int:
    args = build_web_parser().parse_args(argv)
    app = create_app(library_dir=args.library_dir.expanduser())
    browser_url = _browser_url(args.host, args.port)
    _log(
        "web server starting",
        host=args.host,
        port=args.port,
        library_dir=args.library_dir.expanduser(),
        endpoint=DEFAULT_ENDPOINT,
    )

    def open_browser() -> None:
        webbrowser.open(browser_url)

    app.router.on_startup.append(open_browser)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()
    return 0
