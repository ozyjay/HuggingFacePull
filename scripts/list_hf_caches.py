#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ENV_CACHE_VARS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_DATASETS_CACHE",
)

PROJECT_CACHE_PATHS = (
    ".cache/huggingface",
    "hf-test",
    "models",
    "cache",
    "library",
)


def discover_caches(
    *,
    home: Path | None = None,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    home = Path.home() if home is None else Path(home).expanduser()
    project_root = Path.cwd() if project_root is None else Path(project_root).expanduser()
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for name in ENV_CACHE_VARS:
        configured = os.environ.get(name)
        if configured:
            _add_entry(entries, seen, name, Path(configured).expanduser())
        else:
            entries.append(_entry(name, None, source="env", configured=False))

    hf_home = Path(os.environ.get("HF_HOME", home / ".cache" / "huggingface")).expanduser()
    defaults = (
        ("default: huggingface home", hf_home),
        ("default: huggingface hub", hf_home / "hub"),
        ("default: datasets", hf_home / "datasets"),
        ("default: transformers", hf_home / "transformers"),
    )
    for label, path in defaults:
        _add_entry(entries, seen, label, path)

    for relative in PROJECT_CACHE_PATHS:
        _add_entry(entries, seen, f"project: {relative}", project_root / relative)

    return entries


def _add_entry(
    entries: list[dict[str, Any]],
    seen: set[Path],
    label: str,
    path: Path,
) -> None:
    try:
        key = path.resolve()
    except OSError:
        key = path.absolute()
    if key in seen:
        return
    seen.add(key)
    entries.append(_entry(label, path))


def _entry(
    label: str,
    path: Path | None,
    *,
    source: str = "path",
    configured: bool = True,
) -> dict[str, Any]:
    if path is None:
        return {
            "label": label,
            "path": None,
            "exists": False,
            "file_count": 0,
            "size_bytes": 0,
            "size": "0 B",
            "status": "unset",
            "source": source,
            "configured": configured,
        }

    exists = path.exists()
    file_count = _file_count(path) if exists else 0
    size_bytes = _directory_size(path) if exists else 0
    return {
        "label": label,
        "path": str(path),
        "exists": exists,
        "file_count": file_count,
        "size_bytes": size_bytes,
        "size": _format_bytes(size_bytes),
        "status": "present" if exists else "missing",
        "source": source,
        "configured": configured,
    }


def _file_count(path: Path) -> int:
    if path.is_file():
        return 1
    count = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                count += 1
        except OSError:
            continue
    return count


def _directory_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def _format_bytes(size: int) -> str:
    value = float(size)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def print_table(entries: list[dict[str, Any]]) -> None:
    columns = ("Label", "Status", "Files", "Size", "Path")
    rows = [
        (
            entry["label"],
            entry["status"],
            str(entry["file_count"]),
            entry["size"],
            entry["path"] or "",
        )
        for entry in entries
    ]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]
    print("  ".join(column.ljust(widths[index]) for index, column in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List Hugging Face cache locations and local cache candidates."
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--home", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--project-root", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = discover_caches(home=args.home, project_root=args.project_root)
    if args.json:
        print(json.dumps(entries, indent=2, sort_keys=True))
    else:
        print_table(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
