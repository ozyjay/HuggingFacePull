# HuggingFacePull

Local FastAPI tool for searching, queueing, and downloading Hugging Face Hub model snapshots.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

This project expects `python3` from the active `pyenv` version.

## Run the Web UI

```bash
hfpull-web
```

The web UI binds to `127.0.0.1:8019` by default and opens the local browser.

## Pull a Repo

```bash
hfpull Qwen/Qwen3-Embedding-0.6B --allow "*.json" --allow "*.safetensors"
```

Use `--dry-run` to verify the target path without downloading files:

```bash
hfpull openai-community/gpt2 --allow config.json --dry-run
```

## Cleanup

```bash
hfpull gc
hfpull gc --delete --include-partials --older-than-days 7
```

By default, snapshots are written under `~/.cache/huggingfacepull/library`.
Set `HUGGINGFACE_PULL_LIBRARY=/path/to/library` to use another location.

## Development

```bash
python3 -m pytest -v
python3 -m py_compile src/huggingface_pull/*.py tests/*.py
node --check src/huggingface_pull/web/app.js
```
