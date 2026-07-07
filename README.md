# HuggingFacePull

Local FastAPI tool for searching, queueing, and downloading Hugging Face Hub model snapshots.

## Setup

On Linux or macOS:

```bash
./scripts/install.sh
```

On Fedora, including Framework desktops, the installer can also install missing
system Python packages before creating the virtual environment:

```bash
./scripts/install.sh --install-system-deps
```

On Windows or PowerShell:

```powershell
./scripts/setup.ps1
```

This project expects `python3` from the active `pyenv` version.

## Run the Web UI

```powershell
./scripts/run.ps1
```

The web UI binds to `127.0.0.1:8019` by default and opens the local browser.

## Run the Desktop App

The desktop app is an Electron wrapper around the same local FastAPI server and
web UI.

```bash
npm install
npm run desktop:start
```

Build a standalone Fedora/Linux x64 desktop folder with:

```bash
npm run desktop:package
```

The packaged app is written to `out/HuggingFacePull-linux-x64/` and can be run
from that folder with:

```bash
./out/HuggingFacePull-linux-x64/huggingfacepull
```

Packaging creates `build/backend/.venv`, installs the runtime Python backend
there, and copies it into `resources/backend` so the packaged app does not need
the repo checkout or a manually prepared `.venv`. Local development still uses
the repo `.venv`, so run `./scripts/install.sh` before `npm run desktop:start`.

## Pull a Repo

```bash
hfpull Qwen/Qwen3-Embedding-0.6B --allow "*.json" --allow "*.safetensors"
```

Downloads use `huggingface_hub.snapshot_download()` with low concurrency by default.
Set `HUGGINGFACE_PULL_MAX_WORKERS` or pass `--max-workers` to change the worker count:

```bash
HUGGINGFACE_PULL_MAX_WORKERS=1 hfpull Qwen/Qwen3-Embedding-0.6B --allow "*.json" --allow "*.safetensors"
hfpull Qwen/Qwen3-Embedding-0.6B --allow "*.json" --allow "*.safetensors" --max-workers 1
```

Enable Xet for a model that requires it with:

```bash
hfpull huge-org/huge-model --xet
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

By default, model payloads are stored in the standard Hugging Face cache:
`~/.cache/huggingface/hub`.

HuggingFacePull keeps small `.huggingfacepull.json` metadata markers under
`~/.cache/huggingfacepull/library`. Set `HUGGINGFACE_PULL_LIBRARY=/path/to/library`
to move those metadata markers.

## Troubleshooting Downloads

First verify the Hugging Face client path directly:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download

with TemporaryDirectory(prefix="hfpull-smoke-") as temp_dir:
    snapshot_download(
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        allow_patterns=["*.json", "*.safetensors"],
        local_dir=Path(temp_dir) / "Qwen3-Embedding-0.6B",
        max_workers=1,
    )
```

HuggingFacePull uses plain HTTP/non-Xet transfers by default because that is
the reliable fallback for most downloads. For very large repos that require
Xet, tick **Use Xet transfer for this download** in the desktop/web UI or pass
`--xet` on the CLI.

The web launcher logs these diagnostics before the app starts:

```text
HF_HUB_DISABLE_XET
HF_HUB_DOWNLOAD_TIMEOUT
HF_HUB_ETAG_TIMEOUT
HUGGINGFACE_PULL_MAX_WORKERS
```

If a download repeatedly stalls or fails, clean stale partial files:

```bash
hfpull gc --include-partials
hfpull gc --delete --include-partials --older-than-days 0
```

For Open Day demos, use pre-cached models. Do not rely on live downloads during the demo.

## Development

```powershell
./scripts/test.ps1
```

Use `./scripts/test.ps1 -Install` to refresh the editable dev install before running tests.

If running tests directly with the active `pyenv` interpreter, install the dev dependencies there too:

```bash
python3 -m pip install -e ".[dev]"
```

`httpx2` must be available in whichever interpreter imports `starlette.testclient`; otherwise pytest emits a Starlette deprecation warning about falling back to `httpx`.
