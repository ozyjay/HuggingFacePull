# HuggingFacePull

Local FastAPI tool for searching, queueing, and downloading Hugging Face Hub model snapshots.

## Setup

```powershell
./scripts/setup.ps1
```

This project expects `python3` from the active `pyenv` version.

## Run the Web UI

```powershell
./scripts/run.ps1
```

The web UI binds to `127.0.0.1:8019` by default and opens the local browser.

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

## Troubleshooting Downloads

First verify the Hugging Face client path directly:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-Embedding-0.6B",
    allow_patterns=["*.json", "*.safetensors"],
    local_dir="hf-test/Qwen3-Embedding-0.6B",
    max_workers=1,
)
```

HuggingFacePull forces `HF_HUB_DISABLE_XET=1` during transfers. Xet has been tried;
plain HTTP/non-Xet mode is the reliable fallback for this app.

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
