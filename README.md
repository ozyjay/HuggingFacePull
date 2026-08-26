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
Live Hub search results automatically group clearly related variants from the
same model family while keeping every repository available to queue separately.
If a stale cached Hub token is rejected, public search retries anonymously;
private repositories and downloads still require a valid Hugging Face login.

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

On Fedora, build and install the desktop RPM in one step:

```bash
./scripts/package-install-fedora.sh
```

On a new Fedora development machine, the script can install its build tools too:

```bash
./scripts/package-install-fedora.sh --install-build-deps
```

Pass `--yes` to make the `dnf` operations non-interactive. The script runs
`npm ci`, builds the bundled desktop app and backend, creates the RPM, and then
installs (or reinstalls) that exact artifact. Because local development RPMs are
unsigned, the script disables signature checking for that command-line RPM only;
packages from configured repositories are still checked. To reuse an existing
`node_modules` directory, pass `--skip-npm-ci`.

The equivalent manual commands are:

```bash
sudo dnf install rpm-build
npm ci
npm run desktop:package:rpm
sudo dnf --setopt=localpkg_gpgcheck=0 install ./out/huggingfacepull-0.1.0-1.*.x86_64.rpm
```

The RPM installs the bundled app under `/opt/huggingfacepull`, adds a command at
`/usr/bin/huggingfacepull`, and registers HuggingFacePull in the desktop app menu.
In the desktop app, use **Storage → Choose folder…** to select the Hugging Face
cache directory. The selection is remembered per user and takes effect after the
automatic app restart; existing cache contents are not moved.

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

Kernel repositories use the same queue and cache flow. Pin executable kernel code to an
expected resolved commit and narrow the downloaded build variant, for example:

```bash
hfpull kernels-community/finegrained-fp8 --repo-type kernel --revision v3 \
  --expected-commit fcf89a79d85eab78182c62fb986ed01f2cbf7422 \
  --allow 'build/torch-rocm/*'
```

The request fails before download if the tag resolves to a different commit. The installed
metadata records both the requested tag and verified resolved revision.

## Reviewed Qwen3.5 Q8_0 presets

HuggingFacePull provides acquisition-only presets for the approved local Qwen3.5
Q8_0 GGUF files. Each preset pins both the requested revision and expected commit,
selects exactly one GGUF file, and uses the application's default non-Xet HTTP
transfer path. Runtime selection and execution remain the responsibility of
downstream applications.

```bash
hfpull preset list
hfpull preset pull qwen3.5-0.8b-q8_0
hfpull preset pull qwen3.5-2b-q8_0
hfpull preset pull qwen3.5-4b-q8_0
hfpull preset pull qwen3.5-9b-q8_0
```

| Preset | Repository | File | Pinned commit |
|---|---|---|---|
| `qwen3.5-0.8b-q8_0` | `bartowski/Qwen_Qwen3.5-0.8B-GGUF` | `Qwen_Qwen3.5-0.8B-Q8_0.gguf` | `f36b1ea49a332ede8fe5f389bbf5b3575ef71f48` |
| `qwen3.5-2b-q8_0` | `bartowski/Qwen_Qwen3.5-2B-GGUF` | `Qwen_Qwen3.5-2B-Q8_0.gguf` | `7d26695454df6de5fbcce2e58681e62dae06ce43` |
| `qwen3.5-4b-q8_0` | `bartowski/Qwen_Qwen3.5-4B-GGUF` | `Qwen_Qwen3.5-4B-Q8_0.gguf` | `4168f45a16a1290d65a4ec0fa312ae917a4c15d6` |
| `qwen3.5-9b-q8_0` | `bartowski/Qwen_Qwen3.5-9B-GGUF` | `Qwen_Qwen3.5-9B-Q8_0.gguf` | `182be2fd6c7bc44887d88a91cb03ff009cc9f549` |

## Completion markers

Each completed pull writes `.huggingfacepull.json` in the configured library.
New markers use the versioned `huggingfacepull-completion` format, version 2.
They are written atomically only after every selected file exists, has the
recorded size, and — when Hub LFS SHA-256 metadata is available — has been
hashed and checked locally.

The mandatory top-level fields are `format`, `version`, `repo_id`, `repo_type`,
`requested_revision`, `revision`, `expected_commit`, `resolved_revision`,
`snapshot_path`, `xet_enabled`, `size`, and `files`. `revision` is retained for
legacy consumers and equals `requested_revision`. `resolved_revision` is always
the immutable 40-character commit used for tree inspection and downloading.
`size` is the sum of the selected files, not the whole cache snapshot.

Every selected file entry contains `path`, `size`, `blob_id`, `lfs_sha256`,
`lfs_size`, `xet_hash`, and `verification`. Unavailable metadata is represented
as `null`. `verification` is `sha256` only when the local content matched the
recorded LFS SHA-256; otherwise it is `size_only`. `blob_id` and `xet_hash` are
identifiers, not interchangeable content checksums.

Consumers can locate markers under:

```text
<library>/<repo-id-with-slashes-replaced-by-->/<requested-revision>/.huggingfacepull.json
```

They can parse and validate a v2 marker without network access using the marker,
the indicated snapshot, and the recorded file sizes and SHA-256 values. Markers
without `format` and `version` are legacy markers and remain readable. Unknown
marker formats or versions are rejected rather than guessed.

To upgrade existing markers without downloading or rehashing files, use:

```bash
hfpull upgrade-metadata
```

The upgrade is best-effort. It enriches entries from local Hugging Face cached
tree metadata when present, preserves `size_only` verification for legacy
content, and skips markers that cannot be safely tied to an immutable cache
snapshot.

## Cleanup

```bash
hfpull gc
hfpull gc --delete --include-partials --older-than-days 7
```

By default, model payloads are stored in the standard Hugging Face cache:
`~/.cache/huggingface/hub`.
HuggingFacePull also honours `HF_HOME` and `HF_HUB_CACHE`, with
`HF_HUB_CACHE` taking precedence when both are set.

HuggingFacePull keeps small `.huggingfacepull.json` metadata markers under
`~/.cache/huggingfacepull/library`. Set `HUGGINGFACE_PULL_LIBRARY=/path/to/library`
to move those metadata markers.

In the web UI, **Remove from list** deletes only the HuggingFacePull metadata
record and leaves the cached snapshot available locally. **Delete from disk**
deletes the cached snapshot and its metadata record; cached blobs that are still
used by another revision are preserved. Both actions require confirmation.

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
