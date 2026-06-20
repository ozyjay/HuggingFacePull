# File-Aware Downloads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pause/stop able to take effect after the current Hugging Face file finishes instead of waiting for a whole snapshot download.

**Architecture:** Replace the main `snapshot_download` pull path with a manifest-driven `hf_hub_download` loop. Keep the existing hard process stop as an escape hatch for blocked transfers.

**Tech Stack:** Python 3.12, `huggingface_hub.HfApi.model_info`, `huggingface_hub.hf_hub_download`, pytest, existing queue progress events.

---

### Task 1: File-Aware Pull Path

**Files:**
- Modify: `src/huggingface_pull/hub.py`
- Test: `tests/test_hub.py`

- [ ] **Step 1: Write the failing test**

Add a test that monkeypatches `HfApi` and `hf_hub_download`, requests stop after the first completed file, and asserts only the first file downloads and no install marker is written.

- [ ] **Step 2: Run the failing test**

Run: `python3 -m pytest tests/test_hub.py::test_pull_snapshot_stops_after_current_file_before_next_download -q`

Expected: fail because `pull_snapshot` still calls `snapshot_download`.

- [ ] **Step 3: Implement file planning and sequential file download**

Import `hf_hub_download`, fetch `model_info(..., files_metadata=True)`, filter siblings through `allow_patterns` and `ignore_patterns`, emit `model-plan`, `file-start`, and `file-complete`, call `hf_hub_download` once per file with `local_dir=target`, and check `stop_after_file()` after each file completes.

- [ ] **Step 4: Run focused hub tests**

Run: `python3 -m pytest tests/test_hub.py -q`

Expected: pass.

### Task 2: Queue Pause Semantics

**Files:**
- Modify: `src/huggingface_pull/queue.py`
- Test: `tests/test_queue.py`

- [ ] **Step 1: Write or adjust queue tests**

Assert cooperative stop/pause still marks the active item `stopped` when the pull function raises `DownloadStoppedAfterFile` after a file boundary.

- [ ] **Step 2: Keep queue implementation compatible**

No queue API change should be needed; `pull_snapshot` continues to use the existing `stop_after_file` callback.

- [ ] **Step 3: Run queue tests**

Run: `python3 -m pytest tests/test_queue.py -q`

Expected: pass.

### Task 3: Full Verification

**Files:**
- Test: full repository test suite

- [ ] **Step 1: Run all tests**

Run: `python3 -m pytest -q`

Expected: pass.

- [ ] **Step 2: Restart local server**

Restart the local `hfpull-web` process so the updated downloader is loaded.
