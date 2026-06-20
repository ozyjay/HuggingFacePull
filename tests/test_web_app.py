import subprocess
import textwrap


def test_search_install_state_matches_repo_revision_and_type():
    script = textwrap.dedent(
        """
        const assert = require("node:assert/strict");
        const fs = require("node:fs");
        const vm = require("node:vm");

        const context = {
          window: {},
          document: {
            addEventListener() {},
            getElementById() { return null; },
          },
          setInterval() {},
          fetch() {},
        };
        vm.createContext(context);
        vm.runInContext(fs.readFileSync("src/huggingface_pull/web/app.js", "utf8"), context);

        const installed = [
          { repo_id: "Qwen/Qwen3", revision: "main", repo_type: "model" },
          { repo_id: "Qwen/Qwen3", revision: "v2", repo_type: "dataset" },
        ];
        const cached = [
          { repo_id: "Qwen/Qwen2.5-0.5B", repo_type: "model" },
        ];

        assert.equal(
          context.window.HuggingFacePull.isInstalledSnapshot(
            installed,
            cached,
            "Qwen/Qwen3",
            "main",
            "model",
          ),
          true,
        );
        assert.equal(
          context.window.HuggingFacePull.isInstalledSnapshot(
            installed,
            cached,
            "Qwen/Qwen3",
            "v2",
            "model",
          ),
          false,
        );
        assert.equal(
          context.window.HuggingFacePull.isInstalledSnapshot(
            installed,
            cached,
            "Qwen/Qwen3",
            "v2",
            "dataset",
          ),
          true,
        );
        assert.equal(
          context.window.HuggingFacePull.isInstalledSnapshot(
            installed,
            cached,
            "Qwen/Qwen2.5-0.5B",
            "main",
            "model",
          ),
          true,
        );
        """
    )

    subprocess.run(["node", "-e", script], check=True)


def test_download_status_helpers_show_running_and_unknown_total_details():
    script = textwrap.dedent(
        """
        const assert = require("node:assert/strict");
        const fs = require("node:fs");
        const vm = require("node:vm");

        const context = {
          window: {},
          document: {
            addEventListener() {},
            getElementById() { return null; },
          },
          setInterval() {},
          fetch() {},
        };
        vm.createContext(context);
        vm.runInContext(fs.readFileSync("src/huggingface_pull/web/app.js", "utf8"), context);

        assert.equal(
          context.window.HuggingFacePull.downloadStatusLine({
            status: "running",
            progress: {
              phase: "downloading",
              overall: {
                downloaded: 5242880,
                total: 10485760,
                percent: 50,
                bytes_per_second: 1048576,
                eta_seconds: 5,
              },
              current_file: { name: "snapshot" },
            },
          }),
          "Downloading snapshot | 50.0% | 5.00 MB / 10.0 MB | 1.00 MB/s | ETA 5s",
        );

        assert.equal(
          context.window.HuggingFacePull.downloadStatusLine({
            status: "running",
            progress: {
              phase: "downloading",
              overall: { downloaded: 1536, total: null, percent: null },
              current_file: { path: "weights.safetensors" },
            },
          }),
          "Downloading weights.safetensors | Downloaded 1.50 KB | total calculating...",
        );
        """
    )

    subprocess.run(["node", "-e", script], check=True)


def test_cleanup_summary_mentions_incomplete_snapshots():
    script = textwrap.dedent(
        """
        const assert = require("node:assert/strict");
        const fs = require("node:fs");
        const vm = require("node:vm");

        const context = {
          window: {},
          document: {
            addEventListener() {},
            getElementById() { return null; },
          },
          setInterval() {},
          fetch() {},
        };
        vm.createContext(context);
        vm.runInContext(fs.readFileSync("src/huggingface_pull/web/app.js", "utf8"), context);

        assert.equal(
          context.window.HuggingFacePull.cleanupSummaryLine({
            dry_run: true,
            stale_partials: [{ path: "one" }],
            incomplete_snapshots: [{ path: "snapshot" }, { path: "other" }],
          }),
          "Scan found 1 stale file and 2 incomplete snapshots.",
        );

        assert.equal(
          context.window.HuggingFacePull.cleanupSummaryLine({
            dry_run: false,
            stale_partials: [],
            incomplete_snapshots: [{ path: "snapshot" }],
          }),
          "Deleted 0 stale files and 1 incomplete snapshot.",
        );
        """
    )

    subprocess.run(["node", "-e", script], check=True)


def test_queue_run_state_distinguishes_pause_and_idle_states():
    script = textwrap.dedent(
        """
        const assert = require("node:assert/strict");
        const fs = require("node:fs");
        const vm = require("node:vm");

        const context = {
          window: {},
          document: {
            addEventListener() {},
            getElementById() { return null; },
          },
          setInterval() {},
          fetch() {},
        };
        vm.createContext(context);
        vm.runInContext(fs.readFileSync("src/huggingface_pull/web/app.js", "utf8"), context);

        assert.equal(
          context.window.HuggingFacePull.queueRunState({
            running: false,
            pause_requested: false,
            stop_after_file_requested: false,
          }),
          "idle",
        );
        assert.equal(
          context.window.HuggingFacePull.queueRunState({
            running: false,
            pause_requested: true,
            stop_after_file_requested: false,
          }),
          "paused",
        );
        assert.equal(
          context.window.HuggingFacePull.downloadStatusLine({
            status: "stopped",
            progress: {
              phase: "stopped",
              overall: { downloaded: 10, total: 100, percent: 10 },
              current_file: { path: "model.safetensors" },
            },
          }),
          "Stopped model.safetensors | 10.0% | 10 B / 100 B",
        );
        assert.equal(
          JSON.stringify(context.window.HuggingFacePull.queueControlState({
            running: true,
            pause_requested: false,
            stop_after_file_requested: false,
          })),
          JSON.stringify({ startDisabled: true, pauseDisabled: false, stopDisabled: false }),
        );
        assert.equal(
          JSON.stringify(context.window.HuggingFacePull.queueControlState({
            running: true,
            pause_requested: true,
            stop_after_file_requested: true,
          })),
          JSON.stringify({ startDisabled: true, pauseDisabled: true, stopDisabled: true }),
        );
        assert.equal(
          JSON.stringify(context.window.HuggingFacePull.queueControlState({
            running: false,
            pause_requested: false,
            stop_after_file_requested: false,
          })),
          JSON.stringify({ startDisabled: false, pauseDisabled: true, stopDisabled: true }),
        );
        """
    )

    subprocess.run(["node", "-e", script], check=True)
