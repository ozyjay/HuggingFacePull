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
