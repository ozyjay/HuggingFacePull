(function () {
  const pollMs = 1000;
  const state = {
    snapshot: null,
    selectedItemId: null,
    searchResults: [],
    searchError: "",
    fileResults: null,
    cleanupResult: null,
    notice: "",
    busy: false,
  };

  const els = {};

  document.addEventListener("DOMContentLoaded", () => {
    bindElements();
    bindEvents();
    refresh();
    window.setInterval(refresh, pollMs);
  });

  function bindElements() {
    [
      "runtimeSummary",
      "startQueue",
      "pauseQueue",
      "stopAfterFile",
      "searchForm",
      "searchInput",
      "searchStatus",
      "searchResults",
      "addForm",
      "repoIdInput",
      "revisionInput",
      "repoTypeInput",
      "allowPatternsInput",
      "ignorePatternsInput",
      "inspectFiles",
      "fileResults",
      "queueSummary",
      "queueRows",
      "installedSummary",
      "installedList",
      "detailStatus",
      "detailPanel",
      "includePartialsInput",
      "olderThanInput",
      "scanCleanup",
      "deleteCleanup",
      "cleanupResult",
      "notice",
    ].forEach((id) => {
      els[id] = document.getElementById(id);
    });
  }

  function bindEvents() {
    els.startQueue.addEventListener("click", () => postAndRefresh("/api/start", "Queue started"));
    els.pauseQueue.addEventListener("click", () => postAndRefresh("/api/pause", "Pause requested"));
    els.stopAfterFile.addEventListener("click", () => postAndRefresh("/api/stop-after-file", "Stop requested after current file or snapshot"));
    els.searchForm.addEventListener("submit", onSearch);
    els.addForm.addEventListener("submit", onAddRepo);
    els.inspectFiles.addEventListener("click", onInspectFiles);
    els.scanCleanup.addEventListener("click", () => onCleanup(false));
    els.deleteCleanup.addEventListener("click", () => onCleanup(true));
  }

  async function api(path, options) {
    const request = {
      ...options,
      headers: { Accept: "application/json", ...(options && options.headers ? options.headers : {}) },
    };
    if (request.body && !request.headers["Content-Type"]) {
      request.headers["Content-Type"] = "application/json";
    }

    let response;
    try {
      response = await fetch(path, request);
    } catch (error) {
      throw new Error(`Network error: ${error.message}`);
    }

    const text = await response.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (error) {
        if (!response.ok) {
          throw new Error(response.statusText || `HTTP ${response.status}`);
        }
        throw new Error("The server returned invalid JSON.");
      }
    }

    if (!response.ok) {
      const detail = data && (data.detail || data.error || data.message);
      throw new Error(formatErrorDetail(detail) || `HTTP ${response.status}`);
    }
    return data;
  }

  async function refresh() {
    try {
      state.snapshot = await api("/api/state");
      state.notice = "";
      render();
    } catch (error) {
      showNotice(error.message, true);
      render();
    }
  }

  function render() {
    const snapshot = state.snapshot || {};
    const items = Array.isArray(snapshot.items) ? snapshot.items : [];
    const installed = Array.isArray(snapshot.installed_models) ? snapshot.installed_models : [];
    const selected = selectedItem(items);

    els.runtimeSummary.textContent = snapshot.library_dir
      ? `Metadata directory: ${snapshot.library_dir} | Model cache: ~/.cache/huggingface/hub | Hub endpoint: ${snapshot.endpoint || "unknown"}`
      : "Loading local state...";
    els.queueSummary.textContent = `${items.length} item${items.length === 1 ? "" : "s"} | ${queueRunState(snapshot)}`;
    els.installedSummary.textContent = `${installed.length} snapshot${installed.length === 1 ? "" : "s"}`;
    els.searchStatus.textContent = state.searchError || `${state.searchResults.length} result${state.searchResults.length === 1 ? "" : "s"}`;
    els.detailStatus.textContent = selected ? selected.status : "";
    els.notice.textContent = state.notice;
    els.notice.classList.toggle("error", state.notice.startsWith("Error:"));
    renderQueueControls(snapshot);

    renderSearchResults();
    renderFileResults();
    renderQueue(items);
    renderInstalled(installed);
    renderDetail(selected);
    renderCleanup();
  }

  async function onSearch(event) {
    event.preventDefault();
    const query = els.searchInput.value.trim();
    state.searchError = "";
    state.searchResults = [];
    if (!query) {
      render();
      return;
    }

    try {
      const result = await api(`/api/search?q=${encodeURIComponent(query)}`);
      state.searchResults = Array.isArray(result.results) ? result.results : [];
      state.searchError = result.available === false ? result.error || "Search is unavailable." : "";
      render();
    } catch (error) {
      state.searchError = error.message;
      showNotice(error.message, true);
      render();
    }
  }

  async function onAddRepo(event) {
    event.preventDefault();
    try {
      const item = await addRepoFromForm();
      state.selectedItemId = item.id;
      showNotice(`Queued ${item.repo_id}`);
      await refresh();
    } catch (error) {
      showNotice(error.message, true);
    }
  }

  async function onInspectFiles() {
    const repoId = els.repoIdInput.value.trim();
    const revision = els.revisionInput.value.trim() || "main";
    const repoType = els.repoTypeInput.value;
    if (!repoId) {
      showNotice("Enter an HF repo ID before inspecting files.", true);
      return;
    }

    state.fileResults = { loading: true, files: [], error: "" };
    render();
    try {
      const path = `/api/models/${repoPath(repoId)}/files?revision=${encodeURIComponent(revision)}&repo_type=${encodeURIComponent(repoType)}`;
      const result = await api(path);
      state.fileResults = { loading: false, files: result.files || [], error: "" };
      render();
    } catch (error) {
      state.fileResults = { loading: false, files: [], error: error.message };
      showNotice(error.message, true);
      render();
    }
  }

  async function onCleanup(deleteMatches) {
    const payload = cleanupPayload();
    try {
      state.cleanupResult = await api(deleteMatches ? "/api/cleanup/delete" : "/api/cleanup/scan", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      showNotice(deleteMatches ? "Cleanup delete finished" : "Cleanup scan finished");
      await refresh();
    } catch (error) {
      showNotice(error.message, true);
      render();
    }
  }

  async function postAndRefresh(path, message) {
    try {
      await api(path, { method: "POST" });
      showNotice(message);
      await refresh();
    } catch (error) {
      showNotice(error.message, true);
    }
  }

  async function addRepoFromForm(repoIdOverride) {
    const payload = {
      repo_id: repoIdOverride || els.repoIdInput.value.trim(),
      revision: els.revisionInput.value.trim() || "main",
      repo_type: els.repoTypeInput.value,
      allow_patterns: splitPatterns(els.allowPatternsInput.value),
      ignore_patterns: splitPatterns(els.ignorePatternsInput.value),
      local_dir: null,
    };
    if (!payload.repo_id) {
      throw new Error("HF repo ID is required.");
    }
    return api("/api/queue", { method: "POST", body: JSON.stringify(payload) });
  }

  function renderSearchResults() {
    if (!state.searchResults.length && !state.searchError) {
      els.searchResults.innerHTML = `<p class="empty">Search for public Hub repos, or add an exact HF repo ID directly.</p>`;
      return;
    }
    if (state.searchError) {
      els.searchResults.innerHTML = `<p class="error-text">${escapeHtml(state.searchError)}</p>`;
      return;
    }

    els.searchResults.innerHTML = state.searchResults.map((result) => {
      const repoId = result.repo_id || result.name || "";
      const installState = snapshotInstallState(
        (state.snapshot && state.snapshot.installed_models) || [],
        (state.snapshot && state.snapshot.cached_models) || [],
        repoId,
        els.revisionInput.value.trim() || "main",
        els.repoTypeInput.value || "model",
      );
      const meta = [
        result.pipeline_tag,
        formatCount(result.downloads, "download"),
        formatCount(result.likes, "like"),
      ].filter(Boolean).join(" | ");
      return `
        <article class="result-row">
          <div>
            <strong>${escapeHtml(repoId)}</strong>
            <small>${escapeHtml(meta || "Hub repo")}</small>
          </div>
          ${installState === "installed"
            ? `<button type="button" class="secondary" disabled>Installed</button>`
            : `<button type="button" data-add-search="${escapeAttr(repoId)}">${installState === "cached" ? "Add from cache" : "Add"}</button>`}
        </article>
      `;
    }).join("");

    els.searchResults.querySelectorAll("[data-add-search]").forEach((button) => {
      button.addEventListener("click", async () => {
        els.repoIdInput.value = button.getAttribute("data-add-search") || "";
        try {
          const item = await addRepoFromForm(els.repoIdInput.value);
          state.selectedItemId = item.id;
          showNotice(`Queued ${item.repo_id}`);
          await refresh();
        } catch (error) {
          showNotice(error.message, true);
        }
      });
    });
  }

  function renderFileResults() {
    const result = state.fileResults;
    if (!result) {
      els.fileResults.innerHTML = "";
      return;
    }
    if (result.loading) {
      els.fileResults.innerHTML = `<p class="empty">Loading files...</p>`;
      return;
    }
    if (result.error) {
      els.fileResults.innerHTML = `<p class="error-text">${escapeHtml(result.error)}</p>`;
      return;
    }
    const files = result.files || [];
    if (!files.length) {
      els.fileResults.innerHTML = `<p class="empty">No files returned for this revision.</p>`;
      return;
    }
    els.fileResults.innerHTML = `
      <div class="file-summary">${files.length} file${files.length === 1 ? "" : "s"}</div>
      <ul>
        ${files.slice(0, 80).map((file) => `
          <li>
            <span>${escapeHtml(file.path || file.name || "unnamed file")}</span>
            <small>${formatBytes(file.size)}</small>
          </li>
        `).join("")}
      </ul>
    `;
  }

  function renderQueue(items) {
    if (!items.length) {
      els.queueRows.innerHTML = `<p class="empty">No queued snapshots yet.</p>`;
      return;
    }

    els.queueRows.innerHTML = items.map((item) => {
      const overall = (item.progress && item.progress.overall) || {};
      const percent = normalisePercent(overall.percent);
      const statusLine = downloadStatusLine(item);
      return `
        <article class="queue-row ${state.selectedItemId === item.id ? "selected" : ""}" data-select-item="${escapeAttr(item.id)}">
          <div class="queue-main">
            <div class="row-title">
              <strong>${escapeHtml(item.repo_id)}</strong>
              <span class="badge ${statusClass(item.status)}">${escapeHtml(item.status)}</span>
            </div>
            <p class="download-status">${escapeHtml(statusLine)}</p>
            <div class="row-meta">
              <span>${escapeHtml(item.revision || "main")}</span>
              <span>${formatProgressAmount(overall)}</span>
              <span>${formatPercent(overall.percent)}</span>
              <span>${formatSpeed(overall.bytes_per_second)}</span>
              <span>ETA ${formatDuration(overall.eta_seconds)}</span>
            </div>
            <div class="progress" aria-label="Progress">
              <span style="width: ${percent}%"></span>
            </div>
          </div>
          <div class="row-actions">
            ${item.status === "failed" ? `<button type="button" data-retry="${escapeAttr(item.id)}">Retry</button>` : ""}
            ${item.status !== "running" ? `<button type="button" class="secondary" data-remove="${escapeAttr(item.id)}">Remove</button>` : ""}
          </div>
        </article>
      `;
    }).join("");

    els.queueRows.querySelectorAll("[data-select-item]").forEach((row) => {
      row.addEventListener("click", () => {
        state.selectedItemId = row.getAttribute("data-select-item");
        render();
      });
    });
    els.queueRows.querySelectorAll("[data-retry]").forEach((button) => {
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        await postAndRefresh(`/api/retry/${encodeURIComponent(button.getAttribute("data-retry"))}`, "Retry queued");
      });
    });
    els.queueRows.querySelectorAll("[data-remove]").forEach((button) => {
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        await postAndRefresh(`/api/remove/${encodeURIComponent(button.getAttribute("data-remove"))}`, "Queue item removed");
      });
    });
  }

  function renderInstalled(installed) {
    if (!installed.length) {
      els.installedList.innerHTML = `<p class="empty">No installed snapshots found in metadata or the HF cache.</p>`;
      return;
    }

    els.installedList.innerHTML = installed.map((item) => `
      <article class="installed-row">
        <div>
          <strong>${escapeHtml(item.repo_id)}</strong>
          <small>${escapeHtml(item.revision || "main")} | ${formatBytes(item.size)} | ${escapeHtml(item.snapshot_path || "")}</small>
        </div>
        <button type="button" class="danger" data-remove-installed="${escapeAttr(item.repo_id)}" data-revision="${escapeAttr(item.revision || "main")}" data-repo-type="${escapeAttr(item.repo_type || "model")}">Delete record</button>
      </article>
    `).join("");

    els.installedList.querySelectorAll("[data-remove-installed]").forEach((button) => {
      button.addEventListener("click", async () => {
        const payload = {
          repo_id: button.getAttribute("data-remove-installed"),
          revision: button.getAttribute("data-revision") || "main",
          repo_type: button.getAttribute("data-repo-type") || "model",
        };
        try {
          await api("/api/installed/remove", { method: "POST", body: JSON.stringify(payload) });
          showNotice(`Deleted ${payload.repo_id}`);
          await refresh();
        } catch (error) {
          showNotice(error.message, true);
        }
      });
    });
  }

  function renderDetail(item) {
    if (!item) {
      els.detailPanel.innerHTML = `<p class="empty">Select a queue row to inspect messages and current file.</p>`;
      return;
    }
    const progress = item.progress || {};
    const overall = progress.overall || {};
    const current = progress.current_file;
    const messages = Array.isArray(item.messages) ? item.messages.slice(-12) : [];
    els.detailPanel.innerHTML = `
      <div class="detail-stack">
        <div>
          <h3>${escapeHtml(item.repo_id)}</h3>
          <p>${escapeHtml(item.revision || "main")} | ${escapeHtml(item.repo_type || "model")}</p>
        </div>
        <p class="download-status detail-status-line">${escapeHtml(downloadStatusLine(item))}</p>
        <div class="progress detail-progress" aria-label="Overall progress">
          <span style="width: ${normalisePercent(overall.percent)}%"></span>
        </div>
        <dl>
          <div><dt>Phase</dt><dd>${escapeHtml(progress.phase || item.status)}</dd></div>
          <div><dt>Overall</dt><dd>${formatProgressAmount(overall)}</dd></div>
          <div><dt>Percent</dt><dd>${formatPercent(overall.percent)}</dd></div>
          <div><dt>Speed</dt><dd>${formatSpeed(overall.bytes_per_second)}</dd></div>
          <div><dt>ETA</dt><dd>${formatDuration(overall.eta_seconds)}</dd></div>
          <div><dt>Current file</dt><dd>${current ? escapeHtml(currentFileLabel(current)) : "None"}</dd></div>
          <div><dt>File progress</dt><dd>${current ? `${formatBytes(current.downloaded)} / ${formatBytes(current.total)}` : "None"}</dd></div>
          <div><dt>File update</dt><dd>${current ? formatTimestamp(current.updated_at || item.updated_at) : "unknown"}</dd></div>
          <div><dt>Last update</dt><dd>${formatTimestamp(item.updated_at)}</dd></div>
        </dl>
        ${item.error ? `<p class="error-text">${escapeHtml(item.error)}</p>` : ""}
        <ol class="message-list">
          ${messages.length ? messages.map((message) => `<li>${escapeHtml(messageLine(message))}</li>`).join("") : "<li>No messages yet.</li>"}
        </ol>
      </div>
    `;
  }

  function renderCleanup() {
    const result = state.cleanupResult;
    if (!result) {
      els.cleanupResult.innerHTML = `<p class="empty">Run a scan before deleting stale partial files.</p>`;
      return;
    }
    const stale = result.stale_partials || [];
    const incomplete = result.incomplete_snapshots || [];
    els.cleanupResult.innerHTML = `
      <p>${cleanupSummaryLine(result)}</p>
      ${result.deleted && result.deleted.length ? `<p>${result.deleted.length} deleted.</p>` : ""}
      ${result.deleted_snapshots && result.deleted_snapshots.length ? `<p>${result.deleted_snapshots.length} snapshots deleted.</p>` : ""}
      <ul>
        ${stale.slice(0, 30).map((item) => {
          const source = item.source === "huggingface_cache" ? "HF cache" : item.source === "library" ? "Library" : "File";
          return `<li><span>${escapeHtml(item.name || item.path)}</span><small>${escapeHtml(source)} - ${formatBytes(item.size)}</small></li>`;
        }).join("")}
        ${incomplete.slice(0, 30).map((item) => {
          return `<li><span>${escapeHtml(item.path)}</span><small>Incomplete snapshot - ${formatBytes(item.size)}</small></li>`;
        }).join("")}
      </ul>
    `;
  }

  function cleanupSummaryLine(result) {
    const staleCount = (result.stale_partials || []).length;
    const incompleteCount = (result.incomplete_snapshots || []).length;
    const action = result.dry_run ? "Scan found" : "Deleted";
    return `${action} ${staleCount} stale file${staleCount === 1 ? "" : "s"} and ${incompleteCount} incomplete snapshot${incompleteCount === 1 ? "" : "s"}.`;
  }

  function selectedItem(items) {
    if (state.selectedItemId) {
      const found = items.find((item) => item.id === state.selectedItemId);
      if (found) {
        return found;
      }
    }
    return items.find((item) => item.status === "running") || items[items.length - 1] || null;
  }

  function cleanupPayload() {
    return {
      include_partials: els.includePartialsInput.checked,
      older_than_days: Number.parseInt(els.olderThanInput.value || "0", 10),
    };
  }

  function splitPatterns(value) {
    return value.split(",").map((part) => part.trim()).filter(Boolean);
  }

  function queueRunState(snapshot) {
    if (snapshot.stop_after_file_requested) {
      return "stopping after current file or snapshot";
    }
    if (snapshot.pause_requested) {
      return snapshot.running ? "pausing after current download" : "paused";
    }
    return snapshot.running ? "running" : "idle";
  }

  function queueControlState(snapshot) {
    const running = Boolean(snapshot && snapshot.running);
    const stopping = Boolean(snapshot && snapshot.stop_after_file_requested);
    const pausing = Boolean(snapshot && snapshot.pause_requested);
    return {
      startDisabled: running || stopping,
      pauseDisabled: !running || pausing || stopping,
      stopDisabled: !running || stopping,
    };
  }

  function renderQueueControls(snapshot) {
    const controls = queueControlState(snapshot);
    els.startQueue.disabled = controls.startDisabled;
    els.pauseQueue.disabled = controls.pauseDisabled;
    els.stopAfterFile.disabled = controls.stopDisabled;
  }

  function downloadStatusLine(item) {
    const progress = item.progress || {};
    const overall = progress.overall || {};
    const current = progress.current_file || {};
    const phase = progress.phase || item.status || "waiting";
    const subject = currentFileLabel(current);
    const lastUpdate = current.updated_at || item.updated_at;
    const quietSeconds = progressQuietSeconds(item, progress, current);
    const parts = [`${titleCase(phase)} ${subject}`];
    if (lastUpdate) {
      parts.push(`last update ${formatTimestamp(lastUpdate)}`);
    }
    if (quietSeconds !== null) {
      parts.push(`no update for ${formatDuration(quietSeconds)}`);
    }
    const percent = formatPercent(overall.percent);
    if (percent !== "calculating") {
      parts.push(percent);
    }
    parts.push(formatProgressAmount(overall));
    const speed = formatSpeed(overall.bytes_per_second);
    if (speed !== "no speed") {
      parts.push(speed);
    }
    const eta = formatDuration(overall.eta_seconds);
    if (eta !== "unknown") {
      parts.push(`ETA ${eta}`);
    }
    return parts.join(" | ");
  }

  function currentFileLabel(current) {
    const name = current.path || current.name || current.digest || "snapshot";
    if (current.index && current.total_files) {
      return `file ${current.index}/${current.total_files} ${name}`;
    }
    return name;
  }

  function messageLine(message) {
    if (!message || typeof message !== "object") {
      return String(message);
    }
    const timestamp = message.timestamp ? `${formatTimestamp(message.timestamp)} ` : "";
    return `${timestamp}${message.text || ""}`;
  }

  function progressQuietSeconds(item, progress, current) {
    const explicit = progress.stall_seconds || current.stall_seconds;
    if (progress.stalled || current.stalled) {
      return explicit || 0;
    }
    if (item.status !== "running") {
      return null;
    }
    const lastUpdate = current.updated_at || item.updated_at;
    if (!lastUpdate) {
      return null;
    }
    const quietFor = Math.max(0, Math.floor(Date.now() / 1000 - Number(lastUpdate)));
    return quietFor >= 60 ? quietFor : null;
  }

  function titleCase(value) {
    const text = String(value || "").replace(/[-_]/g, " ");
    return text ? text.charAt(0).toUpperCase() + text.slice(1) : "";
  }

  function isInstalledSnapshot(installed, cached, repoId, revision, repoType) {
    return snapshotInstallState(installed, cached, repoId, revision, repoType) === "installed";
  }

  function snapshotInstallState(installed, cached, repoId, revision, repoType) {
    const expectedRevision = revision || "main";
    const expectedRepoType = repoType || "model";
    const managedMatch = (installed || []).some((item) => (
      item.repo_id === repoId
      && (item.revision || "main") === expectedRevision
      && (item.repo_type || "model") === expectedRepoType
    ));
    if (managedMatch) {
      return "installed";
    }
    const cacheMatch = (cached || []).some((item) => (
      item.repo_id === repoId
      && item.revision === expectedRevision
      && (item.repo_type || "model") === expectedRepoType
    ));
    return cacheMatch ? "cached" : "available";
  }

  function repoPath(repoId) {
    return repoId.split("/").map(encodeURIComponent).join("/");
  }

  function showNotice(message, error) {
    state.notice = `${error ? "Error: " : ""}${message}`;
    render();
  }

  function formatErrorDetail(detail) {
    if (!detail) {
      return "";
    }
    if (Array.isArray(detail)) {
      return detail.map((item) => item.msg || item.message || JSON.stringify(item)).join("; ");
    }
    if (typeof detail === "object") {
      return detail.message || JSON.stringify(detail);
    }
    return String(detail);
  }

  function formatBytes(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "unknown";
    }
    const bytes = Number(value);
    if (bytes < 1024) {
      return `${bytes} B`;
    }
    const units = ["KB", "MB", "GB", "TB"];
    let amount = bytes / 1024;
    let index = 0;
    while (amount >= 1024 && index < units.length - 1) {
      amount /= 1024;
      index += 1;
    }
    return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${units[index]}`;
  }

  function formatProgressAmount(overall) {
    const downloaded = overall && overall.downloaded;
    const total = overall && overall.total;
    if (downloaded === null || downloaded === undefined || Number.isNaN(Number(downloaded))) {
      return total ? `0 B / ${formatBytes(total)}` : "waiting for bytes";
    }
    if (total === null || total === undefined || Number.isNaN(Number(total))) {
      return `Downloaded ${formatBytes(downloaded)} | total calculating...`;
    }
    return `${formatBytes(downloaded)} / ${formatBytes(total)}`;
  }

  function formatPercent(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "calculating";
    }
    return `${normalisePercent(value).toFixed(1)}%`;
  }

  function formatSpeed(value) {
    if (!value) {
      return "no speed";
    }
    return `${formatBytes(value)}/s`;
  }

  function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) {
      return "unknown";
    }
    const value = Math.max(0, Number(seconds));
    if (value < 60) {
      return `${Math.round(value)}s`;
    }
    const minutes = Math.floor(value / 60);
    const remainder = Math.round(value % 60);
    return `${minutes}m ${remainder}s`;
  }

  function formatTimestamp(value) {
    if (!value) {
      return "unknown";
    }
    return new Date(Number(value) * 1000).toLocaleTimeString();
  }

  function formatCount(value, label) {
    if (value === null || value === undefined) {
      return "";
    }
    const count = Number(value).toLocaleString();
    return `${count} ${label}${Number(value) === 1 ? "" : "s"}`;
  }

  function normalisePercent(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return 0;
    }
    return Math.max(0, Math.min(100, Number(value)));
  }

  function statusClass(status) {
    return `status-${String(status || "waiting").replace(/[^a-z0-9_-]/gi, "").toLowerCase()}`;
  }

  function escapeHtml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function escapeAttr(value) {
    return escapeHtml(value);
  }

  window.HuggingFacePull = {
    api,
    refresh,
    render,
    isInstalledSnapshot,
    snapshotInstallState,
    downloadStatusLine,
    cleanupSummaryLine,
    queueControlState,
    queueRunState,
  };
}());
