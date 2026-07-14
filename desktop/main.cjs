const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const net = require("node:net");
const path = require("node:path");

const HOST = "127.0.0.1";
const DEFAULT_PORT = 8019;
const SERVER_READY_TIMEOUT_MS = 30000;
const SERVER_POLL_INTERVAL_MS = 250;
const PYTHON_WEB_LAUNCHER = "import sys; from huggingface_pull.cli import run_web; raise SystemExit(run_web(sys.argv[1:]))";

app.setName("HuggingFacePull");
if (process.platform === "linux") {
  app.setDesktopName("huggingfacepull.desktop");
  app.commandLine.appendSwitch("class", "huggingfacepull");
}

let backendProcess = null;
let mainWindow = null;

function settingsPath() {
  return path.join(app.getPath("userData"), "settings.json");
}

function readSettings() {
  try {
    return JSON.parse(fs.readFileSync(settingsPath(), "utf8"));
  } catch (error) {
    return {};
  }
}

function writeSettings(settings) {
  fs.mkdirSync(path.dirname(settingsPath()), { recursive: true });
  fs.writeFileSync(settingsPath(), `${JSON.stringify(settings, null, 2)}\n`, "utf8");
}

function configuredCacheDirectory() {
  const configured = readSettings().hfHubCache;
  return typeof configured === "string" && configured.trim() ? configured : null;
}

function projectRoot() {
  return path.resolve(__dirname, "..");
}

function executableExists(filePath) {
  try {
    fs.accessSync(filePath, fs.constants.X_OK);
    return true;
  } catch (error) {
    return false;
  }
}

function commandExists(command) {
  return !command.includes(path.sep) || executableExists(command);
}

function venvBinDir(venvDir) {
  return path.join(venvDir, process.platform === "win32" ? "Scripts" : "bin");
}

function venvScript(venvDir, scriptName) {
  const executableName = process.platform === "win32" ? `${scriptName}.exe` : scriptName;
  return path.join(venvBinDir(venvDir), executableName);
}

function venvPython(venvDir) {
  const executableName = process.platform === "win32" ? "python.exe" : "python";
  return path.join(venvBinDir(venvDir), executableName);
}

function pythonBackendCandidate(venvDir, cwd) {
  const python = venvPython(venvDir);
  if (!executableExists(python)) {
    return null;
  }
  return {
    command: python,
    args: ["-c", PYTHON_WEB_LAUNCHER],
    cwd,
    label: python,
  };
}

function scriptBackendCandidate(venvDir, cwd) {
  const command = venvScript(venvDir, "hfpull-web");
  if (!executableExists(command)) {
    return null;
  }
  return {
    command,
    args: [],
    cwd,
    label: command,
  };
}

function environmentBackendCandidate(root) {
  if (!process.env.HFPULL_BACKEND_COMMAND) {
    return null;
  }
  return {
    command: process.env.HFPULL_BACKEND_COMMAND,
    args: [],
    cwd: root,
    label: process.env.HFPULL_BACKEND_COMMAND,
  };
}

function backendCandidates(root) {
  const packagedBackend = path.join(process.resourcesPath || root, "backend");
  const packagedVenv = path.join(packagedBackend, ".venv");
  const devVenv = path.join(root, ".venv");

  return [
    app.isPackaged ? pythonBackendCandidate(packagedVenv, packagedBackend) : null,
    !app.isPackaged ? scriptBackendCandidate(devVenv, root) : null,
    scriptBackendCandidate(packagedVenv, packagedBackend),
    scriptBackendCandidate(devVenv, root),
    environmentBackendCandidate(root),
    { command: "hfpull-web", args: [], cwd: root, label: "hfpull-web" },
  ].filter(Boolean);
}

function findBackendCommand(root) {
  for (const candidate of backendCandidates(root)) {
    if (commandExists(candidate.command)) {
      return candidate;
    }
  }

  return null;
}

function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, HOST, () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

function portAcceptsConnections(port) {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host: HOST, port });
    socket.once("connect", () => {
      socket.destroy();
      resolve(true);
    });
    socket.once("error", () => {
      resolve(false);
    });
  });
}

function getState(port, timeout = 2000) {
  return new Promise((resolve, reject) => {
    const request = http.get({ host: HOST, port, path: "/api/state", timeout }, (response) => {
      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        body += chunk;
      });
      response.on("end", () => {
        if (!response.statusCode || response.statusCode >= 500) {
          reject(new Error(`HTTP ${response.statusCode || "unknown"}`));
          return;
        }
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(new Error("Server did not return HuggingFacePull state."));
        }
      });
    });

    request.on("timeout", () => {
      request.destroy();
      reject(new Error("Timed out waiting for HuggingFacePull state."));
    });
    request.on("error", reject);
  });
}

function isHuggingFacePullState(state) {
  return Boolean(
    state &&
    Array.isArray(state.items) &&
    Array.isArray(state.installed_models) &&
    typeof state.library_dir === "string" &&
    typeof state.endpoint === "string"
  );
}

async function probeExistingServer(port) {
  try {
    const state = await getState(port, 750);
    return isHuggingFacePullState(state);
  } catch (error) {
    return false;
  }
}

function waitForServer(port) {
  const deadline = Date.now() + SERVER_READY_TIMEOUT_MS;

  return new Promise((resolve, reject) => {
    const retry = () => {
      if (Date.now() >= deadline) {
        reject(new Error("Timed out waiting for HuggingFacePull to start."));
        return;
      }
      setTimeout(poll, SERVER_POLL_INTERVAL_MS);
    };

    const poll = () => {
      getState(port)
        .then((state) => {
          if (isHuggingFacePullState(state)) {
            resolve();
            return;
          }
          retry();
        })
        .catch(retry);
    };

    poll();
  });
}

async function backendPort() {
  const configured = Number(process.env.HFPULL_DESKTOP_PORT || DEFAULT_PORT);
  if (Number.isInteger(configured) && configured > 0) {
    return configured;
  }
  return findFreePort();
}

function startBackend(root, port) {
  const backend = findBackendCommand(root);
  if (!backend) {
    throw new Error("Could not find hfpull-web. Run ./scripts/install.sh first, or rebuild the desktop package.");
  }

  const args = [...backend.args, "--host", HOST, "--port", String(port), "--no-browser"];
  const env = {
    ...process.env,
    HF_HUB_DISABLE_XET: "1",
  };
  const cacheDirectory = configuredCacheDirectory();
  if (cacheDirectory) {
    env.HF_HUB_CACHE = cacheDirectory;
  }
  delete env.HF_XET_HIGH_PERFORMANCE;
  delete env.HF_XET_CHUNK_CACHE_SIZE_BYTES;
  delete env.HF_XET_SHARD_CACHE_SIZE_LIMIT;

  backendProcess = spawn(backend.command, args, {
    cwd: backend.cwd,
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });

  backendProcess.stdout.on("data", (chunk) => {
    process.stdout.write(`[hfpull-web] ${chunk}`);
  });
  backendProcess.stderr.on("data", (chunk) => {
    process.stderr.write(`[hfpull-web] ${chunk}`);
  });
  backendProcess.on("error", (error) => {
    process.stderr.write(`[hfpull-web] failed to start ${backend.label}: ${error.message}\n`);
  });
  backendProcess.on("exit", (code, signal) => {
    backendProcess = null;
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("backend-exited", { code, signal });
    }
  });
}

function stopBackend() {
  if (!backendProcess) {
    return;
  }

  const processToStop = backendProcess;
  backendProcess = null;
  processToStop.kill("SIGTERM");
}

function isAllowedAppUrl(targetUrl, appUrl) {
  try {
    const target = new URL(targetUrl);
    const appOrigin = new URL(appUrl).origin;
    return target.origin === appOrigin;
  } catch (error) {
    return false;
  }
}

function createWindow(appUrl) {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 960,
    minHeight: 640,
    title: "HuggingFacePull",
    backgroundColor: "#f7f7f3",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (!isAllowedAppUrl(url, appUrl)) {
      shell.openExternal(url);
      return { action: "deny" };
    }
    return { action: "allow" };
  });

  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (isAllowedAppUrl(url, appUrl)) {
      return;
    }
    event.preventDefault();
    shell.openExternal(url);
  });

  mainWindow.loadURL(appUrl);
}

ipcMain.handle("select-hf-cache-directory", async () => {
  const current = configuredCacheDirectory();
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select Hugging Face cache directory",
    defaultPath: current || app.getPath("home"),
    buttonLabel: "Use this folder",
    properties: ["openDirectory", "createDirectory"],
  });
  if (result.canceled || !result.filePaths[0]) {
    return { changed: false, path: current };
  }

  const selected = path.resolve(result.filePaths[0]);
  writeSettings({ ...readSettings(), hfHubCache: selected });
  return { changed: selected !== current, path: selected };
});

ipcMain.handle("restart-for-cache-directory", () => {
  app.relaunch();
  app.quit();
});

async function main() {
  const root = projectRoot();
  let port = await backendPort();
  let ownsBackend = false;

  if (!(await probeExistingServer(port))) {
    if (await portAcceptsConnections(port)) {
      port = await findFreePort();
    }
    if (port !== DEFAULT_PORT) {
      startBackend(root, port);
    } else {
      try {
        startBackend(root, port);
      } catch (error) {
        port = await findFreePort();
        startBackend(root, port);
      }
    }
    ownsBackend = true;
    await waitForServer(port);
  }

  const appUrl = `http://${HOST}:${port}/`;

  createWindow(appUrl);
  mainWindow.on("closed", () => {
    mainWindow = null;
    if (ownsBackend) {
      stopBackend();
    }
  });
}

app.whenReady().then(() => {
  main().catch((error) => {
    dialog.showErrorBox("HuggingFacePull failed to start", error.message);
    app.quit();
  });
});

app.on("before-quit", stopBackend);

app.on("window-all-closed", () => {
  app.quit();
});
