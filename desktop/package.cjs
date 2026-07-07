const { packager } = require("@electron/packager");
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const root = process.cwd();
const buildDir = path.join(root, "build");
const backendDir = path.join(buildDir, "backend");
const backendVenv = path.join(backendDir, ".venv");

function run(command, args, options = {}) {
  console.log(`+ ${[command, ...args].join(" ")}`);
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    env: {
      ...process.env,
      HF_HUB_DISABLE_XET: "1",
    },
    ...options,
  });

  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`${command} exited with status ${result.status}`);
  }
}

function venvPython() {
  return path.join(backendVenv, process.platform === "win32" ? "Scripts" : "bin", process.platform === "win32" ? "python.exe" : "python");
}

function buildBackend() {
  fs.rmSync(backendDir, { recursive: true, force: true });
  fs.mkdirSync(backendDir, { recursive: true });

  const python = process.env.PYTHON || "python3";
  run(python, ["-m", "venv", backendVenv]);
  run(venvPython(), ["-m", "pip", "install", "."]);
}

async function main() {
  buildBackend();

  const appPaths = await packager({
    dir: root,
    name: "HuggingFacePull",
    executableName: "huggingfacepull",
    out: "out",
    overwrite: true,
    prune: true,
    platform: "linux",
    arch: "x64",
    extraResource: backendDir,
    ignore: [
      /^\/\.cache($|\/)/,
      /^\/\.git($|\/)/,
      /^\/\.pytest_cache($|\/)/,
      /^\/\.venv($|\/)/,
      /(^|\/)__pycache__($|\/)/,
      /^\/build($|\/)/,
      /^\/cache($|\/)/,
      /^\/docs($|\/)/,
      /^\/hf-test($|\/)/,
      /^\/library($|\/)/,
      /^\/models($|\/)/,
      /^\/node_modules($|\/)/,
      /^\/out($|\/)/,
      /^\/tests($|\/)/,
      /^\/.*\.egg-info($|\/)/,
    ],
  });

  for (const appPath of appPaths) {
    console.log(appPath);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
