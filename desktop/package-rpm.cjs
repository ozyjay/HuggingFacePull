const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const root = process.cwd();
const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const version = pkg.version;
const rpmRoot = path.join(root, "build", "rpm");
const sourceName = `huggingfacepull-${version}`;
const sourceRoot = path.join(rpmRoot, "staging", sourceName);
const portableApp = path.join(root, "out", "HuggingFacePull-linux-x64");

function run(command, args, cwd = root) {
  console.log(`+ ${[command, ...args].join(" ")}`);
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} exited with status ${result.status}`);
}

function copyFile(source, destination) {
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.copyFileSync(source, destination);
}

function prepareSource() {
  fs.rmSync(rpmRoot, { recursive: true, force: true });
  for (const directory of ["BUILD", "BUILDROOT", "RPMS", "SOURCES", "SPECS", "SRPMS"]) {
    fs.mkdirSync(path.join(rpmRoot, directory), { recursive: true });
  }
  fs.mkdirSync(sourceRoot, { recursive: true });
  fs.cpSync(portableApp, path.join(sourceRoot, "app"), {
    recursive: true,
    verbatimSymlinks: true,
  });
  copyFile(path.join(root, "desktop", "huggingfacepull.desktop"), path.join(sourceRoot, "huggingfacepull.desktop"));
  copyFile(path.join(root, "desktop", "huggingfacepull.svg"), path.join(sourceRoot, "huggingfacepull.svg"));
  run("tar", ["-czf", path.join(rpmRoot, "SOURCES", `${sourceName}.tar.gz`), "-C", path.join(rpmRoot, "staging"), sourceName]);
}

function writeSpec() {
  const python = path.join(portableApp, "resources", "backend", ".venv", "bin", "python");
  const pythonResult = spawnSync(python, ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], {
    encoding: "utf8",
  });
  if (pythonResult.status !== 0) throw new Error("Could not determine the bundled backend Python ABI");
  const pythonAbi = pythonResult.stdout.trim();
  const spec = `%global debug_package %{nil}

Name:           huggingfacepull
Version:        ${version}
Release:        1%{?dist}
Summary:        Desktop app for downloading Hugging Face model snapshots
License:        MIT
URL:            https://github.com/cpjjh/HuggingFacePull
Source0:        %{name}-%{version}.tar.gz
BuildArch:      x86_64
AutoReqProv:    no
Requires:       alsa-lib, at-spi2-core, cups-libs, gtk3, libX11, libXcomposite, libXdamage, libXext, libXfixes, libXrandr, mesa-libgbm, nss, python(abi) = ${pythonAbi}

%description
HuggingFacePull is a local desktop tool for searching, queueing, and downloading
Hugging Face Hub model snapshots.

%prep
%setup -q

%build

%install
mkdir -p %{buildroot}/opt/huggingfacepull
cp -a app/. %{buildroot}/opt/huggingfacepull/
mkdir -p %{buildroot}%{_bindir}
ln -s /opt/huggingfacepull/huggingfacepull %{buildroot}%{_bindir}/huggingfacepull
install -Dm0644 huggingfacepull.desktop %{buildroot}%{_datadir}/applications/huggingfacepull.desktop
install -Dm0644 huggingfacepull.svg %{buildroot}%{_datadir}/icons/hicolor/scalable/apps/huggingfacepull.svg

%files
/opt/huggingfacepull
%{_bindir}/huggingfacepull
%{_datadir}/applications/huggingfacepull.desktop
%{_datadir}/icons/hicolor/scalable/apps/huggingfacepull.svg

%changelog
* Sat Jul 11 2026 HuggingFacePull maintainers - ${version}-1
- Initial Fedora desktop package
`;
  fs.writeFileSync(path.join(rpmRoot, "SPECS", "huggingfacepull.spec"), spec, "utf8");
}

function main() {
  if (process.env.HFPULL_SKIP_DESKTOP_PACKAGE !== "1") {
    run(process.execPath, [path.join(root, "desktop", "package.cjs")]);
  }
  prepareSource();
  writeSpec();
  run("rpmbuild", ["--define", `_topdir ${rpmRoot}`, "-bb", path.join(rpmRoot, "SPECS", "huggingfacepull.spec")]);

  const rpmDirectory = path.join(rpmRoot, "RPMS", "x86_64");
  const rpm = fs.readdirSync(rpmDirectory).find((name) => name.endsWith(".rpm"));
  if (!rpm) throw new Error("rpmbuild completed without producing an RPM");
  const output = path.join(root, "out", rpm);
  fs.copyFileSync(path.join(rpmDirectory, rpm), output);
  console.log(`RPM written to ${output}`);
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
