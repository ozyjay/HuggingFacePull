#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/package-install-fedora.sh [options]

Build the HuggingFacePull desktop RPM and install it with dnf. The locally
built RPM is unsigned, so its signature check is disabled for this transaction;
repository package signature checks remain enabled.

Options:
  --install-build-deps  Install missing Fedora build tools with dnf.
  --skip-npm-ci         Reuse the existing node_modules directory.
  -y, --yes             Pass --assumeyes to dnf.
  -h, --help            Show this help.
EOF
}

log() {
    printf '%s\n' "$*"
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

run() {
    log "+ $*"
    "$@"
}

have() {
    command -v "$1" >/dev/null 2>&1
}

root_dir() {
    local script_dir
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    cd -- "$script_dir/.." && pwd
}

dnf_command() {
    if ((EUID == 0)); then
        DNF=(dnf)
    else
        have sudo || die "sudo is required to install Fedora packages"
        DNF=(sudo dnf)
    fi
}

main() {
    local install_build_deps=0
    local skip_npm_ci=0
    local assume_yes=0

    while (($#)); do
        case "$1" in
            --install-build-deps)
                install_build_deps=1
                ;;
            --skip-npm-ci)
                skip_npm_ci=1
                ;;
            -y|--yes)
                assume_yes=1
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown option: $1"
                ;;
        esac
        shift
    done

    [[ -r /etc/os-release ]] || die "Cannot identify this operating system"
    # shellcheck disable=SC1091
    . /etc/os-release
    case "${ID:-} ${ID_LIKE:-}" in
        *fedora*) ;;
        *) die "This script supports Fedora and Fedora-based distributions only" ;;
    esac

    [[ "$(uname -m)" == "x86_64" ]] || die "Desktop RPM packaging currently supports x86_64 only"
    have dnf || die "dnf was not found"
    dnf_command

    if [[ "$install_build_deps" == 1 ]]; then
        local -a dependency_args=(install rpm-build nodejs npm python3 tar)
        [[ "$assume_yes" == 1 ]] && dependency_args+=(--assumeyes)
        run "${DNF[@]}" "${dependency_args[@]}"
    fi

    local tool
    for tool in node npm python3 rpmbuild rpm tar; do
        have "$tool" || die "$tool was not found; rerun with --install-build-deps"
    done

    local root
    root="$(root_dir)"
    cd -- "$root"

    if [[ "$skip_npm_ci" == 0 ]]; then
        run npm ci
    elif [[ ! -d node_modules ]]; then
        die "node_modules does not exist; omit --skip-npm-ci to install dependencies"
    fi

    run npm run desktop:package:rpm

    local version dist rpm_path
    version="$(node -p "require('./package.json').version")"
    dist="$(rpm --eval '%{?dist}')"
    rpm_path="$root/out/huggingfacepull-${version}-1${dist}.x86_64.rpm"
    [[ -f "$rpm_path" ]] || die "Expected RPM was not created: $rpm_path"

    local action=install
    local installed_nevra="${version}-1${dist}.x86_64"
    if rpm -q --qf '%{VERSION}-%{RELEASE}.%{ARCH}' huggingfacepull 2>/dev/null | grep -Fxq "$installed_nevra"; then
        action=reinstall
    fi

    # rpmbuild does not sign local development artifacts. Limit the exception to
    # command-line RPMs so dependencies from configured repositories are still checked.
    local -a install_args=(--setopt=localpkg_gpgcheck=0 "$action" "$rpm_path")
    [[ "$assume_yes" == 1 ]] && install_args+=(--assumeyes)
    run "${DNF[@]}" "${install_args[@]}"

    log ""
    log "Installed $rpm_path"
    log "Launch it from the desktop app menu or run: huggingfacepull"
}

main "$@"
