#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/install.sh [options]

Create the project virtual environment and install HuggingFacePull.

Options:
  --runtime               Install runtime dependencies only.
  --dev                   Install development dependencies. This is the default.
  --recreate              Remove and recreate .venv.
  --install-system-deps   Install missing platform packages when a supported
                          package manager is available.
  -y, --yes               Do not prompt when installing system packages.
  -h, --help              Show this help.
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

detect_platform() {
    local kernel
    kernel="$(uname -s 2>/dev/null || printf unknown)"

    case "$kernel" in
        Linux)
            if [[ -r /etc/os-release ]]; then
                # shellcheck disable=SC1091
                . /etc/os-release
                PLATFORM_ID="${ID:-linux}"
                PLATFORM_ID_LIKE="${ID_LIKE:-}"
                PLATFORM_NAME="${PRETTY_NAME:-Linux}"
            else
                PLATFORM_ID="linux"
                PLATFORM_ID_LIKE=""
                PLATFORM_NAME="Linux"
            fi
            ;;
        Darwin)
            PLATFORM_ID="macos"
            PLATFORM_ID_LIKE=""
            PLATFORM_NAME="macOS"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            PLATFORM_ID="windows"
            PLATFORM_ID_LIKE=""
            PLATFORM_NAME="Windows"
            ;;
        *)
            PLATFORM_ID="unknown"
            PLATFORM_ID_LIKE=""
            PLATFORM_NAME="$kernel"
            ;;
    esac
}

package_install_command() {
    case "$PLATFORM_ID $PLATFORM_ID_LIKE" in
        *fedora*|*rhel*|*centos*)
            printf 'sudo dnf install python3 python3-pip'
            ;;
        *debian*|*ubuntu*)
            printf 'sudo apt-get update && sudo apt-get install python3 python3-venv python3-pip'
            ;;
        *arch*)
            printf 'sudo pacman -S python python-pip'
            ;;
        *opensuse*|*suse*)
            printf 'sudo zypper install python3 python3-pip'
            ;;
        macos*)
            if have brew; then
                printf 'brew install python'
            else
                printf 'Install Homebrew from https://brew.sh, then run: brew install python'
            fi
            ;;
        windows*)
            printf 'Run scripts/setup.ps1 in PowerShell, or install Python from https://www.python.org/downloads/windows/'
            ;;
        *)
            printf 'Install Python 3.10+ and pip with your platform package manager'
            ;;
    esac
}

install_system_deps() {
    local assume_yes="$1"

    case "$PLATFORM_ID $PLATFORM_ID_LIKE" in
        *fedora*|*rhel*|*centos*)
            if have dnf; then
                if [[ "$assume_yes" == 1 ]]; then
                    run sudo dnf install -y python3 python3-pip
                else
                    run sudo dnf install python3 python3-pip
                fi
                return
            fi
            ;;
        *debian*|*ubuntu*)
            if have apt-get; then
                run sudo apt-get update
                if [[ "$assume_yes" == 1 ]]; then
                    run sudo apt-get install -y python3 python3-venv python3-pip
                else
                    run sudo apt-get install python3 python3-venv python3-pip
                fi
                return
            fi
            ;;
        *arch*)
            if have pacman; then
                if [[ "$assume_yes" == 1 ]]; then
                    run sudo pacman -S --needed --noconfirm python python-pip
                else
                    run sudo pacman -S --needed python python-pip
                fi
                return
            fi
            ;;
        *opensuse*|*suse*)
            if have zypper; then
                if [[ "$assume_yes" == 1 ]]; then
                    run sudo zypper --non-interactive install python3 python3-pip
                else
                    run sudo zypper install python3 python3-pip
                fi
                return
            fi
            ;;
        macos*)
            if have brew; then
                run brew install python
                return
            fi
            ;;
    esac

    die "Automatic system dependency installation is not supported here. Suggested action: $(package_install_command)"
}

python_version_ok() {
    "$1" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

find_python() {
    local candidate
    for candidate in python3 python python3.13 python3.12 python3.11 python3.10; do
        if have "$candidate" && python_version_ok "$candidate"; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    return 1
}

ensure_venv_support() {
    local python="$1"
    if "$python" -m venv --help >/dev/null 2>&1; then
        return
    fi

    die "Python venv support is unavailable. Suggested action: $(package_install_command)"
}

main() {
    local install_target=".[dev]"
    local recreate=0
    local install_deps=0
    local assume_yes=0

    while (($#)); do
        case "$1" in
            --runtime)
                install_target="."
                ;;
            --dev)
                install_target=".[dev]"
                ;;
            --recreate)
                recreate=1
                ;;
            --install-system-deps)
                install_deps=1
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

    detect_platform
    log "Detected platform: $PLATFORM_NAME"

    if [[ "$PLATFORM_ID" == "windows" ]]; then
        die "Use PowerShell on Windows: ./scripts/setup.ps1"
    fi

    if [[ "$install_deps" == 1 ]]; then
        install_system_deps "$assume_yes"
    fi

    local root
    root="$(root_dir)"
    cd -- "$root"

    local python
    if ! python="$(find_python)"; then
        die "Python 3.10+ was not found. Suggested action: $(package_install_command)"
    fi

    ensure_venv_support "$python"

    if [[ "$recreate" == 1 && -d .venv ]]; then
        run rm -rf .venv
    fi

    if [[ ! -d .venv ]]; then
        run "$python" -m venv .venv
    fi

    local venv_python=".venv/bin/python"
    [[ -x "$venv_python" ]] || die "Virtual environment Python was not created at $venv_python"

    run "$venv_python" -m pip install --upgrade pip
    run "$venv_python" -m pip install -e "$install_target"

    log ""
    log "Install complete."
    log "Python: $("$venv_python" -c 'import sys; print(sys.executable)')"
    log "Run the web UI with: ./scripts/run.ps1"
    log "Or run directly with: .venv/bin/hfpull-web --host 127.0.0.1 --port 8019"
}

main "$@"
