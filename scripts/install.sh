#!/usr/bin/env bash
set -Eeuo pipefail

export HF_HUB_DISABLE_XET=1
unset HF_XET_HIGH_PERFORMANCE
unset HF_XET_CHUNK_CACHE_SIZE_BYTES
unset HF_XET_SHARD_CACHE_SIZE_LIMIT

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
  --install-mac-app       Build and install HuggingFacePullMac.app in
                          ~/Applications (macOS only).
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

install_macos_app() {
    local root="$1"
    local python="$2"
    local mac_root="$root/mac/HuggingFacePullMac"
    local staging_app="$root/build/HuggingFacePullMac.app"
    local backend_dir="$staging_app/Contents/Resources/backend"
    local backend_venv="$backend_dir/.venv"
    local asset_catalog="$mac_root/Assets.xcassets"
    local applications_dir="$HOME/Applications"
    local installed_app="$applications_dir/HuggingFacePullMac.app"
    local bin_path

    [[ "$PLATFORM_ID" == "macos" ]] || die "--install-mac-app is supported on macOS only"
    have swift || die "Swift is required for --install-mac-app. Install Xcode or the Xcode Command Line Tools."
    [[ -f "$mac_root/Package.swift" ]] || die "Mac app package not found: $mac_root"
    [[ -f "$mac_root/Info.plist" ]] || die "Mac app Info.plist template not found"
    [[ -f "$mac_root/HuggingFacePullMac-launcher.sh" ]] || die "Mac app launcher template not found"
    [[ -d "$asset_catalog/AppIcon.appiconset" ]] || die "Mac app icon assets not found"

    run swift build --configuration release --package-path "$mac_root" --product HuggingFacePullMac
    bin_path="$(swift build --configuration release --package-path "$mac_root" --show-bin-path)"
    [[ -x "$bin_path/HuggingFacePullMac" ]] || die "Mac app executable was not produced"

    run rm -rf "$staging_app"
    run mkdir -p "$staging_app/Contents/MacOS" "$backend_dir"
    run install -m 755 "$bin_path/HuggingFacePullMac" "$staging_app/Contents/MacOS/HuggingFacePullMac.bin"
    run install -m 755 "$mac_root/HuggingFacePullMac-launcher.sh" "$staging_app/Contents/MacOS/HuggingFacePullMac"
    run install -m 644 "$mac_root/Info.plist" "$staging_app/Contents/Info.plist"
    run xcrun actool --compile "$staging_app/Contents/Resources" --output-partial-info-plist "$staging_app/Contents/Resources/asset-info.plist" --platform macosx --minimum-deployment-target 13.0 --app-icon AppIcon "$asset_catalog"
    run cp -R "$root/src" "$backend_dir/src"

    run "$python" -m venv --copies "$backend_venv"
    run "$backend_venv/bin/python" -m pip install --upgrade pip
    run "$backend_venv/bin/python" -m pip install "$root"

    run mkdir -p "$applications_dir"
    if [[ -e "$installed_app" ]]; then
        local backup_app="$applications_dir/HuggingFacePullMac.app.backup-$(date +%Y%m%d-%H%M%S)"
        run mv "$installed_app" "$backup_app"
        log "Previous app moved to: $backup_app"
    fi
    run mv "$staging_app" "$installed_app"

    if have codesign; then
        run codesign --force --sign - "$installed_app/Contents/MacOS/HuggingFacePullMac.bin"
    fi
    if [[ -x /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister ]]; then
        run /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$installed_app"
    fi

    log "Mac app installed: $installed_app"
}

main() {
    local install_target=".[dev]"
    local recreate=0
    local install_deps=0
    local install_mac_app=0
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
            --install-mac-app)
                install_mac_app=1
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

    if [[ "$install_mac_app" == 1 ]]; then
        install_macos_app "$root" "$python"
    fi

    log ""
    log "Install complete."
    log "Python: $("$venv_python" -c 'import sys; print(sys.executable)')"
    log "Run the web UI with: ./scripts/run.ps1"
    log "Or run directly with: .venv/bin/hfpull-web --host 127.0.0.1 --port 8019"
    if [[ "$install_mac_app" == 1 ]]; then
        log "Open the Mac app from ~/Applications/HuggingFacePullMac.app"
    fi
}

main "$@"
