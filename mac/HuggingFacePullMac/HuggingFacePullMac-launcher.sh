#!/bin/bash
set -euo pipefail

contents_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
backend_dir="$contents_dir/Resources/backend"

export PYTHONDONTWRITEBYTECODE=1
cd "$backend_dir"
exec "$contents_dir/MacOS/HuggingFacePullMac.bin" "$@"
