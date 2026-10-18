#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
script_path="$(cygpath -w "$script_dir/win-make.ps1")"

exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$script_path" "$@"
