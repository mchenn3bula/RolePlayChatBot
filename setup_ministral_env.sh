#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
venv="${MINISTRAL_VENV:-$HOME/.venvs/roleplay-ministral}"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
[[ -e /dev/dxg ]] || { echo 'WSL GPU compute interface missing.'; exit 1; }
python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install -r requirements-ministral-rocm.lock.txt
"$venv/bin/python" -m pip check
