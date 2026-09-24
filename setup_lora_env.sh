#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
venv="${LORA_VENV:-$HOME/.venvs/roleplay-lora}"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
[[ -e /dev/dxg ]] || { echo 'WSL GPU interface missing.'; exit 1; }
python3 -m venv "$venv"
"$venv/bin/python" -m pip install -r requirements-lora-rocm.lock.txt
"$venv/bin/python" -m pip check
