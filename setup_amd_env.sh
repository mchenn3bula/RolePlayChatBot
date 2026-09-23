#!/usr/bin/env bash
# Run as the ordinary WSL user, from the Windows project directory.
set -euo pipefail
cd -- "$(dirname -- "$0")"
venv="${ROLEPLAY_VENV:-$HOME/.venvs/roleplay-rocm}"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
[[ -e /dev/dxg ]] || { echo 'WSL GPU compute interface is missing.'; exit 1; }
python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install -r requirements-rocm.txt
"$venv/bin/python" -m pip check
export HSA_ENABLE_DXG_DETECTION=1
"$venv/bin/python" -c 'import torch; print(torch.__version__, torch.version.hip); assert torch.version.hip and torch.cuda.is_available(); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
