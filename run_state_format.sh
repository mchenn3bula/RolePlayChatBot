#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HSA_ENABLE_DXG_DETECTION=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
venv="${LORA_VENV:-$HOME/.venvs/roleplay-lora}"
mkdir -p reports/logs
log="$(mktemp "reports/logs/ministral-state-format-v1-XXXXXXXX.log")"
echo "State-format inference comparison | technical log $log" | tee -a "$log"
for mode in prepare generate pair; do
    "$venv/bin/python" -u -m posttraining.state_format "$mode" 2>&1 | tee -a "$log"
done
