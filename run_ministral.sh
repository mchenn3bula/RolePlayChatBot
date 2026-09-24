#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HSA_ENABLE_DXG_DETECTION=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
venv="${MINISTRAL_VENV:-$HOME/.venvs/roleplay-ministral}"
run_name="${1:-ministral-p1-$(date +%Y%m%d-%H%M%S)}"
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid run name.'; exit 1; }
profile="${2:-P1}"
case "$profile" in
    P0|P1) config="configs/ministral_${profile,,}.json" ;;
    *) echo "Invalid profile."; exit 1 ;;
esac
args=(--config "$config" --output-dir "reports/$run_name")
if [[ -n "${3:-}" ]]; then
    args+=(--adapter "$3")
    venv="${LORA_VENV:-$HOME/.venvs/roleplay-lora}"
fi
mkdir -p reports/logs
log="$(mktemp "reports/logs/${run_name}-XXXXXXXX.log")"
echo "Ministral $profile inference (file-output only) | log: $log" | tee -a "$log"
"$venv/bin/python" -u -m posttraining.evaluate "${args[@]}" 2>&1 | tee -a "$log"
