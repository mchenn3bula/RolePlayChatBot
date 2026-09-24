#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HSA_ENABLE_DXG_DETECTION=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
venv="${LORA_VENV:-$HOME/.venvs/roleplay-lora}"
run_name="${1:?Run name required}"
mode="${2:-Train}"
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid run name.'; exit 1; }
args=(--output-dir "checkpoints/$run_name")
case "$mode" in
    Train) ;;
    Smoke) args+=(--smoke) ;;
    Resume)
        checkpoint="${3:?Checkpoint directory name required}"
        [[ "$checkpoint" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 1
        args+=(--resume "checkpoints/$run_name/$checkpoint") ;;
    *) echo 'Invalid mode.'; exit 1 ;;
esac
mkdir -p reports/logs
log="$(mktemp "reports/logs/${run_name}-${mode}-XXXXXXXX.log")"
echo "LoRA $mode | technical log: $log" | tee -a "$log"
"$venv/bin/python" -u -m posttraining.lora_train "${args[@]}" 2>&1 | tee -a "$log"
