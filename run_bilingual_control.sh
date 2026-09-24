#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HSA_ENABLE_DXG_DETECTION=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
venv="${LORA_VENV:-$HOME/.venvs/roleplay-lora}"
arm="${1:?sft or dpo required}"
run="${2:?Run name required}"
mode="${3:-Train}"
[[ "$arm" == sft || "$arm" == dpo ]] || exit 1
[[ "$run" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 1
args=(--config "configs/ministral_v2_${arm}.json" --output-dir "checkpoints/$run")
case "$mode" in
    Train) ;;
    Smoke) args+=(--smoke) ;;
    Resume)
        checkpoint="${4:?Checkpoint required}"
        [[ "$checkpoint" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 1
        args+=(--resume "checkpoints/$run/$checkpoint") ;;
    *) exit 1 ;;
esac
mkdir -p reports/logs
log="$(mktemp "reports/logs/${run}-${mode}-XXXXXXXX.log")"
echo "Bilingual $arm control $mode | log $log" | tee -a "$log"
"$venv/bin/python" -u -m posttraining.control_train "${args[@]}" 2>&1 | tee -a "$log"
