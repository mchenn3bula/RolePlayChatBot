#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HSA_ENABLE_DXG_DETECTION=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
venv="${MINISTRAL_VENV:-$HOME/.venvs/roleplay-ministral}"
profile="${2:-P1}"
case "$profile" in
    P0|P1) config="configs/ministral_${profile,,}.json" ;;
    *) echo "Invalid profile."; exit 1 ;;
esac
args=(--language "${1:-en}" --config "$config")
if (( $# >= 2 )); then shift 2; else set --; fi
for arg in "$@"; do
    if [[ "$arg" == '--adapter' ]]; then venv="${LORA_VENV:-$HOME/.venvs/roleplay-lora}"; fi
done
args+=("$@")
exec "$venv/bin/python" -u -m posttraining.chat "${args[@]}"
