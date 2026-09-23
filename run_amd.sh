#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HSA_ENABLE_DXG_DETECTION=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
venv="${ROLEPLAY_VENV:-$HOME/.venvs/roleplay-rocm}"
python="$venv/bin/python"
[[ -x "$python" ]] || { echo 'Run setup_amd_env.sh first.'; exit 1; }
mode="${1:-Smoke}"
run_name="${2:-rx7900xtx-smoke-$(date +%Y%m%d-%H%M%S)}"
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid run name.'; exit 1; }
profile="${3:-Small}"
config=configs/colab_t4.json
epochs=3
time_args=()
case "$profile" in
  Small) ;;
  Overnight)
    config=configs/overnight_124m.json
    epochs=4
    time_args=(--max-hours 6) ;;
  Modern)
    config=configs/overnight_rope_swiglu.json
    epochs=4
    time_args=(--max-hours 6) ;;
  *) echo 'Profile must be Small, Overnight, or Modern.'; exit 1 ;;
esac
mkdir -p reports/logs
log="$(mktemp "reports/logs/${run_name}-${mode}-XXXXXXXX.log")"
run_logged() {
  echo "Profile: $profile | config: $config | terminal log: $log" | tee -a "$log"
  "$python" -u amd_run.py "$@" 2>&1 | tee -a "$log"
}
case "$mode" in
  Test)
    run_logged -m unittest discover -s tests -v ;;
  Smoke)
    run_logged smoke_amd.py --config "$config" --output-dir "checkpoints/$run_name" ;;
  Benchmark)
    [[ ! -e "reports/$run_name.json" ]] || { echo 'Benchmark report already exists.'; exit 1; }
    run_logged benchmark.py --config "$config" --tokenizer tokenizer --precision fp16 --micro-batch 8 --grad-accum 4 --warmup-steps 8 --steps 256 --output "reports/$run_name.json" ;;
  Train|Resume)
    [[ -n "${2:-}" ]] || { echo 'Specify a distinct run name for training or resume.'; exit 1; }
    args=(--config "$config" --tokenizer tokenizer
      --data-dir data/bluemoon_train_tok_ds --validation-dir data/bluemoon_validation_tok_ds
      --output-dir "checkpoints/$run_name" --epochs "$epochs" --micro-batch 8 --grad-accum 4
      --lr 0.00015 --warmup-updates 500 --precision fp16 --device cuda --seed 42 --log-every 25
      "${time_args[@]}")
    [[ "$mode" != Resume ]] || args+=(--resume "checkpoints/$run_name/latest.pt")
    run_logged train.py "${args[@]}" ;;
  *) echo 'Mode must be Test, Smoke, Benchmark, Train, or Resume.'; exit 1 ;;
esac
