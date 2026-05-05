#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/traffic/home/dingkuiye/DPOT}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/traffic/home/dingkuiye/.conda/envs/ot/bin/python}"
SESSION="${SESSION:-dpot_sweep}"
EPOCHS="${EPOCHS:-20}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-20}"

cd "$ROOT_DIR"

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

run_cmd() {
  local gpu="$1"
  local output_tag="$2"
  shift 2
  printf 'cd %q && %q scripts/run_dpot_20epoch_sweep.py --gpu %q --epochs %q --warmup_epochs %q --batch_size %q --only' \
    "$ROOT_DIR" "$PYTHON_BIN" "$gpu" "$EPOCHS" "$WARMUP_EPOCHS" "$BATCH_SIZE"
  for exp in "$@"; do
    printf ' %q' "$exp"
  done
  printf ' --output_csv %q --output_md %q 2>&1 | tee %q' \
    "dpot_20epoch_sweep_results_${output_tag}.csv" \
    "dpot_20epoch_sweep_results_${output_tag}.md" \
    "sweep20_${output_tag}.tmux.log"
}

tmux new-session -d -s "$SESSION" -n gpu0 "$(run_cmd 0 gpu0 ar semi_freq_0.1)"
tmux new-window -t "$SESSION" -n gpu1 "$(run_cmd 1 gpu1 semi_freq_0.2 semi_freq_0.3)"
tmux new-window -t "$SESSION" -n gpu2 "$(run_cmd 2 gpu2 semi_freq_0.4 semi_freq_0.5)"
tmux new-window -t "$SESSION" -n gpu3 "$(run_cmd 3 gpu3 semi_freq_0.6 semi_freq_0.7)"
tmux new-window -t "$SESSION" -n gpu5 "$(run_cmd 5 gpu5 semi_freq_0.8 semi_freq_0.9 semi_freq_1.0)"

tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
tmux list-windows -t "$SESSION"
