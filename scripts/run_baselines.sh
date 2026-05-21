#!/usr/bin/env bash
# Baseline runs: GRU / LSTM / S4D on all 4 POPGym tasks at 15M steps.
# One seed each. Runs sequentially; parallelise manually if multiple GPUs available.
# Usage: bash scripts/run_baselines.sh

set -e
cd "$(dirname "$0")/.."

RUNS_DIR="runs/gate/F1/baselines"
CFG="benchmarks/phase_a/ablation"
LOG_DIR="$RUNS_DIR/logs"
mkdir -p "$LOG_DIR"

run() {
  local cfg="$1"
  local env_dir="$2"
  mkdir -p "$RUNS_DIR/$env_dir"
  echo "[$(date '+%H:%M:%S')] Starting $cfg -> $env_dir"
  python train.py \
    --config "$CFG/${cfg}.yaml" \
    --seed 0 \
    --runs-dir "$RUNS_DIR/$env_dir/" \
    > "$LOG_DIR/${cfg}.log" 2>&1
  echo "[$(date '+%H:%M:%S')] Done    $cfg"
}

# # ── AutoencodeMedium ──────────────────────────────────────────────────────────
# run gru_autoencode_medium_15M   autoencode_medium_15M
# run lstm_autoencode_medium_15M  autoencode_medium_15M
# run s4d_autoencode_medium_15M   autoencode_medium_15M

# # ── CountRecallMedium ─────────────────────────────────────────────────────────
# run gru_countrecall_medium_15M   countrecall_medium_15M
# run lstm_countrecall_medium_15M  countrecall_medium_15M
# run s4d_countrecall_medium_15M   countrecall_medium_15M

# # ── RepeatPreviousMedium ──────────────────────────────────────────────────────
# run gru_repeat_previous_medium_15M   repeat_previous_medium_15M
run lstm_repeat_previous_medium_15M  repeat_previous_medium_15M
run s4d_repeat_previous_medium_15M   repeat_previous_medium_15M

# # ── BattleshipEasy ────────────────────────────────────────────────────────────
# run gru_battleship_easy_15M   battleship_easy_15M
# run lstm_battleship_easy_15M  battleship_easy_15M
# run s4d_battleship_easy_15M   battleship_easy_15M

echo "All baseline runs complete."
