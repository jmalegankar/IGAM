#!/usr/bin/env bash
# RP-Medium long-horizon test (15M steps) — kicks off 3 arms in parallel:
#   LMU, GatedLMU softsign, GatedLMU none
# Output under runs/gate/F1/repeat_previous_medium_15M/, logs sibling.
#
# Total CPU contention: 3 simultaneous processes. Each ~7-12h wall depending
# on contention. Designed for overnight unattended; if your laptop sleeps
# the run will pause until wake.
#
# Usage (when ready to fire):
#   nohup bash scripts/run_rp_medium_15M.sh >/dev/null 2>&1 &
#   tail -f runs/gate/F1/repeat_previous_medium_15M/_suite.log

set -uo pipefail

cd "$(dirname "$0")/.."

# Python path — override with `PY=/path/to/python bash scripts/run_rp_medium_15M.sh`
# on machines where the venv lives somewhere other than ./.venv/.
PY="${PY:-.venv/bin/python}"
RUN_DIR=runs/gate/F1/repeat_previous_medium_15M
SUITE_LOG="${RUN_DIR}/_suite.log"

mkdir -p "${RUN_DIR}"
echo "=== RP-Medium 15M suite started at $(date) ===" | tee -a "${SUITE_LOG}"

# Device sanity check — SB3 PPO auto-detects CUDA. Log what we'll actually use.
"${PY}" -c "import torch; print(f'  torch={torch.__version__}  cuda_available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}  device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1 | tee -a "${SUITE_LOG}"

# Three arms, parallel
ARMS=(
    "lmu_medium_tuned_15M"
    "gated_lmu_medium_tuned_15M"
    "gated_lmu_medium_tuned_no_gate_15M"
)

pids=()
for stem in "${ARMS[@]}"; do
    cfg="benchmarks/phase_a/ablation/${stem}.yaml"
    log="${RUN_DIR}/${stem}.log"
    "${PY}" train.py --config "${cfg}" --seed 0 --runs-dir "${RUN_DIR}" > "${log}" 2>&1 &
    pid=$!
    pids+=(${pid})
    echo "  Started ${stem} (pid ${pid})" | tee -a "${SUITE_LOG}"
done

# Wait for all 3, record exit codes
echo "Waiting for all 3 arms..." | tee -a "${SUITE_LOG}"
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    stem=${ARMS[$i]}
    wait ${pid}; rc=$?
    echo "  [$(date)] ${stem} exited ${rc}" | tee -a "${SUITE_LOG}"
done

echo "=== RP-Medium 15M suite finished at $(date) ===" | tee -a "${SUITE_LOG}"
