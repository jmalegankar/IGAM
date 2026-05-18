#!/usr/bin/env bash
# F1-suite launcher.
#
# For each new env (Autoencode, CountRecall, Battleship), runs LMU and
# GatedLMU IN PARALLEL — both at the env's tuned hyperparameters (matched
# memory_size, θ scaled to the env's episode length, same PPO geometry).
# Pairs are run sequentially across envs so CPU contention stays bounded
# to 2 simultaneous processes.
#
# Output layout:
#   runs/gate/F1/<env_short>/lmu_<cfg>/LMU/seed_0_<ts>/
#   runs/gate/F1/<env_short>/gated_lmu_<cfg>/GatedLMU/seed_0_<ts>/
#
# RepeatPreviousMedium is already at runs/gate/F1/repeat_previous_medium/.
#
# Wall time estimate: ~5-7h per pair (2× CPU contention) × 3 pairs ≈ 15-20h.
#
# Usage:  bash scripts/run_f1_suite.sh
#         tail -f runs/gate/F1/_suite.log

set -uo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
F1_DIR=runs/gate/F1
SUITE_LOG="${F1_DIR}/_suite.log"

mkdir -p "${F1_DIR}"
echo "=== F1-suite started at $(date) ===" | tee -a "${SUITE_LOG}"

# Each entry: <env_short> <lmu_cfg> <gated_cfg>
PAIRS=(
    "autoencode_medium       lmu_autoencode_medium_tuned    gated_lmu_autoencode_medium_tuned"
    "countrecall_medium      lmu_countrecall_medium_tuned   gated_lmu_countrecall_medium_tuned"
    "battleship_easy         lmu_battleship_easy_tuned      gated_lmu_battleship_easy_tuned"
)

for entry in "${PAIRS[@]}"; do
    read -r env_short lmu_stem gated_stem <<< "${entry}"
    env_dir="${F1_DIR}/${env_short}"
    mkdir -p "${env_dir}"

    lmu_cfg="benchmarks/phase_a/ablation/${lmu_stem}.yaml"
    gated_cfg="benchmarks/phase_a/ablation/${gated_stem}.yaml"

    echo "" | tee -a "${SUITE_LOG}"
    echo "=== [$(date)] Starting pair ${env_short} (LMU + GatedLMU in parallel) ===" | tee -a "${SUITE_LOG}"

    "${PY}" train.py --config "${lmu_cfg}"   --seed 0 --runs-dir "${env_dir}" \
        > "${env_dir}/${lmu_stem}.log" 2>&1 &
    pid_lmu=$!

    "${PY}" train.py --config "${gated_cfg}" --seed 0 --runs-dir "${env_dir}" \
        > "${env_dir}/${gated_stem}.log" 2>&1 &
    pid_gated=$!

    echo "    LMU pid=${pid_lmu}  GatedLMU pid=${pid_gated}" | tee -a "${SUITE_LOG}"

    # Wait for both arms before moving to next env.
    wait ${pid_lmu};   lmu_rc=$?
    wait ${pid_gated}; gated_rc=$?

    echo "=== [$(date)] DONE ${env_short}  (LMU exit ${lmu_rc}, GatedLMU exit ${gated_rc}) ===" | tee -a "${SUITE_LOG}"
done

echo "" | tee -a "${SUITE_LOG}"
echo "=== F1-suite finished at $(date) ===" | tee -a "${SUITE_LOG}"
