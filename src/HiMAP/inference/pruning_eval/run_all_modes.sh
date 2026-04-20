#!/usr/bin/env bash
set -euo pipefail

# Run all pruning modes for MME.
# Examples:
#   bash ./src/HiMAP/inference/pruning_eval/run_all_modes.sh
#   GPU_ID=1 bash ./src/HiMAP/inference/pruning_eval/run_all_modes.sh

MODES=${MODES:-baseline himap fastv jsd_entropy}

for mode in ${MODES}; do
  echo ""
  echo "=============================="
  echo "[Batch] MME MODE=${mode}"
  echo "=============================="
  MODE="${mode}" bash ./src/HiMAP/inference/pruning_eval/run_pruning_eval.sh
done

echo ""
echo "[Done] Completed all modes for MME"
