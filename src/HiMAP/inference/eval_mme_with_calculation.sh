#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODE=baseline bash ./src/HiMAP/inference/eval_mme_with_calculation.sh
#   MODE=himap bash ./src/HiMAP/inference/eval_mme_with_calculation.sh
#   MODE=fastv bash ./src/HiMAP/inference/eval_mme_with_calculation.sh
#   MODE=jsd_entropy bash ./src/HiMAP/inference/eval_mme_with_calculation.sh

MODE=${MODE:-jsd_entropy}  # baseline|himap|fastv|jsd_entropy
MODEL_PATH=${MODEL_PATH:-/root/nfs/model/llava-v1.5-7b}
QUESTION_FILE=${QUESTION_FILE:-/root/nfs/code/HiMAP/data/MME/MME_test.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/root/nfs/code/HiMAP/data/MME/images/test}
GPU_ID=${GPU_ID:-1}
CALC_SCRIPT=${CALC_SCRIPT:-./src/HiMAP/eval_tool/calculation.py}
JSD_MODE=${JSD_MODE:-prompt_image}  # prompt_image|global
JSD_TARGET_TOKENS=${JSD_TARGET_TOKENS:-64}
JSD_N0=${JSD_N0:-576}
JSD_PHASE1_LAYER=${JSD_PHASE1_LAYER:-3}
JSD_PHASE2_LAYER=${JSD_PHASE2_LAYER:-8}
JSD_PHASE3_LAYER=${JSD_PHASE3_LAYER:-16}
JSD_N_BASE_192=${JSD_N_BASE_192:-200}
JSD_N_BASE_128=${JSD_N_BASE_128:-140}
JSD_N_BASE_64=${JSD_N_BASE_64:-80}
JSD_MU_H=${JSD_MU_H:-0.620257}
JSD_SIGMA_H=${JSD_SIGMA_H:-0.030169}
JSD_MU_W=${JSD_MU_W:-0.667733}
JSD_SIGMA_W=${JSD_SIGMA_W:-0.038618}
JSD_USE_ONLY_PROMPT2IMAGE=${JSD_USE_ONLY_PROMPT2IMAGE:-True}
JSD_USE_ADAPTIVE_KEEP_RATIO=${JSD_USE_ADAPTIVE_KEEP_RATIO:-True}
if [[ "${MODE}" == "jsd_entropy" ]]; then
  JSD_N_BASE_VAR="JSD_N_BASE_${JSD_TARGET_TOKENS}"
  CALC_RESULTS_DIR=${CALC_RESULTS_DIR:-./mme_calres_${MODE}_${JSD_TARGET_TOKENS}_base${!JSD_N_BASE_VAR}_adaptratio${JSD_USE_ADAPTIVE_KEEP_RATIO}}
else
  CALC_RESULTS_DIR=${CALC_RESULTS_DIR:-./mme_calculation_results_${MODE}}
fi


case "${JSD_TARGET_TOKENS}" in
  192|128)
    JSD_ALPHA=${JSD_ALPHA:-24}
    JSD_BETA=${JSD_BETA:-16}
    ;;
  64)
    JSD_ALPHA=${JSD_ALPHA:-9}
    JSD_BETA=${JSD_BETA:-5}
    ;;
  *)
    JSD_ALPHA=${JSD_ALPHA:-0}
    JSD_BETA=${JSD_BETA:-0}
    ;;
esac

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "[Run] MODE=${MODE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[Run] CALC_SCRIPT=${CALC_SCRIPT}"
echo "[Run] CALC_RESULTS_DIR=${CALC_RESULTS_DIR}"

if [[ "${MODE}" == "himap" ]]; then
  python ./src/HiMAP/inference/eval_mme_with_calculation.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --calculation-script "${CALC_SCRIPT}" \
    --calculation-results-dir "${CALC_RESULTS_DIR}" \
    --use-hmap-v \
    --sys-length 35 \
    --img-length 576 \
    --hmap-v-attn-txt-layer 2 \
    --hmap-v-attn-img-layer 8 \
    --hmap-v-attn-txt-rank 288 \
    --hmap-v-attn-img-rank 128
elif [[ "${MODE}" == "fastv" ]]; then
  python ./src/HiMAP/inference/eval_mme_with_calculation.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --calculation-script "${CALC_SCRIPT}" \
    --calculation-results-dir "${CALC_RESULTS_DIR}" \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank 128 \
    --fast-v-agg-layer 2
elif [[ "${MODE}" == "jsd_entropy" ]]; then
  python ./src/HiMAP/inference/eval_mme_with_calculation.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --calculation-script "${CALC_SCRIPT}" \
    --calculation-results-dir "${CALC_RESULTS_DIR}" \
    --use-jsd-entropy-prune \
    --jsd-entropy-topk-attention-mode "${JSD_MODE}" \
    --jsd-entropy-sys-length 35 \
    --jsd-entropy-img-length 576 \
    --jsd-entropy-target-tokens "${JSD_TARGET_TOKENS}" \
    --jsd-entropy-n0 "${JSD_N0}" \
    --jsd-entropy-phase1-prune-layer "${JSD_PHASE1_LAYER}" \
    --jsd-entropy-phase2-prune-layer "${JSD_PHASE2_LAYER}" \
    --jsd-entropy-phase3-prune-layer "${JSD_PHASE3_LAYER}" \
    --jsd-entropy-n-base-192 "${JSD_N_BASE_192}" \
    --jsd-entropy-n-base-128 "${JSD_N_BASE_128}" \
    --jsd-entropy-n-base-64 "${JSD_N_BASE_64}" \
    --jsd-entropy-mu-h "${JSD_MU_H}" \
    --jsd-entropy-sigma-h "${JSD_SIGMA_H}" \
    --jsd-entropy-mu-w "${JSD_MU_W}" \
    --jsd-entropy-sigma-w "${JSD_SIGMA_W}" \
    --jsd-entropy-use-only-prompt2image-scoring "${JSD_USE_ONLY_PROMPT2IMAGE}" \
    --jsd-entropy-use-adaptive-keep-ratio "${JSD_USE_ADAPTIVE_KEEP_RATIO}" \
    --jsd-entropy-alpha "${JSD_ALPHA}" \
    --jsd-entropy-beta "${JSD_BETA}" \
    --jsd-entropy-topk-percent 10 \
    --jsd-entropy-stage-ranges 1-4,5-15,16-25
elif [[ "${MODE}" == "baseline" ]]; then
  python ./src/HiMAP/inference/eval_mme_with_calculation.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --calculation-script "${CALC_SCRIPT}" \
    --calculation-results-dir "${CALC_RESULTS_DIR}"
else
  echo "[Error] Unsupported MODE=${MODE}. Use one of: baseline|himap|fastv|jsd_entropy"
  exit 1
fi
