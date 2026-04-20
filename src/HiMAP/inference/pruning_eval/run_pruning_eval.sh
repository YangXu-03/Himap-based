#!/usr/bin/env bash
set -euo pipefail

# MME-only evaluation script using eval_mme.py data loading style:
#   1) eval_mme.py inference (same question/image loading as eval_mme.sh)
#   2) convert detailed predictions to eval_tool text format
#   3) eval_tool/calculation.py scoring
#
# Examples:
#   MODE=baseline EXP_NAME=llava-v1.5-7b-baseline bash ./src/HiMAP/inference/pruning_eval/run_pruning_eval.sh
#   MODE=himap EXP_NAME=llava-v1.5-7b-himap bash ./src/HiMAP/inference/pruning_eval/run_pruning_eval.sh
#   MODE=fastv EXP_NAME=llava-v1.5-7b-fastv bash ./src/HiMAP/inference/pruning_eval/run_pruning_eval.sh
#   MODE=jsd_entropy EXP_NAME=llava-v1.5-7b-jsd bash ./src/HiMAP/inference/pruning_eval/run_pruning_eval.sh

MODE=${MODE:-baseline}
GPU_ID=${GPU_ID:-1}
MODEL_PATH=${MODEL_PATH:-/root/nfs/model/llava-v1.5-7b}
CONV_MODE=${CONV_MODE:-vicuna_v1}
TEMPERATURE=${TEMPERATURE:-0.0}

# MME data defaults (aligned with eval_mme.sh)
QUESTION_FILE=${QUESTION_FILE:-/root/nfs/code/HiMAP/data/MME/MME_test.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/root/nfs/code/HiMAP/data/MME/images/test}

# Output defaults
EXP_NAME=${EXP_NAME:-llava-v1.5-7b-${MODE}}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/nfs/code/HiMAP/output/mme_evaltool}
DETAIL_DIR=${DETAIL_DIR:-${OUTPUT_ROOT}/detailed}
EVALTOOL_RESULTS_BASE=${EVALTOOL_RESULTS_BASE:-${OUTPUT_ROOT}/answers}
DETAILED_RESULTS_FILE=${DETAILED_RESULTS_FILE:-${DETAIL_DIR}/${EXP_NAME}.json}
EVALTOOL_RESULTS_DIR=${EVALTOOL_RESULTS_DIR:-${EVALTOOL_RESULTS_BASE}/${EXP_NAME}}

# Shared defaults
SYS_LENGTH=${SYS_LENGTH:-35}
IMG_LENGTH=${IMG_LENGTH:-576}

# HiMAP defaults
HMAP_TXT_LAYER=${HMAP_TXT_LAYER:-2}
HMAP_IMG_LAYER=${HMAP_IMG_LAYER:-8}
HMAP_TXT_RANK=${HMAP_TXT_RANK:-128}
HMAP_IMG_RANK=${HMAP_IMG_RANK:-72}

# FastV defaults
FASTV_RANK=${FASTV_RANK:-128}
FASTV_AGG_LAYER=${FASTV_AGG_LAYER:-2}

# JSD+Entropy defaults
JSD_TOPK_PERCENT=${JSD_TOPK_PERCENT:-10}
JSD_STAGE_RANGES=${JSD_STAGE_RANGES:-2-8,9-20,21-31}
JSD_STAGE_PRUNE_RATIOS=${JSD_STAGE_PRUNE_RATIOS:-0.1,0.4,0.5}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "[Run] MME MODE=${MODE} GPU=${CUDA_VISIBLE_DEVICES}"
echo "[Run] EXP_NAME=${EXP_NAME}"

if [[ ! -f "${QUESTION_FILE}" ]]; then
  echo "[Error] QUESTION_FILE not found: ${QUESTION_FILE}"
  exit 1
fi

if [[ ! -d "${IMAGE_FOLDER}" ]]; then
  echo "[Error] IMAGE_FOLDER not found: ${IMAGE_FOLDER}"
  exit 1
fi

if [[ ! -f "./src/HiMAP/inference/pruning_eval/convert_mme_results_to_evaltool.py" ]]; then
  echo "[Error] converter not found: ./src/HiMAP/inference/pruning_eval/convert_mme_results_to_evaltool.py"
  exit 1
fi

mkdir -p "${DETAIL_DIR}" "${EVALTOOL_RESULTS_BASE}"

eval_cmd=(
  python ./src/HiMAP/inference/eval_mme.py
  --model-path "${MODEL_PATH}"
  --question-file "${QUESTION_FILE}"
  --image-folder "${IMAGE_FOLDER}"
  --temperature "${TEMPERATURE}"
  --conv-mode "${CONV_MODE}"
  --save-detailed-results-file "${DETAILED_RESULTS_FILE}"
)

case "${MODE}" in
    baseline)
      ;;
    himap)
      eval_cmd+=(
        --use-hmap-v
        --sys-length "${SYS_LENGTH}"
        --img-length "${IMG_LENGTH}"
        --hmap-v-attn-txt-layer "${HMAP_TXT_LAYER}"
        --hmap-v-attn-img-layer "${HMAP_IMG_LAYER}"
        --hmap-v-attn-txt-rank "${HMAP_TXT_RANK}"
        --hmap-v-attn-img-rank "${HMAP_IMG_RANK}"
      )
      ;;
    fastv)
      eval_cmd+=(
        --use-fast-v
        --fast-v-sys-length "${SYS_LENGTH}"
        --fast-v-image-token-length "${IMG_LENGTH}"
        --fast-v-attention-rank "${FASTV_RANK}"
        --fast-v-agg-layer "${FASTV_AGG_LAYER}"
      )
      ;;
    jsd_entropy)
      eval_cmd+=(
        --use-jsd-entropy-prune
        --jsd-entropy-sys-length "${SYS_LENGTH}"
        --jsd-entropy-img-length "${IMG_LENGTH}"
        --jsd-entropy-topk-percent "${JSD_TOPK_PERCENT}"
        --jsd-entropy-stage-ranges "${JSD_STAGE_RANGES}"
        --jsd-entropy-stage-prune-ratios "${JSD_STAGE_PRUNE_RATIOS}"
      )
      ;;
    *)
      echo "[Error] Unsupported MODE=${MODE}. Use baseline|himap|fastv|jsd_entropy"
      exit 1
      ;;
esac

echo "[Step 1/3] Run eval_mme.py inference"
"${eval_cmd[@]}"

echo "[Step 2/3] Convert detailed results to eval_tool format"
python ./src/HiMAP/inference/pruning_eval/convert_mme_results_to_evaltool.py \
  --detailed-results "${DETAILED_RESULTS_FILE}" \
  --output-dir "${EVALTOOL_RESULTS_DIR}"

echo "[Step 3/3] Calculate MME metrics via eval_tool/calculation.py"
python ./src/HiMAP/eval_tool/calculation.py --results_dir "${EVALTOOL_RESULTS_DIR}"

echo "[Done] MME evaluation finished."
