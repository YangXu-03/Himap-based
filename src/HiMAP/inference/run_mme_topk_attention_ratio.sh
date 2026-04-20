#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODEL_PATH=${MODEL_PATH:-/root/nfs/model/llava-v1.5-7b}
QUESTION_FILE=${QUESTION_FILE:-/root/nfs/code/HiMAP/data/MME/MME_test.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/root/nfs/code/HiMAP/data/MME/images/test}
GPU_ID=${GPU_ID:-1}
ATTN_MODE=${ATTN_MODE:-prompt_image}  # prompt_image|global
SYS_LENGTH=${SYS_LENGTH:-35}
IMG_LENGTH=${IMG_LENGTH:-576}
TOPK_VALUES=${TOPK_VALUES:-1,10,20,50}
NUM_SAMPLES=${NUM_SAMPLES:-0}  # 0 means full set
OUTPUT_JSON=${OUTPUT_JSON:-${SCRIPT_DIR}/mme_topk_attention_ratio.json}
OUTPUT_PLOT=${OUTPUT_PLOT:-${SCRIPT_DIR}/mme_topk_attention_ratio.png}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

CMD=(
  python ./src/HiMAP/inference/analyze_mme_topk_attention_ratio.py
  --model-path "${MODEL_PATH}"
  --question-file "${QUESTION_FILE}"
  --image-folder "${IMAGE_FOLDER}"
  --attention-mode "${ATTN_MODE}"
  --sys-length "${SYS_LENGTH}"
  --img-length "${IMG_LENGTH}"
  --topk-values "${TOPK_VALUES}"
  --output-json "${OUTPUT_JSON}"
  --output-plot "${OUTPUT_PLOT}"
)

if [[ "${NUM_SAMPLES}" -gt 0 ]]; then
  CMD+=(--num-samples "${NUM_SAMPLES}")
fi

echo "[Run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, mode=${ATTN_MODE}, topk=${TOPK_VALUES}"
"${CMD[@]}"
