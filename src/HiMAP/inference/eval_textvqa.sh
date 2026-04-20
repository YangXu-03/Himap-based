#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   MODE=baseline bash ./src/HiMAP/inference/eval_textvqa.sh
#   MODE=himap bash ./src/HiMAP/inference/eval_textvqa.sh
#   MODE=fastv bash ./src/HiMAP/inference/eval_textvqa.sh
#   MODE=jsd_entropy bash ./src/HiMAP/inference/eval_textvqa.sh

MODE=${MODE:-jsd_entropy}  # baseline|himap|fastv|jsd_entropy
MODEL_PATH=${MODEL_PATH:-/root/nfs/model/llava-v1.5-7b}
ANNOTATION_FILE=${ANNOTATION_FILE:-/root/nfs/code/HiMAP/data/textvqa/TextVQA_0.5.1_val.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/root/nfs/code/HiMAP/data/textvqa/train_images}
OUTPUT_DIR=${OUTPUT_DIR:-/root/nfs/code/HiMAP/data/textvqa/answers}
GPU_ID=${GPU_ID:-2}
NUM_SAMPLES=${NUM_SAMPLES:--1}
JSD_MODE=${JSD_MODE:-prompt_image}  # prompt_image|global
JSD_TARGET_TOKENS=${JSD_TARGET_TOKENS:-64}
JSD_N0=${JSD_N0:-576}
JSD_PHASE1_LAYER=${JSD_PHASE1_LAYER:-3}
JSD_PHASE2_LAYER=${JSD_PHASE2_LAYER:-8}
JSD_PHASE3_LAYER=${JSD_PHASE3_LAYER:-16}
JSD_MU_H=${JSD_MU_H:-0.620257}
JSD_SIGMA_H=${JSD_SIGMA_H:-0.030169}
JSD_MU_W=${JSD_MU_W:-0.667733}
JSD_SIGMA_W=${JSD_SIGMA_W:-0.038618}

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

mkdir -p "${OUTPUT_DIR}"
ANSWERS_FILE="${OUTPUT_DIR}/textvqa_${MODE}.jsonl"

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "[Run] MODE=${MODE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

declare -a BASE_ARGS=(
  --model-path "${MODEL_PATH}"
  --annotation-file "${ANNOTATION_FILE}"
  --image-folder "${IMAGE_FOLDER}"
  --answers-file "${ANSWERS_FILE}"
  --single-pred-prompt
  --num-samples "${NUM_SAMPLES}"
)

if [[ "${MODE}" == "himap" ]]; then
  python ./src/HiMAP/inference/eval_textvqa.py \
    "${BASE_ARGS[@]}" \
    --use-hmap-v \
    --sys-length 35 \
    --img-length 576 \
    --hmap-v-attn-txt-layer 2 \
    --hmap-v-attn-img-layer 8 \
    --hmap-v-attn-txt-rank 288 \
    --hmap-v-attn-img-rank 128
elif [[ "${MODE}" == "fastv" ]]; then
  python ./src/HiMAP/inference/eval_textvqa.py \
    "${BASE_ARGS[@]}" \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank 100 \
    --fast-v-agg-layer 2
elif [[ "${MODE}" == "jsd_entropy" ]]; then
  python ./src/HiMAP/inference/eval_textvqa.py \
    "${BASE_ARGS[@]}" \
    --use-jsd-entropy-prune \
    --jsd-entropy-topk-attention-mode "${JSD_MODE}" \
    --jsd-entropy-sys-length 35 \
    --jsd-entropy-img-length 576 \
    --jsd-entropy-target-tokens "${JSD_TARGET_TOKENS}" \
    --jsd-entropy-n0 "${JSD_N0}" \
    --jsd-entropy-phase1-prune-layer "${JSD_PHASE1_LAYER}" \
    --jsd-entropy-phase2-prune-layer "${JSD_PHASE2_LAYER}" \
    --jsd-entropy-phase3-prune-layer "${JSD_PHASE3_LAYER}" \
    --jsd-entropy-mu-h "${JSD_MU_H}" \
    --jsd-entropy-sigma-h "${JSD_SIGMA_H}" \
    --jsd-entropy-mu-w "${JSD_MU_W}" \
    --jsd-entropy-sigma-w "${JSD_SIGMA_W}" \
    --jsd-entropy-alpha "${JSD_ALPHA}" \
    --jsd-entropy-beta "${JSD_BETA}" \
    --jsd-entropy-topk-percent 10
elif [[ "${MODE}" == "baseline" ]]; then
  python ./src/HiMAP/inference/eval_textvqa.py \
    "${BASE_ARGS[@]}"
else
  echo "[Error] Unsupported MODE=${MODE}. Use one of: baseline|himap|fastv|jsd_entropy"
  exit 1
fi

echo "[Eval] Running official TextVQA evaluator"
python -m llava.eval.eval_textvqa \
  --annotation-file "${ANNOTATION_FILE}" \
  --result-file "${ANSWERS_FILE}"

echo "[Done] Result file: ${ANSWERS_FILE}"
