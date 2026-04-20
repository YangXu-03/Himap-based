#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
MODEL_PATH=${MODEL_PATH:-/root/nfs/model/llava-v1.5-7b}

SCIENCEQA_QUESTION_FILE=${SCIENCEQA_QUESTION_FILE:-/root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json}
SCIENCEQA_IMAGE_FOLDER=${SCIENCEQA_IMAGE_FOLDER:-/root/nfs/code/HiMAP/data/scienceqa/images/test}

DEFAULT_TEXTVQA_QUESTION_FILE_1=/root/nfs/dataset/TextVQA/TextVQA_0.5.1_val.json
DEFAULT_TEXTVQA_QUESTION_FILE_2=/root/nfs/code/dataset/TextVQA/TextVQA_0.5.1_val.json

if [[ -z "${TEXTVQA_QUESTION_FILE:-}" ]]; then
  if [[ -f "$DEFAULT_TEXTVQA_QUESTION_FILE_1" ]]; then
    TEXTVQA_QUESTION_FILE="$DEFAULT_TEXTVQA_QUESTION_FILE_1"
  else
    TEXTVQA_QUESTION_FILE="$DEFAULT_TEXTVQA_QUESTION_FILE_2"
  fi
fi

TEXTVQA_IMAGE_FOLDER=${TEXTVQA_IMAGE_FOLDER:-/root/nfs/dataset/TextVQA/train_images}
if [[ ! -d "$TEXTVQA_IMAGE_FOLDER" ]]; then
  TEXTVQA_IMAGE_FOLDER=/root/nfs/code/dataset/TextVQA/train_images
fi

if [[ ! -f "$TEXTVQA_QUESTION_FILE" ]]; then
  echo "[Error] TextVQA question file not found: $TEXTVQA_QUESTION_FILE"
  echo "Set TEXTVQA_QUESTION_FILE manually, e.g.:"
  echo "  export TEXTVQA_QUESTION_FILE=/root/nfs/dataset/TextVQA/TextVQA_0.5.1_val.json"
  exit 1
fi

NUM_SAMPLES_PER_DATASET=${NUM_SAMPLES_PER_DATASET:-100}
FAST_V_SYS_LENGTH=${FAST_V_SYS_LENGTH:-35}
FAST_V_IMAGE_TOKEN_LENGTH=${FAST_V_IMAGE_TOKEN_LENGTH:-576}
CONV_MODE=${CONV_MODE:-vicuna_v1}

OUTPUT_FILE=${OUTPUT_FILE:-layerwise_full_image_prune_results.json}
OUTPUT_PLOT=${OUTPUT_PLOT:-layerwise_full_image_prune_plot.png}

python /root/nfs/code/HiMAP/src/HiMAP/inference/layerwise_full_image_prune_eval.py \
  --model-path "$MODEL_PATH" \
  --conv-mode "$CONV_MODE" \
  --scienceqa-question-file "$SCIENCEQA_QUESTION_FILE" \
  --scienceqa-image-folder "$SCIENCEQA_IMAGE_FOLDER" \
  --textvqa-question-file "$TEXTVQA_QUESTION_FILE" \
  --textvqa-image-folder "$TEXTVQA_IMAGE_FOLDER" \
  --num-samples-per-dataset "$NUM_SAMPLES_PER_DATASET" \
  --fast-v-sys-length "$FAST_V_SYS_LENGTH" \
  --fast-v-image-token-length "$FAST_V_IMAGE_TOKEN_LENGTH" \
  --output-file "$OUTPUT_FILE" \
  --output-plot "$OUTPUT_PLOT"
