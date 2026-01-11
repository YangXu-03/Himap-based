#!/bin/bash

# AdaptInfer evaluation script for ScienceQA
# Usage: bash eval_scivqa_adaptinfer.sh

# Set your paths here
MODEL_PATH="liuhaotian/llava-v1.5-7b"  # or your local model path
IMAGE_FOLDER="./data/scienceqa/images/test"  # path to ScienceQA images
QUESTION_FILE="./data/scienceqa/himap-inference-MCQ.json"
# AdaptInfer parameters
SYS_LENGTH=35
IMG_LENGTH=576
PRUNING_LAYERS=(1 10 20)  # layers to apply pruning
KEEP_RATIO=0.25

# Run evaluation
python ./src/HiMAP/inference/eval_scivqa_adaptinfer.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --conv-mode vicuna_v1 \
    --use-ours \
    --adaptinfer-sys-length "$SYS_LENGTH" \
    --adaptinfer-img-length "$IMG_LENGTH" \
    --adaptinfer-pruning-layers "${PRUNING_LAYERS[@]}" \
    --adaptinfer-keep-ratio "$KEEP_RATIO" \
    --single-pred-prompt
