#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
MODEL_PATH="liuhaotian/llava-v1.5-7b"
QUESTION_FILE="/root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json"
IMAGE_FOLDER="./data/scienceqa/images/test"
NUM_SAMPLES=10

echo "Running FastV with max_head..."

python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --single-pred-prompt \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank 288 \
    --fast-v-agg-layer 12 \
    --fast-v-token-selection-method max_head \
    --num-samples $NUM_SAMPLES
