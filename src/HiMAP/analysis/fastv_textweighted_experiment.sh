#!/bin/bash

# Text-Weighted Pruning Experiment Script
# 使用文本token重要性加权的图像token剪枝策略

MODEL_PATH="liuhaotian/llava-v1.5-7b"
IMAGE_FOLDER="/root/nfs/code/HiMAP/data/scienceqa/images/test"
QUESTION_FILE="/root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ-VC-7B.json"
CONV_MODE="vicuna_v1"
NUM_SAMPLES=300  # 每个rank测试的样本数

# FastV参数
SYS_LENGTH=35
IMAGE_TOKEN_LENGTH=576

# 剪枝层（在第几层应用文本加权剪枝策略）
PRUNING_LAYER=3
export CUDA_VISIBLE_DEVICES=0
python /root/nfs/code/HiMAP/src/HiMAP/analysis/fastv_textweighted_experiment.py \
    --model-path $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --question-file $QUESTION_FILE \
    --conv-mode $CONV_MODE \
    --num-samples $NUM_SAMPLES \
    --fast-v-sys-length $SYS_LENGTH \
    --fast-v-image-token-length $IMAGE_TOKEN_LENGTH \
    --pruning-layer $PRUNING_LAYER \
    --single-pred-prompt

echo "Text-weighted pruning experiment completed!"
