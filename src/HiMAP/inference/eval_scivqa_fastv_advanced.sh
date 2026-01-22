#!/bin/bash

# FastV Advanced 推理测试脚本
# 测试三种不同的 token 选择策略

export CUDA_VISIBLE_DEVICES=2

# 基础配置
MODEL_PATH="liuhaotian/llava-v1.5-7b"
QUESTION_FILE="/root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json"
IMAGE_FOLDER="./data/scienceqa/images/test"
NUM_SAMPLES=100

echo "========================================"
echo "FastV Advanced 推理测试"
echo "========================================"

# 1. 基线测试（不使用任何剪枝）
echo ""
echo "1. 运行基线测试（无剪枝）..."
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --single-pred-prompt \
    --num-samples $NUM_SAMPLES

# 2. FastV - max_head 策略
echo ""
echo "2. 运行 FastV Advanced - max_head 策略..."
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --single-pred-prompt \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank 288 \
    --fast-v-agg-layer 2 \
    --fast-v-token-selection-method max_head \
    --num-samples $NUM_SAMPLES

# 3. FastV - avg_all_heads 策略（原始 FastV）
echo ""
echo "3. 运行 FastV Advanced - avg_all_heads 策略（原始FastV）..."
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --single-pred-prompt \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank 288 \
    --fast-v-agg-layer 2 \
    --fast-v-token-selection-method avg_all_heads \
    --num-samples $NUM_SAMPLES

# 4. FastV - weighted_combination 策略 (alpha=0.3)
echo ""
echo "4. 运行 FastV Advanced - weighted_combination 策略 (alpha=0.3)..."
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --single-pred-prompt \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank 288 \
    --fast-v-agg-layer 2 \
    --fast-v-token-selection-method weighted_combination \
    --fast-v-weighted-alpha 0.3 \
    --num-samples $NUM_SAMPLES

# 5. FastV - weighted_combination 策略 (alpha=0.5)
echo ""
echo "5. 运行 FastV Advanced - weighted_combination 策略 (alpha=0.5)..."
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --single-pred-prompt \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank 288 \
    --fast-v-agg-layer 2 \
    --fast-v-token-selection-method weighted_combination \
    --fast-v-weighted-alpha 0.5 \
    --num-samples $NUM_SAMPLES

# 6. FastV - weighted_combination 策略 (alpha=0.7)
echo ""
echo "6. 运行 FastV Advanced - weighted_combination 策略 (alpha=0.7)..."
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --single-pred-prompt \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank 288 \
    --fast-v-agg-layer 2 \
    --fast-v-token-selection-method weighted_combination \
    --fast-v-weighted-alpha 0.7 \
    --num-samples $NUM_SAMPLES

echo ""
echo "========================================"
echo "所有测试完成！"
echo "========================================"
echo ""
echo "结果文件："
echo "  - scienceqa_results_baseline.json"
echo "  - scienceqa_results_fastv_max_head.json"
echo "  - scienceqa_results_fastv_avg_all_heads.json"
echo "  - scienceqa_results_fastv_weighted_combination_alpha0.3.json"
echo "  - scienceqa_results_fastv_weighted_combination_alpha0.5.json"
echo "  - scienceqa_results_fastv_weighted_combination_alpha0.7.json"
echo ""
