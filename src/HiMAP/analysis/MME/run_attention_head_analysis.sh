#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# 分析 LLaVA 模型在 MME 数据集上的注意力头交互情况

# 设置默认参数
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/root/nfs/code/HiMAP/data/MME/images/test}"
QUESTION_FILE="${QUESTION_FILE:-/root/nfs/code/HiMAP/data/MME/MME_test.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/nfs/code/HiMAP/mme_attention_head_analysis}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"

echo "=========================================="
echo "MME Attention Head Analysis"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Image Folder: $IMAGE_FOLDER"
echo "Question File: $QUESTION_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "Conversation Mode: $CONV_MODE"
echo "Number of Samples: $NUM_SAMPLES"
echo "=========================================="
echo

# 运行分析脚本
python /root/nfs/code/HiMAP/src/HiMAP/analysis/MME/analyze_mme_attention_heads.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --conv-mode "$CONV_MODE" \
    --num-samples "$NUM_SAMPLES"

echo
echo "=========================================="
echo "Analysis completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
