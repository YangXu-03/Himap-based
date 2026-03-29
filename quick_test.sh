#!/bin/bash

# 快速测试不同策略 - 只运行2个样本

export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT=https://hf-mirror.com

MODEL_PATH="/root/nfs/model/llava-v1.5-7b"
QUESTION_FILE="/root/nfs/code/HiMAP/data/MME/MME_test.json"
IMAGE_FOLDER="/root/nfs/code/HiMAP/data/MME/images/test"
ATTENTION_RANK=128

echo "========================================"
echo "快速测试 - 只运行2个样本"
echo "========================================"

# 测试 max_head
echo ""
echo "1. 测试 max_head 策略..."
python ./src/HiMAP/inference/eval_mme_fastv_advanced.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --single-pred-prompt \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank $ATTENTION_RANK \
    --fast-v-agg-layer 12 \
    --fast-v-token-selection-method max_head \
    --num-samples 2 \
    --output-file test_max_head.json 2>&1 | grep -E "(DEBUG|token_selection|配置|验证)"

# 测试 avg_all_heads
echo ""
echo "2. 测试 avg_all_heads 策略..."
python ./src/HiMAP/inference/eval_mme_fastv_advanced.py \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --single-pred-prompt \
    --use-fast-v \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --fast-v-attention-rank $ATTENTION_RANK \
    --fast-v-agg-layer 12 \
    --fast-v-token-selection-method avg_all_heads \
    --num-samples 2 \
    --output-file test_avg_all_heads.json 2>&1 | grep -E "(DEBUG|token_selection|配置|验证)"

echo ""
echo "比较结果..."
python -c "
import json

with open('test_max_head.json') as f:
    d1 = json.load(f)
with open('test_avg_all_heads.json') as f:
    d2 = json.load(f)

pred1 = [x['pred'] for x in d1['predictions']]
pred2 = [x['pred'] for x in d2['predictions']]

print(f'max_head 预测: {pred1}')
print(f'avg_all_heads 预测: {pred2}')
print(f'结果相同: {pred1 == pred2}')
"
