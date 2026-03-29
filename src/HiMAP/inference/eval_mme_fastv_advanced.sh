#!/bin/bash

# FastV Advanced MME 推理测试脚本
# 测试五种不同的 token 选择策略

export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT=https://hf-mirror.com

# 基础配置。/root/nfs/model/llava-v1.5-7b".  "liuhaotian/llava-v1.5-7b"
MODEL_PATH="/root/nfs/model/llava-v1.5-7b"
QUESTION_FILE="/root/nfs/code/HiMAP/data/MME/MME_test.json"
IMAGE_FOLDER="/root/nfs/code/HiMAP/data/MME/images/test"
ATTENTION_RANK=128

# 样本数量设置 (设置为空或注释掉以使用全部样本)
# NUM_SAMPLES=200  # 快速测试用 50 个样本
NUM_SAMPLES=""    # 使用全部样本

if [ -n "$NUM_SAMPLES" ]; then
    SAMPLE_ARG="--num-samples $NUM_SAMPLES"
    echo "注意: 将只测试前 $NUM_SAMPLES 个样本"
else
    SAMPLE_ARG=""
    echo "注意: 将测试全部样本"
fi

echo "========================================"
echo "FastV Advanced MME 推理测试"
echo "========================================"

# # 1. 基线测试（不使用任何剪枝）
# echo ""
# echo "1. 运行基线测试（无剪枝）..."
# python ./src/HiMAP/inference/eval_mme_fastv_advanced.py \
#     --model-path $MODEL_PATH \
#     --question-file $QUESTION_FILE \
#     --image-folder $IMAGE_FOLDER \
#     --single-pred-prompt \
#     $SAMPLE_ARG \
#     --output-file mme_results_baseline.json

# 2. FastV - max_head 策略
echo ""
echo "2. 运行 FastV Advanced - max_head 策略..."
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
    $SAMPLE_ARG \
    --output-file mme_results_fastv_max_head.json

# 3. FastV - avg_all_heads 策略（原始 FastV）
echo ""
echo "3. 运行 FastV Advanced - avg_all_heads 策略（原始FastV）..."
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
    $SAMPLE_ARG \
    --output-file mme_results_fastv_avg_all_heads.json

# 4. FastV - text_weighted 策略
echo ""
echo "4. 运行 FastV Advanced - text_weighted 策略..."
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
    --fast-v-token-selection-method text_weighted \
    $SAMPLE_ARG \
    --output-file mme_results_fastv_text_weighted.json

# 5. FastV - text_weighted_max_head 策略
echo ""
echo "5. 运行 FastV Advanced - text_weighted_max_head 策略..."
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
    --fast-v-token-selection-method text_weighted_max_head \
    $SAMPLE_ARG \
    --output-file mme_results_fastv_text_weighted_max_head.json

# 6. FastV - weighted_combination 策略 (alpha=0.7)
echo ""
echo "6. 运行 FastV Advanced - weighted_combination 策略 (alpha=0.7)..."
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
    --fast-v-token-selection-method weighted_combination \
    --fast-v-weighted-alpha 0.7 \
    $SAMPLE_ARG \
    --output-file mme_results_fastv_weighted_combination_alpha0.7.json

   #  6. FastV - weighted_combination 策略 (alpha=0.5)
echo ""
echo "6. 运行 FastV Advanced - weighted_combination 策略 (alpha=0.5)..."
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
    --fast-v-token-selection-method weighted_combination \
    --fast-v-weighted-alpha 0.5 \
    $SAMPLE_ARG \
    --output-file mme_results_fastv_weighted_combination_alpha0.5.json

    # 6. FastV - weighted_combination 策略 (alpha=0.3)
echo ""
echo "6. 运行 FastV Advanced - weighted_combination 策略 (alpha=0.3)..."
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
    --fast-v-token-selection-method weighted_combination \
    --fast-v-weighted-alpha 0.3 \
    $SAMPLE_ARG \
    --output-file mme_results_fastv_weighted_combination_alpha0.3.json

echo ""
echo "========================================"
echo "所有测试完成！"
echo "========================================"
echo ""
echo "结果文件："
echo "  - mme_results_baseline.json"
echo "  - mme_results_fastv_max_head.json"
echo "  - mme_results_fastv_avg_all_heads.json"
echo "  - mme_results_fastv_text_weighted.json"
echo "  - mme_results_fastv_text_weighted_max_head.json"
echo "  - mme_results_fastv_weighted_combination_alpha0.7.json"
echo ""
echo "运行可视化脚本来生成图表："
echo "  python ./src/HiMAP/inference/visualize_mme_fastv_results.py"
echo ""
