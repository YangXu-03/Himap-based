#!/bin/bash
# 测试版运行脚本 - 仅处理少量样本用于快速测试
# 用于验证代码逻辑和参数配置

echo "=========================================="
echo "LLaVA Attention Head Analysis - Test Run"
echo "=========================================="

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=2
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# 设置Python路径
export PYTHONPATH=/root/nfs/code/HiMAP/src
echo "PYTHONPATH: $PYTHONPATH"

# 模型配置
MODEL_PATH="/root/nfs/model/llava-v1.5-7b"
IMAGE_FOLDER="/root/nfs/code/HiMAP/data/MME/images/test"
QUESTION_FILE="/root/nfs/code/HiMAP/data/MME/MME_test.json"
OUTPUT_DIR="mme_attention_observation_test"
CONV_MODE="vicuna_v1"
# NUM_SAMPLES=50  # 仅处理50个样本用于测试

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Images: $IMAGE_FOLDER"
echo "  Questions: $QUESTION_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Mode: Test run (only $NUM_SAMPLES samples)"
echo ""

# 检查文件是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Error: Image folder does not exist: $IMAGE_FOLDER"
    exit 1
fi

if [ ! -f "$QUESTION_FILE" ]; then
    echo "Error: Question file does not exist: $QUESTION_FILE"
    exit 1
fi

# 记录开始时间
START_TIME=$(date +%s)
echo "Start time: $(date)"
echo ""

# 运行Python脚本
python src/HiMAP/analysis/observe_mme_attention_heads.py \
  --model-path "$MODEL_PATH" \
  --image-folder "$IMAGE_FOLDER" \
  --question-file "$QUESTION_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --conv-mode "$CONV_MODE" \
  --num-samples $NUM_SAMPLES

# 检查执行状态
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Test run completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Error: Test run failed with exit code $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "End time: $(date)"
echo "Total duration: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "View results:"
echo "  - Overall: $OUTPUT_DIR/mme_overall_stacked_hist.png"
echo "  - Per-layer: $OUTPUT_DIR/mme_layer*_stacked_hist.png"
echo "  - By category: $OUTPUT_DIR/category_*/"
echo ""
echo "If test results look good, run the full analysis with:"
echo "  bash src/HiMAP/analysis/run_mme_attention_full.sh"
echo ""
