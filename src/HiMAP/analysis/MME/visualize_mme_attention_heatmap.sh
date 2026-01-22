#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
# MME数据集注意力热力图可视化脚本
# 从每个子任务中选取样本，绘制文本-视觉注意力热力图

MODEL_PATH="liuhaotian/llava-v1.5-7b"
IMAGE_FOLDER="/root/nfs/code/HiMAP/data/MME/images/test"
QUESTION_FILE="/root/nfs/code/HiMAP/data/MME/MME_test.json"
OUTPUT_DIR="mme_attention_heatmap_visualization"
SAMPLES_PER_CATEGORY=1  # 每个类别选择的样本数
mkdir -p "$OUTPUT_DIR"

echo "======================================"
echo "MME Attention Heatmap Visualization"
echo "======================================"
echo "Model: $MODEL_PATH"
echo "Image folder: $IMAGE_FOLDER"
echo "Question file: $QUESTION_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Samples per category: $SAMPLES_PER_CATEGORY"
echo "======================================"

cd /root/nfs/code/HiMAP

python src/HiMAP/analysis/MME/visualize_mme_attention_heatmap.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --samples-per-category "$SAMPLES_PER_CATEGORY" \
    --conv-mode "vicuna_v1"

echo ""
echo "======================================"
echo "Visualization complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "======================================"
