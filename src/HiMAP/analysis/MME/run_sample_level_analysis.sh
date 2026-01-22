#!/bin/bash

# 样本级注意力分析运行脚本
# 从MME每个子任务中提取前8个样本进行分析

# 设置工作目录
export CUDA_VISIBLE_DEVICES=2
cd /root/nfs/code/HiMAP

# 设置参数
MODEL_PATH="liuhaotian/llava-v1.5-7b"
IMAGE_FOLDER="/root/nfs/code/HiMAP/data/MME/images/test"
QUESTION_FILE="/root/nfs/code/HiMAP/data/MME/MME_test.json"
OUTPUT_DIR="mme_sample_level_attention"
SAMPLES_PER_CATEGORY=8

echo "======================================"
echo "样本级注意力分析实验"
echo "======================================"
echo "模型: ${MODEL_PATH}"
echo "图片目录: ${IMAGE_FOLDER}"
echo "问题文件: ${QUESTION_FILE}"
echo "每个类别样本数: ${SAMPLES_PER_CATEGORY}"
echo "输出目录: ${OUTPUT_DIR}"
echo "======================================"

# 运行分析脚本
python src/HiMAP/analysis/MME/sample_level_attention_analysis.py \
    --model-path "${MODEL_PATH}" \
    --image-folder "${IMAGE_FOLDER}" \
    --question-file "${QUESTION_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --samples-per-category ${SAMPLES_PER_CATEGORY} \
    --conv-mode vicuna_v1 \
    --plot-individual

echo ""
echo "======================================"
echo "分析完成！"
echo "结果保存在: ${OUTPUT_DIR}/"
echo "======================================"
echo ""
echo "生成的文件："
echo "  - sample_level_records.pt: 原始记录数据"
echo "  - sample_level_statistics.json: 统计数据"
echo "  - category_*_overview.png: 每个类别的概览图"
echo ""
echo "如需为每个样本生成独立图表，请添加 --plot-individual 参数"
