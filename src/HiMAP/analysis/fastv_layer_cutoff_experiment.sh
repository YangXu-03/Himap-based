#!/bin/bash

# FastV Layer Cutoff Experiment on ScienceQA
# 测试从第2层到最后一层完全剪除图像token的准确率
export CUDA_VISIBLE_DEVICES=2

MODEL_PATH="/root/nfs/model/llava-v1.5-7b"  #"liuhaotian/llava-v1.5-7b"
IMAGE_FOLDER="/root/nfs/code/HiMAP/data/scienceqa/images/test"
QUESTION_FILE="/root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json"
CONV_MODE="vicuna_v1"
NUM_SAMPLES=-1  # 使用100个样本进行快速测试，设置为-1使用全部样本

OUTPUT_FILE="fastv_layer_cutoff_results.json"
OUTPUT_PLOT="fastv_layer_cutoff_plot.png"

# 如果需要测试更大的数据集，可以修改 NUM_SAMPLES=-1
# 如果使用13B模型，修改 MODEL_PATH="liuhaotian/llava-v1.5-13b"

python ./src/HiMAP/analysis/fastv_layer_cutoff_experiment.py \
    --model-path ${MODEL_PATH} \
    --image-folder ${IMAGE_FOLDER} \
    --question-file ${QUESTION_FILE} \
    --conv-mode ${CONV_MODE} \
    --single-pred-prompt \
    --num-samples ${NUM_SAMPLES} \
    --fast-v-sys-length 35 \
    --fast-v-image-token-length 576 \
    --output-file ${OUTPUT_FILE} \
    --output-plot ${OUTPUT_PLOT}

echo "Experiment completed!"
echo "Results saved to: ${OUTPUT_FILE}"
echo "Plot saved to: ${OUTPUT_PLOT}"
