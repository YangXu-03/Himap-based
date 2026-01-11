#!/bin/bash

# Visual Token Pruning Experiment Script
# This script runs the visual token pruning experiment on ScienceQA test set

# Default paths - modify these according to your setup
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
SCIENCEQA_DIR="${SCIENCEQA_DIR:-./data/scienceqa}"
OUTPUT_DIR="${OUTPUT_DIR:-./pruning_results}"
CONV_MODE="${CONV_MODE:-llava_v1}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

# Check if required paths exist
if [ ! -d "$SCIENCEQA_DIR" ]; then
    echo "Warning: ScienceQA directory not found at $SCIENCEQA_DIR"
    echo "Please set SCIENCEQA_DIR environment variable to the correct path"
fi

echo "Running Visual Token Pruning Experiment"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "ScienceQA Dir: $SCIENCEQA_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Conv Mode: $CONV_MODE"
echo "Num Samples: $NUM_SAMPLES"
echo ""

python /root/nfs/code/HiMAP/src/LLaVA/scripts/pruning_experiment.py \
    --model-path "$MODEL_PATH" \
    --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
    --image-folder ./data/scienceqa/images/test \
    --base-dir "$SCIENCEQA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --conv-mode "$CONV_MODE" \
    --single-pred-prompt \
    --num-samples "$NUM_SAMPLES"

echo ""
echo "Experiment completed. Results saved to $OUTPUT_DIR"
