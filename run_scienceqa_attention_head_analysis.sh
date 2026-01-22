#!/bin/bash

# Script to analyze attention heads on ScienceQA dataset

# Set paths
MODEL_PATH="liuhaotian/llava-v1.5-7b"
IMAGE_FOLDER="./data/scienceqa/images/test"
QUESTION_FILE="./data/scienceqa/himap-inference-MCQ.json"
OUTPUT_DIR="./scienceqa_attention_head_analysis"

mkdir -p "$OUTPUT_DIR"
# Set number of samples (use -1 for all samples)
NUM_SAMPLES=100  # Change to -1 to process all samples

# Run analysis
python -m src.HiMAP.analysis.ScienceQA.analyze_scienceqa_attention_heads \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --conv-mode "vicuna_v1"

echo "Analysis complete! Results saved to $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - attention_head_records.pt: Sample-level records"
echo "  - statistics.json: Overall statistics"
echo "  - max_attention_head_distribution.png: Per-layer histograms"
echo "  - max_attention_head_heatmap.png: Heatmap across all layers"
