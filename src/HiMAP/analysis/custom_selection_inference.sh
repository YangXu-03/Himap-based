
export CUDA_VISIBLE_DEVICES=2

python src/HiMAP/analysis/custom_selection_inference.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder data/scienceqa/images/test \
    --question-file data/scienceqa/himap-inference-MCQ.json \
    --baseline \
    --num-samples 100 \
    --output-file results_baseline.json


# 使用FPS算法
# python src/HiMAP/analysis/custom_selection_inference.py \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --image-folder data/scienceqa/images/test \
#     --question-file data/scienceqa/himap-inference-MCQ.json \
#     --num-samples 100 \
#     --custom-agg-layer 32 \
#     --custom-kept-tokens 576 \
#     --custom-selection-method fps \
#     --output-file results_fps.json

# 使用ToMe算法
# python src/HiMAP/analysis/custom_selection_inference.py \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --image-folder data/scienceqa/images/test \
#     --question-file data/scienceqa/himap-inference-MCQ.json \
#     --custom-kept-tokens 8 \
#     --custom-selection-method tome \
#     --output-file results_tome.json