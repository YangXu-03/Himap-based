# 使用FPS算法
export CUDA_VISIBLE_DEVICES=0


python src/HiMAP/analysis/custom_selection_inference.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder data/scienceqa/images/test \
    --question-file data/scienceqa/himap-inference-MCQ.json \
    --custom-kept-tokens 8 \
    --custom-selection-method fps \
    --output-file results_fps.json

# 使用ToMe算法
# python src/HiMAP/analysis/custom_selection_inference.py \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --image-folder data/scienceqa/images/test \
#     --question-file data/scienceqa/himap-inference-MCQ.json \
#     --custom-kept-tokens 8 \
#     --custom-selection-method tome \
#     --output-file results_tome.json