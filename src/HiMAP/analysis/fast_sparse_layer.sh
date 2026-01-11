export CUDA_VISIBLE_DEVICES=0

# ——————————————————————----fastv----——————————————————————————
# python src/HiMAP/analysis/fastv_sparse_layer_experiment.py \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --image-folder data/scienceqa/images/test \
#     --question-file data/scienceqa/himap-inference-MCQ.json \
#     --kept-tokens 8 \
#     --num-samples 50 \
#     --output-file fastv_sparse_8tokens_results.json \
#     --output-plot fastv_sparse_8tokens_plot.png

# ——————————————————————----custom selection fps----——————————————————————————
python src/HiMAP/analysis/fastv_custom_selection_experiment.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder data/scienceqa/images/test \
    --question-file data/scienceqa/himap-inference-MCQ.json \
    --kept-tokens 8 \
    --selection-method fps \
    --num-samples 50 \
    --output-file custom_fps_results.json \
    --output-plot custom_fps_plot.png


# ——————————————————————----custom selection tome----——————————————————————————
# python src/HiMAP/analysis/fastv_custom_selection_experiment.py \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --image-folder data/scienceqa/images/test \
#     --question-file data/scienceqa/himap-inference-MCQ.json \
#     --kept-tokens 8 \
#     --selection-method tome \
#     --num-samples 50 \
#     --output-file custom_tome_results.json \
#     --output-plot custom_tome_plot.png