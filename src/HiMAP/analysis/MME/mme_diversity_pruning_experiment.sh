export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/root/nfs/code/HiMAP/src 
python src/HiMAP/analysis/MME/mme_diversity_pruning_experiment.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder data/MME/images/test \
    --question-file data/MME/MME_test.json \
    --keep-tokens 8 \
    --num-samples 10 \
    --layer-start 1 \
    --layer-end 32 \
    --output-file mme_diversity_pruning_results.json \
    --output-plot mme_diversity_pruning_plot.png \
    --methods fps,tome \
    --metric cosine
