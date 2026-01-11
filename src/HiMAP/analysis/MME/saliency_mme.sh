export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export PYTHONPATH=/root/nfs/code/HiMAP/src 

# PYTHONPATH=/root/nfs/code/HiMAP/src
python ./src/HiMAP/analysis/MME/saliency_mme.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./data/MME/MME_test.json \
    --image-folder ./data/MME/images/test \
    --result-file ./output_example/mme_saliency_results.pt \
    --output-dir ./output_example/mme_saliency_plots \
    --conv-mode vicuna_v1
