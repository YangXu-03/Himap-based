export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export PYTHONPATH=/root/nfs/code/HiMAP/src 

python ./src/HiMAP/analysis/MME/observe_mme.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./data/MME/MME_test.json \
    --image-folder ./data/MME/images/test \
    --output-dir ./output_example/mme_observation_results \
    --conv-mode vicuna_v1 \
    --num-samples 0
