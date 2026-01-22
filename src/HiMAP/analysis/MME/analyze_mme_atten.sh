export CUDA_VISIBLE_DEVICES=0

python /root/nfs/code/HiMAP/src/HiMAP/analysis/MME/analyze_mme_attention_distribution.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder /root/nfs/code/HiMAP/data/MME/images/test \
    --question-file /root/nfs/code/HiMAP/data/MME/MME_test.json \
    --output-dir "mme_attention_distribution" \
    --num-samples 100