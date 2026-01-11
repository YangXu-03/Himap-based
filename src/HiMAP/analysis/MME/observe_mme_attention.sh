export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/root/nfs/code/HiMAP/src 

python src/HiMAP/analysis/MME/observe_mme_attention.py \
  --model-path /root/nfs/model/llava-v1.5-7b \
  --image-folder /root/nfs/code/HiMAP/data/MME/images/test \
  --question-file /root/nfs/code/HiMAP/data/MME/MME_test.json \
  --output-dir mme_attention_observation \
  --conv-mode vicuna_v1