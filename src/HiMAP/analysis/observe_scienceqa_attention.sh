export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/root/nfs/code/HiMAP/src 

python src/HiMAP/analysis/observe_scienceqa_attention.py \
  --model-path /root/nfs/model/llava-v1.5-7b \
  --image-folder /root/nfs/code/HiMAP/data/scienceqa/images/test \
  --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
  --output-dir scienceqa_attention_observation \
  --conv-mode vicuna_v1