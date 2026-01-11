export CUDA_VISIBLE_DEVICES=2
python src/HiMAP/analysis/fastv_rank_experiment.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
    --image-folder ./data/scienceqa/images/test \
    --single-pred-prompt \
    --num-samples 300