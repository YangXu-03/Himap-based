export CUDA_VISIBLE_DEVICES=2
PYTHONPATH=/root/nfs/code/HiMAP/src
# 使用Qwen 3B模型；如需LLaVA请改为：liuhaotian/llava-v1.5-7b           --model-path Qwen/Qwen2.5-3B-Instruct \
python ./src/HiMAP/analysis/saliency_scivqa.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./data/scienceqa/himap-inference-MCQ.json \
    --result-file ./output_example/scivqa_atten-7b.pt \
    --image-folder ./data/scienceqa/images/test \
    --single-pred-prompt \
    --conv-mode vicuna_v1