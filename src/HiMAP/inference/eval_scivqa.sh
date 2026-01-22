# # 使用HuggingFace模型（需要网络连接）. liuhaotian/llava-v1.5-7b.  /root/nfs/model/llava-v1.5-7b
export CUDA_VISIBLE_DEVICES=2
# python ./src/HiMAP/inference/eval_scivqa.py \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
#     --image-folder ./data/scienceqa/images/test \
#     --single-pred-prompt \
#     --use-hmap-v \
#     --sys-length 35 \
#     --img-length 576 \
#     --num-samples 100 \
#     --hmap-v-attn-txt-layer 2 \
#     --hmap-v-attn-img-layer 8 \
#     --hmap-v-attn-txt-rank 288 \
#     --hmap-v-attn-img-rank 72 \
#     --cut-off-layer 32

# # fastv
# export CUDA_VISIBLE_DEVICES=2
# python ./src/HiMAP/inference/eval_scivqa.py \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
#     --image-folder ./data/scienceqa/images/test \
#     --single-pred-prompt \
#     --use-fast-v \
#     --fast-v-sys-length 35 \
#     --fast-v-image-token-length 576 \
#     --fast-v-attention-rank 288 \
#     --fast-v-agg-layer 1

# # # 基线模型
export CUDA_VISIBLE_DEVICES=2
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
    --image-folder ./data/scienceqa/images/test \
    --single-pred-prompt \
    --num-samples 100



# 使用本地模型路径（如果模型已下载到本地）
# export CUDA_VISIBLE_DEVICES=0
# python ./src/HiMAP/inference/eval_scivqa.py \
#     --model-path /code/FasterV/models/llava-v1.5-7b \
#     --question-file ./data/scienceqa/himap-inference-MCQ.json \
#     --image-folder ./data/scienceqa/images/test \
#     --single-pred-prompt