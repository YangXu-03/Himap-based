# HiMAP
export CUDA_VISIBLE_DEVICES=0. #liuhaotian/llava-v1.5-7b  /root/nfs/model/llava-v1.5-7b
python ./src/HiMAP/inference/eval_mme.py \
    --model-path /root/nfs/model/llava-v1.5-7b \
    --question-file /root/nfs/code/HiMAP/data/MME/MME_test.json \
    --image-folder /root/nfs/code/HiMAP/data/MME/images/test \
    --use-hmap-v \
    --sys-length 35 \
    --img-length 576 \
    --hmap-v-attn-txt-layer 2 \
    --hmap-v-attn-img-layer 8 \
    --hmap-v-attn-txt-rank 288 \
    --hmap-v-attn-img-rank 72 \
   --cut-off-layer 0

# FastV
# export CUDA_VISIBLE_DEVICES=0
# python ./src/HiMAP/inference/eval_mme.py \
#     --model-path /root/nfs/model/llava-v1.5-7b \
#     --question-file /root/nfs/code/HiMAP/data/MME/MME_test.json \
#     --image-folder /root/nfs/code/HiMAP/data/MME/images/test \
#     --use-fast-v \
#     --fast-v-sys-length 35 \
#     --fast-v-image-token-length 576 \
#     --fast-v-attention-rank 288 \
#     --fast-v-agg-layer 2

# Baseline
# export CUDA_VISIBLE_DEVICES=0
# python ./src/HiMAP/inference/eval_mme.py \
#     --model-path /root/nfs/model/llava-v1.5-7b \
#     --question-file /root/nfs/code/HiMAP/data/MME/MME_test.json \
#     --image-folder /root/nfs/code/HiMAP/data/MME/images/test
