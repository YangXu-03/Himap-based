# # 使用HuggingFace模型（需要网络连接）. liuhaotian/llava-v1.5-7b.  /root/nfs/model/llava-v1.5-7b
export CUDA_VISIBLE_DEVICES=2
# python ./src/HiMAP/inference/eval_scivqa.py \
#     --model-path /root/nfs/model/llava-v1.5-7b \
#     --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
#     --image-folder ./data/scienceqa/images/test \
#     --single-pred-prompt \
#     --use-hmap-v \
#     --sys-length 35 \
#     --img-length 576 \
#     --num-samples -1 \
#     --hmap-v-attn-txt-layer 2 \
#     --hmap-v-attn-img-layer 8 \
#     --hmap-v-attn-txt-rank 288 \
#     --hmap-v-attn-img-rank 128 
# #     --cut-off-layer 32

# # fastv
# export CUDA_VISIBLE_DEVICES=2
# python ./src/HiMAP/inference/eval_scivqa.py \
#     --model-path /root/nfs/model/llava-v1.5-7b \
#     --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
#     --image-folder ./data/scienceqa/images/test \
#     --single-pred-prompt \
#     --use-fast-v \
#     --fast-v-sys-length 35 \
#     --fast-v-image-token-length 576 \
#     --fast-v-attention-rank 100 \
#     --fast-v-agg-layer 2 \
#     --num-samples -1

# # jsd+entropy adaptive 3-stage pruning
GPU_ID=${GPU_ID:-1}
JSD_MODE=${JSD_MODE:-prompt_image}  # prompt_image|global
JSD_TARGET_TOKENS=${JSD_TARGET_TOKENS:-128}
JSD_N0=${JSD_N0:-576}
JSD_PHASE1_LAYER=${JSD_PHASE1_LAYER:-3}
JSD_PHASE2_LAYER=${JSD_PHASE2_LAYER:-8}
JSD_PHASE3_LAYER=${JSD_PHASE3_LAYER:-16}
JSD_N_BASE_192=${JSD_N_BASE_192:-200}
JSD_N_BASE_128=${JSD_N_BASE_128:-140}
JSD_N_BASE_64=${JSD_N_BASE_64:-80}
JSD_MU_H=${JSD_MU_H:-0.620257}
JSD_SIGMA_H=${JSD_SIGMA_H:-0.030169}
JSD_MU_W=${JSD_MU_W:-0.667733}
JSD_SIGMA_W=${JSD_SIGMA_W:-0.038618}
JSD_USE_ONLY_PROMPT2IMAGE=${JSD_USE_ONLY_PROMPT2IMAGE:-True}
JSD_USE_ADAPTIVE_KEEP_RATIO=${JSD_USE_ADAPTIVE_KEEP_RATIO:-True}

case "${JSD_TARGET_TOKENS}" in
    192|128)
        JSD_ALPHA=${JSD_ALPHA:-24}
        JSD_BETA=${JSD_BETA:-16}
        ;;
    64)
        JSD_ALPHA=${JSD_ALPHA:-9}
        JSD_BETA=${JSD_BETA:-5}
        ;;
    *)
        JSD_ALPHA=${JSD_ALPHA:-0}
        JSD_BETA=${JSD_BETA:-0}
        ;;
esac

export CUDA_VISIBLE_DEVICES=${GPU_ID}
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path /root/nfs/model/llava-v1.5-7b \
    --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
    --image-folder ./data/scienceqa/images/test \
    --single-pred-prompt \
    --use-jsd-entropy-prune \
    --jsd-entropy-topk-attention-mode "${JSD_MODE}" \
    --jsd-entropy-sys-length 35 \
    --jsd-entropy-img-length 576 \
    --jsd-entropy-target-tokens "${JSD_TARGET_TOKENS}" \
    --jsd-entropy-n0 "${JSD_N0}" \
    --jsd-entropy-phase1-prune-layer "${JSD_PHASE1_LAYER}" \
    --jsd-entropy-phase2-prune-layer "${JSD_PHASE2_LAYER}" \
    --jsd-entropy-phase3-prune-layer "${JSD_PHASE3_LAYER}" \
    --jsd-entropy-n-base-192 "${JSD_N_BASE_192}" \
    --jsd-entropy-n-base-128 "${JSD_N_BASE_128}" \
    --jsd-entropy-n-base-64 "${JSD_N_BASE_64}" \
    --jsd-entropy-mu-h "${JSD_MU_H}" \
    --jsd-entropy-sigma-h "${JSD_SIGMA_H}" \
    --jsd-entropy-mu-w "${JSD_MU_W}" \
    --jsd-entropy-sigma-w "${JSD_SIGMA_W}" \
    --jsd-entropy-use-only-prompt2image-scoring "${JSD_USE_ONLY_PROMPT2IMAGE}" \
    --jsd-entropy-use-adaptive-keep-ratio "${JSD_USE_ADAPTIVE_KEEP_RATIO}" \
    --jsd-entropy-alpha "${JSD_ALPHA}" \
    --jsd-entropy-beta "${JSD_BETA}" \
    --jsd-entropy-topk-percent 10 \
    --num-samples -1

# # # 基线模型
# export CUDA_VISIBLE_DEVICES=2
# python ./src/HiMAP/inference/eval_scivqa.py \
#     --model-path /root/nfs/model/llava-v1.5-7b \
#     --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
#     --image-folder ./data/scienceqa/images/test \
#     --single-pred-prompt \
#     --num-samples -1



# 使用本地模型路径（如果模型已下载到本地）
# export CUDA_VISIBLE_DEVICES=0
# python ./src/HiMAP/inference/eval_scivqa.py \
#     --model-path /code/FasterV/models/llava-v1.5-7b \
#     --question-file ./data/scienceqa/himap-inference-MCQ.json \
#     --image-folder ./data/scienceqa/images/test \
#     --single-pred-prompt