export CUDA_VISIBLE_DEVICES=0
python /root/nfs/code/HiMAP/src/HiMAP/analysis/himap_cutoff_sweep.py \
    --dataset mme \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /root/nfs/code/HiMAP/data/MME/MME_test.json \
    --image-folder /root/nfs/code/HiMAP/data/MME/images/test \
    --use-hmap-v \
    --sys-length 35 \
    --img-length 576 \
    --hmap-v-attn-txt-layer 2 \
    --hmap-v-attn-img-layer 8 \
    --hmap-v-attn-txt-rank 288 \
    --hmap-v-attn-img-rank 72 \
    --cutoff-start 9 \
    --cutoff-end 32 \
    --output-dir /root/nfs/code/HiMAP/output_example \
    --tag llava-7B-mme
