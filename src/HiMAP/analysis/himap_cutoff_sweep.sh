export CUDA_VISIBLE_DEVICES=2 
python /root/nfs/code/HiMAP/src/HiMAP/analysis/himap_cutoff_sweep.py \
    --dataset scienceqa \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json \
    --image-folder /root/nfs/code/HiMAP/data/scienceqa/images/test \
    --use-hmap-v \
    --sys-length 35 \
    --img-length 576 \
    --hmap-v-attn-txt-layer 2 \
    --hmap-v-attn-img-layer 8 \
    --hmap-v-attn-txt-rank 288 \
    --hmap-v-attn-img-rank 72 \
    --single-pred-prompt \
    --cutoff-start 9 \
    --cutoff-end 32 \
    --output-dir /root/nfs/code/HiMAP/output_example \
    --tag llava-7B