export CUDA_VISIBLE_DEVICES=2
python src/HiMAP/analysis/MME/mme_cutoff_experiment.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder data/MME/images/test \
    --question-file data/MME/MME_test.json \
    --output-file mme_layer_cutoff_results.json \
    --output-plot mme_layer_cutoff_plot.png 
    # --num-samples 10


