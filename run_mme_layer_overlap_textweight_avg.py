
import argparse
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add src to path
import sys
project_root = "/root/nfs/code/HiMAP"
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "src/LLaVA"))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
from layer_overlap_utils import (
    get_text_weighted_scores,
    get_topk_index_set,
    resolve_attention_spans,
)

def load_mme_samples(mme_json_path):
    """
    Load MME data and select one sample per category (subtask).
    Returns a dictionary: {category: sample_item}
    """
    print(f"Loading MME data from {mme_json_path}...")
    try:
        with open(mme_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: MME file not found at {mme_json_path}")
        return {}
    
    samples_by_category = {}
    for item in data:
        cat = item.get('category', 'unknown')
        # Modified to collect ALL samples for averaging
        if cat not in samples_by_category:
            samples_by_category[cat] = []
        samples_by_category[cat].append(item)
    
    print(f"Selected {len(samples_by_category)} categories from {len(data)} samples.")
    return samples_by_category

def get_overlap_ratio(set_a, set_b):
    """
    Compute overlap ratio: size(intersection) / size(set_a)
    Assumes size(set_a) == size(set_b) roughly, or normalized by set_a size.
    """
    if len(set_a) == 0:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    return intersection / len(set_a)

def visualize_heatmap(matrix, labels, title, output_path):
    """
    Generate and save a heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    
    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap to {output_path}")

def format_category_desc(category, current_idx, total_count):
    """
    Format tqdm description for per-category sample progress.
    """
    display_name = str(category)
    if len(display_name) > 30:
        display_name = display_name[:27] + "..."
    return f"[{current_idx}/{total_count}] {display_name}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, default="2", help="GPU ID to use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Configuration
    model_path = "/root/nfs/model/llava-v1.5-7b"
    mme_json_path = os.path.join(project_root, "data/MME/MME_test.json")
    image_folder = os.path.join(project_root, "data/MME/images/test")
    output_dir = os.path.join(project_root, "mme_layer_overlap_results_textweight")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading LLaVA model...")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Disable FastV during inference to get full attention maps (we want to analyze unmodified behavior)
    if hasattr(model.config, "use_fast_v"):
        print("Disabling FastV to capture full attention maps...")
        model.config.use_fast_v = False
        if hasattr(model, "model") and hasattr(model.model, "reset_fastv"):
            model.model.reset_fastv()
    
    # Parameters for image token extraction
    image_token_len = 576
    fallback_sys_length = getattr(model.config, "fast_v_sys_length", 36)
    
    # Get grouped samples
    samples_by_category = load_mme_samples(mme_json_path)

    # Iterate over categories
    total_categories = len(samples_by_category)
    for category_idx, (category, samples_list) in enumerate(
        tqdm(samples_by_category.items(), desc="Processing categories", total=total_categories),
        start=1,
    ):
        # print(f"Processing category: {category} ({len(samples_list)} samples)")
        
        accum_matrix_1 = None
        accum_matrix_10 = None
        count_samples = 0
        
        sample_iterator = tqdm(
            samples_list,
            desc=format_category_desc(category, category_idx, total_categories),
            leave=False,
            position=1,
        )

        for sample in sample_iterator:
            image_file = sample.get('image_file')
            question = sample.get('question')
            
            if not image_file or not question:
                continue
                
            # Construct image path
            image_path = os.path.join(image_folder, image_file)
            if not os.path.exists(image_path):
                # print(f"Warning: Image {image_path} not found. Skipping {category}.")
                continue
                
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
                
            # Prepare inputs
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            # Prepare prompt
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv = conv_templates['vicuna_v1'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            # Run inference
            with torch.inference_mode():
                outputs = model(
                    input_ids,
                    images=image_tensor,
                    output_attentions=True,
                    return_dict=True
                )
                
            attentions = outputs.attentions # Tuple of (batch, head, seq, seq)
            num_layers = len(attentions)
            
            if accum_matrix_1 is None:
                accum_matrix_1 = np.zeros((num_layers, num_layers))
                accum_matrix_10 = np.zeros((num_layers, num_layers))
            
            # Collect top indices per layer
            top1_indices = {}
            top10_indices = {}
            
            # Calculate K for 1% and 10%
            k_1pct = max(1, int(image_token_len * 0.01))
            k_10pct = max(1, int(image_token_len * 0.10))
            
            valid_sample = True
            for layer_idx, layer_attn in enumerate(attentions):
                # layer_attn: (batch=1, heads, seq_len, seq_len)
                seq_len = layer_attn.shape[-1]
                image_start, image_end, prompt_start, prompt_end = resolve_attention_spans(
                    input_ids,
                    seq_len,
                    image_token_len,
                    fallback_sys_length=fallback_sys_length,
                )

                if image_end - image_start < image_token_len:
                    # print(f"Warning: Sequence length {seq_len} too short for category {category}. Skipping.")
                    valid_sample = False
                    break

                image_attn_scores = get_text_weighted_scores(
                    layer_attn,
                    image_start,
                    image_end,
                    prompt_start,
                    prompt_end,
                )
                
                # Get indices with highest attention scores using topk
                top1_indices[layer_idx] = get_topk_index_set(image_attn_scores, k_1pct)
                top10_indices[layer_idx] = get_topk_index_set(image_attn_scores, k_10pct)
            
            if not valid_sample:
                continue

            matrix_1 = np.zeros((num_layers, num_layers))
            matrix_10 = np.zeros((num_layers, num_layers))
            
            for i in range(num_layers):
                for j in range(num_layers):
                    if i in top1_indices and j in top1_indices:
                        matrix_1[i, j] = get_overlap_ratio(top1_indices[i], top1_indices[j])
                        matrix_10[i, j] = get_overlap_ratio(top10_indices[i], top10_indices[j])
            
            accum_matrix_1 += matrix_1
            accum_matrix_10 += matrix_10
            count_samples += 1

        if count_samples > 0:
            avg_matrix_1 = accum_matrix_1 / count_samples
            avg_matrix_10 = accum_matrix_10 / count_samples
        
            # Generate clean filename from category
            safe_cat = category.replace("/", "_").replace(" ", "_")
            
            # Save heatmaps
            visualize_heatmap(avg_matrix_1, list(range(num_layers)), 
                            f"{category} - Top 1% Token Overlap (Avg)", 
                            os.path.join(output_dir, f"{safe_cat}_top1_heatmap.png"))
            
            visualize_heatmap(avg_matrix_10, list(range(num_layers)), 
                            f"{category} - Top 10% Token Overlap (Avg)", 
                            os.path.join(output_dir, f"{safe_cat}_top10_heatmap.png"))
        
    print(f"\nProcessing complete. All results saved to {output_dir}")

if __name__ == "__main__":
    main()
