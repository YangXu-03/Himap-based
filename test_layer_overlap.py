
import argparse
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

# Add src to path
import sys
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/LLaVA"))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

def load_mme_samples(mme_json_path, image_root_path):
    """
    Load MME data and select one sample per category (subtask).
    """
    with open(mme_json_path, 'r') as f:
        data = json.load(f)
    
    samples_by_category = {}
    for item in data:
        cat = item['category']
        if cat not in samples_by_category:
            samples_by_category[cat] = item
    
    print(f"Loaded {len(samples_by_category)} categories from MME.")
    return samples_by_category

def get_overlap_ratio(set_a, set_b):
    """
    Compute overlap ratio: size(intersection) / size(set_a)
    (Assuming sets are same size, so symmetric)
    """
    intersection = len(set_a.intersection(set_b))
    return intersection / len(set_a) if len(set_a) > 0 else 0

def visualize_heatmap(matrix, layers, title, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="YlGnBu", xticklabels=layers, yticklabels=layers, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    model_path = "/root/nfs/model/llava-v1.5-7b"
    mme_json_path = "/root/nfs/code/HiMAP/data/MME/MME_test.json"
    image_folder = "/root/nfs/code/HiMAP/data/MME/images/test"
    output_dir = "mme_attention_overlap_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )
    
    # Ensure FastV is disabled to get full attention maps
    if hasattr(model.config, "use_fast_v"):
        model.config.use_fast_v = False
        if hasattr(model, "model") and hasattr(model.model, "reset_fastv"):
            model.model.reset_fastv()
    
    # Parameters
    sys_length = 35 # Default for LLaVA-1.5 Vicuna
    image_token_len = 576
    
    # Get samples
    samples = load_mme_samples(mme_json_path, image_folder)
    
    # Iterate over samples
    for category, sample in tqdm(samples.items(), desc="Processing categories"):
        image_file = sample['image_file']
        question = sample['question']
        
        # Load image
        image_path = os.path.join(image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping...")
            continue
            
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
            
        # Prepare inputs
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
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
        
        # Collect top indices per layer
        top1_indices = {}
        top10_indices = {}
        
        # Number of tokens to select
        k_1pct = max(1, int(image_token_len * 0.01))
        k_10pct = max(1, int(image_token_len * 0.10))
        
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn: (batch, heads, seq_len, seq_len)
            avg_attn = layer_attn[0].mean(dim=0) # (seq_len, seq_len)
            
            # Attention of the last token (before generation)
            last_token_attn = avg_attn[-1, :] # (seq_len,)
            
            # Extract image part
            # Important: Adjust sys_length if needed. For standard LLaVA 1.5 prompt, image starts at 35.
            # We can double check if input_ids has -200 (IMAGE_TOKEN_INDEX) but here inputs are embeddings.
            # Assuming 35 is correct as per FastV config.
            
            if last_token_attn.shape[0] < sys_length + image_token_len:
                print(f"Warning: Sequence length {last_token_attn.shape[0]} too short for category {category}")
                break

            image_attn = last_token_attn[sys_length : sys_length + image_token_len]
            
            # Get indices descending
            sorted_indices = torch.argsort(image_attn, descending=True).cpu().numpy()
            
            top1_indices[layer_idx] = set(sorted_indices[:k_1pct])
            top10_indices[layer_idx] = set(sorted_indices[:k_10pct])

        # Compute overlap matrices
        matrix_1 = np.zeros((num_layers, num_layers))
        matrix_10 = np.zeros((num_layers, num_layers))
        
        for i in range(num_layers):
            for j in range(num_layers):
                if i in top1_indices and j in top1_indices:
                    matrix_1[i, j] = get_overlap_ratio(top1_indices[i], top1_indices[j])
                    matrix_10[i, j] = get_overlap_ratio(top10_indices[i], top10_indices[j])
        
        # Plot
        layer_ticks = list(range(num_layers))
        
        safe_cat = category.replace("/", "_").replace(" ", "_")
        visualize_heatmap(matrix_1, layer_ticks, 
                         f"Layer Overlap (Top 1%) - {category}", 
                         os.path.join(output_dir, f"{safe_cat}_top1_heatmap.png"))
        
        visualize_heatmap(matrix_10, layer_ticks, 
                         f"Layer Overlap (Top 10%) - {category}", 
                         os.path.join(output_dir, f"{safe_cat}_top10_heatmap.png"))
        
    print(f"Done. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
