
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
project_root = "/root/nfs/code/HiMAP"
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "src/LLaVA"))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

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
        # Select the first sample encountered for this category
        if cat not in samples_by_category:
            samples_by_category[cat] = item
    
    print(f"Selected {len(samples_by_category)} samples (one per category).")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU ID to use")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # Configuration
    model_path = "/root/nfs/model/llava-v1.5-7b"
    mme_json_path = os.path.join(project_root, "data/MME/MME_test.json")
    image_folder = os.path.join(project_root, "data/MME/images/test")
    output_dir = os.path.join(project_root, "mme_layer_overlap_results_text")
    
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
    # Standard LLaVA-1.5 uses 576 tokens.
    # System prompt length is typically 35.
    sys_length = 35 
    image_token_len = 576
    
    # Get one sample per subtask
    samples = load_mme_samples(mme_json_path)
    
    # Iterate over samples
    for category, sample in tqdm(samples.items(), desc="Processing categories"):
        image_file = sample.get('image_file')
        question = sample.get('question')
        
        if not image_file or not question:
            continue
            
        # Construct image path
        image_path = os.path.join(image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping {category}.")
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
        # We only need the forward pass to get attentions, we don't need to generate text generally.
        # But let's run a single forward pass.
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
        
        # Calculate K for 1% and 10%
        k_1pct = max(1, int(image_token_len * 0.30))
        k_10pct = max(1, int(image_token_len * 0.50))
        
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn: (batch=1, heads, seq_len, seq_len)
            
            seq_len = layer_attn.shape[-1]
            if seq_len < sys_length + image_token_len:
                print(f"Warning: Sequence length {seq_len} too short for category {category}. Skipping.")
                break

            # Calculate Text-to-Image Attention
            # Query: Text tokens (sys_length + image_token_len :)
            # Key: Image tokens (sys_length : sys_length + image_token_len)
            text_start = sys_length + image_token_len
            image_start = sys_length
            image_end = sys_length + image_token_len
            
            # (heads, text_len, image_len)
            text_to_image_attn = layer_attn[0, :, text_start:, image_start:image_end]
            
            # Sum over heads and text queries -> (image_len,)
            image_attn_scores = text_to_image_attn.sum(dim=0).sum(dim=0)
            
            # Get indices with highest attention scores using topk
            _, top1_indices_t = torch.topk(image_attn_scores, k_1pct)
            top1_indices[layer_idx] = set(top1_indices_t.cpu().numpy())
            
            _, top10_indices_t = torch.topk(image_attn_scores, k_10pct)
            top10_indices[layer_idx] = set(top10_indices_t.cpu().numpy())

        # Compute overlap matrices
        matrix_1 = np.zeros((num_layers, num_layers))
        matrix_10 = np.zeros((num_layers, num_layers))
        
        for i in range(num_layers):
            for j in range(num_layers):
                if i in top1_indices and j in top1_indices:
                     # Jaccard or simple Intersection/Size?
                     # Request says "Compare pairwise overlap". Let's use Intersection / Size (which is constant).
                     # This represents "What fraction of Layer I's top tokens are also in Layer J's top tokens?"
                    matrix_1[i, j] = get_overlap_ratio(top1_indices[i], top1_indices[j])
                    matrix_10[i, j] = get_overlap_ratio(top10_indices[i], top10_indices[j])
        
        # Generate clean filename from category
        safe_cat = category.replace("/", "_").replace(" ", "_")
        
        # Save heatmaps
        visualize_heatmap(matrix_1, list(range(num_layers)), 
                         f"{category} - Top 30% Token Overlap", 
                         os.path.join(output_dir, f"{safe_cat}_top1_heatmap.png"))
        
        visualize_heatmap(matrix_10, list(range(num_layers)), 
                         f"{category} - Top 50% Token Overlap", 
                         os.path.join(output_dir, f"{safe_cat}_top10_heatmap.png"))
        
    print(f"\nProcessing complete. All results saved to {output_dir}")

if __name__ == "__main__":
    main()
