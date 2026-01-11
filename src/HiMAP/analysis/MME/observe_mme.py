import argparse
import torch
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

def calculate_entropy(attn_matrix):
    # attn_matrix: (num_heads, seq_len, seq_len)
    # We want entropy of rows.
    # Add epsilon to avoid log(0)
    epsilon = 1e-10
    probs = attn_matrix + epsilon
    # Normalize to ensure it sums to 1 (it should already, but if we take submatrix it won't)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return entropy.mean().item()

def calculate_rank(attn_matrix):
    # attn_matrix: (num_heads, seq_len, seq_len)
    # Calculate rank for each head and average
    ranks = []
    for h in range(attn_matrix.shape[0]):
        try:
            # matrix_rank requires float32 or complex
            r = torch.linalg.matrix_rank(attn_matrix[h].float())
            ranks.append(r.item())
        except Exception as e:
            print(f"Error calculating rank: {e}")
            ranks.append(0)
    return np.mean(ranks)

def process_sample(model, tokenizer, image_processor, line, args):
    qs = line['question']
    image_file = line["image_file"]
    
    # Prepare Image
    image_path = os.path.join(args.image_folder, image_file)
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    images = image_tensor.unsqueeze(0).half().cuda()

    # Prepare Prompt
    if getattr(model.config, 'mm_use_im_start_end', False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    qs = qs + '\n' + "Answer the question using a single word or phrase."

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # Find image token index in input_ids
    # This helps us locate visual tokens in the expanded sequence
    image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
    # Assuming one image
    if len(image_token_indices) > 0:
        img_start_idx_input = image_token_indices[0].item()
    else:
        # Should not happen for multimodal inputs
        return None

    with torch.no_grad():
        outputs = model(
            input_ids,
            images=images,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states # Tuple of (batch, seq_len, hidden_dim), len = num_layers + 1
    attentions = outputs.attentions # Tuple of (batch, num_heads, seq_len, seq_len), len = num_layers

    # Determine indices in expanded sequence
    # The model expands IMAGE_TOKEN_INDEX into num_patches tokens.
    # We need to know num_patches.
    # Usually it's 576 for 336px images (24*24).
    # We can infer it from the sequence length difference.
    seq_len_input = input_ids.shape[1]
    seq_len_output = hidden_states[0].shape[1]
    num_patches = seq_len_output - seq_len_input + 1 # +1 because 1 token is replaced by N tokens
    
    img_start = img_start_idx_input
    img_end = img_start + num_patches
    
    # Indices
    vis_indices = list(range(img_start, img_end))
    text_indices = list(range(0, img_start)) + list(range(img_end, seq_len_output))
    
    metrics = {
        "layer_idx": [],
        "vis_sim": [],
        "text_sim": [],
        "vis_attn_entropy": [],
        "vis_attn_rank": [],
        "text_to_vis_ratio": [],
        "text_to_vis_slope": [] # Will calculate later or here? 
        # Slope is derivative w.r.t layer. We can just store the ratio and compute slope in post-processing.
    }

    # 1. Cosine Similarity
    # hidden_states[0] is embeddings. hidden_states[1] is output of layer 0.
    # We want similarity between layer i and layer i-1.
    # hidden_states has L+1 elements.
    # Layer 0 input is hidden_states[0]. Layer 0 output is hidden_states[1].
    # We compare output of layer i (hidden_states[i+1]) with input of layer i (hidden_states[i]).
    # Or output of layer i vs output of layer i-1?
    # Usually "layer i and previous layer" means H_i vs H_{i-1}.
    # Let's use H_{i} vs H_{i-1} where H_0 is embeddings.
    
    for i in range(1, len(hidden_states)):
        h_curr = hidden_states[i].squeeze(0) # (seq_len, dim)
        h_prev = hidden_states[i-1].squeeze(0)
        
        # Visual Sim
        sim_vis = F.cosine_similarity(h_curr[vis_indices], h_prev[vis_indices], dim=-1).mean().item()
        
        # Text Sim
        sim_text = F.cosine_similarity(h_curr[text_indices], h_prev[text_indices], dim=-1).mean().item()
        
        metrics["vis_sim"].append(sim_vis)
        metrics["text_sim"].append(sim_text)
        metrics["layer_idx"].append(i-1) # 0-indexed layer

    # 2. Attention Metrics
    # attentions has L elements. attentions[i] is for layer i.
    for i, attn in enumerate(attentions):
        # attn: (batch, heads, seq, seq)
        attn = attn.squeeze(0) # (heads, seq, seq)
        
        # Visual Attention Matrix (Visual -> Visual)
        # Submatrix where query is visual and key is visual
        attn_vis = attn[:, vis_indices, :][:, :, vis_indices] # (heads, num_vis, num_vis)
        
        entropy = calculate_entropy(attn_vis)
        rank = calculate_rank(attn_vis)
        
        metrics["vis_attn_entropy"].append(entropy)
        metrics["vis_attn_rank"].append(rank)
        
        # Text to Visual Attention
        # Query: Text, Key: Visual
        attn_text_vis = attn[:, text_indices, :][:, :, vis_indices] # (heads, num_text, num_vis)
        # Sum over visual keys to get total attention paid to image
        total_attn_to_vis = attn_text_vis.sum(dim=-1) # (heads, num_text)
        # Average over heads and text tokens
        ratio = total_attn_to_vis.mean().item()
        
        metrics["text_to_vis_ratio"].append(ratio)

    # Calculate slope for text_to_vis_ratio
    # Slope at layer i = Ratio[i] - Ratio[i-1] (for i > 0)
    # For i=0, maybe 0 or Ratio[0]
    ratios = metrics["text_to_vis_ratio"]
    slopes = [0] * len(ratios)
    for i in range(1, len(ratios)):
        slopes[i] = ratios[i] - ratios[i-1]
    metrics["text_to_vis_slope"] = slopes

    return metrics

def plot_metrics(results_by_cat, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to plot
    metric_names = ["vis_sim", "text_sim", "vis_attn_entropy", "vis_attn_rank", "text_to_vis_ratio", "text_to_vis_slope"]
    titles = {
        "vis_sim": "Visual Token Cosine Similarity (Layer vs Prev)",
        "text_sim": "Text Token Cosine Similarity (Layer vs Prev)",
        "vis_attn_entropy": "Visual Attention Entropy",
        "vis_attn_rank": "Visual Attention Rank",
        "text_to_vis_ratio": "Text-to-Visual Attention Ratio",
        "text_to_vis_slope": "Text-to-Visual Attention Slope"
    }
    
    # Aggregate all results first to get global average if needed, or just plot per category
    # We will plot one figure per metric, with lines for each category (or selected categories)
    
    # Define colors/styles
    cmap = plt.get_cmap('tab20')
    
    for m_name in metric_names:
        plt.figure(figsize=(12, 8))
        
        for idx, (cat, cat_results) in enumerate(results_by_cat.items()):
            # cat_results is a list of metrics dicts
            # Average over samples
            layers = cat_results[0]["layer_idx"]
            values = np.array([r[m_name] for r in cat_results]) # (num_samples, num_layers)
            mean_values = values.mean(axis=0)
            
            plt.plot(layers, mean_values, label=cat, color=cmap(idx % 20))
        
        plt.title(titles[m_name])
        plt.xlabel("Layer")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{m_name}.png"))
        plt.close()

    # Also plot grouped by sub-task (Perception vs Cognition) if possible
    # But MME has many categories.
    
    # Special request: "按子任务分类绘图" (Plot categorized by sub-task)
    # The above loop does exactly that (one line per category).

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="observation_results")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-samples", type=int, default=-1)
    args = parser.parse_args()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # 关闭hmap_v并重置相关参数，防止None报错
    model.config.use_hmap_v = False
    if hasattr(model.model, 'reset_hmapv'):
        model.model.reset_hmapv()
    # 强制兜底，防止None
    if not hasattr(model.model, 'hmap_v_sys_length') or model.model.hmap_v_sys_length is None:
        model.model.hmap_v_sys_length = 0
    if not hasattr(model.model, 'hmap_v_img_length') or model.model.hmap_v_img_length is None:
        model.model.hmap_v_img_length = 0
    
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    if args.num_samples > 0:
        questions = questions[:args.num_samples]

    results_by_cat = {}

    for line in tqdm(questions):
        cat = line['category']
        metrics = process_sample(model, tokenizer, image_processor, line, args)
        
        if metrics is None:
            continue
            
        if cat not in results_by_cat:
            results_by_cat[cat] = []
        results_by_cat[cat].append(metrics)

    # Save raw results
    torch.save(results_by_cat, os.path.join(args.output_dir, "observation_results.pt"))
    
    # Plot
    plot_metrics(results_by_cat, args.output_dir)

if __name__ == "__main__":
    main()
