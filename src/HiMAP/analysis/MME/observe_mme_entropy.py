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
    row_sums = probs.sum(dim=-1, keepdim=True)
    if (row_sums == 0).any():
        print("Warning: row_sums has zeros")
    probs = probs / row_sums
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    
    if torch.isnan(entropy).any():
        print("Warning: entropy has NaNs")
        # print("probs min:", probs.min().item(), "max:", probs.max().item())
        # print("row_sums min:", row_sums.min().item(), "max:", row_sums.max().item())
        
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
        "text_vis_attn_entropy": [],
    }

    # 只计算跨模态注意力熵
    for i, attn in enumerate(attentions):
        attn = attn.squeeze(0) # (heads, seq, seq)
        
        if torch.isnan(attn).any():
            print(f"Layer {i}: attn has NaNs")
            
        attn_text_vis = attn[:, text_indices, :][:, :, vis_indices] # (heads, num_text, num_vis)
        
        if attn_text_vis.numel() == 0:
             print(f"Layer {i}: attn_text_vis is empty. text_indices len: {len(text_indices)}, vis_indices len: {len(vis_indices)}")
             metrics["layer_idx"].append(i)
             metrics["text_vis_attn_entropy"].append(0.0) # Or NaN
             continue

        entropy_text_vis = calculate_entropy(attn_text_vis)
        metrics["layer_idx"].append(i)
        metrics["text_vis_attn_entropy"].append(entropy_text_vis)
    return metrics

def plot_metrics(results_by_cat, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 只绘制跨模态注意力熵
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('tab20')
    for idx, (cat, cat_results) in enumerate(results_by_cat.items()):
        layers = cat_results[0]["layer_idx"]
        values = np.array([r["text_vis_attn_entropy"] for r in cat_results]) # (num_samples, num_layers)
        mean_values = values.mean(axis=0)
        plt.plot(layers, mean_values, label=cat, color=cmap(idx % 20))
    plt.title("Text-Visual Cross-Modal Attention Entropy")
    plt.xlabel("Layer")
    plt.ylabel("Entropy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_vis_attn_entropy.png"))
    plt.close()

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
