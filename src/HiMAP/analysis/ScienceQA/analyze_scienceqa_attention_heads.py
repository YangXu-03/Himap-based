#!/usr/bin/env python3
"""
分析 LLaVA 模型在 ScienceQA 数据集上各注意力头的文本到图像token最大注意力值情况
1. 保存每个样本每一层文本到图像注意力最大的注意力头序号
2. 绘制所有样本每一层最大文本到图像注意力值头的序号分布直方图
"""

import argparse
import os
import json
import math
from typing import Dict, List, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def find_max_text_to_image_attention_heads(
    attentions: List[torch.Tensor], 
    image_token_start: int, 
    image_token_end: int,
    sys_length: int = 0
) -> List[int]:
    """
    找到每一层中文本到图像token注意力最大的注意力头序号
    
    Args:
        attentions: List of (batch=1, num_heads, seq_len, seq_len) tensors for each layer
        image_token_start: 图像token的起始位置
        image_token_end: 图像token的结束位置（不包含）
        sys_length: 系统提示的长度
        
    Returns:
        max_head_indices: List of integers, 每一层文本到图像注意力最大的头序号
    """
    max_head_indices = []
    
    for layer_idx, attn in enumerate(attentions):
        # attn shape: (batch=1, num_heads, seq_len, seq_len)
        attn = attn.squeeze(0)  # (num_heads, seq_len, seq_len)
        
        seq_len = attn.shape[-1]
        
        # 定义文本token的位置：系统提示 + 图像token之后的所有token
        # 排除图像token本身
        text_positions = []
        # 系统提示部分
        if sys_length > 0:
            text_positions.extend(range(0, min(sys_length, image_token_start)))
        # 图像token之后的文本
        if image_token_end < seq_len:
            text_positions.extend(range(image_token_end, seq_len))
        
        if len(text_positions) == 0:
            # 如果没有文本token，使用整体最大值
            max_values_per_head = attn.max(dim=-1)[0].max(dim=-1)[0]
            max_head_idx = max_values_per_head.argmax().item()
            max_head_indices.append(max_head_idx)
            continue
        
        # 图像token的位置
        image_positions = list(range(image_token_start, image_token_end))
        
        # 提取文本到图像的注意力: (num_heads, len(text_positions), len(image_positions))
        text_to_image_attn = attn[:, text_positions, :][:, :, image_positions]
        
        # 计算每个头的文本到图像注意力最大值
        max_values_per_head = text_to_image_attn.max(dim=-1)[0].max(dim=-1)[0]  # (num_heads,)
        
        # 找到最大值对应的头序号
        max_head_idx = max_values_per_head.argmax().item()
        max_head_indices.append(max_head_idx)
    
    return max_head_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=-1, help="Number of samples to use for testing (-1 for all)")
    parser.add_argument("--output-dir", type=str, default="./scienceqa_attention_head_analysis", help="Output directory for results")
    parser.add_argument("--sys-length", type=int, default=35, help="Length of system prompt tokens")
    parser.add_argument("--img-length", type=int, default=576, help="Length of image tokens")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Load questions
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.num_samples > 0:
        questions = questions[:args.num_samples]

    # Get number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    print(f"Model has {num_layers} layers and {num_heads} attention heads per layer")
    print(f"Processing {len(questions)} samples...")

    # Data structures to store results
    sample_records = []  # List of dicts, each containing sample_id and max_head_per_layer
    layer_head_counts = defaultdict(lambda: defaultdict(int))  # layer_idx -> head_idx -> count

    # Process each sample
    for i, line in enumerate(tqdm(questions, desc="Processing samples")):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()

        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if torch.cuda.is_available():
            images = image_tensor.unsqueeze(0).half().cuda()
        else:
            images = image_tensor.unsqueeze(0).float()

        # Prepare input
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        # Generate with attention output
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                max_new_tokens=1024,
                use_cache=False,
                stopping_criteria=stopping_criteria,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # Extract attentions from first forward pass (prefill stage)
        if hasattr(output_ids, 'attentions') and len(output_ids.attentions) > 0:
            # output_ids.attentions is a tuple of tuples
            # Each element corresponds to one generation step
            # We focus on the first step (prefill), which contains all input tokens
            first_step_attentions = output_ids.attentions[0]  # Tuple of tensors, one per layer
            
            # Determine image token positions
            # Image tokens typically start after system prompt (args.sys_length)
            image_token_start = args.sys_length
            image_token_end = args.sys_length + args.img_length
            
            # Find max text-to-image attention heads for this sample
            max_head_indices = find_max_text_to_image_attention_heads(
                first_step_attentions, 
                image_token_start, 
                image_token_end,
                args.sys_length
            )
            
            # Store sample record
            sample_record = {
                'sample_id': idx,
                'image_file': image_file,
                'max_head_per_layer': max_head_indices,
                'image_token_start': image_token_start,
                'image_token_end': image_token_end
            }
            sample_records.append(sample_record)
            
            # Update layer-head counts
            for layer_idx, head_idx in enumerate(max_head_indices):
                layer_head_counts[layer_idx][head_idx] += 1

    # Save sample-level records
    records_file = os.path.join(args.output_dir, 'attention_head_records.pt')
    torch.save(sample_records, records_file)
    print(f"\nSaved sample records to {records_file}")

    # Save statistics as JSON
    statistics = {
        'num_samples': len(sample_records),
        'num_layers': num_layers,
        'num_heads': num_heads,
        'sys_length': args.sys_length,
        'img_length': args.img_length,
        'analysis_type': 'text_to_image_attention',
        'layer_head_counts': {
            layer_idx: dict(head_counts) 
            for layer_idx, head_counts in layer_head_counts.items()
        }
    }
    
    stats_file = os.path.join(args.output_dir, 'statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"Saved statistics to {stats_file}")

    # Plot histograms for each layer
    print("\nGenerating histograms...")
    
    # Create a figure with subplots for each layer
    n_cols = 4
    n_rows = math.ceil(num_layers / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if num_layers > 1 else [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        # Get head counts for this layer
        head_counts = layer_head_counts.get(layer_idx, {})
        
        # Prepare data for histogram
        head_indices = list(range(num_heads))
        counts = [head_counts.get(head_idx, 0) for head_idx in head_indices]
        
        # Plot
        ax.bar(head_indices, counts, color='steelblue', alpha=0.7)
        ax.set_xlabel('Attention Head Index')
        ax.set_ylabel('Count')
        ax.set_title(f'Layer {layer_idx} (Text→Image)')
        ax.grid(True, alpha=0.3)
        
        # Highlight the most frequent head
        if counts:
            max_count = max(counts)
            max_head = counts.index(max_count)
            ax.bar(max_head, max_count, color='red', alpha=0.7, 
                   label=f'Most frequent: Head {max_head}')
            ax.legend()
    
    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    histogram_file = os.path.join(args.output_dir, 'max_attention_head_distribution.png')
    plt.savefig(histogram_file, dpi=150, bbox_inches='tight')
    print(f"Saved histogram to {histogram_file}")
    plt.close()

    # Create an overall summary plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Prepare data: layer x head matrix
    matrix = np.zeros((num_layers, num_heads))
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            matrix[layer_idx, head_idx] = layer_head_counts.get(layer_idx, {}).get(head_idx, 0)
    
    # Plot heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Attention Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title('Max Text→Image Attention Head Distribution Across Layers', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=12)
    
    plt.tight_layout()
    heatmap_file = os.path.join(args.output_dir, 'max_attention_head_heatmap.png')
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {heatmap_file}")
    plt.close()

    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
