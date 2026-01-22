"""
LLaVA 模型在 MME 数据集上的注意力头可视化分析脚本

功能描述：
    观测 LLaVA 多模态大模型在 MME 数据集上推理时，各个注意力头（Attention Heads）
    的"视觉-文本"交互情况，并生成堆叠柱状图进行可视化分析。

核心计算逻辑：
    1. Text-to-Visual (T2V): 
       - Query为文本token，Key为图像token
       - 提取注意力子矩阵 attn[text_idx, vis_idx]
       - 先沿Key(图像)维度求和，得到每个文本token对所有图像的总关注度（范围0~1）
       - 再沿Query(文本)维度取平均，得到该头的T2V值
    
    2. Visual-to-Text (V2T):
       - Query为图像token，Key为文本token
       - 提取注意力子矩阵 attn[vis_idx, text_idx]
       - 先沿Key(文本)维度求和，得到每个图像token对所有文本的总关注度（范围0~1）
       - 再沿Query(图像)维度取平均，得到该头的V2T值

可视化输出：
    1. 总体图 (Overall Plot): 所有样本所有层所有头的统计分布
    2. 分层图 (Per-layer Plots): 每一层的注意力头分布
    3. 分类图 (Category Plots): 按MME数据集类别分别统计
    
    堆叠柱状图特征：
    - X轴: 注意力值 (0.0~1.0, bins步长0.05)
    - Y轴: 占所有样本所有头总数的比例
    - 颜色: 蓝色(T2V) 和 橙色(V2T) 堆叠显示

使用方法：
    python observe_mme_attention_heads.py \\
        --model-path liuhaotian/llava-v1.5-7b \\
        --image-folder /path/to/MME/images \\
        --question-file /path/to/MME_test.json \\
        --output-dir mme_attention_observation \\
        --num-samples 100

作者: AI Assistant
日期: 2026-01-08
"""

import argparse
import os
import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

# 视觉-文本交互类型
INTERACTION_TYPES = [
    "text_to_visual",
    "visual_to_text",
    "total",  # 新增：两者总和
]

TYPE_LABELS = {
    "text_to_visual": "Text→Visual",
    "visual_to_text": "Visual→Text",
    "total": "Total (T2V + V2T)",
}

COLORS = {
    "text_to_visual": "#1f77b4",
    "visual_to_text": "#ff7f0e",
    "total": "#2ca02c",
}

def compute_token_groups(conv, prompt: str, tokenizer, input_ids: torch.Tensor, attentions: List[torch.Tensor]):
    image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
    if len(image_token_indices) == 0:
        return [], [], []
    img_start_idx_input = image_token_indices[0].item()
    seq_len_output = attentions[0].shape[-1]
    num_patches = seq_len_output - input_ids.shape[1] + 1
    vis_indices = list(range(img_start_idx_input, img_start_idx_input + num_patches))
    system_tokens = tokenizer(conv.system + conv.sep, add_special_tokens=False).input_ids
    system_start = 1 if input_ids[0, 0].item() == tokenizer.bos_token_id else 0
    system_end = min(system_start + len(system_tokens), seq_len_output)
    system_indices = list(range(system_start, system_end))
    text_indices = [i for i in range(seq_len_output) if i not in vis_indices and i not in system_indices]
    return system_indices, text_indices, vis_indices

def process_sample(model, tokenizer, image_processor, line: Dict, args, device: torch.device):
    qs = line["question"]
    image_file = line["image_file"]
    image_path = os.path.join(args.image_folder, image_file)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None
    
    try:
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        images = image_tensor.unsqueeze(0)
        if device.type == "cuda":
            images = images.half()
        images = images.to(device)
        
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        qs = qs + "\n" + "Answer the question using a single word or phrase."
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids,
                images=images,
                output_attentions=True,
                use_cache=True,
                return_dict=True,
            )
        
        attentions = outputs.attentions
        if attentions is None or len(attentions) == 0:
            return None
        
        system_idx, text_idx, vis_idx = compute_token_groups(conv, prompt, tokenizer, input_ids, attentions)
        if len(text_idx) == 0 or len(vis_idx) == 0:
            return None
        
        # 统计每层每个固定头的视觉-文本交互
        # 返回格式: {layer: {type: {head_idx: value}}}
        per_layer = []
        for attn in attentions:
            attn = attn.squeeze(0)  # (heads, seq, seq)
            n_heads = attn.shape[0]
            head_stats = {t: {} for t in INTERACTION_TYPES}
            
            for h in range(n_heads):
                # text_to_visual
                t2v_matrix = attn[h][text_idx][:, vis_idx]
                t2v_sum_over_key = t2v_matrix.sum(dim=1)
                t2v = t2v_sum_over_key.mean().item()
                
                # visual_to_text
                v2t_matrix = attn[h][vis_idx][:, text_idx]
                v2t_sum_over_key = v2t_matrix.sum(dim=1)
                v2t = v2t_sum_over_key.mean().item()
                
                # total
                total = t2v + v2t
                
                head_stats["text_to_visual"][h] = t2v
                head_stats["visual_to_text"][h] = v2t
                head_stats["total"][h] = total
            
            per_layer.append(head_stats)
        
        return {
            "category": line.get("category", "unknown"),
            "per_layer": per_layer,  # list of dict: {type: {head_idx: value}}
        }
    except Exception as e:
        print(f"Error processing sample {image_file}: {e}")
        return None

def collect_all_head_values(records: List[Dict]):
    """
    收集所有样本在32个固定头上的注意力值
    返回: {layer: {type: {head_idx: [所有样本在该头的值]}}}
    """
    if not records:
        return {}, {}, 0.0
    
    n_layers = len(records[0]["per_layer"])
    n_heads = len(records[0]["per_layer"][0]["text_to_visual"])  # 通常是32
    
    # 初始化: all_layers[layer][type][head_idx] = []
    all_layers = {l: {t: {h: [] for h in range(n_heads)} for t in INTERACTION_TYPES} for l in range(n_layers)}
    cat_layers = {}
    
    # 用于计算全局最大平均值
    global_max = 0.0
    
    for rec in records:
        cat = rec["category"]
        if cat not in cat_layers:
            cat_layers[cat] = {l: {t: {h: [] for h in range(n_heads)} for t in INTERACTION_TYPES} for l in range(n_layers)}
        
        for l, layer_stat in enumerate(rec["per_layer"]):
            for t in INTERACTION_TYPES:
                for h in range(n_heads):
                    val = layer_stat[t][h]
                    all_layers[l][t][h].append(val)
                    cat_layers[cat][l][t][h].append(val)
                    # 更新全局最大值
                    global_max = max(global_max, val)
    
    return all_layers, cat_layers, global_max

def plot_stacked_hist_per_layer(all_layers, output_dir, prefix, global_y_max):
    """
    绘制每一层的柱状图，X轴为32个注意力头，Y轴为该头在所有样本上的平均注意力值
    为每种交互类型（T2V, V2T, Total）绘制一张包含所有层的大图
    
    Args:
        global_y_max: 全局最大y轴值，用于统一所有图的y轴范围
    """
    os.makedirs(output_dir, exist_ok=True)
    n_layers = len(all_layers)
    n_heads = len(all_layers[0]["text_to_visual"])
    
    y_max = global_y_max * 1.1 if global_y_max > 0 else 0.1
    
    # 为每种交互类型绘制一张大图
    for interaction_type in ["text_to_visual", "visual_to_text", "total"]:
        n_cols = 4
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        if n_layers == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for l in range(n_layers):
            # 计算每个头的平均值
            head_means = []
            for h in range(n_heads):
                values = all_layers[l][interaction_type][h]
                if len(values) > 0:
                    head_means.append(np.mean(values))
                else:
                    head_means.append(0.0)
            
            # 绘制柱状图
            head_indices = np.arange(n_heads)
            axes[l].bar(head_indices, head_means, color=COLORS[interaction_type], 
                       alpha=0.8, edgecolor='black', linewidth=0.5)
            
            axes[l].set_xlabel("Attention Head Index", fontsize=10)
            axes[l].set_ylabel("Average Attention Value", fontsize=10)
            axes[l].set_title(f"Layer {l}", fontsize=11)
            axes[l].set_ylim(0, y_max)
            axes[l].grid(axis='y', alpha=0.3)
            axes[l].set_xticks(np.arange(0, n_heads, 4))
        
        # 隐藏多余的子图
        for l in range(n_layers, len(axes)):
            axes[l].axis('off')
        
        fig.suptitle(f"{TYPE_LABELS[interaction_type]} - All Layers (32 Heads)", fontsize=16, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        output_path = os.path.join(output_dir, f"{prefix}_all_layers_{interaction_type}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {interaction_type} all-layers plot: {output_path}")

def plot_stacked_hist_overall(all_layers, output_dir, prefix, num_samples, global_y_max):
    """
    绘制总体柱状图，X轴为32个注意力头，Y轴为该头在所有样本所有层上的平均注意力值
    
    Args:
        global_y_max: 全局最大y轴值，用于统一所有图的y轴范围
    """
    n_heads = len(all_layers[0]["text_to_visual"])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    y_max = global_y_max * 1.1 if global_y_max > 0 else 0.1
    
    for idx, interaction_type in enumerate(["text_to_visual", "visual_to_text", "total"]):
        # 对每个头，收集所有层所有样本的值
        head_all_values = {h: [] for h in range(n_heads)}
        for l in all_layers:
            for h in range(n_heads):
                head_all_values[h].extend(all_layers[l][interaction_type][h])
        
        # 计算每个头的平均值
        head_means = [np.mean(head_all_values[h]) if len(head_all_values[h]) > 0 else 0.0 
                      for h in range(n_heads)]
        
        # 绘制柱状图
        head_indices = np.arange(n_heads)
        axes[idx].bar(head_indices, head_means, color=COLORS[interaction_type], 
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        axes[idx].set_xlabel("Attention Head Index", fontsize=12)
        axes[idx].set_ylabel("Average Attention Value", fontsize=12)
        axes[idx].set_title(f"{TYPE_LABELS[interaction_type]}\n({num_samples} samples)", fontsize=13)
        axes[idx].set_ylim(0, y_max)
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_xticks(np.arange(0, n_heads, 4))
    
    plt.suptitle("Overall Visual-Textual Attention Distribution (32 Heads)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_overall_three_types.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overall histogram: {prefix}_overall_three_types.png")

def plot_stacked_hist_by_category(cat_layers, output_dir, prefix, global_y_max):
    """
    为每个类别绘制柱状图，X轴为32个注意力头
    
    Args:
        global_y_max: 全局最大y轴值，用于统一所有图的y轴范围
    """
    y_max = global_y_max * 1.1 if global_y_max > 0 else 0.1
    
    for cat, layers in cat_layers.items():
        if len(layers) > 0:
            n_heads = len(layers[0]["text_to_visual"])
            cat_output_dir = os.path.join(output_dir, f"category_{cat}")
            os.makedirs(cat_output_dir, exist_ok=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            for idx, interaction_type in enumerate(["text_to_visual", "visual_to_text", "total"]):
                # 对每个头，收集该类别所有层所有样本的值
                head_all_values = {h: [] for h in range(n_heads)}
                for l in layers:
                    for h in range(n_heads):
                        head_all_values[h].extend(layers[l][interaction_type][h])
                
                # 计算每个头的平均值
                head_means = [np.mean(head_all_values[h]) if len(head_all_values[h]) > 0 else 0.0 
                              for h in range(n_heads)]
                
                # 绘制柱状图
                head_indices = np.arange(n_heads)
                axes[idx].bar(head_indices, head_means, color=COLORS[interaction_type], 
                             alpha=0.8, edgecolor='black', linewidth=0.5)
                
                axes[idx].set_xlabel("Attention Head Index", fontsize=12)
                axes[idx].set_ylabel("Average Attention Value", fontsize=12)
                axes[idx].set_title(f"{TYPE_LABELS[interaction_type]}", fontsize=13)
                axes[idx].set_ylim(0, y_max)
                axes[idx].grid(axis='y', alpha=0.3)
                axes[idx].set_xticks(np.arange(0, n_heads, 4))
            
            plt.suptitle(f"Category: {cat} (32 Heads)", fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(cat_output_dir, f"{prefix}_{cat}_three_types.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved category histogram: {cat}")

def main():
    parser = argparse.ArgumentParser(description="Observe attention heads in LLaVA on MME dataset")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="Path to the LLaVA model")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path (for LoRA models)")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to the folder containing images")
    parser.add_argument("--question-file", type=str, required=True,
                        help="Path to the JSON file containing questions")
    parser.add_argument("--output-dir", type=str, default="mme_attention_observation",
                        help="Directory to save results and plots")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1",
                        help="Conversation mode for the model")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Number of samples to process (-1 for all)")
    args = parser.parse_args()
    
    # 初始化模型
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    print(f"Loading model from {model_path}...")
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载问题
    print(f"Loading questions from {args.question_file}...")
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    if args.num_samples > 0:
        questions = questions[: args.num_samples]
    print(f"Processing {len(questions)} samples...")
    
    # 处理样本
    records = []
    for line in tqdm(questions, desc="Observing attention heads"):
        rec = process_sample(model, tokenizer, image_processor, line, args, device)
        if rec is not None:
            records.append(rec)
    
    print(f"\nSuccessfully processed {len(records)} samples")
    
    # 保存原始记录
    records_path = os.path.join(args.output_dir, "mme_attention_head_records.pt")
    torch.save(records, records_path)
    print(f"Saved raw attention records to {records_path}")
    
    # 收集所有头的值
    all_layers, cat_layers, global_y_max = collect_all_head_values(records)
    
    print(f"\nGlobal max attention value: {global_y_max:.4f}")
    print(f"Y-axis range for all plots: [0, {global_y_max * 1.1:.4f}]")
    
    # 绘制图表
    print("\nGenerating plots...")
    plot_stacked_hist_overall(all_layers, args.output_dir, "mme", len(records), global_y_max)
    plot_stacked_hist_per_layer(all_layers, args.output_dir, "mme", global_y_max)
    plot_stacked_hist_by_category(cat_layers, args.output_dir, "mme", global_y_max)
    
    print(f"\n{'='*50}")
    print(f"All results saved to: {args.output_dir}")
    print(f"Total samples processed: {len(records)}")
    print(f"Total categories: {len(cat_layers)}")
    print(f"Categories: {', '.join(sorted(cat_layers.keys()))}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
