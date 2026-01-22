#!/usr/bin/env python3
"""
分析 LLaVA 模型在 MME 数据集上各注意力头的视觉-文本交互情况
绘制 Text-to-Visual 和 Visual-to-Text 的堆叠直方图
"""

import argparse
import os
import json
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
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path


def compute_t2v_score(attn: torch.Tensor, text_idx: List[int], vis_idx: List[int]) -> torch.Tensor:
    """
    计算 Text-to-Visual 注意力分数
    
    Args:
        attn: (num_heads, seq_len, seq_len) 注意力矩阵
        text_idx: 文本 token 的索引列表
        vis_idx: 视觉 token 的索引列表
        
    Returns:
        (num_heads,) 每个头的 T2V 分数
    """
    if len(text_idx) == 0 or len(vis_idx) == 0:
        return torch.zeros(attn.size(0))
    
    # 提取 Query 为文本、Key 为图像的子矩阵
    # sub: (num_heads, len(text_idx), len(vis_idx))
    sub = attn[:, text_idx, :][:, :, vis_idx]
    
    # 沿 Key (视觉) 维度求和：每个文本 token 对图像的总关注度
    # weights: (num_heads, len(text_idx))
    weights = sub.sum(dim=-1)
    
    # 沿 Query (文本) 维度取平均
    # scores: (num_heads,)
    scores = weights.mean(dim=-1)
    
    return scores


def compute_v2t_score(attn: torch.Tensor, vis_idx: List[int], text_idx: List[int]) -> torch.Tensor:
    """
    计算 Visual-to-Text 注意力分数
    
    Args:
        attn: (num_heads, seq_len, seq_len) 注意力矩阵
        vis_idx: 视觉 token 的索引列表
        text_idx: 文本 token 的索引列表
        
    Returns:
        (num_heads,) 每个头的 V2T 分数
    """
    if len(vis_idx) == 0 or len(text_idx) == 0:
        return torch.zeros(attn.size(0))
    
    # 提取 Query 为图像、Key 为文本的子矩阵
    # sub: (num_heads, len(vis_idx), len(text_idx))
    sub = attn[:, vis_idx, :][:, :, text_idx]
    
    # 沿 Key (文本) 维度求和：每个视觉 token 对文本的总关注度
    # weights: (num_heads, len(vis_idx))
    weights = sub.sum(dim=-1)
    
    # 沿 Query (视觉) 维度取平均
    # scores: (num_heads,)
    scores = weights.mean(dim=-1)
    
    return scores


def compute_token_groups(conv, prompt: str, tokenizer, input_ids: torch.Tensor, attentions: List[torch.Tensor]) -> Tuple[List[int], List[int], List[int]]:
    """
    计算系统、文本和视觉 token 的索引
    
    Returns:
        system_indices, text_indices, vis_indices
    """
    image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
    if len(image_token_indices) == 0:
        return [], [], []
    
    img_start_idx_input = image_token_indices[0].item()
    seq_len_output = attentions[0].shape[-1]
    num_patches = seq_len_output - input_ids.shape[1] + 1
    vis_indices = list(range(img_start_idx_input, img_start_idx_input + num_patches))

    # 系统 tokens: 前缀系统提示词 + 第一个分隔符
    system_tokens = tokenizer(conv.system + conv.sep, add_special_tokens=False).input_ids
    system_start = 1 if input_ids[0, 0].item() == tokenizer.bos_token_id else 0
    system_end = min(system_start + len(system_tokens), seq_len_output)
    system_indices = list(range(system_start, system_end))

    # 文本 tokens: 除系统和视觉之外的所有 tokens
    text_indices = [i for i in range(seq_len_output) if i not in vis_indices and i not in system_indices]
    
    return system_indices, text_indices, vis_indices


def process_sample(model, tokenizer, image_processor, line: Dict, args, device: torch.device) -> Dict:
    """
    处理单个样本，计算所有层所有头的 T2V 和 V2T 分数
    
    Returns:
        {
            "category": str,
            "t2v_scores": List[List[float]],  # [layer][head]
            "v2t_scores": List[List[float]],  # [layer][head]
        }
        或 None（如果处理失败）
    """
    qs = line["question"]
    image_file = line["image_file"]

    # 加载图片
    image_path = os.path.join(args.image_folder, image_file)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # 预处理图片
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    images = image_tensor.unsqueeze(0)
    if device.type == "cuda":
        images = images.half()
    images = images.to(device)

    # 准备带图片占位符的提示词
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

    # 前向推理，获取注意力权重
    with torch.no_grad():
        outputs = model(
            input_ids,
            images=images,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )

    attentions = outputs.attentions  # List[(batch, heads, seq, seq)]
    if attentions is None or len(attentions) == 0:
        return None

    # 计算 token 分组
    system_idx, text_idx, vis_idx = compute_token_groups(conv, prompt, tokenizer, input_ids, attentions)
    if len(text_idx) == 0 or len(vis_idx) == 0:
        return None

    # 计算每层每个头的 T2V 和 V2T 分数
    t2v_scores = []
    v2t_scores = []
    
    for attn in attentions:
        attn = attn.squeeze(0)  # (num_heads, seq, seq)
        
        # 计算 T2V 和 V2T
        t2v = compute_t2v_score(attn, text_idx, vis_idx)
        v2t = compute_v2t_score(attn, vis_idx, text_idx)
        
        t2v_scores.append(t2v.cpu().tolist())
        v2t_scores.append(v2t.cpu().tolist())

    return {
        "category": line["category"],
        "t2v_scores": t2v_scores,
        "v2t_scores": v2t_scores,
    }


def collect_all_scores(records: List[Dict], score_type: str = "t2v") -> np.ndarray:
    """
    收集所有样本所有层所有头的分数
    
    Args:
        records: 样本记录列表
        score_type: "t2v" 或 "v2t"
        
    Returns:
        所有分数的 1D numpy 数组
    """
    all_scores = []
    key = "t2v_scores" if score_type == "t2v" else "v2t_scores"
    
    for rec in records:
        for layer_scores in rec[key]:
            all_scores.extend(layer_scores)
    
    return np.array(all_scores)


def collect_scores_by_category(records: List[Dict], score_type: str = "t2v") -> Dict[str, np.ndarray]:
    """
    按类别收集所有层所有头的分数
    
    Returns:
        {category: np.ndarray} 字典
    """
    category_scores = defaultdict(list)
    key = "t2v_scores" if score_type == "t2v" else "v2t_scores"
    
    for rec in records:
        cat = rec["category"]
        for layer_scores in rec[key]:
            category_scores[cat].extend(layer_scores)
    
    return {cat: np.array(scores) for cat, scores in category_scores.items()}


def collect_scores_by_layer(records: List[Dict], score_type: str = "t2v") -> List[np.ndarray]:
    """
    按层收集所有样本的所有头的分数
    
    Args:
        records: 样本记录列表
        score_type: "t2v" 或 "v2t"
        
    Returns:
        List[np.ndarray]: 每个元素是一层的所有头分数
    """
    if not records:
        return []
    
    key = "t2v_scores" if score_type == "t2v" else "v2t_scores"
    num_layers = len(records[0][key])
    
    layer_scores = [[] for _ in range(num_layers)]
    
    for rec in records:
        for layer_idx, head_scores in enumerate(rec[key]):
            layer_scores[layer_idx].extend(head_scores)
    
    return [np.array(scores) for scores in layer_scores]


def collect_scores_by_category_and_layer(records: List[Dict], score_type: str = "t2v") -> Dict[str, List[np.ndarray]]:
    """
    按类别和层收集分数
    
    Args:
        records: 样本记录列表
        score_type: "t2v" 或 "v2t"
        
    Returns:
        {category: List[np.ndarray]} 字典，每个类别对应一个列表，列表中每个元素是一层的所有头分数
    """
    if not records:
        return {}
    
    key = "t2v_scores" if score_type == "t2v" else "v2t_scores"
    num_layers = len(records[0][key])
    
    # 按类别分组
    category_layer_scores = defaultdict(lambda: [[] for _ in range(num_layers)])
    
    for rec in records:
        cat = rec["category"]
        for layer_idx, head_scores in enumerate(rec[key]):
            category_layer_scores[cat][layer_idx].extend(head_scores)
    
    # 转换为 numpy 数组
    return {cat: [np.array(scores) for scores in layer_list] 
            for cat, layer_list in category_layer_scores.items()}


def plot_histogram(scores: np.ndarray, title: str, output_path: str, bins=50):
    """
    绘制堆叠直方图
    
    Args:
        scores: 分数数组
        title: 图表标题
        output_path: 输出路径
        bins: 直方图的 bins 数量
    """
    # 检查空数组
    if len(scores) == 0:
        print(f"Warning: Empty scores array for {output_path}, skipping plot.")
        return
    
    # 创建 bins: 从 0.0 到 1.0，步长 0.02
    bin_edges = np.arange(0.0, 1.02, 0.02)
    
    # 计算直方图
    counts, _ = np.histogram(scores, bins=bin_edges)
    
    # 计算比例 (占总数的比例)
    total_heads = len(scores)
    proportions = counts / total_heads
    
    # 绘图
    plt.figure(figsize=(12, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, proportions, width=0.018, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    plt.xlabel("Attention Score", fontsize=12)
    plt.ylabel("Proportion of Total Heads", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(0.0, 1.0)
    
    # 设置y轴范围，避免相同值
    max_prop = np.max(proportions) if len(proportions) > 0 else 0
    y_max = max(max_prop * 1.1, 0.01)  # 至少0.01避免相同的上下限
    plt.ylim(0, y_max)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    stats_text = f"Mean: {mean_score:.4f}\nMedian: {median_score:.4f}\nStd: {std_score:.4f}\nTotal Heads: {total_heads}"
    plt.text(0.98, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved plot to {output_path}")


def plot_category_histograms(category_scores: Dict[str, np.ndarray], score_type: str, output_dir: str):
    """
    为每个类别绘制直方图
    
    Args:
        category_scores: {category: scores} 字典
        score_type: "t2v" 或 "v2t"
        output_dir: 输出目录
    """
    categories = sorted(category_scores.keys())
    n_cats = len(categories)
    
    if n_cats == 0:
        return
    
    # 计算网格布局
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()
    
    bin_edges = np.arange(0.0, 1.02, 0.02)
    
    # 首先计算所有类别的最大proportion，以统一y轴范围
    global_max_proportion = 0
    for cat in categories:
        scores = category_scores[cat]
        if len(scores) > 0:
            counts, _ = np.histogram(scores, bins=bin_edges)
            proportions = counts / len(scores)
            global_max_proportion = max(global_max_proportion, np.max(proportions))
    
    global_y_max = global_max_proportion * 1.1 if global_max_proportion > 0 else 0.1
    
    for idx, cat in enumerate(categories):
        scores = category_scores[cat]
        ax = axes[idx]
        
        # 计算直方图
        counts, _ = np.histogram(scores, bins=bin_edges)
        total_heads = len(scores)
        proportions = counts / total_heads if total_heads > 0 else counts
        
        # 绘图
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, proportions, width=0.018, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_title(f"{cat} (n={total_heads})", fontsize=10, fontweight='bold')
        ax.set_xlabel("Attention Score", fontsize=9)
        ax.set_ylabel("Proportion", fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0, global_y_max)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加均值线
        mean_score = np.mean(scores)
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_score:.3f}')
        ax.legend(fontsize=8, loc='upper right')
    
    # 隐藏未使用的子图
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    
    score_label = "Text-to-Visual" if score_type == "t2v" else "Visual-to-Text"
    fig.suptitle(f"{score_label} Attention Distribution by Category", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{score_type}_by_category.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved category plots to {output_path}")


def plot_layer_histograms(layer_scores: List[np.ndarray], score_type: str, output_dir: str):
    """
    为每一层绘制直方图子图，合并为一张大图
    
    Args:
        layer_scores: List[np.ndarray]，每个元素是一层的所有头分数
        score_type: "t2v" 或 "v2t"
        output_dir: 输出目录
    """
    num_layers = len(layer_scores)
    if num_layers == 0:
        return
    
    # 计算网格布局 (8行4列可以容纳32层)
    n_cols = 4
    n_rows = (num_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()
    
    bin_edges = np.arange(0.0, 1.02, 0.02)
    
    # 计算全局 y 轴范围以保持一致性
    max_proportion = 0
    for scores in layer_scores:
        if len(scores) > 0:
            counts, _ = np.histogram(scores, bins=bin_edges)
            proportions = counts / len(scores)
            max_proportion = max(max_proportion, np.max(proportions))
    
    for layer_idx, scores in enumerate(layer_scores):
        ax = axes[layer_idx]
        
        # 计算直方图
        counts, _ = np.histogram(scores, bins=bin_edges)
        total_heads = len(scores)
        proportions = counts / total_heads if total_heads > 0 else counts
        
        # 绘图
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, proportions, width=0.018, alpha=0.7, edgecolor='black', linewidth=0.3)
        
        # 计算统计量
        if len(scores) > 0:
            mean_score = np.mean(scores)
            median_score = np.median(scores)
            std_score = np.std(scores)
            
            # 添加均值线
            ax.axvline(mean_score, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
            
            # 添加中值线
            ax.axvline(median_score, color='green', linestyle='-.', linewidth=1.2, alpha=0.7)
            
            # 添加统计信息文本
            stats_text = f'μ={mean_score:.3f}\nmed={median_score:.3f}\nσ={std_score:.3f}'
            ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        ax.set_title(f"Layer {layer_idx + 1} (n={total_heads})", fontsize=9, fontweight='bold')
        ax.set_xlabel("Score", fontsize=8)
        ax.set_ylabel("Proportion", fontsize=8)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0, max_proportion * 1.1)
        ax.grid(True, alpha=0.2, axis='y')
        ax.tick_params(labelsize=7)
    
    # 隐藏未使用的子图
    for j in range(num_layers, len(axes)):
        axes[j].axis("off")
    
    score_label = "Text-to-Visual" if score_type == "t2v" else "Visual-to-Text"
    fig.suptitle(f"{score_label} Attention Distribution Across Layers", 
                 fontsize=18, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    
    output_path = os.path.join(output_dir, f"{score_type}_by_layer.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved layer plots to {output_path}")


def save_category_layer_statistics(category_layer_scores: Dict[str, List[np.ndarray]], score_type: str, output_dir: str):
    """
    计算并保存每个类别每一层的详细统计数据
    
    Args:
        category_layer_scores: {category: List[np.ndarray]} 字典
        score_type: "t2v" 或 "v2t"
        output_dir: 输出目录
    """
    if not category_layer_scores:
        return
    
    # 创建统计数据字典
    all_stats = {}
    
    for cat in sorted(category_layer_scores.keys()):
        layer_scores = category_layer_scores[cat]
        cat_stats = {}
        
        for layer_idx, scores in enumerate(layer_scores):
            if len(scores) == 0:
                layer_stat = {
                    "layer": layer_idx + 1,
                    "num_heads": 0,
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "variance": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "q10": 0.0,
                    "q20": 0.0,
                }
            else:
                layer_stat = {
                    "layer": layer_idx + 1,
                    "num_heads": int(len(scores)),
                    "mean": float(np.mean(scores)),
                    "median": float(np.median(scores)),
                    "std": float(np.std(scores)),
                    "variance": float(np.var(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "q10": float(np.percentile(scores, 10)),
                    "q20": float(np.percentile(scores, 20)),
                }
            
            cat_stats[f"layer_{layer_idx + 1}"] = layer_stat
        
        all_stats[cat] = cat_stats
    
    # 保存为 JSON 文件
    stats_path = os.path.join(output_dir, f"{score_type}_category_layer_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved {score_type} category-layer statistics to {stats_path}")


def plot_category_layer_histograms(category_layer_scores: Dict[str, List[np.ndarray]], score_type: str, output_dir: str):
    """
    为每个类别绘制32层的直方图子图，每个类别生成一张独立的大图
    
    Args:
        category_layer_scores: {category: List[np.ndarray]} 字典
        score_type: "t2v" 或 "v2t"
        output_dir: 输出目录
    """
    if not category_layer_scores:
        return
    
    # 创建类别子目录
    category_dir = os.path.join(output_dir, f"category_layers_{score_type}")
    os.makedirs(category_dir, exist_ok=True)
    
    bin_edges = np.arange(0.0, 1.02, 0.02)
    score_label = "Text-to-Visual" if score_type == "t2v" else "Visual-to-Text"
    
    # 首先计算所有类别所有层的全局最大 proportion，以统一所有图的 y 轴范围
    global_max_proportion = 0
    for cat, layer_scores in category_layer_scores.items():
        for scores in layer_scores:
            if len(scores) > 0:
                counts, _ = np.histogram(scores, bins=bin_edges)
                proportions = counts / len(scores)
                global_max_proportion = max(global_max_proportion, np.max(proportions))
    
    global_y_max = global_max_proportion * 1.1 if global_max_proportion > 0 else 0.1
    print(f"  Global max proportion for {score_type}: {global_max_proportion:.4f}, y_max: {global_y_max:.4f}")
    
    for cat in sorted(category_layer_scores.keys()):
        layer_scores = category_layer_scores[cat]
        num_layers = len(layer_scores)
        
        if num_layers == 0:
            continue
        
        # 计算网格布局 (8行4列可以容纳32层)
        n_cols = 4
        n_rows = (num_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        axes = axes.flatten()
        
        for layer_idx, scores in enumerate(layer_scores):
            ax = axes[layer_idx]
            
            # 计算直方图
            counts, _ = np.histogram(scores, bins=bin_edges)
            total_heads = len(scores)
            proportions = counts / total_heads if total_heads > 0 else counts
            
            # 绘图
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(bin_centers, proportions, width=0.018, alpha=0.7, edgecolor='black', linewidth=0.3)
            
            # 计算统计量
            if len(scores) > 0:
                mean_score = np.mean(scores)
                median_score = np.median(scores)
                std_score = np.std(scores)
                
                # 添加均值线
                ax.axvline(mean_score, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
                
                # 添加中值线
                ax.axvline(median_score, color='green', linestyle='-.', linewidth=1.2, alpha=0.7)
                
                # 添加统计信息文本
                stats_text = f'μ={mean_score:.3f}\nmed={median_score:.3f}\nσ={std_score:.3f}'
                ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=7, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
            
            ax.set_title(f"Layer {layer_idx + 1} (n={total_heads})", fontsize=9, fontweight='bold')
            ax.set_xlabel("Score", fontsize=8)
            ax.set_ylabel("Proportion", fontsize=8)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0, global_y_max)
            ax.grid(True, alpha=0.2, axis='y')
            ax.tick_params(labelsize=7)
        
        # 隐藏未使用的子图
        for j in range(num_layers, len(axes)):
            axes[j].axis("off")
        
        fig.suptitle(f"{score_label} - {cat}\nAttention Distribution Across Layers", 
                     fontsize=18, fontweight='bold', y=0.998)
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        
        # 保存到类别子目录
        output_path = os.path.join(category_dir, f"{cat}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"  Saved {cat} layer plots to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="分析 LLaVA 模型在 MME 数据集上的注意力头交互情况")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="LLaVA 模型路径")
    parser.add_argument("--model-base", type=str, default=None,
                        help="模型基础路径 (可选)")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="MME 图片目录路径")
    parser.add_argument("--question-file", type=str, required=True,
                        help="MME 问题 JSON 文件路径")
    parser.add_argument("--output-dir", type=str, default="mme_attention_head_analysis",
                        help="输出目录")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1",
                        help="对话模板模式")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="处理的样本数量 (-1 表示全部)")
    args = parser.parse_args()

    # 初始化模型
    print("Loading model...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    # load_pretrained_model 已经将模型加载到正确的设备上了
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    model.eval()
    print(f"Model loaded on {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载问题
    print("Loading questions...")
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    if args.num_samples > 0:
        questions = questions[:args.num_samples]
    print(f"Loaded {len(questions)} questions")

    # 处理所有样本
    print("Processing samples...")
    records = []
    for line in tqdm(questions, desc="Analyzing attention heads"):
        rec = process_sample(model, tokenizer, image_processor, line, args, device)
        if rec is not None:
            records.append(rec)

    print(f"\nSuccessfully processed {len(records)} samples")
    
    if len(records) == 0:
        print("Error: No valid samples were processed. Exiting.")
        return

    # 保存原始数据
    save_path = os.path.join(args.output_dir, "attention_head_records.pt")
    torch.save(records, save_path)
    print(f"Saved records to {save_path}")

    # 绘制总体直方图 - T2V
    print("\nGenerating overall T2V histogram...")
    t2v_all = collect_all_scores(records, "t2v")
    plot_histogram(t2v_all, "Text-to-Visual Attention Distribution (Overall)", 
                   os.path.join(args.output_dir, "t2v_overall.png"))

    # 绘制总体直方图 - V2T
    print("Generating overall V2T histogram...")
    v2t_all = collect_all_scores(records, "v2t")
    plot_histogram(v2t_all, "Visual-to-Text Attention Distribution (Overall)", 
                   os.path.join(args.output_dir, "v2t_overall.png"))

    # 按类别绘制直方图 - T2V
    print("\nGenerating category-wise T2V histograms...")
    t2v_by_cat = collect_scores_by_category(records, "t2v")
    plot_category_histograms(t2v_by_cat, "t2v", args.output_dir)

    # 按类别绘制直方图 - V2T
    print("Generating category-wise V2T histograms...")
    v2t_by_cat = collect_scores_by_category(records, "v2t")
    plot_category_histograms(v2t_by_cat, "v2t", args.output_dir)

    # 按层绘制直方图 - T2V
    print("\nGenerating layer-wise T2V histograms...")
    t2v_by_layer = collect_scores_by_layer(records, "t2v")
    plot_layer_histograms(t2v_by_layer, "t2v", args.output_dir)

    # 按层绘制直方图 - V2T
    print("Generating layer-wise V2T histograms...")
    v2t_by_layer = collect_scores_by_layer(records, "v2t")
    plot_layer_histograms(v2t_by_layer, "v2t", args.output_dir)

    # 按类别和层绘制直方图 - T2V
    print("\nGenerating category and layer-wise T2V histograms...")
    t2v_by_cat_layer = collect_scores_by_category_and_layer(records, "t2v")
    plot_category_layer_histograms(t2v_by_cat_layer, "t2v", args.output_dir)
    save_category_layer_statistics(t2v_by_cat_layer, "t2v", args.output_dir)

    # 按类别和层绘制直方图 - V2T
    print("Generating category and layer-wise V2T histograms...")
    v2t_by_cat_layer = collect_scores_by_category_and_layer(records, "v2t")
    plot_category_layer_histograms(v2t_by_cat_layer, "v2t", args.output_dir)
    save_category_layer_statistics(v2t_by_cat_layer, "v2t", args.output_dir)

    # 生成统计报告
    print("\nGenerating statistics report...")
    
    # 安全计算统计量
    def safe_stats(scores):
        if len(scores) == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        return {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }
    
    stats = {
        "total_samples": len(records),
        "total_heads": len(t2v_all),
        "t2v_stats": safe_stats(t2v_all),
        "v2t_stats": safe_stats(v2t_all),
        "category_stats": {}
    }
    
    for cat in sorted(t2v_by_cat.keys()):
        t2v_cat = t2v_by_cat[cat]
        v2t_cat = v2t_by_cat[cat]
        stats["category_stats"][cat] = {
            "num_heads": len(t2v_cat),
            "t2v_mean": float(np.mean(t2v_cat)) if len(t2v_cat) > 0 else 0.0,
            "v2t_mean": float(np.mean(v2t_cat)) if len(v2t_cat) > 0 else 0.0,
        }
    
    stats_path = os.path.join(args.output_dir, "statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    print("\n" + "="*60)
    print("Analysis completed!")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total heads analyzed: {stats['total_heads']}")
    print(f"\nT2V - Mean: {stats['t2v_stats']['mean']:.4f}, Std: {stats['t2v_stats']['std']:.4f}")
    print(f"V2T - Mean: {stats['v2t_stats']['mean']:.4f}, Std: {stats['v2t_stats']['std']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
