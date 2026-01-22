#!/usr/bin/env python3
"""
样本级注意力分析：从MME每个子任务中提取前N个样本
记录每个样本的：
  1. 6种注意力值在每一层的变化
  2. 注意力头的 text-visual (T2V 和 V2T) 值在每一层的变化
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


# 6种注意力类型
ATTN_TYPES = [
    "text_to_text",
    "text_to_visual",
    "visual_to_text",
    "visual_to_visual",
    "system_to_text",
    "system_to_visual",
]

TYPE_LABELS = {
    "text_to_text": "Text → Text",
    "text_to_visual": "Text → Visual",
    "visual_to_text": "Visual → Text",
    "visual_to_visual": "Visual → Visual",
    "system_to_text": "System → Text",
    "system_to_visual": "System → Visual",
}


def attention_share(attn: torch.Tensor, query_idx: List[int], key_idx: List[int]) -> float:
    """
    计算从 queries 到 keys 的平均注意力质量
    
    Args:
        attn: (num_heads, seq_len, seq_len) 注意力矩阵
        query_idx: query token 的索引列表
        key_idx: key token 的索引列表
        
    Returns:
        float: 平均注意力分数
    """
    if len(query_idx) == 0 or len(key_idx) == 0:
        return 0.0
    sub = attn[:, query_idx, :]
    sub = sub[:, :, key_idx]
    weights = sub.sum(dim=-1)  # (heads, q_len)
    return weights.mean().item()


def compute_t2v_scores_per_head(attn: torch.Tensor, text_idx: List[int], vis_idx: List[int]) -> List[float]:
    """
    计算每个注意力头的 Text-to-Visual 分数
    
    Args:
        attn: (num_heads, seq_len, seq_len) 注意力矩阵
        text_idx: 文本 token 的索引列表
        vis_idx: 视觉 token 的索引列表
        
    Returns:
        List[float]: 每个头的 T2V 分数
    """
    if len(text_idx) == 0 or len(vis_idx) == 0:
        return [0.0] * attn.size(0)
    
    # 提取 Query 为文本、Key 为图像的子矩阵
    sub = attn[:, text_idx, :][:, :, vis_idx]
    
    # 沿 Key (视觉) 维度求和，再沿 Query (文本) 维度取平均
    weights = sub.sum(dim=-1)  # (num_heads, len(text_idx))
    scores = weights.mean(dim=-1)  # (num_heads,)
    
    return scores.cpu().tolist()


def compute_v2t_scores_per_head(attn: torch.Tensor, vis_idx: List[int], text_idx: List[int]) -> List[float]:
    """
    计算每个注意力头的 Visual-to-Text 分数
    
    Args:
        attn: (num_heads, seq_len, seq_len) 注意力矩阵
        vis_idx: 视觉 token 的索引列表
        text_idx: 文本 token 的索引列表
        
    Returns:
        List[float]: 每个头的 V2T 分数
    """
    if len(vis_idx) == 0 or len(text_idx) == 0:
        return [0.0] * attn.size(0)
    
    # 提取 Query 为图像、Key 为文本的子矩阵
    sub = attn[:, vis_idx, :][:, :, text_idx]
    
    # 沿 Key (文本) 维度求和，再沿 Query (视觉) 维度取平均
    weights = sub.sum(dim=-1)  # (num_heads, len(vis_idx))
    scores = weights.mean(dim=-1)  # (num_heads,)
    
    return scores.cpu().tolist()


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
    处理单个样本，记录：
    1. 6种注意力值在每一层的变化
    2. 每一层每个注意力头的 T2V 和 V2T 分数
    
    Returns:
        {
            "category": str,
            "image_file": str,
            "question": str,
            "answer": str,
            # 6种注意力类型在每一层的值
            "attention_flows": {
                "text_to_text": List[float],  # 每层的值
                "text_to_visual": List[float],
                "visual_to_text": List[float],
                "visual_to_visual": List[float],
                "system_to_text": List[float],
                "system_to_visual": List[float],
            },
            # 每一层每个头的 T2V 和 V2T 分数
            "head_scores": {
                "t2v": List[List[float]],  # [layer][head]
                "v2t": List[List[float]],  # [layer][head]
            }
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

    # 初始化记录结构
    attention_flows = {k: [] for k in ATTN_TYPES}
    t2v_head_scores = []
    v2t_head_scores = []

    # 遍历每一层
    for attn in attentions:
        attn = attn.squeeze(0)  # (num_heads, seq, seq)
        
        # 记录6种注意力流
        attention_flows["text_to_text"].append(attention_share(attn, text_idx, text_idx))
        attention_flows["text_to_visual"].append(attention_share(attn, text_idx, vis_idx))
        attention_flows["visual_to_text"].append(attention_share(attn, vis_idx, text_idx))
        attention_flows["visual_to_visual"].append(attention_share(attn, vis_idx, vis_idx))
        attention_flows["system_to_text"].append(attention_share(attn, system_idx, text_idx))
        attention_flows["system_to_visual"].append(attention_share(attn, system_idx, vis_idx))
        
        # 记录每个头的 T2V 和 V2T 分数
        t2v_scores = compute_t2v_scores_per_head(attn, text_idx, vis_idx)
        v2t_scores = compute_v2t_scores_per_head(attn, vis_idx, text_idx)
        
        t2v_head_scores.append(t2v_scores)
        v2t_head_scores.append(v2t_scores)

    return {
        "category": line["category"],
        "image_file": image_file,
        "question": line["question"],
        "answer": line.get("answer", ""),
        "attention_flows": attention_flows,
        "head_scores": {
            "t2v": t2v_head_scores,
            "v2t": v2t_head_scores,
        }
    }


def select_samples_per_category(questions: List[Dict], samples_per_category: int) -> List[Dict]:
    """
    从每个类别中选择前N个样本
    
    Args:
        questions: 所有问题列表
        samples_per_category: 每个类别选择的样本数
        
    Returns:
        选中的样本列表
    """
    # 按类别分组
    category_samples = defaultdict(list)
    for q in questions:
        category_samples[q["category"]].append(q)
    
    # 从每个类别选择前N个样本
    selected = []
    for cat in sorted(category_samples.keys()):
        samples = category_samples[cat][:samples_per_category]
        selected.extend(samples)
        print(f"Category '{cat}': Selected {len(samples)} samples")
    
    return selected


def plot_sample_attention_flows(sample_data: Dict, output_path: str):
    """
    绘制单个样本的6种注意力流在各层的变化
    
    Args:
        sample_data: 样本数据字典
        output_path: 输出路径
    """
    flows = sample_data["attention_flows"]
    num_layers = len(flows["text_to_text"])
    layers = list(range(1, num_layers + 1))
    
    plt.figure(figsize=(12, 6))
    
    for attn_type in ATTN_TYPES:
        plt.plot(layers, flows[attn_type], label=TYPE_LABELS[attn_type], linewidth=2, marker='o', markersize=4)
    
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Attention Mass", fontsize=12)
    plt.title(f"Attention Flows Across Layers\n{sample_data['category']} - {sample_data['image_file']}", 
              fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_sample_head_scores(sample_data: Dict, output_path: str):
    """
    绘制单个样本的注意力头 T2V 和 V2T 分数在各层的分布
    使用箱线图展示每层所有头的分数分布
    
    箱线图元素说明：
    - 箱体：第25-75百分位（Q1-Q3），包含中间50%的数据
    - 红色/蓝色粗线：中位数（Median）
    - 绿色/橙色菱形：平均值（Mean）
    - 须线：延伸到 [Q1-1.5×IQR, Q3+1.5×IQR] 范围内的最远数据点
    - 圆点：异常值，超出须线范围的数据点
    
    Args:
        sample_data: 样本数据字典
        output_path: 输出路径
    """
    t2v_scores = sample_data["head_scores"]["t2v"]  # [layer][head]
    v2t_scores = sample_data["head_scores"]["v2t"]
    num_layers = len(t2v_scores)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # T2V 箱线图
    ax1.boxplot(t2v_scores, positions=range(1, num_layers + 1), widths=0.6, 
                patch_artist=True, showmeans=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(marker='D', markerfacecolor='green', markersize=6))
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("T2V Score", fontsize=11)
    ax1.set_title("Text-to-Visual Attention Head Scores Across Layers", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(0, num_layers + 1)
    ax1.set_ylim(0, 1)
    
    # V2T 箱线图
    ax2.boxplot(v2t_scores, positions=range(1, num_layers + 1), widths=0.6, 
                patch_artist=True, showmeans=True,
                boxprops=dict(facecolor='lightcoral', alpha=0.7),
                medianprops=dict(color='blue', linewidth=2),
                meanprops=dict(marker='D', markerfacecolor='orange', markersize=6))
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("V2T Score", fontsize=11)
    ax2.set_title("Visual-to-Text Attention Head Scores Across Layers", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(0, num_layers + 1)
    ax2.set_ylim(0, 1)
    
    fig.suptitle(f"{sample_data['category']} - {sample_data['image_file']}", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_category_samples_grid(records: List[Dict], category: str, output_dir: str):
    """
    为一个类别的所有样本绘制网格图：
    每个样本一个子图，展示其6种注意力流在各层的变化
    
    Args:
        records: 该类别的所有样本记录
        category: 类别名称
        output_dir: 输出目录
    """
    if not records:
        return
    
    num_samples = len(records)
    num_layers = len(records[0]["attention_flows"]["text_to_text"])
    layers = list(range(1, num_layers + 1))
    
    # 计算子图布局：尽量接近正方形
    ncols = int(np.ceil(np.sqrt(num_samples)))
    nrows = int(np.ceil(num_samples / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    
    # 如果只有一行或一列，确保axes是二维数组
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, rec in enumerate(records):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        flows = rec["attention_flows"]
        
        # 绘制6种注意力流
        for attn_type in ATTN_TYPES:
            ax.plot(layers, flows[attn_type], label=TYPE_LABELS[attn_type], 
                   linewidth=1.5, marker='o', markersize=3)
        
        # 设置子图标题和标签
        img_name = rec["image_file"].split("/")[-1] if "/" in rec["image_file"] else rec["image_file"]
        ax.set_title(f"Sample {idx+1}: {img_name}", fontsize=9, fontweight='bold')
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("Attention Mass", fontsize=8)
        ax.legend(loc='best', fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    
    # 隐藏多余的子图
    for idx in range(num_samples, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    fig.suptitle(f"Category: {category} - All Samples Attention Flows", 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path = os.path.join(output_dir, f"category_{category}_samples_grid.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved category samples grid: {output_path}")


def plot_sample_head_distribution(sample_data: Dict, output_path: str):
    """
    绘制单个样本的注意力头分布直方图
    横坐标：注意力头的T2V/V2T分数值
    纵坐标：注意力头比例（密度）
    每层一个子图，展示该层所有注意力头的分数分布
    
    Args:
        sample_data: 样本数据字典
        output_path: 输出路径
    """
    t2v_scores = sample_data["head_scores"]["t2v"]  # [layer][head]
    v2t_scores = sample_data["head_scores"]["v2t"]
    num_layers = len(t2v_scores)
    
    # 计算子图布局：每行8个子图
    ncols = 8
    nrows = int(np.ceil(num_layers / ncols))
    
    # 创建T2V直方图
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*2.5))
    axes = axes.flatten() if num_layers > 1 else [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        scores = t2v_scores[layer_idx]
        
        # 绘制直方图
        ax.hist(scores, bins=20, range=(0, 1), density=True, 
               alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
        
        ax.set_title(f"Layer {layer_idx+1}", fontsize=9, fontweight='bold')
        ax.set_xlabel("T2V Score", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 10)  # 统一纵坐标范围
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f"Text-to-Visual Head Score Distribution\n{sample_data['category']} - {sample_data['image_file']}", 
                fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    # 创建V2T直方图
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*2.5))
    axes = axes.flatten() if num_layers > 1 else [axes]
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        scores = v2t_scores[layer_idx]
        
        # 绘制直方图
        ax.hist(scores, bins=20, range=(0, 1), density=True, 
               alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
        
        ax.set_title(f"Layer {layer_idx+1}", fontsize=9, fontweight='bold')
        ax.set_xlabel("V2T Score", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 10)  # 统一纵坐标范围
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f"Visual-to-Text Head Score Distribution\n{sample_data['category']} - {sample_data['image_file']}", 
                fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 修改输出路径，在原文件名基础上添加v2t标识
    v2t_output_path = output_path.replace('_distribution.png', '_v2t_distribution.png')
    plt.savefig(v2t_output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_category_samples_overview(records: List[Dict], category: str, output_dir: str):
    """
    为一个类别的所有样本绘制概览图：
    - 上图：所有样本的6种注意力流
    - 下图：所有样本的 T2V 和 V2T 平均分数
    
    Args:
        records: 该类别的所有样本记录
        category: 类别名称
        output_dir: 输出目录
    """
    if not records:
        return
    
    num_layers = len(records[0]["attention_flows"]["text_to_text"])
    layers = list(range(1, num_layers + 1))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 上图：6种注意力流
    ax1 = axes[0]
    for attn_type in ATTN_TYPES:
        # 收集所有样本该类型的注意力流
        all_flows = [rec["attention_flows"][attn_type] for rec in records]
        mean_flow = np.mean(all_flows, axis=0)
        std_flow = np.std(all_flows, axis=0)
        
        ax1.plot(layers, mean_flow, label=TYPE_LABELS[attn_type], linewidth=2.5, marker='o', markersize=5)
        ax1.fill_between(layers, mean_flow - std_flow, mean_flow + std_flow, alpha=0.2)
    
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Attention Mass", fontsize=11)
    ax1.set_title(f"Attention Flows (n={len(records)} samples)", fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 下图：T2V 和 V2T 平均分数
    ax2 = axes[1]
    
    # 计算每层所有样本所有头的平均 T2V 和 V2T
    t2v_means = []
    v2t_means = []
    
    for layer_idx in range(num_layers):
        # 收集该层所有样本所有头的分数
        t2v_layer_all = []
        v2t_layer_all = []
        
        for rec in records:
            t2v_layer_all.extend(rec["head_scores"]["t2v"][layer_idx])
            v2t_layer_all.extend(rec["head_scores"]["v2t"][layer_idx])
        
        t2v_means.append(np.mean(t2v_layer_all))
        v2t_means.append(np.mean(v2t_layer_all))
    
    ax2.plot(layers, t2v_means, label="Text-to-Visual (T2V)", linewidth=2.5, marker='s', markersize=6, color='blue')
    ax2.plot(layers, v2t_means, label="Visual-to-Text (V2T)", linewidth=2.5, marker='^', markersize=6, color='red')
    
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Average Head Score", fontsize=11)
    ax2.set_title(f"Average Attention Head Scores (n={len(records)} samples)", fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    fig.suptitle(f"Category: {category}", fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(output_dir, f"category_{category}_overview.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved category overview: {output_path}")


def save_statistics(records: List[Dict], output_dir: str):
    """
    保存详细统计数据
    
    Args:
        records: 所有样本记录
        output_dir: 输出目录
    """
    # 按类别分组
    category_records = defaultdict(list)
    for rec in records:
        category_records[rec["category"]].append(rec)
    
    # 统计数据结构
    statistics = {
        "total_samples": len(records),
        "categories": {},
    }
    
    for cat in sorted(category_records.keys()):
        cat_recs = category_records[cat]
        num_samples = len(cat_recs)
        num_layers = len(cat_recs[0]["attention_flows"]["text_to_text"])
        
        cat_stats = {
            "num_samples": num_samples,
            "num_layers": num_layers,
            "attention_flows": {},
            "head_scores": {},
        }
        
        # 计算每种注意力流的统计量（跨样本和层）
        for attn_type in ATTN_TYPES:
            all_values = []
            for rec in cat_recs:
                all_values.extend(rec["attention_flows"][attn_type])
            
            cat_stats["attention_flows"][attn_type] = {
                "mean": float(np.mean(all_values)),
                "std": float(np.std(all_values)),
                "min": float(np.min(all_values)),
                "max": float(np.max(all_values)),
            }
        
        # 计算注意力头分数的统计量
        all_t2v = []
        all_v2t = []
        for rec in cat_recs:
            for layer_scores in rec["head_scores"]["t2v"]:
                all_t2v.extend(layer_scores)
            for layer_scores in rec["head_scores"]["v2t"]:
                all_v2t.extend(layer_scores)
        
        cat_stats["head_scores"]["t2v"] = {
            "mean": float(np.mean(all_t2v)),
            "std": float(np.std(all_t2v)),
            "min": float(np.min(all_t2v)),
            "max": float(np.max(all_t2v)),
        }
        
        cat_stats["head_scores"]["v2t"] = {
            "mean": float(np.mean(all_v2t)),
            "std": float(np.std(all_v2t)),
            "min": float(np.min(all_v2t)),
            "max": float(np.max(all_v2t)),
        }
        
        statistics["categories"][cat] = cat_stats
    
    # 保存统计数据
    stats_path = os.path.join(output_dir, "sample_level_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\nSaved statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="样本级注意力分析：记录每个样本的注意力流和注意力头分数")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="LLaVA 模型路径")
    parser.add_argument("--model-base", type=str, default=None,
                        help="模型基础路径 (可选)")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="MME 图片目录路径")
    parser.add_argument("--question-file", type=str, required=True,
                        help="MME 问题 JSON 文件路径")
    parser.add_argument("--output-dir", type=str, default="mme_sample_level_attention",
                        help="输出目录")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1",
                        help="对话模板模式")
    parser.add_argument("--samples-per-category", type=int, default=8,
                        help="每个类别选择的样本数")
    parser.add_argument("--plot-individual", action="store_true",
                        help="是否为每个样本绘制独立的图表")
    args = parser.parse_args()

    # 初始化模型
    print("Loading model...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    model.eval()
    print(f"Model loaded on {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载问题并选择样本
    print("Loading questions...")
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    selected_questions = select_samples_per_category(questions, args.samples_per_category)
    print(f"\nTotal selected samples: {len(selected_questions)}")

    # 处理所有样本
    print("\nProcessing samples...")
    records = []
    for line in tqdm(selected_questions, desc="Analyzing samples"):
        rec = process_sample(model, tokenizer, image_processor, line, args, device)
        if rec is not None:
            records.append(rec)

    print(f"\nSuccessfully processed {len(records)} samples")
    
    if len(records) == 0:
        print("Error: No valid samples were processed. Exiting.")
        return

    # 保存原始数据
    save_path = os.path.join(args.output_dir, "sample_level_records.pt")
    torch.save(records, save_path)
    print(f"\nSaved records to {save_path}")

    # 按类别分组
    category_records = defaultdict(list)
    for rec in records:
        category_records[rec["category"]].append(rec)

    # 为每个类别绘制概览图和网格图
    print("\nGenerating category overview plots...")
    for cat in sorted(category_records.keys()):
        cat_recs = category_records[cat]
        plot_category_samples_overview(cat_recs, cat, args.output_dir)
        plot_category_samples_grid(cat_recs, cat, args.output_dir)

    # 可选：为每个样本绘制独立的图表
    if args.plot_individual:
        print("\nGenerating individual sample plots...")
        sample_plots_dir = os.path.join(args.output_dir, "sample_plots")
        boxplot_dir = os.path.join(sample_plots_dir, "boxplots")
        histogram_dir = os.path.join(sample_plots_dir, "histograms")
        os.makedirs(sample_plots_dir, exist_ok=True)
        os.makedirs(boxplot_dir, exist_ok=True)
        os.makedirs(histogram_dir, exist_ok=True)
        
        for idx, rec in enumerate(tqdm(records, desc="Plotting samples")):
            cat = rec["category"]
            safe_name = rec["image_file"].replace("/", "_").replace("\\", "_")
            
            # 注意力流图
            flow_path = os.path.join(sample_plots_dir, f"{cat}_{safe_name}_flows.png")
            plot_sample_attention_flows(rec, flow_path)
            
            # 注意力头分数箱线图（放入boxplots子文件夹）
            heads_path = os.path.join(boxplot_dir, f"{cat}_{safe_name}_heads.png")
            plot_sample_head_scores(rec, heads_path)
            
            # 注意力头分布直方图（放入histograms子文件夹）
            dist_path = os.path.join(histogram_dir, f"{cat}_{safe_name}_distribution.png")
            plot_sample_head_distribution(rec, dist_path)

    # 保存统计数据
    save_statistics(records, args.output_dir)

    print("\n" + "="*60)
    print("Sample-level analysis completed!")
    print(f"Total samples analyzed: {len(records)}")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
