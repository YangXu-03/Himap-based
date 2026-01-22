#!/usr/bin/env python3
"""
在MME数据集上测试，从每个子任务中选取一个样本，
绘制每一层的文本-视觉注意力热力图和最高注意力头的注意力分布，
并将热力图叠加在原始图像上。
"""

import argparse
import os
import json
from typing import Dict, List, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
# import cv2
from scipy import ndimage

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path


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
    
    # System tokens: prefix system prompt plus the first separator
    system_tokens = tokenizer(conv.system + conv.sep, add_special_tokens=False).input_ids
    system_start = 1 if input_ids[0, 0].item() == tokenizer.bos_token_id else 0
    system_end = min(system_start + len(system_tokens), seq_len_output)
    system_indices = list(range(system_start, system_end))
    
    text_indices = [i for i in range(seq_len_output) if i not in vis_indices and i not in system_indices]
    return system_indices, text_indices, vis_indices


def compute_text_to_visual_attention(attn: torch.Tensor, text_idx: List[int], vis_idx: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算文本到视觉的注意力
    
    Args:
        attn: (num_heads, seq_len, seq_len) 注意力矩阵
        text_idx: 文本 token 的索引列表
        vis_idx: 视觉 token 的索引列表
        
    Returns:
        avg_attention: (num_vis_tokens,) 每个视觉token的平均注意力
        head_attentions: (num_heads, num_vis_tokens) 每个头对每个视觉token的注意力
    """
    if len(text_idx) == 0 or len(vis_idx) == 0:
        return torch.zeros(len(vis_idx)), torch.zeros(attn.size(0), len(vis_idx))
    
    # 提取 Query 为文本、Key 为视觉的子矩阵
    # sub: (num_heads, len(text_idx), len(vis_idx))
    sub = attn[:, text_idx, :][:, :, vis_idx]
    
    # 对所有文本token求平均，得到每个头对每个视觉token的注意力
    # head_attentions: (num_heads, num_vis_tokens)
    head_attentions = sub.mean(dim=1)
    
    # 对所有头求平均，得到每个视觉token的平均注意力
    avg_attention = head_attentions.mean(dim=0)
    
    return avg_attention, head_attentions


def compute_last_text_to_visual_attention(attn: torch.Tensor, text_idx: List[int], vis_idx: List[int]) -> torch.Tensor:
    """
    计算最后一个文本token对视觉的注意力
    
    Args:
        attn: (num_heads, seq_len, seq_len) 注意力矩阵
        text_idx: 文本 token 的索引列表
        vis_idx: 视觉 token 的索引列表
        
    Returns:
        last_text_attention: (num_vis_tokens,) 最后一个文本token对每个视觉token的平均注意力（所有头平均）
    """
    if len(text_idx) == 0 or len(vis_idx) == 0:
        return torch.zeros(len(vis_idx))
    
    # 获取最后一个文本token的索引
    last_text_idx = text_idx[-1]
    
    # 提取最后一个文本token对视觉token的注意力
    # (num_heads, num_vis_tokens)
    last_text_attn = attn[:, last_text_idx, :][:, vis_idx]
    
    # 对所有头求平均
    avg_last_text_attn = last_text_attn.mean(dim=0)
    
    return avg_last_text_attn


def compute_text_importance_weighted_attention(attn: torch.Tensor, text_idx: List[int], vis_idx: List[int]) -> torch.Tensor:
    """
    使用文本-文本注意力计算文本重要性，然后加权文本-视觉注意力
    
    Args:
        attn: (num_heads, seq_len, seq_len) 注意力矩阵
        text_idx: 文本 token 的索引列表
        vis_idx: 视觉 token 的索引列表
        
    Returns:
        weighted_attention: (num_vis_tokens,) 文本重要性加权后的视觉token注意力
    """
    if len(text_idx) == 0 or len(vis_idx) == 0:
        return torch.zeros(len(vis_idx))
    
    # 提取文本-文本注意力矩阵: (num_heads, len(text_idx), len(text_idx))
    text_text_attn = attn[:, text_idx, :][:, :, text_idx]
    
    # 计算每个文本token接收到的注意力（作为重要性）
    # 对每个文本token，计算所有其他文本token对它的注意力之和
    # (num_heads, len(text_idx))
    text_importance = text_text_attn.sum(dim=1)
    
    # 对所有头求平均: (len(text_idx),)
    text_importance = text_importance.mean(dim=0)
    
    # 归一化文本重要性
    text_importance = text_importance / (text_importance.sum() + 1e-10)
    
    # 提取文本-视觉注意力: (num_heads, len(text_idx), len(vis_idx))
    text_vis_attn = attn[:, text_idx, :][:, :, vis_idx]
    
    # 对所有头求平均: (len(text_idx), len(vis_idx))
    text_vis_attn = text_vis_attn.mean(dim=0)
    
    # 使用文本重要性加权: (len(vis_idx),)
    weighted_attention = (text_importance.unsqueeze(1) * text_vis_attn).sum(dim=0)
    
    return weighted_attention


def create_attention_heatmap_overlay(image: Image.Image, attention: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    创建注意力热力图并叠加到原始图像上
    
    Args:
        image: 原始PIL图像
        attention: 注意力值数组，需要重塑为2D形状
        alpha: 热力图透明度
        
    Returns:
        叠加后的PIL图像
    """
    # 将图像转换为numpy数组
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    
    # 计算patch的grid size (假设是正方形)
    num_patches = len(attention)
    grid_size = int(np.sqrt(num_patches))
    
    # 重塑attention为2D grid
    if num_patches == grid_size * grid_size:
        attention_2d = attention.reshape(grid_size, grid_size)
    else:
        # 如果不是完美正方形，尝试找到最接近的矩形
        # LLaVA默认使用336x336图像，每个patch是14x14，所以是24x24的grid
        grid_h = int(np.sqrt(num_patches))
        grid_w = num_patches // grid_h
        if grid_h * grid_w < num_patches:
            grid_w += 1
        attention_2d = np.zeros((grid_h, grid_w))
        attention_2d.flat[:num_patches] = attention
    
    # 归一化到0-1
    if attention_2d.max() > attention_2d.min():
        attention_2d = (attention_2d - attention_2d.min()) / (attention_2d.max() - attention_2d.min())
    
    # 将attention转换为PIL图像并resize（避免ndimage的类型问题）
    attention_img = Image.fromarray((attention_2d * 255).astype(np.uint8))
    attention_resized_img = attention_img.resize((img_w, img_h), Image.BILINEAR)
    attention_resized = np.array(attention_resized_img).astype(np.float32) / 255.0

    # 应用颜色映射 (jet colormap)
    heatmap = plt.cm.jet(attention_resized)[:, :, :3]  # 去掉alpha通道
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 叠加热力图到原始图像
    # overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    overlay = (img_array * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    
    return Image.fromarray(overlay)


def process_sample(model, tokenizer, image_processor, line: Dict, args, device: torch.device, output_dir: str):
    """
    处理单个样本，生成所有层的注意力热力图
    """
    qs = line["question"]
    image_file = line["image_file"]
    category = line["category"]
    question_id = line["question_id"]
    gt_answer = line["answer"]

    image_path = os.path.join(args.image_folder, image_file)
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    images = image_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        images = images.half().cuda()
    else:
        images = images.float()
    images = images.to(device)

    # 准备prompt
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

    # 前向推理，获取注意力
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=images,
            max_new_tokens=128,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
    
    # 获取生成的答案
    output_ids = outputs.sequences[0]
    pred_answer = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # 获取注意力 - 注意generate返回的attentions结构不同
    # outputs.attentions是一个tuple，每个元素对应一个生成步骤
    # 我们使用第一个生成步骤的注意力（对应输入的完整处理）
    if not hasattr(outputs, 'attentions') or outputs.attentions is None or len(outputs.attentions) == 0:
        print(f"No attention found for {image_file}")
        return None
    
    # 第一个生成步骤的注意力
    first_step_attentions = outputs.attentions[0]  # tuple of layers
    if first_step_attentions is None or len(first_step_attentions) == 0:
        print(f"No attention in first step for {image_file}")
        return None
    
    attentions = [attn.squeeze(0) for attn in first_step_attentions]  # list of (heads, seq, seq)
    
    system_idx, text_idx, vis_idx = compute_token_groups(conv, prompt, tokenizer, input_ids, attentions)
    if len(text_idx) == 0 or len(vis_idx) == 0:
        print(f"Empty text or visual indices for {image_file}")
        return None

    # 为每一层计算注意力
    num_layers = len(attentions)
    layer_attentions = []  # 存储每层的平均注意力
    layer_max_head_attentions = []  # 存储每层最高注意力头的注意力
    layer_last_text_attentions = []  # 存储每层最后一个文本token的注意力
    layer_text_importance_weighted_attentions = []  # 存储每层文本重要性加权的注意力
    
    for layer_idx, attn in enumerate(attentions):
        avg_attn, head_attns = compute_text_to_visual_attention(attn, text_idx, vis_idx)
        layer_attentions.append(avg_attn.cpu().numpy())
        
        # 找到注意力最高的头
        head_scores = head_attns.mean(dim=1)  # (num_heads,)
        max_head_idx = head_scores.argmax().item()
        max_head_attn = head_attns[max_head_idx].cpu().numpy()
        layer_max_head_attentions.append(max_head_attn)
        
        # 计算最后一个文本token的注意力
        last_text_attn = compute_last_text_to_visual_attention(attn, text_idx, vis_idx)
        layer_last_text_attentions.append(last_text_attn.cpu().numpy())
        
        # 计算文本重要性加权的注意力
        text_importance_weighted_attn = compute_text_importance_weighted_attention(attn, text_idx, vis_idx)
        layer_text_importance_weighted_attentions.append(text_importance_weighted_attn.cpu().numpy())
    
    # 创建输出目录
    sample_dir = os.path.join(output_dir, f"{category}_{os.path.splitext(image_file)[0]}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 保存原始图像
    orig_img_path = os.path.join(sample_dir, "original.png")
    image.save(orig_img_path)
    
    # 为每一层生成热力图
    num_cols = 8
    num_rows_avg = (num_layers + num_cols - 1) // num_cols
    num_rows_max = (num_layers + num_cols - 1) // num_cols
    
    # 绘制平均注意力热力图网格
    fig_avg, axes_avg = plt.subplots(num_rows_avg, num_cols, figsize=(num_cols * 3, num_rows_avg * 3))
    axes_avg = axes_avg.flatten() if num_layers > 1 else [axes_avg]
    
    for layer_idx in range(num_layers):
        ax = axes_avg[layer_idx]
        overlay = create_attention_heatmap_overlay(image, layer_attentions[layer_idx], alpha=0.6)
        ax.imshow(overlay)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(num_layers, len(axes_avg)):
        axes_avg[idx].axis('off')
    
    fig_avg.suptitle(f"Text→Visual Attention (Average)\n{category} - Q: {qs[:80]}...\nGT: {gt_answer} | Pred: {pred_answer}", 
                     fontsize=12, fontweight='bold')
    plt.tight_layout()
    avg_grid_path = os.path.join(sample_dir, "avg_attention_grid.png")
    plt.savefig(avg_grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig_avg)
    
    # 绘制最高注意力头热力图网格
    fig_max, axes_max = plt.subplots(num_rows_max, num_cols, figsize=(num_cols * 3, num_rows_max * 3))
    axes_max = axes_max.flatten() if num_layers > 1 else [axes_max]
    
    for layer_idx in range(num_layers):
        ax = axes_max[layer_idx]
        overlay = create_attention_heatmap_overlay(image, layer_max_head_attentions[layer_idx], alpha=0.6)
        ax.imshow(overlay)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(num_layers, len(axes_max)):
        axes_max[idx].axis('off')
    
    fig_max.suptitle(f"Text→Visual Attention (Max Head)\n{category} - Q: {qs[:80]}...\nGT: {gt_answer} | Pred: {pred_answer}", 
                     fontsize=12, fontweight='bold')
    plt.tight_layout()
    max_grid_path = os.path.join(sample_dir, "max_head_attention_grid.png")
    plt.savefig(max_grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig_max)
    
    # 绘制最后一个文本token注意力热力图网格
    fig_last, axes_last = plt.subplots(num_rows_max, num_cols, figsize=(num_cols * 3, num_rows_max * 3))
    axes_last = axes_last.flatten() if num_layers > 1 else [axes_last]
    
    for layer_idx in range(num_layers):
        ax = axes_last[layer_idx]
        overlay = create_attention_heatmap_overlay(image, layer_last_text_attentions[layer_idx], alpha=0.6)
        ax.imshow(overlay)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(num_layers, len(axes_last)):
        axes_last[idx].axis('off')
    
    fig_last.suptitle(f"Text→Visual Attention (Last Text Token)\n{category} - Q: {qs[:80]}...\nGT: {gt_answer} | Pred: {pred_answer}", 
                      fontsize=12, fontweight='bold')
    plt.tight_layout()
    last_grid_path = os.path.join(sample_dir, "last_text_token_attention_grid.png")
    plt.savefig(last_grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig_last)
    
    # 绘制文本重要性加权注意力热力图网格
    fig_weighted, axes_weighted = plt.subplots(num_rows_max, num_cols, figsize=(num_cols * 3, num_rows_max * 3))
    axes_weighted = axes_weighted.flatten() if num_layers > 1 else [axes_weighted]
    
    for layer_idx in range(num_layers):
        ax = axes_weighted[layer_idx]
        overlay = create_attention_heatmap_overlay(image, layer_text_importance_weighted_attentions[layer_idx], alpha=0.6)
        ax.imshow(overlay)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(num_layers, len(axes_weighted)):
        axes_weighted[idx].axis('off')
    
    fig_weighted.suptitle(f"Text→Visual Attention (Text-Importance Weighted)\n{category} - Q: {qs[:80]}...\nGT: {gt_answer} | Pred: {pred_answer}", 
                          fontsize=12, fontweight='bold')
    plt.tight_layout()
    weighted_grid_path = os.path.join(sample_dir, "text_importance_weighted_attention_grid.png")
    plt.savefig(weighted_grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig_weighted)
    
    print(f"Saved visualizations for {category}/{image_file} to {sample_dir}")
    
    # 返回结果数据
    return {
        "category": category,
        "question_id": question_id,
        "image_file": image_file,
        "question": qs,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "num_layers": num_layers,
        "output_dir": sample_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize attention heatmaps on MME dataset")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="mme_attention_heatmap_visualization")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--samples-per-category", type=int, default=1,
                       help="Number of samples to select from each category")
    args = parser.parse_args()

    # 初始化模型
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载问题
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    
    # 按类别分组
    category_samples = defaultdict(list)
    for q in questions:
        category_samples[q["category"]].append(q)
    
    print(f"Found {len(category_samples)} categories")
    for cat, samples in sorted(category_samples.items()):
        print(f"  {cat}: {len(samples)} samples")
    
    # 从每个类别选择指定数量的样本
    selected_samples = []
    for cat in sorted(category_samples.keys()):
        samples = category_samples[cat][:args.samples_per_category]
        selected_samples.extend(samples)
        print(f"Selected {len(samples)} sample(s) from {cat}")
    
    print(f"\nTotal selected samples: {len(selected_samples)}")
    
    # 处理样本
    results = []
    for line in tqdm(selected_samples, desc="Processing samples"):
        result = process_sample(model, tokenizer, image_processor, line, args, device, args.output_dir)
        if result is not None:
            results.append(result)
    
    # 保存结果数据
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Processed {len(results)} samples successfully")
    print(f"Results saved to: {results_file}")
    print(f"Visualizations saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
