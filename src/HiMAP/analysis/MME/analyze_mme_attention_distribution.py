"""
MME数据集注意力头分布统计与可视化

功能：
1. 统计LLaVA模型在MME数据集上的文本-视觉注意力分布
2. 按category分类统计每个注意力头在32层的注意力值
3. 为每个category生成热力图，展示32个头×32层的注意力分布

使用方法：
    python analyze_mme_attention_distribution.py \
        --model-path liuhaotian/llava-v1.5-7b \
        --image-folder /path/to/MME/images/test \
        --question-file /path/to/MME_test.json \
        --output-dir mme_attention_distribution \
        --num-samples 100
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
import seaborn as sns

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path


def extract_token_indices(conv, tokenizer, input_ids: torch.Tensor, seq_len: int) -> Tuple[List[int], List[int]]:
    """
    提取文本token和视觉token的索引位置

    Returns:
        text_indices: 文本token的索引列表
        visual_indices: 视觉token的索引列表
    """
    # 找到图像token的位置
    image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
    if len(image_token_indices) == 0:
        return [], []

    # 计算视觉token的范围
    img_start_idx = image_token_indices[0].item()
    num_patches = seq_len - input_ids.shape[1] + 1
    visual_indices = list(range(img_start_idx, img_start_idx + num_patches))

    # 计算系统提示词的范围（需要排除）
    system_tokens = tokenizer(conv.system + conv.sep, add_special_tokens=False).input_ids
    system_start = 1 if input_ids[0, 0].item() == tokenizer.bos_token_id else 0
    system_end = min(system_start + len(system_tokens), seq_len)
    system_indices = list(range(system_start, system_end))

    # 文本token = 全部token - 视觉token - 系统token
    text_indices = [i for i in range(seq_len)
                    if i not in visual_indices and i not in system_indices]

    return text_indices, visual_indices


def compute_text_to_visual_attention(attention_tensor: torch.Tensor,
                                     text_indices: List[int],
                                     visual_indices: List[int]) -> np.ndarray:
    """
    计算文本到视觉的注意力值

    Args:
        attention_tensor: shape (num_heads, seq_len, seq_len)
        text_indices: 文本token的索引
        visual_indices: 视觉token的索引

    Returns:
        head_values: shape (num_heads,)，每个头的平均文本→视觉注意力值
    """
    num_heads = attention_tensor.shape[0]
    head_values = np.zeros(num_heads)

    for h in range(num_heads):
        # 提取子矩阵：text_query × visual_key
        # shape: (len(text_indices), len(visual_indices))
        t2v_matrix = attention_tensor[h][text_indices][:, visual_indices]

        # 沿key维度求和（每个文本token对所有视觉token的总注意力）
        # 然后对所有文本token取平均
        t2v_sum_over_visual = t2v_matrix.sum(dim=1)  # shape: (len(text_indices),)
        head_values[h] = t2v_sum_over_visual.mean().item()

    return head_values


def process_single_sample(model, tokenizer, image_processor, sample_data: Dict,
                          args, device: torch.device) -> Dict:
    """
    处理单个样本，返回各层各头的注意力值

    Returns:
        {
            'category': str,
            'attention_values': np.ndarray, shape (num_layers, num_heads)
        }
        或 None（如果处理失败）
    """
    question = sample_data["question"]
    image_file = sample_data["image_file"]
    category = sample_data.get("category", "unknown")

    # 加载图像
    image_path = os.path.join(args.image_folder, image_file)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None

    try:
        # 预处理图像
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        images = image_tensor.unsqueeze(0)
        if device.type == "cuda":
            images = images.half()
        images = images.to(device)

        # 构造prompt
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + question
        qs = qs + "\n" + "Answer the question using a single word or phrase."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(device)

        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids,
                images=images,
                output_attentions=True,
                use_cache=True,
                return_dict=True,
            )

        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
        if attentions is None or len(attentions) == 0:
            return None

        # 提取token索引
        seq_len = attentions[0].shape[-1]
        text_indices, visual_indices = extract_token_indices(conv, tokenizer, input_ids, seq_len)

        if len(text_indices) == 0 or len(visual_indices) == 0:
            return None

        # 计算每层每个头的文本→视觉注意力
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        attention_values = np.zeros((num_layers, num_heads))

        for layer_idx, attn in enumerate(attentions):
            attn = attn.squeeze(0)  # shape: (heads, seq, seq)
            layer_head_values = compute_text_to_visual_attention(
                attn, text_indices, visual_indices
            )
            attention_values[layer_idx] = layer_head_values

        return {
            'category': category,
            'attention_values': attention_values,  # shape: (num_layers, num_heads)
        }

    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        return None


def aggregate_by_category(results: List[Dict]) -> Dict[str, np.ndarray]:
    """
    按category聚合注意力值

    Returns:
        {
            category: np.ndarray, shape (num_samples_in_category, num_layers, num_heads)
        }
    """
    category_data = defaultdict(list)

    for result in results:
        category = result['category']
        attention_values = result['attention_values']
        category_data[category].append(attention_values)

    # 转换为numpy数组
    aggregated = {}
    for category, values_list in category_data.items():
        aggregated[category] = np.array(values_list)  # shape: (n_samples, n_layers, n_heads)

    return aggregated


def plot_category_heatmap(category_data: Dict[str, np.ndarray], output_dir: str):
    """
    为每个category绘制热力图：32个头 × 32层

    Args:
        category_data: {category: array of shape (n_samples, n_layers, n_heads)}
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 首先计算所有类别的全局最大值，以统一颜色范围
    global_max = 0.0
    for data in category_data.values():
        mean_attention = data.mean(axis=0)
        global_max = max(global_max, mean_attention.max())
    
    print(f"\nGlobal max attention value across all categories: {global_max:.4f}")

    for category, data in category_data.items():
        # 计算平均值：对所有样本取平均
        # shape: (n_layers, n_heads)
        mean_attention = data.mean(axis=0)

        # 创建热力图
        fig, ax = plt.subplots(figsize=(14, 10))

        # 使用seaborn绘制热力图，使用统一的vmax
        sns.heatmap(
            mean_attention.T,  # 转置使得层数在x轴，头数在y轴
            cmap='YlOrRd',
            cbar_kws={'label': 'Text→Visual Attention'},
            ax=ax,
            vmin=0,
            vmax=global_max,  # 使用全局最大值统一颜色范围
            linewidths=0.1,
            linecolor='gray'
        )

        ax.set_xlabel('Layer Index', fontsize=14)
        ax.set_ylabel('Attention Head Index', fontsize=14)
        ax.set_title(f'Text→Visual Attention Distribution\nCategory: {category}\n({data.shape[0]} samples)',
                     fontsize=16, pad=20)

        # 设置刻度
        num_layers, num_heads = mean_attention.shape
        ax.set_xticks(np.arange(0, num_layers, 4) + 0.5)
        ax.set_xticklabels(np.arange(0, num_layers, 4))
        ax.set_yticks(np.arange(0, num_heads, 4) + 0.5)
        ax.set_yticklabels(np.arange(0, num_heads, 4))

        plt.tight_layout()

        # 保存图像
        category_dir = os.path.join(output_dir, f"category_{category}")
        os.makedirs(category_dir, exist_ok=True)
        output_path = os.path.join(category_dir, f"{category}_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved heatmap for category '{category}': {output_path}")


def plot_overall_heatmap(category_data: Dict[str, np.ndarray], output_dir: str):
    """
    绘制总体热力图：所有样本的平均注意力分布
    """
    # 合并所有category的数据
    all_data = np.concatenate(list(category_data.values()), axis=0)
    mean_attention = all_data.mean(axis=0)  # shape: (n_layers, n_heads)

    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(
        mean_attention.T,
        cmap='YlOrRd',
        cbar_kws={'label': 'Text→Visual Attention'},
        ax=ax,
        vmin=0,
        vmax=None,
        linewidths=0.1,
        linecolor='gray'
    )

    ax.set_xlabel('Layer Index', fontsize=14)
    ax.set_ylabel('Attention Head Index', fontsize=14)
    ax.set_title(
        f'Overall Text→Visual Attention Distribution\n({all_data.shape[0]} samples, {len(category_data)} categories)',
        fontsize=16, pad=20)

    num_layers, num_heads = mean_attention.shape
    ax.set_xticks(np.arange(0, num_layers, 4) + 0.5)
    ax.set_xticklabels(np.arange(0, num_layers, 4))
    ax.set_yticks(np.arange(0, num_heads, 4) + 0.5)
    ax.set_yticklabels(np.arange(0, num_heads, 4))

    plt.tight_layout()

    output_path = os.path.join(output_dir, "overall_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved overall heatmap: {output_path}")


def plot_layer_comparison(category_data: Dict[str, np.ndarray], output_dir: str):
    """
    绘制多个category的层级对比图
    """
    categories = sorted(category_data.keys())
    num_categories = len(categories)

    # 计算每个category的层平均值
    layer_means = {}
    for category in categories:
        data = category_data[category]
        # 对每一层，计算所有样本所有头的平均值
        layer_means[category] = data.mean(axis=(0, 2))  # shape: (n_layers,)

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))

    for category in categories:
        means = layer_means[category]
        ax.plot(range(len(means)), means, marker='o', markersize=4,
                label=category, linewidth=2, alpha=0.7)

    ax.set_xlabel('Layer Index', fontsize=14)
    ax.set_ylabel('Average Text→Visual Attention', fontsize=14)
    ax.set_title('Layer-wise Attention Comparison Across Categories', fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, len(means), 4))

    plt.tight_layout()

    output_path = os.path.join(output_dir, "layer_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved layer comparison plot: {output_path}")


def save_statistics(category_data: Dict[str, np.ndarray], output_dir: str):
    """
    保存统计数据到JSON文件
    """
    stats = {}

    for category, data in category_data.items():
        mean_attention = data.mean(axis=0)  # shape: (n_layers, n_heads)

        stats[category] = {
            'num_samples': int(data.shape[0]),
            'mean_per_layer': mean_attention.mean(axis=1).tolist(),  # 每层的平均值
            'mean_per_head': mean_attention.mean(axis=0).tolist(),  # 每个头的平均值
            'overall_mean': float(mean_attention.mean()),
            'overall_std': float(mean_attention.std()),
            'overall_min': float(mean_attention.min()),
            'overall_max': float(mean_attention.max()),
        }

    # 保存到JSON
    output_path = os.path.join(output_dir, "attention_statistics.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved statistics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MME attention distribution by category")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="Path to LLaVA model")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to MME images folder")
    parser.add_argument("--question-file", type=str, required=True,
                        help="Path to MME_test.json")
    parser.add_argument("--output-dir", type=str, default="mme_attention_distribution",
                        help="Output directory")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Number of samples to process (-1 for all)")
    args = parser.parse_args()

    # 初始化模型
    print("=" * 60)
    print("Initializing model...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # 加载数据
    print(f"Loading questions from {args.question_file}...")
    with open(os.path.expanduser(args.question_file), 'r') as f:
        questions = json.load(f)

    if args.num_samples > 0:
        questions = questions[:args.num_samples]

    print(f"Total samples to process: {len(questions)}")
    print("=" * 60)

    # 处理所有样本
    results = []
    for sample in tqdm(questions, desc="Processing samples"):
        result = process_single_sample(model, tokenizer, image_processor,
                                       sample, args, device)
        if result is not None:
            results.append(result)

    print(f"\nSuccessfully processed {len(results)} samples")

    # 按category聚合
    print("\nAggregating data by category...")
    category_data = aggregate_by_category(results)

    print(f"Found {len(category_data)} categories:")
    for cat, data in sorted(category_data.items()):
        print(f"  - {cat}: {data.shape[0]} samples")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存原始数据
    print("\nSaving raw data...")
    torch.save({
        'results': results,
        'category_data': category_data,
    }, os.path.join(args.output_dir, "attention_distribution_data.pt"))

    # 生成可视化
    print("\nGenerating visualizations...")
    plot_overall_heatmap(category_data, args.output_dir)
    plot_category_heatmap(category_data, args.output_dir)
    plot_layer_comparison(category_data, args.output_dir)

    # 保存统计数据
    print("\nSaving statistics...")
    save_statistics(category_data, args.output_dir)

    print("\n" + "=" * 60)
    print(f"All results saved to: {args.output_dir}")
    print(f"Total samples processed: {len(results)}")
    print(f"Total categories: {len(category_data)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
