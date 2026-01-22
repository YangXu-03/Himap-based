#!/usr/bin/env python3
"""
可视化和分析已保存的 ScienceQA 注意力头记录
生成更详细的统计图表
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns


def load_records(records_file):
    """加载保存的记录文件"""
    return torch.load(records_file)


def load_statistics(stats_file):
    """加载统计文件"""
    with open(stats_file, 'r') as f:
        return json.load(f)


def plot_top_k_heads_per_layer(layer_head_counts, num_heads, output_dir, k=5):
    """绘制每层中最常出现的 Top-K 注意力头"""
    num_layers = len(layer_head_counts)
    
    fig, axes = plt.subplots(1, 1, figsize=(16, 8))
    
    # Prepare data
    layers = []
    top_heads_data = []
    
    for layer_idx in sorted(layer_head_counts.keys()):
        head_counts = layer_head_counts[str(layer_idx)]
        # Sort by count
        sorted_heads = sorted(head_counts.items(), key=lambda x: x[1], reverse=True)
        
        layers.append(f"Layer {layer_idx}")
        
        # Get top-k heads
        top_k_heads = sorted_heads[:k]
        heads_str = ", ".join([f"H{h}({c})" for h, c in top_k_heads])
        top_heads_data.append(top_k_heads)
    
    # Plot as heatmap for top-k heads per layer
    # Create matrix: layer x top-k positions
    matrix = np.zeros((num_layers, k))
    labels = [['' for _ in range(k)] for _ in range(num_layers)]
    
    for layer_idx, top_heads in enumerate(top_heads_data):
        for rank, (head_idx, count) in enumerate(top_heads):
            if rank < k:
                matrix[layer_idx, rank] = count
                labels[layer_idx][rank] = f"H{head_idx}"
    
    # Plot
    im = axes.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Set ticks and labels
    axes.set_xticks(range(k))
    axes.set_xticklabels([f"Top {i+1}" for i in range(k)])
    axes.set_yticks(range(num_layers))
    axes.set_yticklabels([f"L{i}" for i in range(num_layers)])
    
    # Add text annotations
    for i in range(num_layers):
        for j in range(k):
            if matrix[i, j] > 0:
                text = axes.text(j, i, f"{labels[i][j]}\n{int(matrix[i, j])}", 
                               ha="center", va="center", color="black", fontsize=8)
    
    axes.set_xlabel('Rank', fontsize=12)
    axes.set_ylabel('Layer', fontsize=12)
    axes.set_title(f'Top-{k} Most Frequent Max Attention Heads per Layer', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes)
    cbar.set_label('Count', fontsize=12)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'top_{k}_heads_per_layer.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved top-{k} heads plot to {output_file}")
    plt.close()


def plot_head_frequency_across_layers(layer_head_counts, num_heads, num_layers, output_dir):
    """绘制每个注意力头在所有层中出现为最大值的总次数"""
    head_total_counts = defaultdict(int)
    
    for layer_idx in range(num_layers):
        head_counts = layer_head_counts.get(str(layer_idx), {})
        for head_idx, count in head_counts.items():
            head_total_counts[int(head_idx)] += count
    
    # Prepare data
    head_indices = list(range(num_heads))
    counts = [head_total_counts.get(head_idx, 0) for head_idx in head_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(head_indices, counts, color='steelblue', alpha=0.7)
    
    # Highlight top heads
    top_k = 5
    top_head_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)[:top_k]
    for idx in top_head_indices:
        bars[idx].set_color('red')
        bars[idx].set_alpha(0.8)
    
    ax.set_xlabel('Attention Head Index', fontsize=12)
    ax.set_ylabel('Total Count Across All Layers', fontsize=12)
    ax.set_title('Attention Head Frequency (Total Count as Max Attention Head)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Regular heads'),
        Patch(facecolor='red', alpha=0.8, label=f'Top-{top_k} heads')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'head_frequency_total.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved head frequency plot to {output_file}")
    plt.close()
    
    # Print top heads
    print(f"\nTop-{top_k} most frequent max attention heads (across all layers):")
    for rank, idx in enumerate(top_head_indices, 1):
        print(f"  {rank}. Head {idx}: {counts[idx]} times")


def plot_layer_diversity(layer_head_counts, num_layers, output_dir):
    """绘制每一层的注意力头多样性（有多少不同的头成为过最大值）"""
    layer_diversity = []
    layer_entropy = []
    
    for layer_idx in range(num_layers):
        head_counts = layer_head_counts.get(str(layer_idx), {})
        num_unique_heads = len(head_counts)
        layer_diversity.append(num_unique_heads)
        
        # Calculate entropy
        total = sum(head_counts.values())
        if total > 0:
            probs = np.array(list(head_counts.values())) / total
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            layer_entropy.append(entropy)
        else:
            layer_entropy.append(0)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Diversity plot
    ax1.bar(range(num_layers), layer_diversity, color='teal', alpha=0.7)
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Number of Unique Max Heads', fontsize=12)
    ax1.set_title('Attention Head Diversity per Layer', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Entropy plot
    ax2.bar(range(num_layers), layer_entropy, color='coral', alpha=0.7)
    ax2.set_xlabel('Layer Index', fontsize=12)
    ax2.set_ylabel('Entropy (bits)', fontsize=12)
    ax2.set_title('Attention Head Distribution Entropy per Layer', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'layer_diversity.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved layer diversity plot to {output_file}")
    plt.close()


def generate_summary_report(statistics, output_dir):
    """生成文本摘要报告"""
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    
    num_samples = statistics['num_samples']
    num_layers = statistics['num_layers']
    num_heads = statistics['num_heads']
    layer_head_counts = statistics['layer_head_counts']
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ScienceQA Attention Head Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total samples analyzed: {num_samples}\n")
        f.write(f"Number of layers: {num_layers}\n")
        f.write(f"Number of attention heads per layer: {num_heads}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("Per-Layer Statistics\n")
        f.write("-" * 80 + "\n\n")
        
        for layer_idx in range(num_layers):
            head_counts = layer_head_counts.get(str(layer_idx), {})
            if not head_counts:
                continue
            
            f.write(f"Layer {layer_idx}:\n")
            
            # Most frequent head
            max_head = max(head_counts.items(), key=lambda x: x[1])
            f.write(f"  Most frequent max head: Head {max_head[0]} ({max_head[1]} times, {max_head[1]/num_samples*100:.1f}%)\n")
            
            # Diversity
            num_unique = len(head_counts)
            f.write(f"  Number of unique max heads: {num_unique}/{num_heads} ({num_unique/num_heads*100:.1f}%)\n")
            
            # Top 3
            top_3 = sorted(head_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            f.write(f"  Top 3 heads: ")
            f.write(", ".join([f"H{h}({c})" for h, c in top_3]))
            f.write("\n\n")
    
    print(f"Saved analysis report to {report_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="./scienceqa_attention_head_analysis",
                        help="Directory containing saved records and statistics")
    args = parser.parse_args()
    
    # Load data
    records_file = os.path.join(args.input_dir, 'attention_head_records.pt')
    stats_file = os.path.join(args.input_dir, 'statistics.json')
    
    if not os.path.exists(records_file) or not os.path.exists(stats_file):
        print(f"Error: Required files not found in {args.input_dir}")
        print(f"Please run analyze_scienceqa_attention_heads.py first.")
        return
    
    print(f"Loading data from {args.input_dir}...")
    sample_records = load_records(records_file)
    statistics = load_statistics(stats_file)
    
    num_samples = statistics['num_samples']
    num_layers = statistics['num_layers']
    num_heads = statistics['num_heads']
    layer_head_counts = statistics['layer_head_counts']
    
    print(f"Loaded {num_samples} samples with {num_layers} layers and {num_heads} heads per layer")
    
    # Generate visualizations
    print("\nGenerating additional visualizations...")
    
    plot_top_k_heads_per_layer(layer_head_counts, num_heads, args.input_dir, k=5)
    plot_head_frequency_across_layers(layer_head_counts, num_heads, num_layers, args.input_dir)
    plot_layer_diversity(layer_head_counts, num_layers, args.input_dir)
    
    # Generate summary report
    generate_summary_report(statistics, args.input_dir)
    
    print(f"\nVisualization complete! All results saved to {args.input_dir}")


if __name__ == "__main__":
    main()
