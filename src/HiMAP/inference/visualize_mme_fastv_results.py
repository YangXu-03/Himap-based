"""
FastV Advanced MME 结果可视化脚本

读取多个 FastV Advanced 方法的 MME 评测结果，生成对比图表：
1. 总分对比柱状图
2. Perception vs Cognition 分数对比
3. 各个子任务详细分数对比
4. 不同方法的相对性能热力图
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 结果文件列表
RESULT_FILES = [
    'mme_results_baseline.json',
    'mme_results_fastv_max_head.json',
    'mme_results_fastv_avg_all_heads.json',
    'mme_results_fastv_text_weighted.json',
    'mme_results_fastv_text_weighted_max_head.json',
    'mme_results_fastv_weighted_combination_alpha0.3.json',
    'mme_results_fastv_weighted_combination_alpha0.5.json',
    'mme_results_fastv_weighted_combination_alpha0.7.json',
]

# 方法名称映射
METHOD_NAMES = {
    'baseline': 'Baseline',
    'max_head': 'Max Head',
    'avg_all_heads': 'Avg All Heads',
    'text_weighted': 'Text Weighted',
    'text_weighted_max_head': 'Text Weighted Max Head',
    'weighted_combination_alpha0.3': 'Weighted Combination (α=0.3)',
    'weighted_combination_alpha0.5': 'Weighted Combination (α=0.5)',
    'weighted_combination_alpha0.7': 'Weighted Combination (α=0.7)',
}

def load_results(result_files):
    """加载所有结果文件"""
    results = {}
    
    print(f"\n检查 {len(result_files)} 个结果文件:")
    for file_path in result_files:
        if not os.path.exists(file_path):
            print(f"  ✗ {file_path} - 文件不存在")
            continue
        print(f"  ✓ {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 提取方法名称（注意：更具体的模式要先匹配）
        if 'baseline' in file_path:
            method_name = 'baseline'
        elif 'text_weighted_max_head' in file_path:  # 必须在 'max_head' 之前
            method_name = 'text_weighted_max_head'
        elif 'max_head' in file_path:
            method_name = 'max_head'
        elif 'avg_all_heads' in file_path:
            method_name = 'avg_all_heads'
        elif 'text_weighted' in file_path:
            method_name = 'text_weighted'
        elif 'weighted_combination_alpha0.3' in file_path:
            method_name = 'weighted_combination_alpha0.3'
        elif 'weighted_combination_alpha0.5' in file_path:
            method_name = 'weighted_combination_alpha0.5'
        elif 'weighted_combination_alpha0.7' in file_path:
            method_name = 'weighted_combination_alpha0.7'
        else:
            method_name = Path(file_path).stem
            
        results[method_name] = data
        
    return results

def plot_total_scores(results, output_dir='mme_fastv_visualizations'):
    """绘制总分对比柱状图"""
    os.makedirs(output_dir, exist_ok=True)
    
    methods = []
    total_scores = []
    perception_scores = []
    cognition_scores = []
    
    # 确保 baseline 排在第一位
    if 'baseline' in results:
        methods.append(METHOD_NAMES.get('baseline', 'baseline'))
        total_scores.append(results['baseline']['total_score'])
        perception_scores.append(results['baseline']['perception_score'])
        cognition_scores.append(results['baseline']['cognition_score'])
    
    # 添加其他方法
    for method, data in results.items():
        if method == 'baseline':
            continue
        methods.append(METHOD_NAMES.get(method, method))
        total_scores.append(data['total_score'])
        perception_scores.append(data['perception_score'])
        cognition_scores.append(data['cognition_score'])
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 总分柱状图
    x = np.arange(len(methods))
    width = 0.6
    
    bars = ax1.bar(x, total_scores, width, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total MME Score', fontsize=12, fontweight='bold')
    ax1.set_title('MME Total Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱子上标注数值
    for i, (bar, score) in enumerate(zip(bars, total_scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Perception vs Cognition 堆叠柱状图
    width = 0.6
    bars1 = ax2.bar(x, perception_scores, width, label='Perception', color='coral', alpha=0.8)
    bars2 = ax2.bar(x, cognition_scores, width, bottom=perception_scores, 
                    label='Cognition', color='skyblue', alpha=0.8)
    
    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Perception vs Cognition Score Breakdown', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    # 标注总分
    for i, (p_score, c_score) in enumerate(zip(perception_scores, cognition_scores)):
        total = p_score + c_score
        ax2.text(i, total, f'{total:.1f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/total_scores_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/total_scores_comparison.pdf', bbox_inches='tight')
    print(f"总分对比图已保存到: {output_dir}/total_scores_comparison.png")
    plt.close()

def plot_category_scores(results, output_dir='mme_fastv_visualizations'):
    """绘制各个子任务的详细分数对比"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有类别
    all_categories = set()
    for data in results.values():
        all_categories.update(data['category_scores'].keys())
    
    categories = sorted(list(all_categories))
    
    # 准备数据
    method_names = []
    scores_matrix = []
    
    # 确保 baseline 排在第一位
    if 'baseline' in results:
        method_names.append(METHOD_NAMES.get('baseline', 'baseline'))
        scores = [results['baseline']['category_scores'].get(cat, {}).get('score', 0) 
                 for cat in categories]
        scores_matrix.append(scores)
    
    # 添加其他方法
    for method, data in results.items():
        if method == 'baseline':
            continue
        method_names.append(METHOD_NAMES.get(method, method))
        scores = [data['category_scores'].get(cat, {}).get('score', 0) 
                 for cat in categories]
        scores_matrix.append(scores)
    
    # 创建分组柱状图
    fig, ax = plt.subplots(figsize=(18, 8))
    
    x = np.arange(len(categories))
    width = 0.13  # 每个柱子的宽度
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(method_names)))
    
    for i, (method, scores, color) in enumerate(zip(method_names, scores_matrix, colors)):
        offset = width * (i - len(method_names)/2 + 0.5)
        ax.bar(x + offset, scores, width, label=method, color=color, alpha=0.8)
    
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Category-wise Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/category_scores_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/category_scores_comparison.pdf', bbox_inches='tight')
    print(f"子任务分数对比图已保存到: {output_dir}/category_scores_comparison.png")
    plt.close()

def plot_heatmap(results, output_dir='mme_fastv_visualizations'):
    """绘制不同方法在各个类别上的相对性能热力图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有类别
    all_categories = set()
    for data in results.values():
        all_categories.update(data['category_scores'].keys())
    
    categories = sorted(list(all_categories))
    
    # 准备数据
    method_names = []
    scores_matrix = []
    
    # 确保 baseline 排在第一位
    if 'baseline' in results:
        method_names.append(METHOD_NAMES.get('baseline', 'baseline'))
        scores = [results['baseline']['category_scores'].get(cat, {}).get('score', 0) 
                 for cat in categories]
        scores_matrix.append(scores)
    
    # 添加其他方法
    for method, data in results.items():
        if method == 'baseline':
            continue
        method_names.append(METHOD_NAMES.get(method, method))
        scores = [data['category_scores'].get(cat, {}).get('score', 0) 
                 for cat in categories]
        scores_matrix.append(scores)
    
    scores_matrix = np.array(scores_matrix)
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(16, 8))
    
    im = ax.imshow(scores_matrix, cmap='YlOrRd', aspect='auto')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_yticklabels(method_names)
    
    # 添加数值标注
    for i in range(len(method_names)):
        for j in range(len(categories)):
            text = ax.text(j, i, f'{scores_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Performance Heatmap Across Categories', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/performance_heatmap.pdf', bbox_inches='tight')
    print(f"性能热力图已保存到: {output_dir}/performance_heatmap.png")
    plt.close()

def plot_relative_performance(results, output_dir='mme_fastv_visualizations'):
    """绘制相对于baseline的性能变化"""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'baseline' not in results:
        print("Warning: No baseline results found, skipping relative performance plot")
        return
    
    baseline_score = results['baseline']['total_score']
    
    methods = []
    relative_scores = []
    colors = []
    
    for method, data in results.items():
        if method == 'baseline':
            continue
        methods.append(METHOD_NAMES.get(method, method))
        diff = data['total_score'] - baseline_score
        relative_scores.append(diff)
        colors.append('green' if diff >= 0 else 'red')
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(methods))
    bars = ax.bar(x, relative_scores, color=colors, alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score Difference vs Baseline', fontsize=12, fontweight='bold')
    ax.set_title('Relative Performance Compared to Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱子上标注数值
    for bar, score in zip(bars, relative_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:+.1f}',
                ha='center', va='bottom' if score >= 0 else 'top',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/relative_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/relative_performance.pdf', bbox_inches='tight')
    print(f"相对性能图已保存到: {output_dir}/relative_performance.png")
    plt.close()

def generate_summary_table(results, output_dir='mme_fastv_visualizations'):
    """生成汇总表格"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备表格数据
    table_data = []
    
    # 确保 baseline 排在第一位
    ordered_methods = ['baseline'] if 'baseline' in results else []
    ordered_methods += [m for m in results.keys() if m != 'baseline']
    
    for method in ordered_methods:
        data = results[method]
        table_data.append([
            METHOD_NAMES.get(method, method),
            f"{data['total_score']:.2f}",
            f"{data['perception_score']:.2f}",
            f"{data['cognition_score']:.2f}",
            f"{data.get('avg_latency', 0):.4f}s" if 'avg_latency' in data else 'N/A'
        ])
    
    # 创建表格图
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data,
                    colLabels=['Method', 'Total Score', 'Perception', 'Cognition', 'Avg Latency'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title('MME FastV Advanced Methods Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(f'{output_dir}/summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/summary_table.pdf', bbox_inches='tight')
    print(f"汇总表格已保存到: {output_dir}/summary_table.png")
    plt.close()

def main():
    print("=" * 60)
    print("FastV Advanced MME 结果可视化")
    print("=" * 60)
    
    # 加载结果
    print("\n正在加载结果文件...")
    results = load_results(RESULT_FILES)
    
    if not results:
        print("Error: No valid result files found!")
        return
    
    print(f"成功加载 {len(results)} 个结果文件:")
    for method in results.keys():
        print(f"  - {METHOD_NAMES.get(method, method)}")
    
    # 生成各种可视化
    print("\n生成可视化图表...")
    output_dir = 'mme_fastv_visualizations'
    
    plot_total_scores(results, output_dir)
    plot_category_scores(results, output_dir)
    plot_heatmap(results, output_dir)
    plot_relative_performance(results, output_dir)
    generate_summary_table(results, output_dir)
    
    print("\n" + "=" * 60)
    print(f"所有可视化图表已保存到: {output_dir}/")
    print("=" * 60)
    
    # 打印简要统计
    print("\n简要统计:")
    print(f"{'Method':<35} {'Total':<12} {'Perception':<12} {'Cognition':<12}")
    print("-" * 71)
    
    # 确保 baseline 排在第一位
    ordered_methods = ['baseline'] if 'baseline' in results else []
    ordered_methods += [m for m in results.keys() if m != 'baseline']
    
    for method in ordered_methods:
        data = results[method]
        print(f"{METHOD_NAMES.get(method, method):<35} "
              f"{data['total_score']:<12.2f} "
              f"{data['perception_score']:<12.2f} "
              f"{data['cognition_score']:<12.2f}")
    
    print("-" * 71)

if __name__ == "__main__":
    main()
