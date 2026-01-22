#!/usr/bin/env python3
"""
比较 FastV Advanced 不同策略的推理结果
"""

import json
import os
from typing import Dict, List
import sys


def load_result(filename: str) -> Dict:
    """加载结果文件"""
    if not os.path.exists(filename):
        return None
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_results():
    """比较所有结果"""
    
    # 要比较的结果文件
    result_files = [
        ("Baseline", "scienceqa_results_baseline.json"),
        ("FastV - max_head", "scienceqa_results_fastv_max_head.json"),
        ("FastV - avg_all_heads", "scienceqa_results_fastv_avg_all_heads.json"),
        ("FastV - weighted α=0.3", "scienceqa_results_fastv_weighted_combination_alpha0.3.json"),
        ("FastV - weighted α=0.5", "scienceqa_results_fastv_weighted_combination_alpha0.5.json"),
        ("FastV - weighted α=0.7", "scienceqa_results_fastv_weighted_combination_alpha0.7.json"),
    ]
    
    print("=" * 100)
    print(" " * 30 + "FastV Advanced 结果比较")
    print("=" * 100)
    
    # 加载所有结果
    results = []
    for name, filename in result_files:
        result = load_result(filename)
        if result is not None:
            results.append((name, result))
        else:
            print(f"⚠️  警告: 未找到 {filename}")
    
    if not results:
        print("❌ 没有找到任何结果文件")
        return
    
    # 表头
    print(f"\n{'策略':<30} {'准确率':<12} {'平均延迟(s)':<15} {'FLOPs比例':<12} {'样本数':<10}")
    print("-" * 100)
    
    # 显示结果
    baseline_acc = None
    for name, result in results:
        accuracy = result.get('accuracy', 0)
        latency = result.get('avg_latency', 0)
        flops = result.get('flops_info', 1.0)
        samples = result.get('total_samples', 0)
        
        if 'Baseline' in name:
            baseline_acc = accuracy
        
        # 计算相对于baseline的变化
        if baseline_acc is not None and baseline_acc > 0 and 'Baseline' not in name:
            acc_diff = accuracy - baseline_acc
            acc_str = f"{accuracy:.4f} ({acc_diff:+.4f})"
        else:
            acc_str = f"{accuracy:.4f}"
        
        print(f"{name:<30} {acc_str:<12} {latency:<15.6f} {flops*100:<11.2f}% {samples:<10}")
    
    print("-" * 100)
    
    # 找出最佳策略
    if len(results) > 1:
        print("\n📊 策略分析:")
        
        # 按准确率排序
        sorted_by_acc = sorted(results, key=lambda x: x[1]['accuracy'], reverse=True)
        print(f"\n最高准确率: {sorted_by_acc[0][0]} ({sorted_by_acc[0][1]['accuracy']:.4f})")
        
        # 按延迟排序
        sorted_by_latency = sorted(results, key=lambda x: x[1]['avg_latency'])
        print(f"最低延迟: {sorted_by_latency[0][0]} ({sorted_by_latency[0][1]['avg_latency']:.6f}s)")
        
        # 综合性能（准确率 vs 延迟）
        print("\n💡 建议:")
        baseline_idx = next((i for i, (name, _) in enumerate(results) if 'Baseline' in name), None)
        if baseline_idx is not None:
            baseline_name, baseline_result = results[baseline_idx]
            baseline_acc = baseline_result['accuracy']
            baseline_latency = baseline_result['avg_latency']
            
            print(f"  Baseline: 准确率={baseline_acc:.4f}, 延迟={baseline_latency:.6f}s")
            
            # 找出在保持相近准确率的情况下最快的策略
            for name, result in results:
                if 'Baseline' not in name:
                    acc_loss = baseline_acc - result['accuracy']
                    speedup = baseline_latency / result['avg_latency'] if result['avg_latency'] > 0 else 1.0
                    
                    if acc_loss <= 0.01:  # 准确率损失 <= 1%
                        print(f"  ✓ {name}: 准确率损失={acc_loss:.4f} ({acc_loss*100:.2f}%), 加速={speedup:.2f}x")
    
    # 详细配置信息
    print("\n" + "=" * 100)
    print("📋 详细配置:")
    print("=" * 100)
    
    for name, result in results:
        config = result.get('model_config', {})
        print(f"\n{name}:")
        
        if config.get('use_fastv'):
            print(f"  - Token Selection Method: {config.get('fast_v_token_selection_method', 'N/A')}")
            if config.get('fast_v_token_selection_method') == 'weighted_combination':
                print(f"  - Weighted Alpha: {config.get('fast_v_weighted_alpha', 'N/A')}")
            print(f"  - System Length: {config.get('fast_v_sys_length', 'N/A')}")
            print(f"  - Image Token Length: {config.get('fast_v_image_token_length', 'N/A')}")
            print(f"  - Attention Rank: {config.get('fast_v_attention_rank', 'N/A')}")
            print(f"  - Aggregation Layer: {config.get('fast_v_agg_layer', 'N/A')}")
            
            img_len = config.get('fast_v_image_token_length', 1)
            rank = config.get('fast_v_attention_rank', 0)
            if img_len > 0:
                pruning_ratio = (1 - rank / img_len) * 100
                print(f"  - Token Pruning Ratio: {pruning_ratio:.1f}%")
        elif config.get('use_himap'):
            print(f"  - Using HiMAP")
        else:
            print(f"  - No pruning (Baseline)")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    compare_results()
