#!/usr/bin/env python3
"""测试可视化脚本是否能正常工作"""

import os
import json

print("=" * 70)
print("检查 MME FastV 可视化准备工作")
print("=" * 70)
print()

# 检查结果文件
result_files = [
    'mme_results_baseline.json',
    'mme_results_fastv_max_head.json',
    'mme_results_fastv_avg_all_heads.json',
    'mme_results_fastv_text_weighted.json',
    'mme_results_fastv_text_weighted_max_head.json',
    'mme_results_fastv_weighted_combination_alpha0.3.json',
    'mme_results_fastv_weighted_combination_alpha0.5.json',
    'mme_results_fastv_weighted_combination_alpha0.7.json',
]

print("检查结果文件:")
existing_files = []
for file_path in result_files:
    if os.path.exists(file_path):
        print(f"  ✓ {file_path}")
        existing_files.append(file_path)
        
        # 检查文件格式
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            required_keys = ['total_score', 'perception_score', 'cognition_score', 'category_scores']
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                print(f"    ⚠ 缺少字段: {missing_keys}")
            else:
                print(f"    - 总分: {data['total_score']:.2f}, "
                      f"感知: {data['perception_score']:.2f}, "
                      f"认知: {data['cognition_score']:.2f}")
        except Exception as e:
            print(f"    ✗ 文件格式错误: {e}")
    else:
        print(f"  ✗ {file_path} (未找到)")

print()
print(f"找到 {len(existing_files)}/{len(result_files)} 个结果文件")
print()

if len(existing_files) >= 2:
    print("✓ 可以运行可视化脚本了！")
    print()
    print("运行命令:")
    print("  python ./src/HiMAP/inference/visualize_mme_fastv_results.py")
    print()
elif len(existing_files) == 0:
    print("✗ 没有找到任何结果文件")
    print()
    print("请先运行评估:")
    print("  bash src/HiMAP/inference/eval_mme_fastv_advanced.sh")
    print()
else:
    print(f"⚠ 只找到 {len(existing_files)} 个结果文件，可视化效果可能不理想")
    print()
    print("建议运行完整评估:")
    print("  bash src/HiMAP/inference/eval_mme_fastv_advanced.sh")
    print()

# 测试可视化脚本的导入
print("检查可视化脚本依赖:")
try:
    import matplotlib.pyplot as plt
    print("  ✓ matplotlib")
except ImportError:
    print("  ✗ matplotlib (未安装)")

try:
    import numpy as np
    print("  ✓ numpy")
except ImportError:
    print("  ✗ numpy (未安装)")

try:
    import seaborn as sns
    print("  ✓ seaborn")
except ImportError:
    print("  ✗ seaborn (未安装)")

print()
print("=" * 70)
