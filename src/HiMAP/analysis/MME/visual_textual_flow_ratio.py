import torch
import numpy as np
import os
from collections import defaultdict

# 结果文件路径
result_file = '/root/nfs/code/HiMAP/output_example/mme_saliency_results.pt'

# 加载结果
results = torch.load(result_file)

# 按子任务分类收集 Visual-Textual Flow（props_img 的第1维）
cat2props = defaultdict(list)
for r in results:
    cat = r['category']
    # props_img: (num_layers, 2), 1: Visual-Textual Flow
    props_img = np.array(r['props_img'])  # (num_layers, 2)
    vt_flow = props_img[:, 1]  # (num_layers,)
    cat2props[cat].append(vt_flow)

# 统计每个子任务的层数
cat2mean = {}
for cat, flows in cat2props.items():
    flows = np.stack(flows, axis=0)  # (N, num_layers)
    mean_flow = flows.mean(axis=0)   # (num_layers,)
    cat2mean[cat] = mean_flow

# 计算每层 Visual-Textual Flow 占当前层及之前所有层总和的比例
cat2ratio = {}
for cat, mean_flow in cat2mean.items():
    cumsum = np.cumsum(mean_flow)
    ratio = mean_flow / cumsum
    cat2ratio[cat] = ratio

# 输出

print('每个子任务的Visual-Textual Flow当前层占累计总和的比例:')
for cat, ratio in cat2ratio.items():
    print(f'\n{cat}')
    for i, r in enumerate(ratio):
        print(f'  Layer {i+1}: {r:.4f}')

# 可选：保存为csv
import csv
csv_path = '/root/nfs/code/HiMAP/output_example/visual_textual_flow_ratio.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    # header
    max_layers = max(len(r) for r in cat2ratio.values())
    writer.writerow(['Category'] + [f'Layer_{i+1}' for i in range(max_layers)])
    for cat, ratio in cat2ratio.items():
        row = [cat] + [f'{r:.4f}' for r in ratio]
        writer.writerow(row)
print(f'已保存csv到: {csv_path}')

# 绘图：每个子任务一张子图，三列拼接
import matplotlib.pyplot as plt

cats = list(cat2ratio.keys())
num_cats = len(cats)
cols = 3
rows = (num_cats + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)

for idx, cat in enumerate(cats):
    row, col = divmod(idx, cols)
    ax = axes[row][col]
    ratio = cat2ratio[cat]
    ax.plot(range(1, len(ratio)+1), ratio, marker='o')
    ax.set_title(cat)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Current/Accumulated')
    ax.grid(True, linestyle='--', alpha=0.5)

# 去除多余子图
for idx in range(num_cats, rows*cols):
    row, col = divmod(idx, cols)
    fig.delaxes(axes[row][col])

fig.tight_layout()
fig.savefig('/root/nfs/code/HiMAP/output_example/visual_textual_flow_ratio_grid.png')
print('已保存拼接大图到: /root/nfs/code/HiMAP/output_example/visual_textual_flow_ratio_grid.png')
