import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# 读取实验结果
with open("/root/nfs/code/HiMAP/mme_layer_cutoff_results.json", "r") as f:
    exp = json.load(f)

# 读取基线分数
from HiMAP.analysis.MME.saliency_mme import calculate_mme_scores
base_results = torch.load("/root/nfs/code/HiMAP/output_example/mme_saliency_results.pt")
_, base_perc, base_cog = calculate_mme_scores(base_results)
base_scores = {k: v for k, v in _ .items()}
base_total = base_perc + base_cog

# 1. 绘制总分、感知分、认知分
layers = exp["layers"] + [33]
total_scores = exp["total_scores"] + [base_total]
perception_scores = exp["perception_scores"] + [base_perc]
cognition_scores = exp["cognition_scores"] + [base_cog]

plt.figure(figsize=(12, 6))
plt.plot(layers, total_scores, marker='o', label='Total MME Score', linewidth=2)
plt.plot(layers, perception_scores, marker='s', label='Perception Score', linewidth=2)
plt.plot(layers, cognition_scores, marker='^', label='Cognition Score', linewidth=2)
plt.xlabel('Layer Index (Image Token Cutoff Layer)', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('MME Scores vs Layer Cutoff (with Baseline)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("/root/nfs/code/HiMAP/output_example/mme_layer_cutoff_plot_with_baseline.png", dpi=300, bbox_inches='tight')

# 2. 各子任务分数
subtasks = list(exp['subtask_scores'][0].keys())
subtask_scores = {k: [d.get(k, 0) for d in exp['subtask_scores']] for k in subtasks}
# 基线各子任务
for k in subtasks:
    subtask_scores[k].append(base_scores.get(k, 0))

plt.figure(figsize=(14, 8))
colors = plt.cm.tab20(np.linspace(0, 1, len(subtasks)))
for i, task in enumerate(subtasks):
    plt.plot(layers, subtask_scores[task], marker='.', label=task, color=colors[i], linewidth=1.5, alpha=0.8)
plt.xlabel('Layer Index (Image Token Cutoff Layer)', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('MME Subtask Scores vs Layer Cutoff (with Baseline)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig("/root/nfs/code/HiMAP/output_example/mme_layer_cutoff_plot_subtasks_with_baseline.png", dpi=300, bbox_inches='tight')
print('Done!')
