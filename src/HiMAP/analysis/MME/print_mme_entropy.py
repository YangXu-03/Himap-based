import torch
import os

# 结果文件路径
result_path = '/root/nfs/code/HiMAP/output_example/mme_observation_results/observation_results.pt'

# 加载结果
data = torch.load(result_path)

print('各子任务的跨模态注意力熵（text_vis_attn_entropy）:')
for cat, metrics_list in data.items():
    # metrics_list: List[Dict]
    # 每个metrics_list元素有 'text_vis_attn_entropy' (list, 每层一个值)
    entropy_arr = [m['text_vis_attn_entropy'] for m in metrics_list if 'text_vis_attn_entropy' in m]
    if not entropy_arr:
        continue
    entropy_arr = torch.tensor(entropy_arr)  # shape: (num_samples, num_layers)
    mean_entropy = entropy_arr.mean(dim=0)   # shape: (num_layers,)
    print(f'子任务: {cat}')
    print('各层平均熵:', ', '.join(f'{v:.4f}' for v in mean_entropy.tolist()))
    print('整体平均熵: {:.4f}'.format(mean_entropy.mean().item()))
    print('-'*40)
