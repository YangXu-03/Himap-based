import os
import matplotlib.pyplot as plt
from PIL import Image

# 感知任务和认知任务分类
perception_tasks = [
    "existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"
]
cognition_tasks = [
    "commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"
]

# 图像文件夹
img_dir = "./output_example/mme_saliency_plots"

# 合并绘图函数
def plot_tasks_grid(tasks, suffix, out_path, ncols=3):
    n = len(tasks)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
    for idx, task in enumerate(tasks):
        row, col = divmod(idx, ncols)
        ax = axes[row, col] if nrows > 1 else axes[col]
        img_path = os.path.join(img_dir, f"{task}_{suffix}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(task, fontsize=16)
            ax.axis('off')
        else:
            ax.axis('off')
            ax.set_title(f"{task}\n(No Image)", fontsize=12)
    # 多余的子图隐藏
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col] if nrows > 1 else axes[col]
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()

# 合并感知+认知任务
tasks_order = perception_tasks + cognition_tasks
plot_tasks_grid(tasks_order, "props4all", os.path.join(img_dir, "all_tasks_props4all_grid.png"))
plot_tasks_grid(tasks_order, "props4img", os.path.join(img_dir, "all_tasks_props4img_grid.png"))
