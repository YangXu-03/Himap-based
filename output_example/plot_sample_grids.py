import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def plot_grid(image_paths, grid_shape, out_path, title=None):
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(grid_shape[1]*4, grid_shape[0]*4))
    for idx, ax in enumerate(axes.flat):
        if idx < len(image_paths):
            img = Image.open(image_paths[idx])
            ax.imshow(img)
            ax.set_title(f"Sample {idx}", fontsize=14)
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    img_dir = '/root/nfs/code/HiMAP/output_example/sample_plots'
    all_imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    
    # 前40个图像
    imgs_1 = all_imgs[:20]
    imgs_2 = all_imgs[20:40]
    
    plot_grid(imgs_1, (5, 4), '/root/nfs/code/HiMAP/output_example/sample_grid_1.png', title='Samples 0-19')
    plot_grid(imgs_2, (5, 4), '/root/nfs/code/HiMAP/output_example/sample_grid_2.png', title='Samples 20-39')
