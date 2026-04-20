import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    source_dir = '/root/nfs/code/HiMAP/attn_heatmaps_promptweighted'
    target_dir = '/root/nfs/code/HiMAP/attn_heatmaps_promptweighted_composed'
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all images
    files = glob.glob(os.path.join(source_dir, 'heatmap_promptweighted_*_top*.png'))
    
    # Extract subtasks
    subtasks = set()
    for f in files:
        basename = os.path.basename(f)
        # Format: heatmap_promptweighted_{subtask}_top{X}pct.png
        prefix = 'heatmap_promptweighted_'
        if basename.startswith(prefix):
            remainder = basename[len(prefix):]
            parts = remainder.rsplit('_', 1)
            if len(parts) == 2:
                subtask = parts[0]
                subtasks.add(subtask)
                
    percentages = ['1pct', '10pct', '20pct', '50pct']
    
    for subtask in subtasks:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f"Subtask: {subtask}", fontsize=16)
        
        for i, pct in enumerate(percentages):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            img_path = os.path.join(source_dir, f'heatmap_promptweighted_{subtask}_top{pct}.png')
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.set_title(f'Top {pct}')
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center')
                ax.set_title(f'Top {pct}')
                ax.axis('off')
                
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        out_path = os.path.join(target_dir, f'composed_{subtask}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
