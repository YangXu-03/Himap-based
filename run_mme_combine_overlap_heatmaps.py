import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

def main():
    project_root = "/root/nfs/code/HiMAP"
    
    # Input directories
    dir_atte = os.path.join(project_root, "mme_layer_overlap_avg_results")
    dir_text = os.path.join(project_root, "mme_layer_overlap_results_text")
    dir_textweight = os.path.join(project_root, "mme_layer_overlap_results_textweight")
    
    # Output directory
    output_dir = os.path.join(project_root, "mme_layer_overlap_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of categories based on files in baseline directory
    # Filenames format: {safe_cat}_top1_heatmap.png
    files = [f for f in os.listdir(dir_atte) if f.endswith("_top1_heatmap.png")]
    categories = [f.replace("_top1_heatmap.png", "") for f in files]
    
    print(f"Found {len(categories)} categories to process.")
    
    for cat in tqdm(categories, desc="Combines Images"):
        # Define file paths for all 6 images
        # Column 1: Baseline (Overlap)
        p_base_1 = os.path.join(dir_atte, f"{cat}_top1_heatmap.png")
        p_base_10 = os.path.join(dir_atte, f"{cat}_top10_heatmap.png")
        
        # Column 2: Text-to-Image
        p_text_1 = os.path.join(dir_text, f"{cat}_top1_heatmap.png")
        p_text_10 = os.path.join(dir_text, f"{cat}_top10_heatmap.png")
        
        # Column 3: Weighted Text-to-Image
        p_weight_1 = os.path.join(dir_textweight, f"{cat}_top1_heatmap.png")
        p_weight_10 = os.path.join(dir_textweight, f"{cat}_top10_heatmap.png")
        
        # Check if all files exist
        all_paths = [p_base_1, p_base_10, p_text_1, p_text_10, p_weight_1, p_weight_10]
        if not all(os.path.exists(p) for p in all_paths):
            print(f"Missing files for category {cat}, skipping.")
            continue
            
        # Create figure
        # 2 Rows, 3 Columns
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Set main title
        fig.suptitle(f"Layer Overlap Analysis: {cat}", fontsize=16)
        
        # Column headers
        cols = ["Global Attention (Baseline)", "Text-to-Image Attention", "Text-Weighted Attention"]
        rows = ["Top 30% Tokens", "Top 50% Tokens"]
        
        # Helper to plot text
        def plot_img(ax, path, title=None, ylabel=None):
            try:
                img = mpimg.imread(path)
                ax.imshow(img)
                ax.axis('off')
                if title:
                    ax.set_title(title, fontsize=14)
                if ylabel:
                    # Creating a "ylabel" effect by placing text to the left
                    ax.text(-0.1, 0.5, ylabel, transform=ax.transAxes, 
                            rotation=90, va='center', ha='right', fontsize=14, fontweight='bold')
            except Exception as e:
                print(f"Error reading {path}: {e}")
                ax.axis('off')

        # Row 1 (Top 1%)
        plot_img(axes[0, 0], p_base_1, title=cols[0], ylabel=rows[0])
        plot_img(axes[0, 1], p_text_1, title=cols[1])
        plot_img(axes[0, 2], p_weight_1, title=cols[2])
        
        # Row 2 (Top 10%)
        plot_img(axes[1, 0], p_base_10, ylabel=rows[1])
        plot_img(axes[1, 1], p_text_10)
        plot_img(axes[1, 2], p_weight_10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        
        out_path = os.path.join(output_dir, f"{cat}_comparison.png")
        plt.savefig(out_path)
        plt.close()

    print(f"Comparison images saved to {output_dir}")

if __name__ == "__main__":
    main()
