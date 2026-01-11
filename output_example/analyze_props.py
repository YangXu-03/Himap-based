import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_single_sample_props4img(data, path, sample_idx):
    # data shape: (2, num_layers)
    
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min > 1e-9:
        data = (data - data_min) / (data_max - data_min)
    else:
        data = data - data_min

    x = np.arange(data.shape[1])
    colors = ['#007c9a', '#965e9b']
    custom_legend_labels = ['Intra-Visual Flow', 'Visual-Textual Flow']

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.bar(x, data[0], label=custom_legend_labels[0], alpha=0.3, color=colors[0], width=0.7)
    ax.bar(x, data[1], label=custom_legend_labels[1], alpha=0.3, color=colors[1], width=0.7)

    ax.set_xlabel('Transformer Layer', fontsize=24, fontfamily='Times New Roman', labelpad=12)
    ax.set_ylabel('Importance Metric', fontsize=24, fontfamily='Times New Roman', labelpad=12)
    ax.set_title(f'Sample {sample_idx}', fontsize=24, fontfamily='Times New Roman')
    plt.xticks(fontsize=24, fontfamily='Times New Roman')
    plt.yticks(fontsize=24, fontfamily='Times New Roman')
    ax.legend(fontsize=24, fancybox=True, loc='upper right',
                   prop={'size':24, 'family': 'Times New Roman','style': 'italic'})

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def analyze_distribution(props_img_array):
    # props_img_array shape: (num_samples, num_layers, 2)
    
    mean_props = np.mean(props_img_array, axis=0) # (num_layers, 2)
    std_props = np.std(props_img_array, axis=0)   # (num_layers, 2)
    
    print(f"Analyzed {props_img_array.shape[0]} samples.")
    print(f"Data shape: {props_img_array.shape}")
    
    print("\n--- Statistics for props_img (Intra-Visual Flow) ---")
    print(f"{'Layer':<10} {'Mean':<15} {'Std':<15} {'CV':<15}")
    for i in range(mean_props.shape[0]):
        m = mean_props[i, 0]
        s = std_props[i, 0]
        cv = s / m if m != 0 else 0
        print(f"{i:<10} {m:<15.6f} {s:<15.6f} {cv:<15.6f}")
        
    print("\n--- Statistics for props_img (Visual-Textual Flow) ---")
    print(f"{'Layer':<10} {'Mean':<15} {'Std':<15} {'CV':<15}")
    for i in range(mean_props.shape[0]):
        m = mean_props[i, 1]
        s = std_props[i, 1]
        cv = s / m if m != 0 else 0
        print(f"{i:<10} {m:<15.6f} {s:<15.6f} {cv:<15.6f}")

if __name__ == "__main__":
    file_path = '/root/nfs/code/HiMAP/output_example/scivqa_props-7b.pt'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit(1)
        
    proportions = torch.load(file_path)

    props_img_list = []
    for i, props in enumerate(proportions):
        _, props4img = props
        props_img_list.append(props4img)

    props_img_array = np.array(props_img_list) # (num_samples, num_layers, 2)
    
    output_dir = '/root/nfs/code/HiMAP/output_example/sample_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_samples_to_plot = min(50, len(props_img_list))
    print(f"Plotting first {num_samples_to_plot} samples to {output_dir}...")
    
    for i in range(num_samples_to_plot):
        # Transpose to (2, num_layers) for plotting
        sample_data = props_img_array[i].transpose(1, 0)
        plot_path = os.path.join(output_dir, f'sample_{i:03d}.png')
        plot_single_sample_props4img(sample_data, plot_path, i)
        
    analyze_distribution(props_img_array)
