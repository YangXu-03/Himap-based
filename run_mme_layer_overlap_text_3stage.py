import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
project_root = "/root/nfs/code/HiMAP"
sys.path.append(os.path.join(project_root, "src"))
sys.path.append(os.path.join(project_root, "src/LLaVA"))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_mme_samples(mme_json_path, sample_index=0):
    """
    Load MME data and select one sample per category (subtask).
    The selected sample is controlled by sample_index per category.
    Returns a dictionary: {category: sample_item}
    """
    print(f"Loading MME data from {mme_json_path}...")
    try:
        with open(mme_json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: MME file not found at {mme_json_path}")
        return {}

    samples_by_category = {}
    for item in data:
        cat = item.get("category", "unknown")
        if cat not in samples_by_category:
            samples_by_category[cat] = []
        samples_by_category[cat].append(item)

    selected_samples = {}
    for cat, items in samples_by_category.items():
        if not items:
            continue

        if sample_index < len(items):
            selected_samples[cat] = items[sample_index]
        else:
            # Fallback to the last sample if requested index is out of range.
            selected_samples[cat] = items[-1]
            print(
                f"Warning: category '{cat}' has only {len(items)} samples; "
                f"fallback to last sample for requested index {sample_index}."
            )

    print(
        f"Selected {len(selected_samples)} samples (one per category), "
        f"sample_index={sample_index}."
    )
    return selected_samples


def get_overlap_ratio(set_a, set_b):
    """
    Compute overlap ratio: size(intersection) / size(set_a)
    """
    if len(set_a) == 0:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    return intersection / len(set_a)


def parse_three_stage_config(stage_boundaries_str, stage_topk_pcts_str):
    boundaries = [int(x.strip()) for x in stage_boundaries_str.split(",") if x.strip()]
    topk_pcts = [float(x.strip()) for x in stage_topk_pcts_str.split(",") if x.strip()]

    if len(boundaries) != 2:
        raise ValueError("--stage-boundaries must contain exactly 2 integers, e.g. 10,20")
    if len(topk_pcts) != 3:
        raise ValueError("--stage-topk-pcts must contain exactly 3 numbers, e.g. 30,50,70")

    b1, b2 = boundaries
    if b1 < 0 or b2 < 0:
        raise ValueError("Stage boundaries must be non-negative")
    if b1 >= b2:
        raise ValueError("Stage boundaries must satisfy boundary1 < boundary2")

    for p in topk_pcts:
        if p <= 0 or p > 100:
            raise ValueError("Each topk percentage must be in (0, 100]")

    return boundaries, topk_pcts


def get_stage_id(layer_idx, boundaries):
    b1, b2 = boundaries
    if layer_idx < b1:
        return 0
    if layer_idx < b2:
        return 1
    return 2


def visualize_heatmap(matrix, labels, title, output_path, boundaries=None):
    """
    Generate and save a heatmap.
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        matrix,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
    )

    if boundaries is not None:
        for b in boundaries:
            # Draw visual separators for stage boundaries.
            ax.axhline(b, color="red", linewidth=1, linestyle="--")
            ax.axvline(b, color="red", linewidth=1, linestyle="--")

    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU ID to use")
    parser.add_argument(
        "--stage-boundaries",
        type=str,
        default="10,20",
        help="Two layer boundaries for three stages, e.g. '10,20'",
    )
    parser.add_argument(
        "--stage-topk-pcts",
        type=str,
        default="70,50,30",
        help="Three topk percentages for stage1,2,3, e.g. '70,50,30'",
    )
    parser.add_argument("--sys-length", type=int, default=35, help="System prompt token length")
    parser.add_argument("--image-token-len", type=int, default=576, help="Image token length")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=20,
        help="Which sample to use per category (0-based).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(project_root, "mme_layer_overlap_results_text_3stage"),
        help="Output directory",
    )
    args = parser.parse_args()

    if args.sample_index < 0:
        print("Invalid sample index: --sample-index must be >= 0")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    try:
        stage_boundaries, stage_topk_pcts = parse_three_stage_config(
            args.stage_boundaries, args.stage_topk_pcts
        )
    except ValueError as e:
        print(f"Invalid stage config: {e}")
        return

    # Configuration
    model_path = "/root/nfs/model/llava-v1.5-7b"
    mme_json_path = os.path.join(project_root, "data/MME/MME_test.json")
    image_folder = os.path.join(project_root, "data/MME/images/test")
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading LLaVA model...")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    try:
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Disable FastV during inference to capture full attention maps.
    if hasattr(model.config, "use_fast_v"):
        print("Disabling FastV to capture full attention maps...")
        model.config.use_fast_v = False
        if hasattr(model, "model") and hasattr(model.model, "reset_fastv"):
            model.model.reset_fastv()

    sys_length = args.sys_length
    image_token_len = args.image_token_len

    print(
        "Three-stage settings: "
        f"boundaries={stage_boundaries}, topk%={stage_topk_pcts}"
    )

    # Get one configurable-index sample per subtask
    samples = load_mme_samples(mme_json_path, sample_index=args.sample_index)

    # Iterate over samples
    for category, sample in tqdm(samples.items(), desc="Processing categories"):
        image_file = sample.get("image_file")
        question = sample.get("question")

        if not image_file or not question:
            continue

        image_path = os.path.join(image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping {category}.")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        qs = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        with torch.inference_mode():
            outputs = model(
                input_ids,
                images=image_tensor,
                output_attentions=True,
                return_dict=True,
            )

        attentions = outputs.attentions
        num_layers = len(attentions)

        # Clamp boundaries if model has fewer layers than expected.
        b1 = min(stage_boundaries[0], num_layers)
        b2 = min(stage_boundaries[1], num_layers)
        effective_boundaries = [b1, b2]

        top_indices = {}
        valid = True

        for layer_idx, layer_attn in enumerate(attentions):
            seq_len = layer_attn.shape[-1]
            if seq_len < sys_length + image_token_len:
                print(f"Warning: Sequence length {seq_len} too short for category {category}. Skipping.")
                valid = False
                break

            text_start = sys_length + image_token_len
            image_start = sys_length
            image_end = sys_length + image_token_len

            text_to_image_attn = layer_attn[0, :, text_start:, image_start:image_end]
            image_attn_scores = text_to_image_attn.sum(dim=0).sum(dim=0)

            stage_id = get_stage_id(layer_idx, effective_boundaries)
            topk_pct = stage_topk_pcts[stage_id]
            k = max(1, int(image_token_len * (topk_pct / 100.0)))

            _, top_idx_t = torch.topk(image_attn_scores, k)
            top_indices[layer_idx] = set(top_idx_t.cpu().numpy())

        if not valid:
            continue

        matrix = np.zeros((num_layers, num_layers))
        for i in range(num_layers):
            for j in range(num_layers):
                if i in top_indices and j in top_indices:
                    matrix[i, j] = get_overlap_ratio(top_indices[i], top_indices[j])

        safe_cat = category.replace("/", "_").replace(" ", "_")
        title = (
            f"{category} - 3-Stage TopK Overlap "
            f"(B=[{b1},{b2}], K%={stage_topk_pcts})"
        )
        out_path = os.path.join(output_dir, f"{safe_cat}_3stage_overlap_heatmap.png")

        visualize_heatmap(
            matrix,
            list(range(num_layers)),
            title,
            out_path,
            boundaries=effective_boundaries,
        )

    print(f"\nProcessing complete. All results saved to {output_dir}")


if __name__ == "__main__":
    main()
