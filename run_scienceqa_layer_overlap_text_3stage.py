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


def parse_three_stage_config(stage_boundaries_str, stage_topk_pcts_str):
    boundaries = [int(x.strip()) for x in stage_boundaries_str.split(",") if x.strip()]
    topk_pcts = [float(x.strip()) for x in stage_topk_pcts_str.split(",") if x.strip()]

    if len(boundaries) != 2:
        raise ValueError("--stage-boundaries must contain exactly 2 integers, e.g. 10,20")
    if len(topk_pcts) != 3:
        raise ValueError("--stage-topk-pcts must contain exactly 3 numbers, e.g. 70,50,30")

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


def get_overlap_ratio(set_a, set_b):
    if len(set_a) == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a)


def visualize_heatmap(matrix, labels, title, output_path, boundaries=None):
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
            ax.axhline(b, color="red", linewidth=1, linestyle="--")
            ax.axvline(b, color="red", linewidth=1, linestyle="--")

    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap to {output_path}")


def extract_question_text(sample):
    conversations = sample.get("conversations", [])
    for turn in conversations:
        if turn.get("from") == "human":
            text = turn.get("value", "")
            # ScienceQA item usually includes <image> token in text.
            text = text.replace("<image>\n", "", 1).replace("<image>", "").strip()
            return text
    return None


def load_scienceqa_samples(question_file, sample_index=0, num_samples=1):
    print(f"Loading ScienceQA data from {question_file}...")
    try:
        with open(question_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: ScienceQA file not found at {question_file}")
        return []

    if not isinstance(data, list):
        print("Error: question file must be a JSON list.")
        return []

    if sample_index < 0:
        print("Error: sample_index must be >= 0")
        return []

    if num_samples <= 0:
        print("Error: num_samples must be >= 1")
        return []

    if sample_index >= len(data):
        print(
            f"Warning: sample_index {sample_index} out of range for {len(data)} samples. "
            "No samples selected."
        )
        return []

    end_idx = min(len(data), sample_index + num_samples)
    selected = data[sample_index:end_idx]

    print(
        f"Selected {len(selected)} ScienceQA samples "
        f"from index range [{sample_index}, {end_idx})."
    )
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU ID to use")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/nfs/model/llava-v1.5-7b",
        help="LLaVA model path",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default="./data/scienceqa/himap-inference-MCQ.json",
        help="ScienceQA question json file",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default="./data/scienceqa/images/test",
        help="ScienceQA image folder",
    )
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
        default=0,
        help="Start sample index in ScienceQA list (0-based).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="How many consecutive samples to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(project_root, "scienceqa_layer_overlap_results_text_3stage"),
        help="Output directory",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    try:
        stage_boundaries, stage_topk_pcts = parse_three_stage_config(
            args.stage_boundaries, args.stage_topk_pcts
        )
    except ValueError as e:
        print(f"Invalid stage config: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading LLaVA model...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    try:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_path, None, model_name
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if hasattr(model.config, "use_fast_v"):
        print("Disabling FastV to capture full attention maps...")
        model.config.use_fast_v = False
        if hasattr(model, "model") and hasattr(model.model, "reset_fastv"):
            model.model.reset_fastv()

    print(
        "Three-stage settings: "
        f"boundaries={stage_boundaries}, topk%={stage_topk_pcts}, "
        f"sample_index={args.sample_index}, num_samples={args.num_samples}"
    )

    samples = load_scienceqa_samples(
        args.question_file, sample_index=args.sample_index, num_samples=args.num_samples
    )
    if not samples:
        print("No valid samples to process.")
        return

    for local_idx, sample in enumerate(tqdm(samples, desc="Processing ScienceQA samples")):
        sample_id = str(sample.get("id", f"idx_{args.sample_index + local_idx}"))
        image_rel = sample.get("image")
        question = extract_question_text(sample)

        if not image_rel or not question:
            print(f"Warning: invalid sample id={sample_id}. Missing image/question, skip.")
            continue

        image_path = os.path.join(args.image_folder, image_rel)
        if not os.path.exists(image_path):
            print(f"Warning: image not found for id={sample_id}: {image_path}")
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

        b1 = min(stage_boundaries[0], num_layers)
        b2 = min(stage_boundaries[1], num_layers)
        effective_boundaries = [b1, b2]

        top_indices = {}
        valid = True

        for layer_idx, layer_attn in enumerate(attentions):
            seq_len = layer_attn.shape[-1]
            if seq_len < args.sys_length + args.image_token_len:
                print(
                    f"Warning: seq_len={seq_len} too short for sample id={sample_id}. Skip sample."
                )
                valid = False
                break

            text_start = args.sys_length + args.image_token_len
            image_start = args.sys_length
            image_end = args.sys_length + args.image_token_len

            text_to_image_attn = layer_attn[0, :, text_start:, image_start:image_end]
            image_attn_scores = text_to_image_attn.sum(dim=0).sum(dim=0)

            stage_id = get_stage_id(layer_idx, effective_boundaries)
            topk_pct = stage_topk_pcts[stage_id]
            k = max(1, int(args.image_token_len * (topk_pct / 100.0)))

            _, top_idx_t = torch.topk(image_attn_scores, k)
            top_indices[layer_idx] = set(top_idx_t.cpu().numpy())

        if not valid:
            continue

        matrix = np.zeros((num_layers, num_layers))
        for i in range(num_layers):
            for j in range(num_layers):
                if i in top_indices and j in top_indices:
                    matrix[i, j] = get_overlap_ratio(top_indices[i], top_indices[j])

        title = (
            f"ScienceQA id={sample_id} - 3-Stage TopK Overlap "
            f"(B=[{b1},{b2}], K%={stage_topk_pcts})"
        )
        out_path = os.path.join(
            args.output_dir,
            f"scienceqa_{sample_id}_idx{args.sample_index + local_idx}_3stage_overlap_heatmap.png",
        )

        visualize_heatmap(
            matrix,
            list(range(num_layers)),
            title,
            out_path,
            boundaries=effective_boundaries,
        )

    print(f"\nProcessing complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
