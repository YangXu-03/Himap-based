#!/usr/bin/env python3
"""
在 MME 数据集上分析不同注意力头对视觉 token 排序的特异性。

默认行为：
1. 按 MME `category`（子任务）分组；
2. 对每个子任务抽取可配置数量的样本；
3. 在指定层（默认 6, 13, 22）提取每个 head 对每个视觉 token 的
    可配置注意力分数；
4. 将每个 head 内部的视觉 token 注意力转换为“排序百分位分数”，
   其中 1 表示该 head 中排名最高的 token，0 表示最低；
5. 对同一任务内样本求平均，并绘制热力图：
   - 横轴：视觉 token index
   - 纵轴：attention head index

同时支持：
- 全局注意力（所有 query 到视觉 token 的聚合）
- Text→Image 注意力（文本 query 到视觉 token 的聚合）
- 使用原始注意力值或排序分数进行可视化。
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[4]
LLAVA_SRC = PROJECT_ROOT / "src" / "LLaVA"
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))
if str(LLAVA_SRC) not in sys.path:
    sys.path.append(str(LLAVA_SRC))

from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MME head-token ranking specificity heatmaps"
    )
    parser.add_argument("--gpu-id", type=str, default="1", help="GPU ID to use")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/nfs/model/llava-v1.5-7b",
        help="Path to the LLaVA model",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image-folder",
        type=str,
        default=str(PROJECT_ROOT / "data" / "MME" / "images" / "test"),
        help="Folder containing MME images",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "MME" / "MME_test.json"),
        help="Path to MME_test.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "mme_head_token_rank_specificity"),
        help="Directory to save figures and metadata",
    )
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[6, 13, 22],
        help="Layer indices to visualize",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=40,
        help="Default number of samples used for each MME task",
    )
    parser.add_argument(
        "--samples-config",
        type=str,
        default=None,
        help="Optional JSON file mapping task name to sample count. Supports '__default__'.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of MME task/category names to process",
    )
    parser.add_argument(
        "--sample-strategy",
        type=str,
        choices=["first", "random"],
        default="first",
        help="How to select samples inside each task",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-type",
        type=str,
        choices=["rank", "attention"],
        default="rank",
        help="Use head-wise rank percentile or raw attention for heatmap values",
    )
    parser.add_argument(
        "--attention-type",
        type=str,
        choices=["global", "text_to_image"],
        default="global",
        help="Attention definition used to score visual tokens for each head",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["mean", "median"],
        default="mean",
        help="Aggregation method across samples within each task",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=-1,
        help="Optional cap on number of tasks to process (-1 means all)",
    )
    parser.add_argument(
        "--token-tick-step",
        type=int,
        default=48,
        help="Tick interval on x-axis for visual token indices",
    )
    return parser.parse_args()


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def resolve_target_device(gpu_id: str) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    gpu_candidates = [part.strip() for part in str(gpu_id).split(",") if part.strip()]
    gpu_index = int(gpu_candidates[0]) if gpu_candidates else 0
    if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu-id {gpu_id!r}; available CUDA device count is {torch.cuda.device_count()}"
        )
    return torch.device(f"cuda:{gpu_index}")


def build_prompt(question: str, model_config, conv_mode: str) -> Tuple[object, str]:
    if getattr(model_config, "mm_use_im_start_end", False):
        prompt_question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        prompt_question = DEFAULT_IMAGE_TOKEN + "\n" + question
    prompt_question += "\nAnswer the question using a single word or phrase."

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_question)
    conv.append_message(conv.roles[1], None)
    return conv, conv.get_prompt()


def compute_token_groups(
    conv,
    tokenizer,
    input_ids: torch.Tensor,
    attentions: Sequence[torch.Tensor],
) -> Tuple[List[int], List[int], List[int]]:
    image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
    if len(image_token_indices) == 0:
        return [], [], []

    img_start_idx_input = image_token_indices[0].item()
    seq_len_output = attentions[0].shape[-1]
    num_patches = seq_len_output - input_ids.shape[1] + 1
    vis_indices = list(range(img_start_idx_input, img_start_idx_input + num_patches))

    system_tokens = tokenizer(conv.system + conv.sep, add_special_tokens=False).input_ids
    system_start = 1 if input_ids[0, 0].item() == tokenizer.bos_token_id else 0
    system_end = min(system_start + len(system_tokens), seq_len_output)
    system_indices = list(range(system_start, system_end))

    text_indices = [
        idx for idx in range(seq_len_output) if idx not in vis_indices and idx not in system_indices
    ]
    return system_indices, text_indices, vis_indices


def compute_head_visual_scores(
    attn: torch.Tensor,
    text_idx: List[int],
    vis_idx: List[int],
    attention_type: str,
) -> np.ndarray:
    """
    Returns:
        ndarray of shape (num_heads, num_visual_tokens)
    """
    if len(text_idx) == 0 or len(vis_idx) == 0:
        return np.zeros((attn.shape[0], len(vis_idx)), dtype=np.float32)

    if attention_type == "global":
        sub = attn[:, :, vis_idx]
        head_visual_scores = sub.sum(dim=1)
    else:
        sub = attn[:, text_idx, :][:, :, vis_idx]
        head_visual_scores = sub.mean(dim=1)

    return head_visual_scores.detach().float().cpu().numpy()


def convert_scores_to_rank_percentiles(scores: np.ndarray) -> np.ndarray:
    """
    Convert per-head token scores into per-head rank percentiles in [0, 1].
    Higher means the token ranks higher within that head.
    """
    num_heads, num_tokens = scores.shape
    if num_tokens <= 1:
        return np.ones((num_heads, num_tokens), dtype=np.float32)

    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    head_indices = np.arange(num_heads)[:, None]
    ranks[head_indices, order] = np.arange(num_tokens)[None, :]
    normalized = 1.0 - ranks.astype(np.float32) / float(num_tokens - 1)
    return normalized


def load_questions(question_file: str) -> List[Dict]:
    with open(os.path.expanduser(question_file), "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_task(questions: Sequence[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for item in questions:
        category = item.get("category", "unknown")
        grouped[category].append(item)
    return dict(grouped)


def load_sample_config(path: Optional[str], default_count: int) -> Dict[str, int]:
    config = {"__default__": default_count}
    if not path:
        return config

    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    if not isinstance(raw_config, dict):
        raise ValueError("samples-config must be a JSON object")

    for key, value in raw_config.items():
        config[str(key)] = int(value)
    return config


def get_task_sample_count(category: str, sample_config: Dict[str, int]) -> int:
    if category in sample_config:
        return sample_config[category]
    return sample_config.get("__default__", 1)


def select_task_samples(
    samples: Sequence[Dict],
    count: int,
    strategy: str,
    rng: random.Random,
) -> List[Dict]:
    if count < 0 or count >= len(samples):
        return list(samples)
    if count == 0:
        return []
    if strategy == "random":
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        return [samples[idx] for idx in indices[:count]]
    return list(samples[:count])


def process_sample(
    model,
    tokenizer,
    image_processor,
    sample: Dict,
    layer_indices: Sequence[int],
    conv_mode: str,
    device: torch.device,
    score_type: str,
    attention_type: str,
    image_folder: str,
) -> Optional[Dict[int, np.ndarray]]:
    image_path = os.path.join(image_folder, sample["image_file"])
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"Failed to load image {image_path}: {exc}")
        return None

    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    images = image_tensor.unsqueeze(0)
    if device.type == "cuda":
        images = images.half()
    images = images.to(device)

    conv, prompt = build_prompt(sample["question"], model.config, conv_mode)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids,
                images=images,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    except Exception as exc:
        print(f"Failed to process sample {sample.get('image_file', 'unknown')}: {exc}")
        return None

    attentions = outputs.attentions
    if attentions is None or len(attentions) == 0:
        return None

    _, text_idx, vis_idx = compute_token_groups(conv, tokenizer, input_ids, attentions)
    if len(text_idx) == 0 or len(vis_idx) == 0:
        return None

    sample_results: Dict[int, np.ndarray] = {}
    num_layers = len(attentions)
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= num_layers:
            continue
        layer_attn = attentions[layer_idx].squeeze(0)
        head_visual_scores = compute_head_visual_scores(
            layer_attn,
            text_idx,
            vis_idx,
            attention_type,
        )
        if score_type == "rank":
            sample_results[layer_idx] = convert_scores_to_rank_percentiles(head_visual_scores)
        else:
            sample_results[layer_idx] = head_visual_scores.astype(np.float32)

    del outputs, attentions, images, input_ids
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return sample_results if sample_results else None


def aggregate_matrices(matrices: Sequence[np.ndarray], method: str) -> np.ndarray:
    stacked = np.stack(matrices, axis=0)
    if method == "median":
        return np.median(stacked, axis=0)
    return stacked.mean(axis=0)


def compute_global_color_range(
    aggregated: Dict[str, Dict[int, np.ndarray]],
    score_type: str,
) -> Tuple[float, float]:
    if score_type == "rank":
        return 0.0, 1.0

    all_values = []
    for layer_data in aggregated.values():
        for matrix in layer_data.values():
            all_values.append(matrix)
    if not all_values:
        return 0.0, 1.0

    concat = np.concatenate([arr.reshape(-1) for arr in all_values], axis=0)
    return float(concat.min()), float(concat.max())


def plot_combined_heatmaps(
    layer_matrices: Dict[int, np.ndarray],
    category: str,
    output_path: str,
    score_type: str,
    attention_type: str,
    layer_sample_counts: Dict[int, int],
    token_tick_step: int,
    vmin: float,
    vmax: float,
) -> None:
    if not layer_matrices:
        return

    sorted_layers = sorted(layer_matrices.keys())
    first_matrix = layer_matrices[sorted_layers[0]]
    num_heads, num_tokens = first_matrix.shape
    num_plots = len(sorted_layers)

    n_cols = min(3, num_plots)
    n_rows = int(np.ceil(num_plots / n_cols))

    subplot_width = max(8, num_tokens / 48)
    subplot_height = max(5, num_heads / 4)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(subplot_width * n_cols, subplot_height * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    if attention_type == "global":
        attention_label = "Global attention"
    else:
        attention_label = "Text2Image attention"

    if score_type == "rank":
        cbar_label = "Rank percentile within each head"
        figure_suffix = f"{attention_label} | head-specific token rank percentile"
    else:
        cbar_label = attention_label
        figure_suffix = f"{attention_label} | head-specific token attention"

    image_handle = None
    for ax, layer_idx in zip(axes_flat, sorted_layers):
        matrix = layer_matrices[layer_idx]
        image_handle = ax.imshow(
            matrix,
            aspect="auto",
            cmap="Reds",
            interpolation="nearest",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )

        sample_count = layer_sample_counts.get(layer_idx, 0)
        ax.set_title(f"Layer {layer_idx} ({sample_count} samples)")
        ax.set_xlabel("Visual token index")
        ax.set_ylabel("Attention head index")

        x_ticks = np.arange(0, num_tokens, max(1, token_tick_step))
        if len(x_ticks) == 0 or x_ticks[-1] != num_tokens - 1:
            x_ticks = np.unique(np.append(x_ticks, num_tokens - 1))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(x)) for x in x_ticks], rotation=45, ha="right")

        y_step = max(1, num_heads // 8)
        y_ticks = np.arange(0, num_heads, y_step)
        if len(y_ticks) == 0 or y_ticks[-1] != num_heads - 1:
            y_ticks = np.unique(np.append(y_ticks, num_heads - 1))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(y)) for y in y_ticks])

    for ax in axes_flat[num_plots:]:
        ax.axis("off")

    fig.suptitle(f"{category} | {figure_suffix}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.94, 0.95])

    if image_handle is not None:
        cbar = fig.colorbar(image_handle, ax=axes_flat.tolist(), fraction=0.02, pad=0.02)
        cbar.set_label(cbar_label)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


    disable_torch_init()
    os.makedirs(args.output_dir, exist_ok=True)
    target_device = resolve_target_device(args.gpu_id)

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device=str(target_device),
        device_map={"": str(target_device)} if target_device.type == "cuda" else None,
    )


    device = target_device
    model.eval()
    print(f"Model loaded on {device}")

    if hasattr(model.config, "use_fast_v"):
        model.config.use_fast_v = False
        if hasattr(model, "model") and hasattr(model.model, "reset_fastv"):
            model.model.reset_fastv()

    questions = load_questions(args.question_file)
    task_to_samples = group_by_task(questions)
    sample_config = load_sample_config(args.samples_config, args.samples_per_task)
    requested_tasks = set(args.tasks) if args.tasks else None
    rng = random.Random(args.seed)


    selected_tasks = []
    for category in sorted(task_to_samples.keys()):
        if requested_tasks is not None and category not in requested_tasks:
            continue
        selected_tasks.append(category)

    if args.max_tasks > 0:
        selected_tasks = selected_tasks[: args.max_tasks]

    print(f"Found {len(selected_tasks)} tasks to process")

    raw_results: Dict[str, Dict[int, List[np.ndarray]]] = {
        category: defaultdict(list) for category in selected_tasks
    }
    task_metadata = {}

    for category in tqdm(selected_tasks, desc="Processing tasks"):
        desired_count = get_task_sample_count(category, sample_config)
        selected_samples = select_task_samples(
            task_to_samples[category],
            desired_count,
            args.sample_strategy,
            rng,
        )

        task_metadata[category] = {
            "available_samples": len(task_to_samples[category]),
            "selected_samples": len(selected_samples),
            "requested_samples": desired_count,
        }

        sample_iterator = tqdm(
            selected_samples,
            desc=f"{category}",
            leave=False,
        )
        for sample in sample_iterator:
            sample_result = process_sample(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                sample=sample,
                layer_indices=args.layers,
                conv_mode=args.conv_mode,
                device=device,
                score_type=args.score_type,
                attention_type=args.attention_type,
                image_folder=args.image_folder,
            )
            if sample_result is None:
                continue
            for layer_idx, matrix in sample_result.items():
                raw_results[category][layer_idx].append(matrix)

    aggregated: Dict[str, Dict[int, np.ndarray]] = {}
    for category, layer_dict in raw_results.items():
        aggregated[category] = {}
        for layer_idx, matrices in layer_dict.items():
            if not matrices:
                continue
            aggregated[category][layer_idx] = aggregate_matrices(matrices, args.aggregate)

    vmin, vmax = compute_global_color_range(aggregated, args.score_type)

    summary = {
        "model_path": model_path,
        "question_file": os.path.expanduser(args.question_file),
        "image_folder": os.path.expanduser(args.image_folder),
        "layers": args.layers,
        "score_type": args.score_type,
        "attention_type": args.attention_type,
        "aggregate": args.aggregate,
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "task_metadata": task_metadata,
        "tasks_with_results": {},
    }

    for category, layer_dict in aggregated.items():
        category_dir = os.path.join(args.output_dir, safe_name(category))
        os.makedirs(category_dir, exist_ok=True)

        summary["tasks_with_results"][category] = {}
        for layer_idx, matrix in sorted(layer_dict.items()):
            np.save(
                os.path.join(
                    category_dir,
                    f"layer_{layer_idx}_{args.attention_type}_{args.score_type}_matrix.npy",
                ),
                matrix,
            )
            summary["tasks_with_results"][category][str(layer_idx)] = {
                "num_samples": len(raw_results[category][layer_idx]),
                "matrix_shape": list(matrix.shape),
                "matrix": os.path.join(
                    category_dir,
                    f"layer_{layer_idx}_{args.attention_type}_{args.score_type}_matrix.npy",
                ),
            }

        combined_output_path = os.path.join(
            category_dir,
            f"combined_layers_{args.attention_type}_{args.score_type}_heatmap.png",
        )
        plot_combined_heatmaps(
            layer_matrices=layer_dict,
            category=category,
            output_path=combined_output_path,
            score_type=args.score_type,
            attention_type=args.attention_type,
            layer_sample_counts={
                layer_idx: len(raw_results[category][layer_idx])
                for layer_idx in layer_dict.keys()
            },
            token_tick_step=args.token_tick_step,
            vmin=vmin,
            vmax=vmax,
        )
        summary["tasks_with_results"][category]["combined_heatmap"] = combined_output_path

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {args.output_dir}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()