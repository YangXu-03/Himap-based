import argparse
import gc
import json
import math
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

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


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_topk_values(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("--topk-values must contain at least one integer")
    if any(v <= 0 or v > 100 for v in values):
        raise ValueError("all top-k% values must be in (0, 100]")
    return values


def extract_global_to_image_attention(attn: Optional[torch.Tensor], sys_length: int, img_length: int) -> torch.Tensor:
    if img_length <= 0:
        return torch.zeros((0,), dtype=torch.float32)
    if attn is None:
        return torch.zeros((img_length,), dtype=torch.float32)

    attn_avg = torch.mean(attn, dim=1)[0]
    seq_len_k = attn_avg.shape[-1]
    img_span_end = min(sys_length + img_length, seq_len_k)
    local_img_len = max(img_span_end - sys_length, 0)
    image_scores = torch.zeros((img_length,), dtype=attn_avg.dtype, device=attn_avg.device)
    if local_img_len <= 0:
        return image_scores

    global_to_img = attn_avg[:, sys_length : sys_length + local_img_len]
    if global_to_img.numel() == 0:
        return image_scores

    global_scores = torch.mean(global_to_img, dim=0)
    image_scores[:local_img_len] = torch.nan_to_num(global_scores, nan=0.0, posinf=0.0, neginf=0.0)
    return image_scores


def extract_prompt_to_image_attention(attn: Optional[torch.Tensor], sys_length: int, img_length: int) -> torch.Tensor:
    if img_length <= 0:
        return torch.zeros((0,), dtype=torch.float32)
    if attn is None:
        return torch.zeros((img_length,), dtype=torch.float32)

    attn_avg = torch.mean(attn, dim=1)[0]
    seq_len_q = attn_avg.shape[-2]
    seq_len_k = attn_avg.shape[-1]

    img_span_end = min(sys_length + img_length, seq_len_k)
    local_img_len = max(img_span_end - sys_length, 0)
    image_scores = torch.zeros((img_length,), dtype=attn_avg.dtype, device=attn_avg.device)
    if local_img_len <= 0:
        return image_scores

    prompt_start = min(sys_length + img_length, seq_len_q)
    if prompt_start < seq_len_q:
        prompt_to_img = attn_avg[prompt_start:seq_len_q, sys_length : sys_length + local_img_len]
        if prompt_to_img.numel() > 0:
            prompt_scores = torch.mean(prompt_to_img, dim=0)
            image_scores[:local_img_len] = torch.nan_to_num(prompt_scores, nan=0.0, posinf=0.0, neginf=0.0)
            return image_scores

    last_token_scores = attn_avg[-1, sys_length : sys_length + local_img_len]
    image_scores[:local_img_len] = torch.nan_to_num(last_token_scores, nan=0.0, posinf=0.0, neginf=0.0)
    return image_scores


def extract_image_attention_scores(attn: Optional[torch.Tensor], sys_length: int, img_length: int, mode: str) -> torch.Tensor:
    if mode == "global":
        return extract_global_to_image_attention(attn, sys_length, img_length)
    return extract_prompt_to_image_attention(attn, sys_length, img_length)


def compute_topk_ratio(image_scores: torch.Tensor, k: int) -> float:
    if image_scores.numel() == 0:
        return 0.0
    image_scores = torch.nan_to_num(image_scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(image_scores.sum().item())
    if total <= 0:
        return 0.0
    k_eff = max(1, int(math.ceil(image_scores.numel() * k / 100.0)))
    k_eff = min(k_eff, image_scores.numel())
    topk_sum = float(torch.topk(image_scores, k_eff).values.sum().item())
    return topk_sum / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None, help="limit sample count for quick diagnostics")
    parser.add_argument(
        "--attention-mode",
        type=str,
        default="prompt_image",
        choices=["prompt_image", "global"],
        help="attention scoring mode for image tokens",
    )
    parser.add_argument("--sys-length", type=int, default=35, help="system token length before image token span")
    parser.add_argument("--img-length", type=int, default=576, help="image token span length")
    parser.add_argument(
        "--topk-values",
        type=str,
        default="1,10,20,50",
        help="comma-separated top-k values in percentage, e.g. 1,10,20,50",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="mme_topk_attention_ratio.json",
        help="output json file for per-layer top-k ratios",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="mme_topk_attention_ratio.png",
        help="output figure path",
    )
    args = parser.parse_args()

    topk_values = parse_topk_values(args.topk_values)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    # Keep baseline behavior for analysis-only runs.
    model.config.use_hmap_v = False
    model.config.use_fast_v = False
    model.config.use_jsd_entropy_pruning = False

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.num_samples is not None and args.num_samples > 0:
        questions = questions[: args.num_samples]

    per_k_layer_sum: Dict[int, List[float]] = {}
    per_k_layer_count: Dict[int, List[int]] = {}
    num_layers = None
    processed = 0

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for line in tqdm(questions, desc="MME attention analysis"):
        qs = line["question"]
        image_file = line["image_file"]
        image_path = os.path.join(args.image_folder, image_file)
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        if torch.cuda.is_available():
            images = image_tensor.unsqueeze(0).half().cuda()
        else:
            images = image_tensor.unsqueeze(0).float()

        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        qs = qs + "\n" + "Answer the question using a single word or phrase."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                images=images,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )

        attentions = outputs.attentions
        if attentions is None or len(attentions) == 0:
            del outputs
            continue

        if num_layers is None:
            num_layers = len(attentions)
            for k in topk_values:
                per_k_layer_sum[k] = [0.0] * num_layers
                per_k_layer_count[k] = [0] * num_layers

        for layer_idx, layer_attn in enumerate(attentions):
            scores = extract_image_attention_scores(
                layer_attn,
                sys_length=args.sys_length,
                img_length=args.img_length,
                mode=args.attention_mode,
            ).detach().float().cpu()

            valid_len = max(0, min(args.img_length, scores.numel()))
            if valid_len <= 0:
                continue
            scores = scores[:valid_len]
            for k in topk_values:
                ratio = compute_topk_ratio(scores, k)
                per_k_layer_sum[k][layer_idx] += ratio
                per_k_layer_count[k][layer_idx] += 1

        processed += 1

        del outputs
        if torch.cuda.is_available() and processed % 32 == 0:
            torch.cuda.empty_cache()

    if num_layers is None or processed == 0:
        raise RuntimeError("No valid sample was processed. Please check dataset path and model settings.")

    per_k_layer_avg: Dict[int, List[float]] = {}
    for k in topk_values:
        ratios = []
        for s, c in zip(per_k_layer_sum[k], per_k_layer_count[k]):
            ratios.append(float(s / c) if c > 0 else 0.0)
        per_k_layer_avg[k] = ratios

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "model_path": args.model_path,
                    "question_file": args.question_file,
                    "image_folder": args.image_folder,
                    "attention_mode": args.attention_mode,
                    "sys_length": args.sys_length,
                    "img_length": args.img_length,
                    "topk_values": topk_values,
                    "processed_samples": processed,
                },
                "layer_topk_ratio": {str(k): v for k, v in per_k_layer_avg.items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.makedirs(os.path.dirname(args.output_plot) or ".", exist_ok=True)
    x = list(range(num_layers))
    plt.figure(figsize=(10, 6))
    for k in topk_values:
        y = [r * 100.0 for r in per_k_layer_avg[k]]
        plt.plot(x, y, marker="o", markersize=3, linewidth=2, label=f"top-{k}%")

    plt.xlabel("Layer Index")
    plt.ylabel("Top-k% Attention Ratio to Image Tokens (%)")
    plt.title(f"MME Per-Layer Top-k% Image Attention Ratio ({args.attention_mode})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=200)

    print(f"Processed samples: {processed}")
    print(f"Saved ratio json: {args.output_json}")
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
