import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from HiMAP.inference.diversity_token_selector import DiversityTokenSelector

# ------------------------- Metrics ---------------------------------#


def calculate_mme_scores(results: List[Dict]) -> Tuple[Dict[str, float], float, float]:
    perception_cats = [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ]
    cognition_cats = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]

    cat_results: Dict[str, List[Dict]] = {}
    for r in results:
        cat = r["category"]
        cat_results.setdefault(cat, []).append(r)

    scores: Dict[str, float] = {}
    perception_score = 0.0
    cognition_score = 0.0

    for cat, items in cat_results.items():
        correct = sum(1 for x in items if x["pred"].lower() == x["gt"].lower())
        acc = correct / len(items) * 100.0

        img_groups: Dict[str, List[bool]] = {}
        for x in items:
            qid = x["question_id"]
            img_groups.setdefault(qid, []).append(x["pred"].lower() == x["gt"].lower())
        correct_pairs = sum(1 for v in img_groups.values() if all(v))
        acc_plus = correct_pairs / len(img_groups) * 100.0

        score = acc + acc_plus
        scores[cat] = score
        if cat in perception_cats:
            perception_score += score
        elif cat in cognition_cats:
            cognition_score += score

    return scores, perception_score, cognition_score


# ----------------------- Pruning hook ------------------------------#


def build_pruning_hook(selector: DiversityTokenSelector, keep: int, method: str):

    def _hook(module, args, kwargs):
        # Use kwargs if provided (register_forward_pre_hook with with_kwargs=True)
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        past_key_value = kwargs.get("past_key_value", None)
        output_attentions = kwargs.get("output_attentions", False)
        use_cache = kwargs.get("use_cache", False)

        if hidden_states is None:
            return None

        keep_idx = selector.select(hidden_states, num_tokens=keep, method=method)
        if (keep_idx < 0).any():
            return None

        bsz, _, _ = hidden_states.shape
        batch_idx = torch.arange(bsz, device=hidden_states.device).unsqueeze(-1)
        pruned_hidden = hidden_states[batch_idx, keep_idx, :]


        # 重新索引 position_ids，确保与 pruned_hidden 对齐，并且防止 rotary embedding shape mismatch
        if position_ids is not None:
            # position_ids: (bsz, seq_len) -> 选取被保留的token位置
            pruned_pos = torch.gather(position_ids, 1, keep_idx)
            # 强制 pruned_pos 连续编号，防止 rotary embedding shape mismatch
            pruned_pos = torch.arange(keep_idx.shape[1], device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
        else:
            # fallback: 直接生成 0~k-1
            pruned_pos = torch.arange(keep_idx.shape[1], device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        if attention_mask is not None:
            pruned_masks = []
            for b in range(bsz):
                idx = keep_idx[b]
                am = attention_mask[b : b + 1]  # (1, 1, seq, seq)
                am = am[:, :, idx, :]
                am = am[:, :, :, idx]
                pruned_masks.append(am)
            new_attention_mask = torch.cat(pruned_masks, dim=0)
        else:
            new_attention_mask = None

        # Return updated args/kwargs
        new_args = ()
        new_kwargs = {
            "hidden_states": pruned_hidden,
            "attention_mask": new_attention_mask,
            "position_ids": pruned_pos,
            "past_key_value": past_key_value,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
        }
        return new_args, new_kwargs

    return _hook


# --------------------- Experiment runner ---------------------------#


def run_single_layer(
    model,
    tokenizer,
    image_processor,
    questions: List[Dict],
    layer_idx: int,
    selector: DiversityTokenSelector,
    keep_tokens: int,
    method: str,
    args,
):
    layer = model.model.layers[layer_idx]
    hook_handle = layer.register_forward_pre_hook(
        build_pruning_hook(selector, keep_tokens, method), with_kwargs=True
    )

    layer_results = []
    for line in questions:
        idx = line.get("question_id")
        qs = line["question"]
        label = line["answer"]
        category = line["category"]
        image_file = line["image_file"]

        image_path = os.path.join(args.image_folder, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        if torch.cuda.is_available():
            images = image_tensor.unsqueeze(0).half().cuda()
        else:
            images = image_tensor.unsqueeze(0).float()

        prompt_qs = qs
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

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output = model.generate(
                input_ids,
                images=images,
                max_new_tokens=10,
                use_cache=False,
                stopping_criteria=stopping_criteria,
                output_attentions=False,
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        pred = outputs
        if "yes" in pred.lower():
            pred = "Yes"
        elif "no" in pred.lower():
            pred = "No"

        layer_results.append({
            "question_id": idx,
            "category": category,
            "pred": pred,
            "gt": label,
        })

    hook_handle.remove()
    return layer_results


# --------------------------- Main ----------------------------------#


def plot_results(results: Dict, output_plot: str):
    plt.figure(figsize=(12, 6))
    for method, data in results.items():
        plt.plot(data["layers"], data["total_scores"], marker="o", label=f"{method.upper()} Total", linewidth=2)
    plt.xlabel("Layer Index (pruning applied at layer)")
    plt.ylabel("Total MME Score")
    plt.title("MME Total Score vs Layer for Diversity Pruning")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"Total score plot saved to {output_plot}")

    plt.figure(figsize=(12, 6))
    for method, data in results.items():
        plt.plot(data["layers"], data["perception_scores"], marker="s", label=f"{method.upper()} Perception", linewidth=2)
        plt.plot(data["layers"], data["cognition_scores"], marker="^", label=f"{method.upper()} Cognition", linewidth=2)
    plt.xlabel("Layer Index (pruning applied at layer)")
    plt.ylabel("Score")
    plt.title("Perception/Cognition Scores vs Layer")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    sub_plot = output_plot.replace(".png", "_pc.png")
    plt.savefig(sub_plot, dpi=300, bbox_inches="tight")
    print(f"Perception/Cognition plot saved to {sub_plot}")


def run_experiment(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    if torch.cuda.is_available():
        model.cuda()

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    if args.num_samples > 0:
        questions = questions[: args.num_samples]
    print(f"Total samples: {len(questions)}")

    num_layers = len(model.model.layers)
    max_layer = min(args.layer_end, num_layers)
    layers_to_test = list(range(args.layer_start - 1, max_layer))  # zero-based for indexing

    selector = DiversityTokenSelector(metric=args.metric)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    experiment_results: Dict[str, Dict] = {}
    for method in methods:
        print(f"\nRunning method: {method} with keep={args.keep_tokens}")
        experiment_results[method] = {
            "layers": [],
            "total_scores": [],
            "perception_scores": [],
            "cognition_scores": [],
            "subtask_scores": [],
        }
        for layer_idx in tqdm(layers_to_test, desc=f"{method} layers"):
            layer_results = run_single_layer(
                model,
                tokenizer,
                image_processor,
                questions,
                layer_idx,
                selector,
                args.keep_tokens,
                method,
                args,
            )
            scores, perception_score, cognition_score = calculate_mme_scores(layer_results)
            experiment_results[method]["layers"].append(layer_idx + 1)  # back to 1-based for reporting
            experiment_results[method]["total_scores"].append(perception_score + cognition_score)
            experiment_results[method]["perception_scores"].append(perception_score)
            experiment_results[method]["cognition_scores"].append(cognition_score)
            experiment_results[method]["subtask_scores"].append(scores)

    with open(args.output_file, "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"Results saved to {args.output_file}")

    plot_results(experiment_results, args.output_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--output-file", type=str, default="mme_diversity_pruning_results.json")
    parser.add_argument("--output-plot", type=str, default="mme_diversity_pruning_plot.png")
    parser.add_argument("--keep-tokens", type=int, default=8)
    parser.add_argument("--methods", type=str, default="fps,tome")
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--layer-start", type=int, default=1)
    parser.add_argument("--layer-end", type=int, default=32)
    args = parser.parse_args()
    run_experiment(args)
