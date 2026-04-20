import argparse
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

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
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_yes_no(text: str) -> str:
    lower = text.strip().lower()
    if "yes" in lower:
        return "yes"
    if "no" in lower:
        return "no"
    return lower


def parse_sqa_choice(text: str) -> str:
    t = text.strip()
    if t in ["A", "B", "C", "D", "E"]:
        return t
    if len(t) >= 3 and t[0] in ["A", "B", "C", "D", "E"] and t[1:3] == ". ":
        return t[0]
    m = re.findall(r"The answer is ([A-Z]).", t)
    if len(m) == 1:
        return m[0]
    return "FAILED"


def decode_generate_output(output_ids, input_ids, tokenizer, stop_str: str) -> str:
    input_token_len = input_ids.shape[1]
    sequences = output_ids["sequences"] if isinstance(output_ids, dict) else output_ids
    seq_token_len = sequences.shape[1]

    if seq_token_len >= input_token_len:
        decode_ids = sequences[:, input_token_len:]
    else:
        decode_ids = sequences

    outputs = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    return outputs.strip()


def sample_items(items: List[dict], num_samples: int, seed: int) -> List[dict]:
    if num_samples <= 0 or num_samples >= len(items):
        return items
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    picked = sorted(indices[:num_samples])
    return [items[i] for i in picked]


def load_scienceqa(path: str, num_samples: int, seed: int) -> List[dict]:
    questions = json.load(open(os.path.expanduser(path), "r"))
    return sample_items(questions, num_samples, seed)


def load_mme(path: str, num_samples: int, seed: int) -> List[dict]:
    questions = json.load(open(os.path.expanduser(path), "r"))
    return sample_items(questions, num_samples, seed)


def load_textvqa(path: str, num_samples: int, seed: int) -> List[dict]:
    resolved = os.path.expanduser(path)
    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    try:
        data = json.loads(raw)
        ann = data.get("data", [])
        return sample_items(ann, num_samples, seed)
    except json.JSONDecodeError as e:
        print(f"[Warning] TextVQA JSON parse failed: {e}")
        print("[Warning] Trying partial recovery from truncated TextVQA file...")

    key_pos = raw.find('"data"')
    if key_pos < 0:
        raise ValueError(f"Invalid TextVQA file (missing 'data' key): {resolved}")

    arr_start = raw.find("[", key_pos)
    if arr_start < 0:
        raise ValueError(f"Invalid TextVQA file (missing data array): {resolved}")

    decoder = json.JSONDecoder()
    items: List[dict] = []
    i = arr_start + 1
    n = len(raw)

    while i < n:
        while i < n and raw[i] in " \t\r\n,":
            i += 1
        if i >= n or raw[i] == "]":
            break
        try:
            obj, j = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            # Most likely truncated tail: keep fully decoded prefix.
            break

        if isinstance(obj, dict):
            items.append(obj)
        i = j

    if len(items) == 0:
        raise ValueError(
            f"TextVQA file is corrupted and no recoverable samples were found: {resolved}"
        )

    print(f"[Warning] Recovered {len(items)} valid TextVQA samples from truncated file.")
    return sample_items(items, num_samples, seed)


def build_multimodal_prompt(raw_question: str, model_config, single_pred_prompt: Optional[str] = None) -> str:
    qs = raw_question.strip()
    if getattr(model_config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    if single_pred_prompt:
        qs = qs + "\n" + single_pred_prompt
    return qs


def make_input_ids(tokenizer, conv_mode: str, prompt_text: str, device: torch.device):
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = input_ids.to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = [KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)] if conv.version == "v0" else None
    return input_ids, stop_str, stopping_criteria


def preprocess_image(image_path: str, image_processor, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    if device.type == "cuda":
        return image_tensor.unsqueeze(0).half().to(device)
    return image_tensor.unsqueeze(0).float().to(device)


def find_textvqa_image_path(image_folder: str, image_id) -> str:
    base = str(image_id)
    candidates = [
        base,
        base + ".jpg",
        base + ".png",
        base + ".jpeg",
    ]
    for cand in candidates:
        p = os.path.join(image_folder, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot resolve TextVQA image for image_id={image_id} under {image_folder}")


def eval_scienceqa_layer(
    model,
    tokenizer,
    image_processor,
    questions: List[dict],
    image_folder: str,
    conv_mode: str,
    device: torch.device,
) -> float:
    correct = 0
    total = 0

    for line in tqdm(questions, desc="ScienceQA", leave=False):
        raw_q = line["conversations"][0]["value"].replace("<image>", "").strip()
        gt = line["conversations"][1]["value"].strip()
        image_file = line["image"]

        prompt_text = build_multimodal_prompt(
            raw_q,
            model.config,
            "Answer with the option's letter from the given choices directly.",
        )
        input_ids, stop_str, stopping_criteria = make_input_ids(tokenizer, conv_mode, prompt_text, device)
        images = preprocess_image(os.path.join(image_folder, image_file), image_processor, device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        pred = parse_sqa_choice(decode_generate_output(output_ids, input_ids, tokenizer, stop_str))
        if pred == gt:
            correct += 1
        total += 1

    return 0.0 if total == 0 else correct / total


def eval_textvqa_layer(
    model,
    tokenizer,
    image_processor,
    questions: List[dict],
    image_folder: str,
    conv_mode: str,
    device: torch.device,
) -> float:
    pred_list = []
    evaluator = TextVQAAccuracyEvaluator()

    for line in tqdm(questions, desc="TextVQA", leave=False):
        raw_q = line["question"].strip()
        gt_answers = line["answers"]

        prompt_text = build_multimodal_prompt(raw_q, model.config, "Answer the question using a single word or phrase.")
        input_ids, stop_str, stopping_criteria = make_input_ids(tokenizer, conv_mode, prompt_text, device)

        image_path = find_textvqa_image_path(image_folder, line["image_id"])
        images = preprocess_image(image_path, image_processor, device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        pred = decode_generate_output(output_ids, input_ids, tokenizer, stop_str)
        pred_list.append({"pred_answer": pred, "gt_answers": gt_answers})

    if len(pred_list) == 0:
        return 0.0
    return evaluator.eval_pred_list(pred_list)


def configure_full_prune(model, agg_layer: int, sys_length: int, image_token_length: int) -> None:
    model.config.use_hmap_v = False
    model.config.use_jsd_entropy_pruning = False
    model.config.use_fast_v = True
    model.config.fast_v_sys_length = sys_length
    model.config.fast_v_image_token_length = image_token_length
    model.config.fast_v_attention_rank = 0
    model.config.fast_v_agg_layer = agg_layer
    if hasattr(model.model, "reset_fastv"):
        model.model.reset_fastv()


def configure_no_prune(model) -> None:
    model.config.use_hmap_v = False
    model.config.use_jsd_entropy_pruning = False
    model.config.use_fast_v = False
    if hasattr(model.model, "reset_fastv"):
        model.model.reset_fastv()


def plot_results(results: Dict, output_plot: str) -> None:
    layers = results["layers"]
    sqa = results["scienceqa_acc"]
    textvqa = results["textvqa_acc"]
    baseline = results.get("baseline", {})

    plt.figure(figsize=(12, 6))
    plt.plot(layers, sqa, marker="o", linewidth=2, label="ScienceQA")
    plt.plot(layers, textvqa, marker="^", linewidth=2, label="TextVQA")

    if "scienceqa_acc" in baseline:
        plt.axhline(
            y=baseline["scienceqa_acc"],
            linestyle="--",
            linewidth=1.5,
            color="C0",
            alpha=0.8,
            label=f"ScienceQA Baseline ({baseline['scienceqa_acc']:.4f})",
        )
    if "textvqa_acc" in baseline:
        plt.axhline(
            y=baseline["textvqa_acc"],
            linestyle="--",
            linewidth=1.5,
            color="C1",
            alpha=0.8,
            label=f"TextVQA Baseline ({baseline['textvqa_acc']:.4f})",
        )

    plt.xlabel("Layer Index (FastV Aggregation Layer)")
    plt.ylabel("Accuracy")
    plt.title("Layer-wise Accuracy with Complete Image Token Pruning")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_plot}")


def run_experiment(args):
    set_seed(args.seed)
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scienceqa_questions = load_scienceqa(args.scienceqa_question_file, args.num_samples_per_dataset, args.seed)
    textvqa_questions = load_textvqa(args.textvqa_question_file, args.num_samples_per_dataset, args.seed)

    print(f"ScienceQA samples: {len(scienceqa_questions)}")
    print(f"TextVQA samples: {len(textvqa_questions)}")

    num_layers = len(model.model.layers)
    default_lo = 0
    default_hi = max(2, num_layers - 1)

    lo = args.min_layer if args.min_layer is not None else default_lo
    hi = args.max_layer if args.max_layer is not None else default_hi
    lo = max(0, lo)
    hi = min(num_layers - 1, hi)
    if lo > hi:
        raise ValueError(f"Invalid layer range: min_layer={lo}, max_layer={hi}, num_layers={num_layers}")
    layers_to_test = list(range(lo, hi + 1))

    results = {
        "layers": [],
        "scienceqa_acc": [],
        "textvqa_acc": [],
        "baseline": {},
        "meta": {
            "model_path": model_path,
            "num_layers": num_layers,
            "num_samples_per_dataset": args.num_samples_per_dataset,
            "seed": args.seed,
            "fast_v_sys_length": args.fast_v_sys_length,
            "fast_v_image_token_length": args.fast_v_image_token_length,
        },
    }

    print("\n=== Evaluating no-pruning baseline ===")
    configure_no_prune(model)
    baseline_sqa = eval_scienceqa_layer(
        model,
        tokenizer,
        image_processor,
        scienceqa_questions,
        args.scienceqa_image_folder,
        args.conv_mode,
        device,
    )
    baseline_textvqa = eval_textvqa_layer(
        model,
        tokenizer,
        image_processor,
        textvqa_questions,
        args.textvqa_image_folder,
        args.conv_mode,
        device,
    )
    results["baseline"] = {
        "scienceqa_acc": baseline_sqa,
        "textvqa_acc": baseline_textvqa,
    }
    print(
        "Baseline | "
        f"ScienceQA: {baseline_sqa:.4f} | TextVQA: {baseline_textvqa:.4f}"
    )

    for layer in tqdm(layers_to_test, desc="Layer Sweep"):
        print(f"\n=== Evaluating layer {layer} (full image token pruning) ===")
        configure_full_prune(
            model,
            agg_layer=layer,
            sys_length=args.fast_v_sys_length,
            image_token_length=args.fast_v_image_token_length,
        )

        sqa_acc = eval_scienceqa_layer(
            model,
            tokenizer,
            image_processor,
            scienceqa_questions,
            args.scienceqa_image_folder,
            args.conv_mode,
            device,
        )
        textvqa_acc = eval_textvqa_layer(
            model,
            tokenizer,
            image_processor,
            textvqa_questions,
            args.textvqa_image_folder,
            args.conv_mode,
            device,
        )

        results["layers"].append(layer)
        results["scienceqa_acc"].append(sqa_acc)
        results["textvqa_acc"].append(textvqa_acc)

        print(
            f"Layer {layer} | ScienceQA: {sqa_acc:.4f} | TextVQA: {textvqa_acc:.4f}"
        )

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {args.output_file}")

    plot_results(results, args.output_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")

    parser.add_argument("--scienceqa-question-file", type=str, required=True)
    parser.add_argument("--scienceqa-image-folder", type=str, required=True)

    parser.add_argument("--textvqa-question-file", type=str, required=True)
    parser.add_argument("--textvqa-image-folder", type=str, required=True)

    parser.add_argument("--num-samples-per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--fast-v-sys-length", type=int, default=35)
    parser.add_argument("--fast-v-image-token-length", type=int, default=576)
    parser.add_argument("--min-layer", type=int, default=None)
    parser.add_argument("--max-layer", type=int, default=None)

    parser.add_argument("--output-file", type=str, default="layerwise_full_image_prune_results.json")
    parser.add_argument("--output-plot", type=str, default="layerwise_full_image_prune_plot.png")

    args = parser.parse_args()
    run_experiment(args)