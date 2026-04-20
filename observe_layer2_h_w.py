import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple



import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample MME/ScienceQA/TextVQA and compute layer-2 H_vis and W=1-JSD."
    )
    parser.add_argument("--model-path", type=str, default="/root/nfs/model/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mme-question-file", type=str, default="/root/nfs/code/HiMAP/data/MME/MME_test.json")
    parser.add_argument("--mme-image-folder", type=str, default="/root/nfs/code/HiMAP/data/MME/images/test")

    parser.add_argument(
        "--scienceqa-question-file",
        type=str,
        default="/root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json",
    )
    parser.add_argument(
        "--scienceqa-image-folder",
        type=str,
        default="/root/nfs/code/HiMAP/data/scienceqa/images/test",
    )

    parser.add_argument(
        "--textvqa-question-file",
        type=str,
        default="/root/nfs/code/dataset/TextVQA/TextVQA_0.5.1_val.json",
    )
    parser.add_argument(
        "--textvqa-image-folder",
        type=str,
        default="/root/nfs/code/dataset/TextVQA/train_images",
    )

    parser.add_argument("--output-json", type=str, default="layer2_h_w_summary.json")
    return parser.parse_args()


def _safe_prob(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    v = torch.nan_to_num(v.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    v = v.clamp_min(0.0)
    s = v.sum()
    if s.item() <= 0:
        return torch.full_like(v, 1.0 / max(1, v.numel()))
    p = (v / s).clamp_min(eps)
    p = p / p.sum()
    return p


def normalized_entropy(v: torch.Tensor) -> float:
    p = _safe_prob(v)
    n = p.numel()
    if n <= 1:
        return 0.0
    ent = -(p * torch.log(p)).sum() / math.log(n)
    return float(ent.item())


def visual_feature_dissimilarity(vis_feats: torch.Tensor, eps: float = 1e-12) -> float:
    """H_vis = 1 - mean cosine similarity among all visual-token pairs."""
    feats = torch.nan_to_num(vis_feats.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if feats.ndim != 2 or feats.shape[0] <= 1:
        return 0.0

    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(eps)
    sim = feats @ feats.transpose(0, 1)
    mean_sim = sim.mean()
    h_vis = 1.0 - float(mean_sim.item())
    return h_vis


def normalized_jsd(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12) -> float:
    p = _safe_prob(v1, eps=eps)
    q = _safe_prob(v2, eps=eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log((p / m).clamp_min(eps)))
    kl_qm = torch.sum(q * torch.log((q / m).clamp_min(eps)))
    jsd = 0.5 * (kl_pm + kl_qm)
    return float((jsd / math.log(2.0)).item())


def _find_textvqa_image_path(image_folder: str, image_id: str) -> str:
    base = str(image_id)
    candidates = [base, f"{base}.jpg", f"{base}.png", f"{base}.jpeg"]
    for c in candidates:
        p = os.path.join(image_folder, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"TextVQA image not found for image_id={image_id} under {image_folder}")


def _sample_list(items: List[dict], n: int, seed: int) -> List[dict]:
    if n <= 0 or n >= len(items):
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:n]]


def load_dataset_samples(args: argparse.Namespace, dataset_name: str) -> List[dict]:
    if dataset_name == "mme":
        with open(args.mme_question_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = _sample_list(data, args.num_samples, args.seed)
        out = []
        for s in samples:
            out.append(
                {
                    "id": str(s.get("question_id", "unknown")),
                    "question": s["question"].strip(),
                    "image_path": os.path.join(args.mme_image_folder, s["image_file"]),
                }
            )
        return out

    if dataset_name == "scienceqa":
        with open(args.scienceqa_question_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = _sample_list(data, args.num_samples, args.seed)
        out = []
        for s in samples:
            q = s["conversations"][0]["value"].replace("<image>", "").strip()
            out.append(
                {
                    "id": str(s.get("id", "unknown")),
                    "question": q,
                    "image_path": os.path.join(args.scienceqa_image_folder, s["image"]),
                }
            )
        return out

    if dataset_name == "textvqa":
        with open(args.textvqa_question_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        data = payload["data"]
        samples = _sample_list(data, args.num_samples, args.seed)
        out = []
        for s in samples:
            out.append(
                {
                    "id": str(s.get("question_id", s.get("image_id", "unknown"))),
                    "question": s["question"].strip(),
                    "image_path": _find_textvqa_image_path(args.textvqa_image_folder, s["image_id"]),
                }
            )
        return out

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_prompt(question: str, model_config, conv_mode: str) -> str:
    if getattr(model_config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    qs = qs + "\n" + "Answer the question using a single word or phrase."
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def extract_layer_vectors(
    model,
    tokenizer,
    image_processor,
    sample: dict,
    conv_mode: str,
    device: torch.device,
) -> Optional[Tuple[List[torch.Tensor], torch.Tensor]]:
    image = Image.open(sample["image_path"]).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    images = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16)
    prompt = build_prompt(sample["question"], model.config, conv_mode)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            images=images,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = outputs.attentions
    if attentions is None or len(attentions) < 3:
        return None

    seq_len = attentions[0].shape[-1]
    orig_seq_len = input_ids.shape[1]
    num_image_tokens = seq_len - orig_seq_len + 1
    if num_image_tokens <= 0:
        return None

    img_token_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)
    if len(img_token_pos[0]) == 0:
        return None

    image_start_idx = int(img_token_pos[0].item())
    image_end_idx = image_start_idx + num_image_tokens
    prompt_indices = torch.arange(image_end_idx, seq_len, device=device)
    if prompt_indices.numel() == 0:
        return None

    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) < 4:
        return None

    # hidden_states[0] is embeddings, hidden_states[3] is output of layer-2 (zero-based indexing).
    vis_feats_l2 = hidden_states[3][0, image_start_idx:image_end_idx, :].detach().float().cpu()
    if vis_feats_l2.numel() == 0:
        return None

    vectors: List[torch.Tensor] = []
    for layer_attn in attentions:
        # [heads, prompt_tokens, image_tokens] -> [image_tokens]
        p2i = layer_attn[0][:, prompt_indices, image_start_idx:image_end_idx]
        p2i = torch.nan_to_num(p2i, nan=0.0, posinf=0.0, neginf=0.0)
        vec = p2i.mean(dim=0).mean(dim=0).detach().float().cpu()
        vectors.append(vec)

    return vectors, vis_feats_l2


def eval_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    model,
    tokenizer,
    image_processor,
    device: torch.device,
) -> Dict[str, float]:
    samples = load_dataset_samples(args, dataset_name)
    h_values: List[float] = []
    w_values: List[float] = []
    jsd_values: List[float] = []

    for s in tqdm(samples, desc=f"{dataset_name} samples"):
        try:
            extract_out = extract_layer_vectors(model, tokenizer, image_processor, s, args.conv_mode, device)
            if extract_out is None:
                continue

            layer_vectors, vis_feats_l2 = extract_out
            if len(layer_vectors) < 3:
                continue

            # layer 2 is the 3rd layer in zero-based indexing
            v_l1 = layer_vectors[1]
            v_l2 = layer_vectors[2]
            h_l2 = visual_feature_dissimilarity(vis_feats_l2)
            jsd_l2 = normalized_jsd(v_l2, v_l1)
            w_l2 = 1.0 - jsd_l2

            h_values.append(h_l2)
            jsd_values.append(jsd_l2)
            w_values.append(w_l2)
        except Exception as e:
            print(f"[Warn] {dataset_name} sample {s.get('id', 'unknown')} skipped: {e}")
            continue

    if len(h_values) == 0 or len(jsd_values) == 0 or len(w_values) == 0:
        raise RuntimeError(f"No valid samples for {dataset_name}")

    h_mean = float(np.mean(h_values))
    h_std = float(np.std(h_values))
    jsd_mean = float(np.mean(jsd_values))
    w_mean = float(np.mean(w_values))
    w_std = float(np.std(w_values))

    return {
        "dataset": dataset_name,
        "requested_samples": int(args.num_samples),
        "valid_samples": int(min(len(h_values), len(jsd_values), len(w_values))),
        "H_vis_mean_layer2": h_mean,
        "H_vis_std_layer2": h_std,
        "JSD_mean_layer2": jsd_mean,
        "W_mean_layer2": w_mean,
        "W_std_layer2": w_std,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no GPU is visible.")

    device = torch.device("cuda:0")
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    model.eval()

    datasets = ["mme", "scienceqa", "textvqa"]
    results: Dict[str, Dict[str, float]] = {}

    for d in datasets:
        print(f"\n[Run] dataset={d}, num_samples={args.num_samples}")
        results[d] = eval_dataset(args, d, model, tokenizer, image_processor, device)

    avg_h = float(np.mean([results[d]["H_vis_mean_layer2"] for d in datasets]))
    avg_h_std = float(np.mean([results[d]["H_vis_std_layer2"] for d in datasets]))
    avg_w = float(np.mean([results[d]["W_mean_layer2"] for d in datasets]))
    avg_w_std = float(np.mean([results[d]["W_std_layer2"] for d in datasets]))

    summary = {
        "layer_index": 2,
        "num_samples_per_dataset": int(args.num_samples),
        "results": results,
        "average_across_datasets": {
            "H_vis_mean": avg_h,
            "H_vis_std_mean": avg_h_std,
            "W_mean": avg_w,
            "W_std_mean": avg_w_std,
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Layer-2 Observation Summary =====")
    for d in datasets:
        row = results[d]
        print(
            f"{d:10s} | valid={row['valid_samples']:3d} | "
            f"H_vis={row['H_vis_mean_layer2']:.6f}+/-{row['H_vis_std_layer2']:.6f} | "
            f"JSD={row['JSD_mean_layer2']:.6f} | "
            f"W={row['W_mean_layer2']:.6f}+/-{row['W_std_layer2']:.6f}"
        )
    print(f"average    | H_vis={avg_h:.6f}+/-{avg_h_std:.6f} | W={avg_w:.6f}+/-{avg_w_std:.6f}")
    print(f"saved to: {args.output_json}")


if __name__ == "__main__":
    main()
