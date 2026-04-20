import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm


def _preparse_gpu_id(argv: Sequence[str], default: str = "2") -> str:
    """Read --gpu-id before importing torch so CUDA_VISIBLE_DEVICES is effective."""
    for i, token in enumerate(argv):
        if token == "--gpu-id" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--gpu-id="):
            return token.split("=", 1)[1]
    return default


# Keep behavior aligned with existing scripts in this repo.
_early_gpu_id = _preparse_gpu_id(sys.argv, default="2")
os.environ["CUDA_VISIBLE_DEVICES"] = _early_gpu_id
# Must be set before importing llava/transformers.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

import torch

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
from PIL import Image


def _safe_prob_from_vector(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vec = vec.clamp_min(0.0)
    s = vec.sum()
    if s.item() <= 0:
        return torch.full_like(vec, 1.0 / vec.numel())
    return (vec / s).clamp_min(eps)


def _normalized_jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = _safe_prob_from_vector(p, eps=eps)
    q = _safe_prob_from_vector(q, eps=eps)
    m = 0.5 * (p + q)

    kl_pm = torch.sum(p * torch.log((p / m).clamp_min(eps)))
    kl_qm = torch.sum(q * torch.log((q / m).clamp_min(eps)))
    jsd = 0.5 * (kl_pm + kl_qm)
    return float((jsd / math.log(2.0)).item())


def _parse_ranges(ranges_str: str, layer_index_base: int) -> List[Tuple[int, int]]:
    """Parse ranges like '2-8,9-20,21-31' to inclusive 0-based tuples."""
    ranges: List[Tuple[int, int]] = []
    for part in ranges_str.split(","):
        s = part.strip()
        if not s:
            continue
        if "-" not in s:
            raise ValueError(f"invalid stage range '{s}', expected lo-hi")
        lo_str, hi_str = s.split("-", 1)
        lo = int(lo_str.strip())
        hi = int(hi_str.strip())
        if layer_index_base == 1:
            lo -= 1
            hi -= 1
        if lo < 0 or hi < 0:
            raise ValueError("stage ranges must be non-negative after base conversion")
        if lo > hi:
            raise ValueError(f"invalid range '{s}', expected lo<=hi")
        ranges.append((lo, hi))

    if len(ranges) != 3:
        raise ValueError("--stage-ranges must contain exactly 3 ranges, e.g. 2-8,9-20,21-31")
    return ranges


def _parse_prune_ratios(ratios_str: str) -> Tuple[float, float, float]:
    """Parse ratios like '0.2,0.3,1.0' (fractions, not percentages)."""
    vals = [float(x.strip()) for x in ratios_str.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("--prune-ratios must contain exactly 3 values, e.g. 0.2,0.3,1.0")

    p1, p2, p3 = vals
    if p1 < 0 or p2 < 0 or p3 < 0 or p1 > 1 or p2 > 1 or p3 > 1:
        raise ValueError("each prune ratio must be in [0, 1]")
    if p1 + p2 > 1:
        raise ValueError("stage1+stage2 prune ratios cannot exceed 1")
    return p1, p2, p3


def _bidirectional_change(arr: torch.Tensor) -> torch.Tensor:
    left = torch.zeros_like(arr)
    right = torch.zeros_like(arr)
    if arr.numel() > 1:
        diff = torch.abs(arr[1:] - arr[:-1])
        left[1:] = diff
        right[:-1] = diff
    return 0.5 * (left + right)


def _max_norm(arr: torch.Tensor) -> torch.Tensor:
    if arr.numel() == 0:
        return arr
    m = torch.max(arr)
    if m.item() <= 1e-12:
        return torch.zeros_like(arr)
    return arr / m


def _ensure_three_increasing_layers(nodes: List[int], max_layer: int) -> List[int]:
    """Make sure we get 3 strictly increasing, in-range layer indices."""
    if max_layer < 3:
        return [1, 2, 3][: max_layer + 1]

    if len(nodes) < 3:
        fallback = [max(1, max_layer // 4), max(2, max_layer // 2), max(3, (3 * max_layer) // 4)]
        nodes = nodes + fallback

    nodes = sorted(nodes[:3])
    nodes[0] = max(1, min(nodes[0], max_layer - 2))
    nodes[1] = max(nodes[0] + 1, min(nodes[1], max_layer - 1))
    nodes[2] = max(nodes[1] + 1, min(nodes[2], max_layer))
    return nodes


def _analyze_three_stages(
    prefill_attns: Sequence[torch.Tensor],
    input_ids: torch.Tensor,
    topk_img_token_percent: float,
    stage_ranges: Sequence[Tuple[int, int]],
) -> Dict[str, object]:
    num_layers = len(prefill_attns)
    orig_seq_len = int(input_ids.shape[1])
    first_layer_seq_len = int(prefill_attns[0].shape[-1])

    image_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if image_pos.numel() == 0:
        raise RuntimeError("failed to locate IMAGE_TOKEN_INDEX in input_ids")

    image_start_idx = int(image_pos[0].item())
    num_image_tokens = first_layer_seq_len - orig_seq_len + 1
    image_end_idx = image_start_idx + num_image_tokens

    layer_topk_masked_vectors: List[torch.Tensor] = []
    layer_topk_entropy_values: List[float] = []
    valid_layers = 0

    for l in range(num_layers):
        attn = prefill_attns[l][0]
        current_seq_len = int(attn.shape[-1])
        if current_seq_len != first_layer_seq_len:
            break

        last_token_attn = attn[:, -1, :]
        avg_attn = last_token_attn.mean(dim=0)
        img_attn = torch.nan_to_num(avg_attn[image_start_idx:image_end_idx], nan=0.0).float().cpu()

        k = max(1, int(math.ceil(img_attn.shape[0] * topk_img_token_percent / 100.0)))
        topk_vals, topk_inds = torch.topk(img_attn, k)

        topk_masked_vec = torch.zeros_like(img_attn)
        topk_masked_vec[topk_inds] = topk_vals
        layer_topk_masked_vectors.append(topk_masked_vec)

        topk_sum = topk_vals.sum()
        if topk_sum.item() > 0 and k > 1:
            topk_probs = (topk_vals / topk_sum).clamp_min(1e-12)
            topk_entropy = float((-(topk_probs * torch.log(topk_probs)).sum() / math.log(k)).item())
        else:
            topk_entropy = 0.0
        layer_topk_entropy_values.append(topk_entropy)
        valid_layers += 1

    if valid_layers < 4:
        raise RuntimeError("too few valid layers for three-stage detection")

    topk_jsd_values = [float("nan")]
    for l in range(1, valid_layers):
        topk_jsd_values.append(
            _normalized_jsd(layer_topk_masked_vectors[l], layer_topk_masked_vectors[l - 1])
        )

    safe_topk_jsd = torch.nan_to_num(torch.tensor(topk_jsd_values, dtype=torch.float32), nan=0.0)
    safe_topk_entropy = torch.nan_to_num(
        torch.tensor(layer_topk_entropy_values[:valid_layers], dtype=torch.float32),
        nan=0.0,
    )

    entropy_bi_change = _bidirectional_change(safe_topk_entropy)
    jsd_bi_change = _bidirectional_change(safe_topk_jsd)
    combined_change_score = 0.5 * _max_norm(jsd_bi_change) + 0.5 * _max_norm(entropy_bi_change)

    stage_layers: List[int] = []
    for lo, hi in stage_ranges:
        lo_idx = max(0, lo)
        hi_idx = min(valid_layers - 1, hi)
        if lo_idx > hi_idx:
            continue
        segment = combined_change_score[lo_idx : hi_idx + 1]
        best_local = int(torch.argmax(segment).item())
        stage_layers.append(lo_idx + best_local)

    stage_layers = _ensure_three_increasing_layers(stage_layers, valid_layers - 1)

    return {
        "stage_layers": stage_layers,
        "valid_layers": valid_layers,
        "num_image_tokens": num_image_tokens,
        "combined_change_score": [float(x) for x in combined_change_score.tolist()],
    }


def _compute_target_ranks(
    total_img_tokens: int,
    prune_ratios: Tuple[float, float, float],
) -> Tuple[int, int]:
    p1, p2, _ = prune_ratios
    prune1 = int(round(total_img_tokens * p1))
    prune2 = int(round(total_img_tokens * p2))

    keep_after_stage1 = max(total_img_tokens - prune1, 0)
    keep_after_stage2 = max(total_img_tokens - prune1 - prune2, 0)
    keep_after_stage2 = min(keep_after_stage2, keep_after_stage1)
    return keep_after_stage1, keep_after_stage2


def calculate_mme_scores(results: List[Dict[str, str]]):
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

    cat_results: Dict[str, List[Dict[str, str]]] = {}
    for r in results:
        cat_results.setdefault(r["category"], []).append(r)

    scores: Dict[str, float] = {}
    perception_score = 0.0
    cognition_score = 0.0

    print(f"\n{'Category':<25} {'Acc':<10} {'Acc+':<10} {'Score':<10}")
    print("-" * 60)

    for cat, items in cat_results.items():
        correct = sum(1 for x in items if x["pred"].lower() == x["gt"].lower())
        acc = correct / len(items) * 100

        img_groups: Dict[str, List[bool]] = {}
        for x in items:
            qid = x["question_id"]
            img_groups.setdefault(qid, []).append(x["pred"].lower() == x["gt"].lower())

        correct_pairs = sum(1 for v in img_groups.values() if all(v))
        acc_plus = correct_pairs / len(img_groups) * 100

        score = acc + acc_plus
        scores[cat] = score

        print(f"{cat:<25} {acc:<10.2f} {acc_plus:<10.2f} {score:<10.2f}")

        if cat in perception_cats:
            perception_score += score
        elif cat in cognition_cats:
            cognition_score += score

    print("-" * 60)
    print(f"Perception Score: {perception_score:.2f}")
    print(f"Cognition Score: {cognition_score:.2f}")
    print(f"Total MME Score: {perception_score + cognition_score:.2f}")

    return scores, perception_score, cognition_score


def split_list(lst: List[dict], n: int) -> List[List[dict]]:
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: List[dict], n: int, k: int) -> List[dict]:
    chunks = split_list(lst, n)
    return chunks[k]


def _to_display_layer(layer_idx: int, layer_index_base: int) -> int:
    return layer_idx + 1 if layer_index_base == 1 else layer_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MME inference with per-sample three-stage dynamic pruning")
    parser.add_argument("--gpu-id", type=str, default="2", help="physical GPU id")
    parser.add_argument("--model-path", type=str, default="/root/nfs/model/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/root/nfs/code/HiMAP/data/MME/images/test")
    parser.add_argument("--question-file", type=str, default="/root/nfs/code/HiMAP/data/MME/MME_test.json")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--single-pred-prompt", action="store_true")

    parser.add_argument("--temperature", type=float, default=0.0)

    # Dynamic three-stage detection parameters.
    parser.add_argument(
        "--topk-img-token-percent",
        type=float,
        default=10.0,
        help="Top-K percent used to build per-layer Top-K vectors for JSD/Entropy",
    )
    parser.add_argument(
        "--stage-ranges",
        type=str,
        default="2-8,9-20,21-31",
        help="three inclusive layer ranges used to pick one abrupt node each",
    )
    parser.add_argument(
        "--layer-index-base",
        type=int,
        choices=[0, 1],
        default=0,
        help="input/output layer index base for --stage-ranges and logs",
    )

    # Pruning ratios are fractions of total image token count.
    parser.add_argument(
        "--prune-ratios",
        type=str,
        default="0.2,0.3,1.0",
        help="stage1,stage2,stage3 prune ratios over total image tokens; default=20%,30%,100%",
    )

    # Prompt system token length can differ slightly by template/model.
    parser.add_argument(
        "--sys-length-fallback",
        type=int,
        default=35,
        help="fallback sys token length if automatic image position extraction fails",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="mme_results_three_stage_dynamic.json",
        help="output json file",
    )

    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_id:
        print(
            f"[Warning] Early GPU binding ({os.environ.get('CUDA_VISIBLE_DEVICES')}) "
            f"!= parsed --gpu-id ({args.gpu_id})."
        )

    stage_ranges = _parse_ranges(args.stage_ranges, layer_index_base=args.layer_index_base)
    prune_ratios = _parse_prune_ratios(args.prune_ratios)
    if abs(prune_ratios[2] - 1.0) > 1e-6:
        print("[Warning] Stage-3 ratio is only used for reporting. Runtime stage-3 always cuts all remaining image tokens.")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"[Device] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"| mapped device=cuda:0 | free={free_bytes / (1024**3):.2f}GiB/{total_bytes / (1024**3):.2f}GiB"
        )
    else:
        print("[Device] CUDA unavailable, fallback to CPU")

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    results: List[Dict[str, str]] = []
    stage_debug: List[Dict[str, object]] = []
    total_latency = 0.0

    import gc

    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    for line in tqdm(questions):
        idx = line.get("question_id")
        qs_raw = line["question"]
        label = line["answer"]
        category = line["category"]
        image_file = line["image_file"]

        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        if use_cuda:
            images = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16)
        else:
            images = image_tensor.unsqueeze(0).float()

        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs_raw
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs_raw

        qs = qs + "\n" + "Answer the question using a single word or phrase."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)
        if use_cuda:
            input_ids = input_ids.to(device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        # Stage detection pass: run without HiMAP pruning to keep full-depth attention maps.
        model.config.use_hmap_v = False
        if hasattr(model, "model") and hasattr(model.model, "reset_hmapv"):
            model.model.reset_hmapv()

        with torch.inference_mode():
            prefill_outputs = model(
                input_ids=input_ids,
                images=images,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        if not hasattr(prefill_outputs, "attentions") or prefill_outputs.attentions is None:
            raise RuntimeError("model forward did not return attentions for stage detection")

        stage_info = _analyze_three_stages(
            prefill_attns=prefill_outputs.attentions,
            input_ids=input_ids,
            topk_img_token_percent=args.topk_img_token_percent,
            stage_ranges=stage_ranges,
        )

        del prefill_outputs
        if use_cuda:
            torch.cuda.empty_cache()

        stage1_layer, stage2_layer, stage3_layer = stage_info["stage_layers"]
        total_img_tokens = int(stage_info["num_image_tokens"])
        txt_rank, img_rank = _compute_target_ranks(total_img_tokens, prune_ratios)

        image_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if image_pos.numel() > 0:
            sys_length = int(image_pos[0].item())
        else:
            sys_length = int(args.sys_length_fallback)

        # Apply dynamic three-stage pruning config for this sample.
        model.config.use_hmap_v = True
        model.config.hmap_v_sys_length = sys_length
        model.config.hmap_v_img_length = total_img_tokens
        model.config.hmap_v_attn_txt_layer = int(stage1_layer)
        model.config.hmap_v_attn_img_layer = int(stage2_layer)
        model.config.hmap_v_attn_txt_rank = int(txt_rank)
        model.config.hmap_v_attn_img_rank = int(img_rank)
        model.config.cut_off_layer = int(stage3_layer)
        if hasattr(model, "model") and hasattr(model.model, "reset_hmapv"):
            model.model.reset_hmapv()

        with torch.inference_mode():
            t0 = time.time()
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else 0.0,
                max_new_tokens=1024,
                use_cache=False,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
                output_attentions=False,
            )
            if use_cuda:
                torch.cuda.synchronize()
            total_latency += time.time() - t0

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids["sequences"][:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")

        outputs = tokenizer.batch_decode(output_ids["sequences"][:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        pred = outputs[:-1] if outputs.endswith(".") else outputs

        results.append(
            {
                "question_id": idx,
                "category": category,
                "pred": pred,
                "gt": label,
            }
        )

        stage_debug.append(
            {
                "question_id": idx,
                "category": category,
                "image_file": image_file,
                "total_img_tokens": total_img_tokens,
                "stage_layers_runtime_0based": [stage1_layer, stage2_layer, stage3_layer],
                "stage_layers_display": [
                    _to_display_layer(stage1_layer, args.layer_index_base),
                    _to_display_layer(stage2_layer, args.layer_index_base),
                    _to_display_layer(stage3_layer, args.layer_index_base),
                ],
                "stage_keep_tokens": [txt_rank, img_rank, 0],
                "prune_ratios": list(prune_ratios),
            }
        )

        # Reset transient states for next sample.
        if hasattr(model, "model") and hasattr(model.model, "reset_hmapv"):
            model.model.reset_hmapv()

    scores, perception_score, cognition_score = calculate_mme_scores(results)

    avg_latency = total_latency / max(len(questions), 1)
    final_results = {
        "scores": scores,
        "perception_score": perception_score,
        "cognition_score": cognition_score,
        "total_score": perception_score + cognition_score,
        "total_samples": len(questions),
        "avg_latency": avg_latency,
        "dynamic_three_stage_config": {
            "topk_img_token_percent": args.topk_img_token_percent,
            "stage_ranges_input": args.stage_ranges,
            "layer_index_base": args.layer_index_base,
            "prune_ratios": list(prune_ratios),
        },
        "sample_stage_details": stage_debug,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output_file}")
