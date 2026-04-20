import argparse
import os
import json
from tqdm import tqdm
import sys
import math
import time
import random


def _preparse_gpu_id(argv, default="2"):
    """Read --gpu-id early so CUDA_VISIBLE_DEVICES is set before importing torch."""
    for i, token in enumerate(argv):
        if token == "--gpu-id" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--gpu-id="):
            return token.split("=", 1)[1]
    return default


# Do not inherit pre-existing CUDA_VISIBLE_DEVICES from shell by default.
# If user does not pass --gpu-id, use GPU 2.
_early_gpu_id = _preparse_gpu_id(sys.argv, default="2")
os.environ["CUDA_VISIBLE_DEVICES"] = _early_gpu_id

import torch

# Must be set before importing llava/transformers to take effect in huggingface_hub constants.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image


def _safe_prob_from_vector(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert non-negative vector to probability vector safely."""
    vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vec = vec.clamp_min(0.0)
    s = vec.sum()
    if s.item() <= 0:
        return torch.full_like(vec, 1.0 / vec.numel())
    return (vec / s).clamp_min(eps)


def _normalized_jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence normalized to [0, 1]."""
    p = _safe_prob_from_vector(p, eps=eps)
    q = _safe_prob_from_vector(q, eps=eps)
    m = 0.5 * (p + q)

    kl_pm = torch.sum(p * torch.log((p / m).clamp_min(eps)))
    kl_qm = torch.sum(q * torch.log((q / m).clamp_min(eps)))
    jsd = 0.5 * (kl_pm + kl_qm)
    return float((jsd / math.log(2.0)).item())


def calculate_mme_scores(results):
    perception_cats = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"]
    cognition_cats = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]

    cat_results = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_results:
            cat_results[cat] = []
        cat_results[cat].append(r)

    scores = {}
    perception_score = 0
    cognition_score = 0

    print(f"\n{'Category':<25} {'Acc':<10} {'Acc+':<10} {'Score':<10}")
    print("-" * 60)

    for cat, items in cat_results.items():
        correct = sum(1 for x in items if x["pred"].lower() == x["gt"].lower())
        acc = correct / len(items) * 100

        img_groups = {}
        for x in items:
            qid = x["question_id"]
            if qid not in img_groups:
                img_groups[qid] = []
            img_groups[qid].append(x["pred"].lower() == x["gt"].lower())

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


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def _sample_items(items, num_samples: int, seed: int):
    if num_samples is None or num_samples <= 0 or num_samples >= len(items):
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    picked = idx[:num_samples]
    return [items[i] for i in picked]


def _detect_dataset_type(question_file: str, payload):
    if isinstance(payload, list) and len(payload) > 0:
        first = payload[0]
        if isinstance(first, dict):
            if "image_file" in first and "question" in first:
                return "mme"
            if "conversations" in first and "image" in first:
                return "scienceqa"
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return "textvqa"

    lower_name = os.path.basename(question_file).lower()
    if "scienceqa" in lower_name:
        return "scienceqa"
    if "textvqa" in lower_name:
        return "textvqa"
    return "mme"


def _load_question_items(question_file: str, dataset_type: str):
    resolved = os.path.expanduser(question_file)
    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        payload = json.load(f)

    detected = _detect_dataset_type(resolved, payload) if dataset_type == "auto" else dataset_type
    if detected == "textvqa":
        if not isinstance(payload, dict) or "data" not in payload:
            raise ValueError(f"TextVQA question file missing 'data' array: {resolved}")
        return payload["data"], detected
    if not isinstance(payload, list):
        raise ValueError(f"Expected list questions for dataset={detected}, got {type(payload)}")
    return payload, detected


def _find_textvqa_image_path(image_folder: str, image_id):
    base = str(image_id)
    candidates = [base, base + ".jpg", base + ".png", base + ".jpeg"]
    for cand in candidates:
        p = os.path.join(image_folder, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot resolve TextVQA image for image_id={image_id} under {image_folder}")


def _normalize_sample(sample: dict, dataset_type: str, image_folder: str):
    if dataset_type == "mme":
        image_file = sample["image_file"]
        return {
            "question_id": sample.get("question_id", sample.get("id", "unknown")),
            "question": sample["question"],
            "answer": sample.get("answer", ""),
            "category": sample.get("category", "mme"),
            "image_path": os.path.join(image_folder, image_file),
        }

    if dataset_type == "scienceqa":
        conv = sample.get("conversations", [])
        if len(conv) < 2:
            raise ValueError("ScienceQA sample conversations is incomplete")
        raw_q = conv[0].get("value", "")
        q = raw_q.replace("<image>", "").strip()
        image_file = sample["image"]
        return {
            "question_id": sample.get("id", "unknown"),
            "question": q,
            "answer": conv[1].get("value", "").strip(),
            "category": sample.get("subject", "scienceqa"),
            "image_path": os.path.join(image_folder, image_file),
        }

    if dataset_type == "textvqa":
        return {
            "question_id": sample.get("question_id", sample.get("image_id", "unknown")),
            "question": sample["question"].strip(),
            "answer": sample.get("answers", [""])[0] if sample.get("answers") else "",
            "category": "textvqa",
            "image_path": _find_textvqa_image_path(image_folder, sample.get("image_id")),
        }

    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def _safe_dirname(name: str) -> str:
    return str(name).strip().lower().replace("/", "_").replace(" ", "_")


def _safe_name_component(name: str) -> str:
    text = str(name).strip()
    if not text:
        return "unknown"
    # Keep only filename-safe chars to avoid accidental path/encoding issues.
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in text)


def _bidirectional_change(arr: np.ndarray) -> np.ndarray:
    left = np.zeros_like(arr)
    right = np.zeros_like(arr)
    if arr.size > 1:
        left[1:] = np.abs(arr[1:] - arr[:-1])
        right[:-1] = np.abs(arr[1:] - arr[:-1])
    return 0.5 * (left + right)


def _max_norm(arr: np.ndarray) -> np.ndarray:
    m = float(np.max(arr)) if arr.size > 0 else 0.0
    if m <= 1e-12:
        return np.zeros_like(arr)
    return arr / m


def _parse_stage_ranges(raw: str):
    ranges = []
    for seg in raw.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "-" in seg:
            lo, hi = seg.split("-", 1)
        elif ":" in seg:
            lo, hi = seg.split(":", 1)
        else:
            lo, hi = seg, seg
        lo_i, hi_i = int(lo), int(hi)
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
        ranges.append((lo_i, hi_i))
    return ranges


def summarize_blocks(overlap_matrix, stages, boundaries):
    n = overlap_matrix.shape[0]
    block_stats = []
    for i, (l_start, l_end) in enumerate(stages, start=1):
        block = overlap_matrix[l_start:l_end + 1, l_start:l_end + 1]
        block_len = l_end - l_start + 1
        if block_len > 1:
            tri = block[np.triu_indices(block_len, k=1)]
            intra_mean = float(np.mean(tri)) if tri.size > 0 else 1.0
        else:
            intra_mean = 1.0

        block_stats.append(
            {
                "block_id": i,
                "layer_start": int(l_start),
                "layer_end": int(l_end),
                "num_layers": int(block_len),
                "intra_overlap_mean": intra_mean,
            }
        )

    boundary_stats = []
    for i, b in enumerate(boundaries):
        adjacent = float(overlap_matrix[b, b + 1]) if 0 <= b < n - 1 else 0.0
        boundary_stats.append(
            {
                "split_after_layer": int(b),
                "left_block": int(i + 1),
                "right_block": int(i + 2),
                "adjacent_overlap": adjacent,
                "boundary_strength": float(1.0 - adjacent),
            }
        )

    return {
        "num_layers": int(n),
        "num_blocks": int(len(stages)),
        "blocks": block_stats,
        "boundaries": boundary_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, default="2", help="使用的物理GPU编号，例如 0 或 1")
    parser.add_argument("--model-path", type=str, default="/root/nfs/model/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/root/nfs/code/dataset/TextVQA/train_images")
    parser.add_argument("--question-file", type=str, default="/root/nfs/code/dataset/TextVQA/TextVQA_0.5.1_val.json")
    # parser.add_argument("--image-folder", type=str, default="/root/nfs/code/HiMAP/data/scienceqa/images/test")
    # parser.add_argument("--question-file", type=str, default="/root/nfs/code/HiMAP/data/scienceqa/himap-inference-MCQ.json")
    # parser.add_argument("--image-folder", type=str, default="/root/nfs/code/HiMAP/data/MME/images/test")
    # parser.add_argument("--question-file", type=str, default="/root/nfs/code/HiMAP/data/MME/MME_test.json")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="textvqa",
        choices=["auto", "mme", "scienceqa", "textvqa"],
        help="数据集类型。auto 会根据 question-file 自动推断。",
    )
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--num-observation-samples", type=int, default=6, help="观测模式采样样本数（SQA/TextVQA建议6）")
    parser.add_argument("--sample-seed", type=int, default=42, help="采样随机种子")

    # HiMAP hyperparameter
    parser.add_argument("--use-hmap-v", default=False, action="store_true")
    parser.add_argument("--sys-length", type=int, required=False)
    parser.add_argument("--img-length", type=int, required=False)
    parser.add_argument("--hmap-v-attn-txt-layer", type=int, required=False)
    parser.add_argument("--hmap-v-attn-img-layer", type=int, required=False)
    parser.add_argument("--hmap-v-attn-txt-rank", type=int, required=False)
    parser.add_argument("--hmap-v-attn-img-rank", type=int, required=False)
    parser.add_argument("--cut-off-layer", type=int, required=False)

    # fastv config
    parser.add_argument("--use-fast-v", default=False, action="store_true")
    parser.add_argument("--fast-v-sys-length", type=int, required=False)
    parser.add_argument("--fast-v-image-token-length", type=int, required=False)
    parser.add_argument("--fast-v-attention-rank", type=int, required=False)
    parser.add_argument("--fast-v-agg-layer", type=int, required=False)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Observation config
    parser.add_argument(
        "--topk-img-token-percent",
        type=float,
        default=10.0,
        help="每一层提取Top-K%%最重要Image Token，K按image token总数百分比计算（例如10表示Top-10%%）",
    )
    parser.add_argument("--heatmap-save-dir", type=str, default="./attn_heatmaps_text2img", help="注意力重叠率热力图保存路径")
    parser.add_argument(
        "--abrupt-stage-ranges",
        type=str,
        default="2-4,5-15,16-20",
        help="分段突变检测区间，格式: 0-4,5-15,16-20",
    )
    parser.add_argument(
        "--block-report-save-dir",
        type=str,
        default="./attn_heatmaps_text2img/block_reports",
        help="对角线分块报告(json)保存目录",
    )

    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_id:
        print(
            f"[Warning] Early GPU binding ({os.environ.get('CUDA_VISIBLE_DEVICES')}) "
            f"!= parsed --gpu-id ({args.gpu_id}). Use CLI --gpu-id only once."
        )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        print(f"[Device] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[Device] Using GPU physical id={args.gpu_id} (mapped to cuda:0)")
        print(f"[Device] cuda:0 name={torch.cuda.get_device_name(device)}")
        print(f"[Device] free={free_gb:.2f} GiB / total={total_gb:.2f} GiB before model load")
    else:
        print("[Device] CUDA unavailable, fallback to CPU")

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if args.use_hmap_v is True:
        model.config.use_hmap_v = True
        model.config.hmap_v_sys_length = args.sys_length
        model.config.hmap_v_img_length = args.img_length
        model.config.hmap_v_attn_txt_layer = args.hmap_v_attn_txt_layer
        model.config.hmap_v_attn_img_layer = args.hmap_v_attn_img_layer
        model.config.hmap_v_attn_txt_rank = args.hmap_v_attn_txt_rank
        model.config.hmap_v_attn_img_rank = args.hmap_v_attn_img_rank
        model.config.cut_off_layer = args.cut_off_layer
        print("HiMAP TECHNIQUE WILL BE USED ------")
        model.model.reset_hmapv()
    elif args.use_fast_v is True:
        model.config.use_fast_v = True
        model.config.fast_v_sys_length = args.fast_v_sys_length
        model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.config.fast_v_agg_layer = args.fast_v_agg_layer
        print("FASTV TECHNIQUE WILL BE USED ------")
        model.model.reset_fastv()
    else:
        model.config.use_hmap_v = False
        print("NO TOKEN PRUNING TECHNIQUE WILL BE USED ------")

    questions, dataset_type = _load_question_items(args.question_file, args.dataset_type)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if dataset_type == "mme":
        # MME 默认保持每类别取1条的观测习惯。
        seen_categories = set()
        obs_questions = []
        for q in questions:
            cat = q.get("category", "mme")
            if cat not in seen_categories:
                seen_categories.add(cat)
                obs_questions.append(q)
        questions = obs_questions
        print(
            f"\n[Observation Mode] dataset=MME，每个类别采样1条，共 {len(questions)} 个样本。"
        )
    else:
        questions = _sample_items(questions, args.num_observation_samples, args.sample_seed)
        print(
            f"\n[Observation Mode] dataset={dataset_type}，采样 {len(questions)} 条样本 "
            f"(requested={args.num_observation_samples}, seed={args.sample_seed})。"
        )

    dataset_dir = _safe_dirname(dataset_type)
    heatmap_output_dir = os.path.join(args.heatmap_save_dir, dataset_dir)
    report_output_dir = os.path.join(args.block_report_save_dir, dataset_dir)
    os.makedirs(heatmap_output_dir, exist_ok=True)
    os.makedirs(report_output_dir, exist_ok=True)
    print(f"[Output] Heatmaps will be saved to: {heatmap_output_dir}")
    print(f"[Output] Block reports will be saved to: {report_output_dir}")

    num_sample = len(questions)
    total_latency = 0.0
    results = []

    import gc

    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    for i, line in enumerate(tqdm(questions)):
        sample = _normalize_sample(line, dataset_type, args.image_folder)
        idx = sample["question_id"]
        qs = sample["question"]
        label = sample["answer"]
        category = sample["category"]
        image_path = sample["image_path"]

        cur_prompt = qs
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        if use_cuda:
            images = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16)
        else:
            images = image_tensor.unsqueeze(0).float()

        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        cur_prompt = "<image>" + "\n" + cur_prompt
        qs = qs + "\n" + "Answer the question using a single word or phrase."
        cur_prompt = cur_prompt + "\n" + "Answer the question using a single word or phrase."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        if use_cuda:
            input_ids = input_ids.to(device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            t0 = time.time()

            # 仅做prefill前向，提取 attention。
            forward_outputs = model(
                input_ids=input_ids,
                images=images,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

            if hasattr(forward_outputs, "attentions") and forward_outputs.attentions is not None:
                prefill_attns = forward_outputs.attentions

                has_nan = any(torch.isnan(l_attn).any().item() for l_attn in prefill_attns)
                print(f"\n[Check] Category: '{category}' - 提取的 Attention Matrix 是否包含 NaN 值: {has_nan}")

                num_layers = len(prefill_attns)
                orig_seq_len = input_ids.shape[1]

                first_layer_seq_len = prefill_attns[0].shape[-1]
                num_image_tokens = first_layer_seq_len - orig_seq_len + 1

                image_start_idx = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
                image_end_idx = image_start_idx + num_image_tokens

                # prompt token 索引 = 序列中 image token 区间后的全部 token。
                prompt_indices = torch.arange(image_end_idx, first_layer_seq_len, device=input_ids.device)


                if prompt_indices.numel() == 0:
                    print(f"[Warning] Category: '{category}' 没有可用的 prompt token，跳过该样本。")
                else:
                    layer_topk_indices = []
                    layer_topk_masked_vectors = []
                    layer_topk_entropy_values = []
                    valid_layers = 0
                    selected_k = None

                    for l in range(num_layers):
                        attn = prefill_attns[l][0]
                        current_seq_len = attn.shape[-1]

                        if current_seq_len != first_layer_seq_len:
                            print(
                                f"[Warning] 层 {l} 的序列长度({current_seq_len})与第一层({first_layer_seq_len})不同！"
                                "这通常由于启用了Token Pruning。截断之后的分析。"
                            )
                            break

                        # 新计算方式: prompt token -> image token 的注意力。
                        # 形状: [num_heads, num_prompt_tokens, num_image_tokens]
                        prompt_to_img_attn = attn[:, prompt_indices, image_start_idx:image_end_idx]

                        if has_nan:
                            prompt_to_img_attn = torch.nan_to_num(prompt_to_img_attn, nan=0.0)

                        # 聚合 heads 与 prompt tokens，得到每个 image token 的重要性分数。
                        img_attn = prompt_to_img_attn.mean(dim=0).mean(dim=0)

                        k = max(1, int(math.ceil(img_attn.shape[0] * args.topk_img_token_percent / 100.0)))
                        if k > 0:
                            topk_vals, topk_inds = torch.topk(img_attn, k)
                            layer_topk_indices.append(set(topk_inds.tolist()))

                            topk_masked_vec = torch.zeros_like(img_attn)
                            topk_masked_vec[topk_inds] = topk_vals
                            layer_topk_masked_vectors.append(topk_masked_vec.detach().float().cpu())

                            topk_sum = topk_vals.sum()
                            if topk_sum.item() > 0 and k > 1:
                                topk_probs = (topk_vals / topk_sum).clamp_min(1e-12)
                                topk_entropy = float((-(topk_probs * torch.log(topk_probs)).sum() / math.log(k)).item())
                            else:
                                topk_entropy = 0.0
                            layer_topk_entropy_values.append(topk_entropy)

                            if selected_k is None:
                                selected_k = k
                            valid_layers += 1

                    if valid_layers > 0:
                        overlap_matrix = np.zeros((valid_layers, valid_layers))
                        for l_i in range(valid_layers):
                            for l_j in range(valid_layers):
                                intersection_size = len(layer_topk_indices[l_i].intersection(layer_topk_indices[l_j]))
                                overlap_matrix[l_i, l_j] = intersection_size / selected_k

                        # 相邻层 Top-K 子空间余弦相似度
                        topk_cosine_sim_values = [np.nan]
                        for l in range(1, valid_layers):
                            prev_topk_vec = layer_topk_masked_vectors[l - 1]
                            curr_topk_vec = layer_topk_masked_vectors[l]
                            topk_cos_sim = torch.nn.functional.cosine_similarity(
                                curr_topk_vec.unsqueeze(0),
                                prev_topk_vec.unsqueeze(0),
                                dim=1,
                                eps=1e-8,
                            ).item()
                            topk_cosine_sim_values.append(float(topk_cos_sim))

                        # 相邻层 Top-K 子空间分布 JSD（归一化到 [0, 1]）
                        topk_jsd_values = [np.nan]
                        for l in range(1, valid_layers):
                            prev_topk_vec = layer_topk_masked_vectors[l - 1]
                            curr_topk_vec = layer_topk_masked_vectors[l]
                            topk_jsd_values.append(_normalized_jsd(curr_topk_vec, prev_topk_vec))

                        topk_entropy_np = np.array(layer_topk_entropy_values[:valid_layers], dtype=np.float32)
                        topk_cosine_np = np.array(topk_cosine_sim_values, dtype=np.float32)
                        topk_jsd_np = np.array(topk_jsd_values, dtype=np.float32)

                        safe_topk_entropy = np.nan_to_num(topk_entropy_np, nan=0.0)
                        safe_topk_jsd = np.nan_to_num(topk_jsd_np, nan=0.0)

                        entropy_bi_change = _bidirectional_change(safe_topk_entropy)
                        jsd_bi_change = _bidirectional_change(safe_topk_jsd)
                        jsd_norm = _max_norm(jsd_bi_change)
                        entropy_change_norm = _max_norm(entropy_bi_change)
                        combined_change_score = 0.5 * jsd_norm + 0.5 * entropy_change_norm

                        target_ranges = _parse_stage_ranges(args.abrupt_stage_ranges)
                        abrupt_nodes = []
                        for lo, hi in target_ranges:
                            lo_idx = max(0, lo)
                            hi_idx = min(valid_layers - 1, hi)
                            if lo_idx > hi_idx:
                                continue
                            segment_idx = np.arange(lo_idx, hi_idx + 1)
                            segment_scores = combined_change_score[segment_idx]
                            best_local = int(np.argmax(segment_scores))
                            abrupt_nodes.append(int(segment_idx[best_local]))

                        boundaries = sorted(list(set(abrupt_nodes)))
                        stages = []
                        cur = 0
                        for b in boundaries:
                            stages.append((cur, b))
                            cur = b + 1
                        stages.append((cur, valid_layers - 1))
                        summary = summarize_blocks(overlap_matrix, stages, boundaries)

                        print(
                            f"[Boundary Detection] Category='{category}' | "
                            f"boundaries={len(boundaries)} "
                            f"(stage-wise absolute change, ranges={args.abrupt_stage_ranges})"
                        )
                        for blk in summary["blocks"]:
                            print(
                                "  "
                                f"- Block {blk['block_id']}: layers {blk['layer_start']}~{blk['layer_end']} "
                                f"(n={blk['num_layers']}), intra_overlap_mean={blk['intra_overlap_mean']:.4f}"
                            )
                        if summary["boundaries"]:
                            for b in summary["boundaries"]:
                                print(
                                    "  "
                                    f"- Boundary after layer {b['split_after_layer']}: "
                                    f"adjacent_overlap={b['adjacent_overlap']:.4f}, "
                                    f"strength={b['boundary_strength']:.4f}"
                                )
                        else:
                            print("  - Boundary: none")

                        if len(boundaries) > 0:
                            print(
                                f"[Observation] Category: '{category}' 分段突变节点({len(boundaries)}个): "
                                + ", ".join(
                                    f"L{int(l)}(score={combined_change_score[int(l)]:.3f})" for l in boundaries
                                )
                            )

                        if combined_change_score.size > 0:
                            print(
                                "  - Formula score peaks: "
                                + ", ".join(
                                    [
                                        f"L{int(i)}={float(s):.3f}"
                                        for i, s in enumerate(combined_change_score)
                                    ]
                                )
                            )

                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(overlap_matrix, cmap="YlGnBu", vmin=0, vmax=1, ax=ax)

                        x_coords = np.arange(valid_layers) + 0.5
                        metric_to_layer = lambda m: m * (valid_layers - 1) + 0.5
                        layer_to_metric = lambda y: (y - 0.5) / (valid_layers - 1)

                        if valid_layers > 1:
                            ax.plot(
                                x_coords,
                                metric_to_layer(topk_entropy_np),
                                color="crimson",
                                marker="^",
                                linewidth=1.8,
                                markersize=4,
                                linestyle="--",
                                label="Top-K Entropy",
                            )
                            ax.plot(
                                x_coords,
                                metric_to_layer(topk_cosine_np),
                                color="darkorange",
                                marker="d",
                                linewidth=1.8,
                                markersize=4,
                                linestyle="--",
                                label="Top-K Cosine",
                            )
                            ax.plot(
                                x_coords,
                                metric_to_layer(topk_jsd_np),
                                color="darkmagenta",
                                marker="P",
                                linewidth=1.8,
                                markersize=5,
                                linestyle=":",
                                label="Top-K JSD",
                            )

                            secax = ax.secondary_yaxis("right", functions=(layer_to_metric, metric_to_layer))
                            secax.set_ylabel("Top-K Entropy / Cosine / JSD", fontsize=11)
                            secax.set_ylim(layer_to_metric(0.0), layer_to_metric(1.0))
                            ax.legend(
                                loc="center right",
                                bbox_to_anchor=(0.98, 0.5),
                                fontsize=8,
                                frameon=True,
                                borderaxespad=0.0,
                            )

                        # 按阶段突变点标注（与 overlap2 一致）
                        for rank, b in enumerate(boundaries, start=1):
                            boundary_pos = b + 0.5
                            y_entropy = metric_to_layer(float(topk_entropy_np[b])) if valid_layers > 1 else 0.5
                            y_jsd = metric_to_layer(float(topk_jsd_np[b])) if valid_layers > 1 else 0.5

                            ax.axvline(boundary_pos, color="black", linestyle="-.", linewidth=0.9, alpha=0.45)
                            ax.scatter([boundary_pos], [y_entropy], color="firebrick", s=22, zorder=6)
                            ax.scatter([boundary_pos], [y_jsd], color="purple", s=22, zorder=6)
                            ax.text(
                                boundary_pos + 0.08,
                                min(y_jsd + 0.12, valid_layers - 0.55),
                                f"N{rank}:L{int(b)}",
                                fontsize=7,
                                color="black",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                            )

                        safe_cat_name = _safe_name_component(category)
                        safe_qid = _safe_name_component(idx)
                        unique_tag = f"{safe_qid}_i{i:04d}"
                        k_percent_str = f"{args.topk_img_token_percent:g}"
                        safe_k_percent = k_percent_str.replace(".", "p")
                        ax.set_title(
                            f"Prompt->Image Token Top-{k_percent_str}% Overlap + Top-K Trends (k={selected_k})\\n"
                            f"Category: {category} | stage abrupt nodes={len(boundaries)}",
                            fontsize=14,
                        )
                        ax.set_xlabel("Layer", fontsize=12)
                        ax.set_ylabel("Layer", fontsize=12)

                        save_path = os.path.join(
                            heatmap_output_dir,
                            f"heatmap_prompt2img_{safe_cat_name}_{unique_tag}_top{safe_k_percent}pct.png",
                        )
                        fig.savefig(save_path, bbox_inches="tight", dpi=150)
                        plt.close(fig)
                        print(f"[Observation] Category: '{category}' 的热力图已保存至: {save_path}")

                        report_path = os.path.join(
                            report_output_dir,
                            f"blocks_prompt2img_{safe_cat_name}_{unique_tag}_top{safe_k_percent}pct.json",
                        )
                        report_obj = {
                            "category": category,
                            "topk_img_token_percent": float(args.topk_img_token_percent),
                            "selected_k": int(selected_k),
                            "abrupt_stage_ranges": args.abrupt_stage_ranges,
                            "chosen_num_boundaries": int(len(boundaries)),
                            "abrupt_nodes": [int(x) for x in boundaries],
                            "formula_scores": {
                                "combined_change_score": combined_change_score.astype(np.float64).tolist(),
                                "entropy_bi_change": entropy_bi_change.astype(np.float64).tolist(),
                                "jsd_bi_change": jsd_bi_change.astype(np.float64).tolist(),
                            },
                            "summary": summary,
                        }
                        with open(report_path, "w", encoding="utf-8") as f:
                            json.dump(report_obj, f, ensure_ascii=False, indent=2)
                        print(f"[Block Detection] Category: '{category}' 的分块报告已保存至: {report_path}")

            del forward_outputs
            if "prefill_attns" in locals():
                del prefill_attns
            if use_cuda:
                torch.cuda.empty_cache()

            if getattr(args, "use_hmap_v", False) and args.use_hmap_v:
                model.model.reset_hmapv()
            elif getattr(args, "use_fast_v", False) and args.use_fast_v:
                model.model.reset_fastv()

    # 保留评测逻辑模板，当前观测脚本默认只做注意力提取与热力图输出。
    _ = (num_sample, total_latency, results, calculate_mme_scores)
