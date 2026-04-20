import argparse
import os
import json
from tqdm import tqdm
import sys
import math
import time


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

# [ADDED FOR OBSERVATION EXPERIMENT] 导入绘图必需的库
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
    """Convert non-negative attention vector to a probability vector safely."""
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

    # Natural-log JSD upper bound is ln(2), normalize to [0, 1].
    return float((jsd / math.log(2.0)).item())

def calculate_mme_scores(results):
    # Define categories
    perception_cats =["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"]
    cognition_cats =["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
    
    cat_results = {}
    for r in results:
        cat = r['category']
        if cat not in cat_results:
            cat_results[cat] = []
        cat_results[cat].append(r)

    scores = {}
    perception_score = 0
    cognition_score = 0
    
    print(f"\n{'Category':<25} {'Acc':<10} {'Acc+':<10} {'Score':<10}")
    print("-" * 60)

    for cat, items in cat_results.items():
        # 1. Accuracy
        correct = sum(1 for x in items if x['pred'].lower() == x['gt'].lower())
        acc = correct / len(items) * 100
        
        # 2. Accuracy+ (Group by question_id/image pair)
        img_groups = {}
        for x in items:
            qid = x['question_id']
            if qid not in img_groups:
                img_groups[qid] = []
            img_groups[qid].append(x['pred'].lower() == x['gt'].lower())
        
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
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=str, default="2", help="使用的物理GPU编号，例如 0 或 1")
    parser.add_argument("--model-path", type=str, default="/root/nfs/model/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/root/nfs/code/HiMAP/data/MME/images/test")
    parser.add_argument("--question-file", type=str, default="/root/nfs/code/HiMAP/data/MME/MME_test.json")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--single-pred-prompt", action="store_true")
    
    # HiMAP hyperparameter
    parser.add_argument('--use-hmap-v', default=False, action='store_true')
    parser.add_argument('--sys-length', type=int, required=False)
    parser.add_argument('--img-length', type=int, required=False)
    parser.add_argument('--hmap-v-attn-txt-layer', type=int, required=False)
    parser.add_argument('--hmap-v-attn-img-layer', type=int, required=False)
    parser.add_argument('--hmap-v-attn-txt-rank', type=int, required=False)
    parser.add_argument('--hmap-v-attn-img-rank', type=int, required=False)
    parser.add_argument('--cut-off-layer', type=int, required=False)
    
    # fastv config
    parser.add_argument('--use-fast-v', default=False, action='store_true')
    parser.add_argument('--fast-v-sys-length', type=int, required=False)
    parser.add_argument('--fast-v-image-token-length', type=int, required=False)
    parser.add_argument('--fast-v-attention-rank', type=int, required=False)
    parser.add_argument('--fast-v-agg-layer', type=int, required=False)
    # JSD+Entropy adaptive 3-stage pruning config
    parser.add_argument('--use-jsd-entropy-pruning', default=False, action='store_true')
    parser.add_argument('--jsd-entropy-sys-length', type=int, required=False)
    parser.add_argument('--jsd-entropy-image-token-length', type=int, required=False)
    parser.add_argument('--jsd-entropy-topk-percent', type=float, default=10.0)
    parser.add_argument('--jsd-entropy-stage-ranges', type=str, default='0-4,5-17,18-20', help='三阶段候选层区间，格式: 2-8,9-20,21-31')
    parser.add_argument('--jsd-entropy-stage-prune-ratios', type=str, default='0.2,0.3,0.5', help='三阶段剪枝比例（针对总image token），格式: 0.2,0.3,0.5')
    parser.add_argument("--temperature", type=float, default=0.0)
    
    # [ADDED FOR OBSERVATION EXPERIMENT] 观测实验额外参数
    parser.add_argument(
        "--topk-img-token-percent",
        type=float,
        default=10.0,
        help="每一层提取Top-K%%最重要Image Token，K按image token总数百分比计算（例如10表示Top-10%%）",
    )
    parser.add_argument("--heatmap-save-dir", type=str, default="./attn_heatmaps", help="注意力重叠率热力图保存路径")
    
    args = parser.parse_args()

    # 由于 torch 导入前已预解析 gpu-id，这里只做一致性检查。
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
    
    if os.path.exists(model_path):
        model_name = get_model_name_from_path(model_path)
    else:
        model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if args.use_hmap_v == True:
        model.config.use_hmap_v = True
        model.config.use_fast_v = False
        model.config.use_jsd_entropy_pruning = False
        model.config.hmap_v_sys_length = args.sys_length
        model.config.hmap_v_img_length = args.img_length
        model.config.hmap_v_attn_txt_layer = args.hmap_v_attn_txt_layer
        model.config.hmap_v_attn_img_layer = args.hmap_v_attn_img_layer
        model.config.hmap_v_attn_txt_rank = args.hmap_v_attn_txt_rank
        model.config.hmap_v_attn_img_rank = args.hmap_v_attn_img_rank
        model.config.cut_off_layer = args.cut_off_layer
        print('HiMAP TECHNIQUE WILL BE USED ------')   
        model.model.reset_hmapv()   
    elif args.use_fast_v == True:
        model.config.use_hmap_v = False
        model.config.use_fast_v = True
        model.config.use_jsd_entropy_pruning = False
        model.config.fast_v_sys_length = args.fast_v_sys_length
        model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.config.fast_v_agg_layer = args.fast_v_agg_layer
        print('FASTV TECHNIQUE WILL BE USED ------')
        model.model.reset_fastv()
    elif args.use_jsd_entropy_pruning == True:
        def _parse_ranges(raw: str):
            ranges = []
            for seg in raw.split(','):
                seg = seg.strip()
                if not seg:
                    continue
                if '-' in seg:
                    lo, hi = seg.split('-', 1)
                elif ':' in seg:
                    lo, hi = seg.split(':', 1)
                else:
                    lo, hi = seg, seg
                ranges.append((int(lo), int(hi)))
            return ranges

        def _parse_ratios(raw: str):
            ratios = [float(x.strip()) for x in raw.split(',') if x.strip()]
            return ratios if len(ratios) > 0 else [0.2, 0.3, 0.5]

        model.config.use_hmap_v = False
        model.config.use_fast_v = False
        model.config.use_jsd_entropy_pruning = True
        model.config.jsd_entropy_sys_length = args.jsd_entropy_sys_length
        model.config.jsd_entropy_image_token_length = args.jsd_entropy_image_token_length
        model.config.jsd_entropy_topk_percent = args.jsd_entropy_topk_percent
        model.config.jsd_entropy_stage_ranges = _parse_ranges(args.jsd_entropy_stage_ranges)
        model.config.jsd_entropy_stage_prune_ratios = _parse_ratios(args.jsd_entropy_stage_prune_ratios)
        print('JSD+ENTROPY ADAPTIVE 3-STAGE PRUNING WILL BE USED ------')
        model.model.reset_jsd_entropy_pruning()
    else:
        model.config.use_hmap_v = False
        model.config.use_fast_v = False
        model.config.use_jsd_entropy_pruning = False
        print('NO TOKEN PRUNING TECHNIQUE WILL BE USED ------')

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    #[ADDED FOR OBSERVATION EXPERIMENT] 确保每个类别仅采样一条进行观测，减少实验时长和图表堆积
    seen_categories = set()
    obs_questions = []
    for q in questions:
        if q["category"] not in seen_categories:
            seen_categories.add(q["category"])
            obs_questions.append(q)
    questions = obs_questions
    print(f"\n[Observation Mode] 筛选完成，共选取 {len(questions)} 个类别样本进行推理和注意力分析。")

    num_sample = len(questions)
    total_latency = 0.0
    results =[]

    # 初始化垃圾回收，防止初始显存碎片
    import gc
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    for i, line in enumerate(tqdm(questions)):
        idx = line.get("question_id")
        qs = line["question"]
        label = line["answer"]
        category = line["category"]
        image_file = line["image_file"]
        
        cur_prompt = qs
        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if use_cuda:
            images = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16)
        else:
            images = image_tensor.unsqueeze(0).float()
            
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
        cur_prompt = '<image>' + '\n' + cur_prompt
        qs = qs + '\n' + "Answer the question using a single word or phrase."
        cur_prompt = cur_prompt + '\n' + "Answer the question using a single word or phrase."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        if use_cuda:
            input_ids = input_ids.to(device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria =[KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            t0 = time.time()
            
            # =================================================================================
            # [MODIFIED] 第一步：单独执行一次前向传播（Prefill阶段），专门提取 Attention 矩阵
            # =================================================================================
            forward_outputs = model(
                input_ids=input_ids,
                images=images,
                output_attentions=True,
                use_cache=False,
                return_dict=True
            )
            
            # 获取 Attention 并做热力图分析
            if hasattr(forward_outputs, 'attentions') and forward_outputs.attentions is not None:
                prefill_attns = forward_outputs.attentions
                
                # ---> [ADDED] 检测 NaN 值并打印 <---
                has_nan = any(torch.isnan(l_attn).any().item() for l_attn in prefill_attns)
                print(f"\n[Check] Category: '{category}' - 提取的 Attention Matrix 是否包含 NaN 值: {has_nan}")
                
                num_layers = len(prefill_attns)
                orig_seq_len = input_ids.shape[1]
                
                # 计算第一层 Attention Matrix 展开后的序列长度
                first_layer_seq_len = prefill_attns[0].shape[-1]
                # LLaVA 替换 IMAGE_TOKEN 后实际增加的 token 数量
                num_image_tokens = first_layer_seq_len - orig_seq_len + 1
                
                image_start_idx = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
                image_end_idx = image_start_idx + num_image_tokens
                
                layer_topk_indices =[]
                layer_img_attn_vectors = []
                layer_entropy_values = []
                layer_topk_masked_vectors = []
                layer_topk_entropy_values = []
                valid_layers = 0
                selected_k = None
                
                for l in range(num_layers):
                    attn = prefill_attns[l][0]
                    current_seq_len = attn.shape[-1]
                    
                    if current_seq_len != first_layer_seq_len:
                        print(f"[Warning] 层 {l} 的序列长度({current_seq_len})与第一层({first_layer_seq_len})不同！这通常由于启用了Token Pruning。截断之后的分析。")
                        break
                    
                    last_token_attn = attn[:, -1, :] 
                    avg_attn = last_token_attn.mean(dim=0) # 跨多头求平均
                    
                    img_attn = avg_attn[image_start_idx:image_end_idx]
                    
                    # 如果有 NaN，需要填 0 防止 topk 报错
                    if has_nan:
                        img_attn = torch.nan_to_num(img_attn, nan=0.0)

                    # 记录每层 image attention 向量，后续用于余弦相似度和信息熵分析
                    layer_img_attn_vectors.append(img_attn.detach().float().cpu())

                    # 归一化信息熵（范围约为 [0, 1]），用于观测层间注意力分布集中程度变化
                    img_attn_sum = img_attn.sum()
                    if img_attn_sum.item() > 0:
                        probs = (img_attn / img_attn_sum).clamp_min(1e-12)
                        entropy = float((-(probs * torch.log(probs)).sum() / math.log(probs.shape[0])).item())
                    else:
                        entropy = 0.0
                    layer_entropy_values.append(entropy)
                        
                    k = max(1, int(math.ceil(img_attn.shape[0] * args.topk_img_token_percent / 100.0)))
                    if k > 0:
                        topk_vals, topk_inds = torch.topk(img_attn, k)
                        layer_topk_indices.append(set(topk_inds.tolist()))

                        # Top-K 子空间向量：仅保留 topk token 的权重，其余位置置 0
                        topk_masked_vec = torch.zeros_like(img_attn)
                        topk_masked_vec[topk_inds] = topk_vals
                        layer_topk_masked_vectors.append(topk_masked_vec.detach().float().cpu())

                        # Top-K token 分布的归一化熵，观测关键 token 的分布离散程度
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

                    # 计算相邻层余弦相似度变化（第 0 层无前驱，置为 NaN 方便绘图跳过）
                    cosine_sim_values = [np.nan]
                    for l in range(1, valid_layers):
                        prev_vec = layer_img_attn_vectors[l - 1]
                        curr_vec = layer_img_attn_vectors[l]
                        cos_sim = torch.nn.functional.cosine_similarity(
                            curr_vec.unsqueeze(0),
                            prev_vec.unsqueeze(0),
                            dim=1,
                            eps=1e-8,
                        ).item()
                        cosine_sim_values.append(float(cos_sim))

                    # 计算相邻层 Top-K 子空间余弦相似度（使用 topk 掩码向量）
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

                    # 计算相邻层全局 image token 分布的 JSD（归一化到 [0, 1]）
                    jsd_values = [np.nan]
                    for l in range(1, valid_layers):
                        prev_vec = layer_img_attn_vectors[l - 1]
                        curr_vec = layer_img_attn_vectors[l]
                        jsd_values.append(_normalized_jsd(curr_vec, prev_vec))

                    # 计算相邻层 Top-K 子空间分布的 JSD（使用 topk 掩码向量）
                    topk_jsd_values = [np.nan]
                    for l in range(1, valid_layers):
                        prev_topk_vec = layer_topk_masked_vectors[l - 1]
                        curr_topk_vec = layer_topk_masked_vectors[l]
                        topk_jsd_values.append(_normalized_jsd(curr_topk_vec, prev_topk_vec))
                    
                    os.makedirs(args.heatmap_save_dir, exist_ok=True)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(overlap_matrix, cmap="YlGnBu", vmin=0, vmax=1, ax=ax)

                    # 在热力图同一坐标区域叠加两条曲线：信息熵(红)与相邻层余弦相似度(橙)
                    if valid_layers > 1:
                        x_coords = np.arange(valid_layers) + 0.5
                        entropy_np = np.array(layer_entropy_values[:valid_layers], dtype=np.float32)
                        cosine_np = np.array(cosine_sim_values, dtype=np.float32)
                        topk_entropy_np = np.array(layer_topk_entropy_values[:valid_layers], dtype=np.float32)
                        topk_cosine_np = np.array(topk_cosine_sim_values, dtype=np.float32)
                        jsd_np = np.array(jsd_values, dtype=np.float32)
                        topk_jsd_np = np.array(topk_jsd_values, dtype=np.float32)

                        # 基于 Top-K JSD 与 Top-K 熵的“双侧变化”联合打分，
                        # 在固定层区间中各选 1 个突变点：0-4、5-15、16-20。
                        safe_topk_jsd = np.nan_to_num(topk_jsd_np, nan=0.0)
                        safe_topk_entropy = np.nan_to_num(topk_entropy_np, nan=0.0)

                        def _bidirectional_change(arr: np.ndarray) -> np.ndarray:
                            left = np.zeros_like(arr)
                            right = np.zeros_like(arr)
                            if arr.size > 1:
                                left[1:] = np.abs(arr[1:] - arr[:-1])
                                right[:-1] = np.abs(arr[1:] - arr[:-1])
                            # 用左右两侧变化的平均值表示该层是否“突变”。
                            return 0.5 * (left + right)

                        entropy_bi_change = _bidirectional_change(safe_topk_entropy)
                        jsd_bi_change = _bidirectional_change(safe_topk_jsd)

                        def _max_norm(arr: np.ndarray) -> np.ndarray:
                            m = float(np.max(arr)) if arr.size > 0 else 0.0
                            if m <= 1e-12:
                                return np.zeros_like(arr)
                            return arr / m

                        jsd_norm = _max_norm(jsd_bi_change)
                        entropy_change_norm = _max_norm(entropy_bi_change)
                        combined_change_score = 0.5 * jsd_norm + 0.5 * entropy_change_norm

                        # 固定区间选点：每个区间取分数最高的层。
                        target_ranges = [(0, 4), (5, 15), (16, 20)]
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

                        metric_to_layer = lambda m: m * (valid_layers - 1) + 0.5
                        layer_to_metric = lambda y: (y - 0.5) / (valid_layers - 1)

                        ax.plot(
                            x_coords,
                            metric_to_layer(entropy_np),
                            color="crimson",
                            marker="o",
                            linewidth=2.0,
                            markersize=4,
                            label="Global Entropy",
                        )
                        ax.plot(
                            x_coords,
                            metric_to_layer(cosine_np),
                            color="darkorange",
                            marker="s",
                            linewidth=2.0,
                            markersize=4,
                            label="Global Cosine",
                        )
                        ax.plot(
                            x_coords,
                            metric_to_layer(topk_entropy_np),
                            color="deepskyblue",
                            marker="^",
                            linewidth=1.8,
                            markersize=4,
                            linestyle="--",
                            label="Top-K Entropy",
                        )
                        ax.plot(
                            x_coords,
                            metric_to_layer(topk_cosine_np),
                            color="limegreen",
                            marker="d",
                            linewidth=1.8,
                            markersize=4,
                            linestyle="--",
                            label="Top-K Cosine",
                        )
                        ax.plot(
                            x_coords,
                            metric_to_layer(jsd_np),
                            color="mediumpurple",
                            marker="x",
                            linewidth=1.8,
                            markersize=5,
                            linestyle=":",
                            label="Global JSD",
                        )
                        ax.plot(
                            x_coords,
                            metric_to_layer(topk_jsd_np),
                            color="sienna",
                            marker="P",
                            linewidth=1.8,
                            markersize=5,
                            linestyle=":",
                            label="Top-K JSD",
                        )

                        # 在图中标注突变最明显的 3 个节点。
                        for rank, layer_idx in enumerate(abrupt_nodes, start=1):
                            x = layer_idx + 0.5
                            y_entropy = metric_to_layer(float(topk_entropy_np[layer_idx]))
                            y_jsd = metric_to_layer(float(topk_jsd_np[layer_idx]))

                            ax.axvline(
                                x=x,
                                color="black",
                                linestyle="-.",
                                linewidth=0.8,
                                alpha=0.4,
                            )
                            ax.scatter([x], [y_entropy], color="navy", s=20, zorder=6)
                            ax.scatter([x], [y_jsd], color="maroon", s=20, zorder=6)
                            ax.text(
                                x + 0.08,
                                min(y_jsd + 0.12, valid_layers - 0.55),
                                f"N{rank}:L{int(layer_idx)}",
                                fontsize=7,
                                color="black",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                            )

                        if len(abrupt_nodes) > 0:
                            print(
                                f"[Observation] Category: '{category}' 分段突变节点({len(abrupt_nodes)}个): "
                                + ", ".join(
                                    f"L{int(l)}(score={combined_change_score[int(l)]:.3f})" for l in abrupt_nodes
                                )
                            )

                        secax = ax.secondary_yaxis('right', functions=(layer_to_metric, metric_to_layer))
                        secax.set_ylabel("Entropy / Cosine / JSD", fontsize=11)
                        secax.set_ylim(layer_to_metric(0.0), layer_to_metric(1.0))
                        ax.legend(
                            loc="center right",
                            bbox_to_anchor=(0.98, 0.5),
                            fontsize=8,
                            frameon=True,
                            borderaxespad=0.0,
                        )
                    
                    safe_cat_name = str(category).replace("/", "_").replace(" ", "_")
                    k_percent_str = f"{args.topk_img_token_percent:g}"
                    safe_k_percent = k_percent_str.replace('.', 'p')
                    ax.set_title(
                        f"Top-{k_percent_str}% Overlap + Global/Top-K Trends (k={selected_k})\\nCategory: {category}",
                        fontsize=14,
                    )
                    ax.set_xlabel("Layer", fontsize=12)
                    ax.set_ylabel("Layer", fontsize=12)
                    
                    save_path = os.path.join(
                        args.heatmap_save_dir,
                        f"heatmap_{safe_cat_name}_top{safe_k_percent}pct.png",
                    )
                    fig.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    print(f"[Observation] Category: '{category}' 的热力图已保存至: {save_path}")

            # =================================================================================
            # [MODIFIED] 释放前向传播产生的大量显存
            # =================================================================================
            del forward_outputs
            if 'prefill_attns' in locals():
                del prefill_attns
            if use_cuda:
                torch.cuda.empty_cache()
            
            # [安全起见] 额外的前向传播可能会导致剪枝算法内部的计数器异常，重置还原
            if getattr(args, 'use_hmap_v', False) and args.use_hmap_v:
                model.model.reset_hmapv()
            elif getattr(args, 'use_fast_v', False) and args.use_fast_v:
                model.model.reset_fastv()
            elif getattr(args, 'use_jsd_entropy_pruning', False) and args.use_jsd_entropy_pruning:
                model.model.reset_jsd_entropy_pruning()

    #         # =================================================================================
    #         # [MODIFIED] 第二步：正常的自回归生成文本，关闭 output_attentions 避免显存爆炸
    #         # =================================================================================
    #         output_ids = model.generate(
    #             input_ids,
    #             images=images,
    #             do_sample=True if args.temperature > 0 else False,
    #             temperature=args.temperature if args.temperature > 0 else 0.0,
    #             max_new_tokens=1024,
    #             use_cache=False, 
    #             stopping_criteria=stopping_criteria,
    #             return_dict_in_generate=True,
    #             output_attentions=False,  # <--- 核心修改：关闭 Attention 提取，避免生成 OOM
    #         )
            
    #     torch.cuda.synchronize()
    #     end_time = time.time()
    #     inference_latency = end_time - t0
    #     total_latency += inference_latency

    #     input_token_len = input_ids.shape[1]
    #     n_diff_input_output = (input_ids != output_ids['sequences'][:, :input_token_len]).sum().item()
    #     if n_diff_input_output > 0:
    #         print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    #     outputs = tokenizer.batch_decode(output_ids['sequences'][:, input_token_len:], skip_special_tokens=True)[0]
    #     outputs = outputs.strip()
    #     if outputs.endswith(stop_str):
    #         outputs = outputs[:-len(stop_str)]
    #     outputs = outputs.strip()
        
    #     pred = outputs
    #     if pred.endswith('.'):
    #         pred = pred[:-1]
            
    #     results.append({
    #         'question_id': idx,
    #         'category': category,
    #         'pred': pred,
    #         'gt': label
    #     })
    
    # scores, perception_score, cognition_score = calculate_mme_scores(results)
    
    # final_results = {
    #     'scores': scores,
    #     'perception_score': perception_score,
    #     'cognition_score': cognition_score,
    #     'total_score': perception_score + cognition_score,
    #     'total_samples': num_sample,
    #     'model_config': {
    #         'use_himap': args.use_hmap_v,
    #         'sys_length': args.sys_length,
    #         'img_length': args.img_length,
    #         'txt_layer': args.hmap_v_attn_txt_layer,
    #         'img_layer': args.hmap_v_attn_img_layer,
    #         'txt_rank': args.hmap_v_attn_txt_rank,
    #         'img_rank': args.hmap_v_attn_img_rank,
    #         'cut_off_layer': args.cut_off_layer
    #     } if args.use_hmap_v else {
    #         'use_fast_v': args.use_fast_v,
    #         'sys_length': args.fast_v_sys_length,
    #         'img_length': args.fast_v_image_token_length,
    #         'attn_rank': args.fast_v_attention_rank,
    #         'agg_layer': args.fast_v_agg_layer
    #     }
    # }
    
    # output_file = f"mme_results_{'himap' if args.use_hmap_v else ('fastv' if args.use_fast_v else 'baseline')}.json"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(final_results, f, indent=2, ensure_ascii=False)
    # print(f"\n结果已保存到: {output_file}")