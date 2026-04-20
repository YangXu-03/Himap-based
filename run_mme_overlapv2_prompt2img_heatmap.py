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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image


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
        default=50.0,
        help="每一层提取Top-K%%最重要Image Token，K按image token总数百分比计算（例如10表示Top-10%%）",
    )
    parser.add_argument("--heatmap-save-dir", type=str, default="./attn_heatmaps_text2img", help="注意力重叠率热力图保存路径")

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

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # 每个类别仅采样一条，便于快速观测。
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
    results = []

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
                            _, topk_inds = torch.topk(img_attn, k)
                            layer_topk_indices.append(set(topk_inds.tolist()))
                            if selected_k is None:
                                selected_k = k
                            valid_layers += 1

                    if valid_layers > 0:
                        overlap_matrix = np.zeros((valid_layers, valid_layers))
                        for l_i in range(valid_layers):
                            for l_j in range(valid_layers):
                                intersection_size = len(layer_topk_indices[l_i].intersection(layer_topk_indices[l_j]))
                                overlap_matrix[l_i, l_j] = intersection_size / selected_k

                        os.makedirs(args.heatmap_save_dir, exist_ok=True)
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(overlap_matrix, cmap="YlGnBu", vmin=0, vmax=1)

                        safe_cat_name = str(category).replace("/", "_").replace(" ", "_")
                        k_percent_str = f"{args.topk_img_token_percent:g}"
                        safe_k_percent = k_percent_str.replace(".", "p")
                        plt.title(
                            f"Prompt->Image Token Top-{k_percent_str}% Overlap (k={selected_k})\\nCategory: {category}",
                            fontsize=14,
                        )
                        plt.xlabel("Layer", fontsize=12)
                        plt.ylabel("Layer", fontsize=12)

                        save_path = os.path.join(
                            args.heatmap_save_dir,
                            f"heatmap_prompt2img_{safe_cat_name}_top{safe_k_percent}pct.png",
                        )
                        plt.savefig(save_path, bbox_inches="tight", dpi=150)
                        plt.close()
                        print(f"[Observation] Category: '{category}' 的热力图已保存至: {save_path}")

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
