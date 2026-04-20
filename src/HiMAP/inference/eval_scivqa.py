import argparse
import torch
import os
import json
import math
from tqdm import tqdm
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image


def summarize_attention_tensor(attn_tensor, topk=20):
    """Summarize one attention tensor with NaN info and top-k values."""
    flat = attn_tensor.detach().float().reshape(-1)
    nan_mask = torch.isnan(flat)
    inf_mask = torch.isinf(flat)
    finite_mask = torch.isfinite(flat)
    finite_vals = flat[finite_mask]

    has_nan = bool(nan_mask.any().item())
    has_inf = bool(inf_mask.any().item())

    if finite_vals.numel() > 0:
        k = min(int(topk), int(finite_vals.numel()))
        top_vals = torch.topk(finite_vals, k=k).values.cpu().tolist()
        mean_val = float(finite_vals.mean().item())
        max_val = float(finite_vals.max().item())
        min_val = float(finite_vals.min().item())
    else:
        top_vals = []
        mean_val = None
        max_val = None
        min_val = None

    return {
        "numel": int(flat.numel()),
        "finite_numel": int(finite_vals.numel()),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "mean": mean_val,
        "max": max_val,
        "min": min_val,
        "top_values": top_vals,
    }


def get_layer_attention_diagnostics(attentions, topk=20):
    """
    Extract per-layer attention diagnostics from model outputs.

    Supports both shapes:
    1) Tuple[num_layers] of tensors (single forward)
    2) Tuple[num_decode_steps] where each item is Tuple[num_layers] (generate output)
    """
    if attentions is None:
        return {}
    if not isinstance(attentions, (list, tuple)) or len(attentions) == 0:
        return {}

    first_item = attentions[0]

    # generate() path: attentions[step][layer] -> tensor
    if isinstance(first_item, (list, tuple)):
        num_layers = len(first_item)
        per_layer_tensors = [[] for _ in range(num_layers)]

        for step_attn in attentions:
            if not isinstance(step_attn, (list, tuple)):
                continue
            for layer_idx, layer_attn in enumerate(step_attn):
                per_layer_tensors[layer_idx].append(layer_attn)

        diagnostics = {}
        for layer_idx, tensor_list in enumerate(per_layer_tensors):
            if len(tensor_list) == 0:
                diagnostics[str(layer_idx)] = {
                    "num_steps": 0,
                    "num_tensors": 0,
                    "has_nan": False,
                    "has_inf": False,
                    "mean": None,
                    "max": None,
                    "min": None,
                    "top_values": [],
                }
                continue

            merged = torch.cat([t.detach().float().reshape(-1) for t in tensor_list], dim=0)
            layer_summary = summarize_attention_tensor(merged, topk=topk)
            layer_summary["num_steps"] = len(attentions)
            layer_summary["num_tensors"] = len(tensor_list)
            diagnostics[str(layer_idx)] = layer_summary

        return diagnostics

    # forward() path: attentions[layer] -> tensor
    diagnostics = {}
    for layer_idx, layer_attn in enumerate(attentions):
        diagnostics[str(layer_idx)] = summarize_attention_tensor(layer_attn, topk=topk)
    return diagnostics

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_stage_ranges(range_text):
    if range_text is None:
        return [(2, 8), (9, 20), (21, 31)]
    ranges = []
    for part in str(range_text).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' not in part:
            raise ValueError(f"Invalid stage range format: {part}")
        lo_s, hi_s = part.split('-', 1)
        ranges.append((int(lo_s), int(hi_s)))
    if not ranges:
        raise ValueError("No valid stage ranges parsed from --jsd-entropy-stage-ranges")
    return ranges


def parse_stage_prune_ratios(ratio_text):
    if ratio_text is None:
        return [0.2, 0.3, 0.5]
    ratios = [float(x.strip()) for x in str(ratio_text).split(',') if x.strip()]
    if len(ratios) == 0:
        raise ValueError("No valid prune ratios parsed from --jsd-entropy-stage-prune-ratios")
    return ratios


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to use for testing (-1 for all)")
    parser.add_argument("--save-attn-diagnostics", action="store_true",
                        help="Save per-layer attention diagnostics (top-k, nan, mean/max/min) to JSON")
    parser.add_argument("--attn-topk", type=int, default=20,
                        help="Top-k attention values to keep for each layer")
    # HiMAP hyperparameter
    parser.add_argument('--use-hmap-v', default=False, action='store_true', help='whether to use hmap-v')
    parser.add_argument('--sys-length', type=int, required=False, help='the length of system prompt')
    parser.add_argument('--img-length', type=int, required=False, help='the length of image token')
    parser.add_argument('--hmap-v-attn-txt-layer', type=int, required=False, help='the layer of pruning accorading to img2txt information')
    parser.add_argument('--hmap-v-attn-img-layer', type=int, required=False, help='the layer of pruning accorading to img2img information')
    parser.add_argument('--hmap-v-attn-txt-rank', type=int, required=False, help='the rank of attn accorading to img2txt information')
    parser.add_argument('--hmap-v-attn-img-rank', type=int, required=False, help='the rank of attn accorading to img2img information')
    parser.add_argument('--cut-off-layer', type=int, required=False, help='the layer index after which all image tokens are removed')
    # fastv config
    parser.add_argument('--use-fast-v', default=False, action='store_true', help='whether to use fast-v')
    parser.add_argument('--fast-v-sys-length', type=int, required=False, help='the length of system prompt for fast-v')
    parser.add_argument('--fast-v-image-token-length', type=int, required=False, help='the length of image token for fast-v')
    parser.add_argument('--fast-v-attention-rank', type=int, required=False, help='the rank of attention for fast-v')
    parser.add_argument('--fast-v-agg-layer', type=int, required=False, help='the aggregation layer for fast-v')
    # FastV Advanced config
    parser.add_argument('--fast-v-token-selection-method', type=str, default='avg_all_heads', 
                       choices=['max_head', 'avg_all_heads', 'weighted_combination', 'text_weighted', 'text_weighted_max_head'],
                       help='token selection strategy: max_head, avg_all_heads, weighted_combination, text_weighted, or text_weighted_max_head')
    parser.add_argument('--fast-v-weighted-alpha', type=float, default=0.5,
                       help='alpha weight for weighted_combination method (0.0 to 1.0)')
    # adaptive JSD+Entropy stage pruning config
    parser.add_argument('--use-jsd-entropy-prune', default=False, action='store_true',
                        help='whether to use adaptive top-k JSD+entropy 3-stage pruning')
    parser.add_argument('--jsd-entropy-sys-length', type=int, required=False,
                        help='system token length for adaptive pruning')
    parser.add_argument('--jsd-entropy-img-length', type=int, required=False,
                        help='image token length for adaptive pruning')
    parser.add_argument('--jsd-entropy-topk-percent', type=float, default=10.0,
                        help='top-k percent used for JSD+entropy stage detection')
    parser.add_argument('--jsd-entropy-topk-attention-mode', '--jsd_entropy_topk_attention_mode',
                        dest='jsd_entropy_topk_attention_mode', type=str, default="prompt_image",
                        choices=["prompt_image", "global"],
                        help='whether to calculate JSD+entropy based top-k attention scores using only prompt+image tokens or all tokens')
    parser.add_argument('--jsd-entropy-stage-ranges', type=str, default='2-8,9-20,21-31',
                        help='layer ranges to pick 3 stage nodes, e.g. 2-8,9-20,21-31')
    parser.add_argument('--jsd-entropy-stage-prune-ratios', type=str, default='0.2,0.3,0.5',
                        help='incremental prune ratios per stage, default 20%,30%,50%')
    parser.add_argument('--jsd-entropy-target-tokens', type=int, required=False,
                        help='target number of image tokens for JSD entropy pruning')
    parser.add_argument('--jsd-entropy-n0', type=int, required=False,
                        help='initial image token length n0 used in stage budgeting')
    parser.add_argument('--jsd-entropy-phase1-prune-layer', type=int, required=False,
                        help='1-based layer index to apply phase-1 pruning')
    parser.add_argument('--jsd-entropy-phase2-prune-layer', type=int, required=False,
                        help='1-based layer index to apply phase-2 pruning')
    parser.add_argument('--jsd-entropy-phase3-prune-layer', type=int, required=False,
                        help='1-based layer index to drop all remaining image tokens')
    parser.add_argument('--jsd-entropy-n-base-192', type=float, required=False,
                        help='phase-1 base keep count when target tokens = 192')
    parser.add_argument('--jsd-entropy-n-base-128', type=float, required=False,
                        help='phase-1 base keep count when target tokens = 128')
    parser.add_argument('--jsd-entropy-n-base-64', type=float, required=False,
                        help='phase-1 base keep count when target tokens = 64')
    parser.add_argument('--jsd-entropy-use-only-prompt2image-scoring', type=str, default='True',
                        help='whether to compute score based only on prompt2image attention')
    parser.add_argument('--jsd-entropy-use-adaptive-keep-ratio', type=str, default='True',
                        help='whether to use adaptive keep ratio')
    parser.add_argument('--jsd-entropy-mu-h', type=float, required=False,
                        help='mean of H metric for z-score normalization')
    parser.add_argument('--jsd-entropy-sigma-h', type=float, required=False,
                        help='std of H metric for z-score normalization')
    parser.add_argument('--jsd-entropy-mu-w', type=float, required=False,
                        help='mean of W metric for z-score normalization')
    parser.add_argument('--jsd-entropy-sigma-w', type=float, required=False,
                        help='std of W metric for z-score normalization')
    parser.add_argument('--jsd-entropy-alpha', type=float, required=False,
                        help='alpha weight for H z-score term in phase-1 keep budget')
    parser.add_argument('--jsd-entropy-beta', type=float, required=False,
                        help='beta weight for W z-score term in phase-1 keep budget')
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    # 检查是否为本地路径
    if os.path.exists(model_path):
        # 本地路径，直接使用
        model_name = get_model_name_from_path(model_path)
    else:
        # 远程路径，使用原始路径
        model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # set model hmapv config
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
        # cut-off layer to drop all image tokens after a specific layer
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
        # FastV Advanced parameters
        model.config.fast_v_token_selection_method = args.fast_v_token_selection_method
        model.config.fast_v_weighted_alpha = args.fast_v_weighted_alpha
        print(f'FASTV TECHNIQUE WILL BE USED ------')
        print(f'  Token Selection Method: {args.fast_v_token_selection_method}')
        if args.fast_v_token_selection_method == 'weighted_combination':
            print(f'  Weighted Alpha: {args.fast_v_weighted_alpha}')
        model.model.reset_fastv()
        print(f"DEBUG: Active Token Selection Method in Model: {model.model.token_selection_method}")
    elif args.use_jsd_entropy_prune == True:
        adaptive_sys_len = args.jsd_entropy_sys_length if args.jsd_entropy_sys_length is not None else args.sys_length
        adaptive_img_len = args.jsd_entropy_img_length if args.jsd_entropy_img_length is not None else args.img_length
        if adaptive_sys_len is None or adaptive_img_len is None:
            raise ValueError('Adaptive pruning requires --jsd-entropy-sys-length/--jsd-entropy-img-length (or fallback --sys-length/--img-length).')

        model.config.use_hmap_v = False
        model.config.use_fast_v = False
        model.config.use_jsd_entropy_pruning = True
        model.config.jsd_entropy_sys_length = adaptive_sys_len
        model.config.jsd_entropy_image_token_length = adaptive_img_len
        model.config.jsd_entropy_topk_percent = args.jsd_entropy_topk_percent
        model.config.jsd_entropy_topk_attention_mode = args.jsd_entropy_topk_attention_mode
        model.config.jsd_entropy_stage_ranges = parse_stage_ranges(args.jsd_entropy_stage_ranges)
        model.config.jsd_entropy_stage_prune_ratios = parse_stage_prune_ratios(args.jsd_entropy_stage_prune_ratios)
        model.config.jsd_entropy_use_only_prompt2image_scoring = args.jsd_entropy_use_only_prompt2image_scoring.lower() in ('true', '1', 't', 'y')
        model.config.jsd_entropy_use_adaptive_keep_ratio = args.jsd_entropy_use_adaptive_keep_ratio.lower() in ('true', '1', 't', 'y')
        if args.jsd_entropy_target_tokens is not None:
            model.config.jsd_entropy_target_tokens = args.jsd_entropy_target_tokens
        if args.jsd_entropy_n0 is not None:
            model.config.jsd_entropy_n0 = args.jsd_entropy_n0
        if args.jsd_entropy_phase1_prune_layer is not None:
            model.config.jsd_entropy_phase1_prune_layer = args.jsd_entropy_phase1_prune_layer
        if args.jsd_entropy_phase2_prune_layer is not None:
            model.config.jsd_entropy_phase2_prune_layer = args.jsd_entropy_phase2_prune_layer
        if args.jsd_entropy_phase3_prune_layer is not None:
            model.config.jsd_entropy_phase3_prune_layer = args.jsd_entropy_phase3_prune_layer
        if args.jsd_entropy_n_base_192 is not None:
            model.config.jsd_entropy_n_base_192 = args.jsd_entropy_n_base_192
        if args.jsd_entropy_n_base_128 is not None:
            model.config.jsd_entropy_n_base_128 = args.jsd_entropy_n_base_128
        if args.jsd_entropy_n_base_64 is not None:
            model.config.jsd_entropy_n_base_64 = args.jsd_entropy_n_base_64
        if args.jsd_entropy_mu_h is not None:
            model.config.jsd_entropy_mu_h = args.jsd_entropy_mu_h
        if args.jsd_entropy_sigma_h is not None:
            model.config.jsd_entropy_sigma_h = args.jsd_entropy_sigma_h
        if args.jsd_entropy_mu_w is not None:
            model.config.jsd_entropy_mu_w = args.jsd_entropy_mu_w
        if args.jsd_entropy_sigma_w is not None:
            model.config.jsd_entropy_sigma_w = args.jsd_entropy_sigma_w
        if args.jsd_entropy_alpha is not None:
            model.config.jsd_entropy_alpha = args.jsd_entropy_alpha
        if args.jsd_entropy_beta is not None:
            model.config.jsd_entropy_beta = args.jsd_entropy_beta

        print('ADAPTIVE JSD+ENTROPY 3-STAGE PRUNING WILL BE USED ------')
        if hasattr(model.model, 'reset_jsd_entropy_pruning'):
            model.model.reset_jsd_entropy_pruning()
    else:
        model.config.use_hmap_v = False
        model.config.use_fast_v = False
        model.config.use_jsd_entropy_pruning = False

        print('NO TOKEN PRUNING TCHNIQUE WILL BE USED ------')

    

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.num_samples > 0:
        questions = questions[:args.num_samples]

    num_sample = len(questions)
    corr_sample = 0
    total_latency = 0.0
    total_flops_ratio_attn_ffn = 0.0
    attention_diagnostics_by_sample = {}

    for i, line in enumerate(tqdm(questions)):
        
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs
        label = line['conversations'][1]['value']

        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        if torch.cuda.is_available():
            images = image_tensor.unsqueeze(0).half().cuda()
        else:
            images = image_tensor.unsqueeze(0).float()
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        cur_prompt = '<image>' + '\n' + cur_prompt


        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        # Reset adaptive stage plan per sample so each sample gets its own stage detection.
        if args.use_jsd_entropy_prune and hasattr(model.model, 'reset_jsd_entropy_pruning'):
            model.model.reset_jsd_entropy_pruning()

        with torch.inference_mode():
            t0 = time.time()
            output_ids = model.generate(
                input_ids,
                images=images,
                max_new_tokens=1024,
                use_cache=False,
                stopping_criteria=stopping_criteria,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        torch.cuda.synchronize()  # 确保GPU操作完成
        end_time = time.time()
        inference_latency = end_time - t0
        total_latency += inference_latency

        input_token_len = input_ids.shape[1]
        sequences = output_ids['sequences']
        seq_token_len = sequences.shape[1]

        # Some model implementations return prompt+generated tokens,
        # while others return generated-only tokens.
        if seq_token_len >= input_token_len:
            n_diff_input_output = (input_ids != sequences[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            decode_ids = sequences[:, input_token_len:]
        else:
            decode_ids = sequences

        outputs = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()

        if args.save_attn_diagnostics:
            sample_attentions = output_ids.get('attentions', None)
            attention_diagnostics_by_sample[str(idx)] = {
                "image": image_file,
                "layer_stats": get_layer_attention_diagnostics(sample_attentions, topk=args.attn_topk)
            }
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        if outputs == label:
            corr_sample += 1
        
        # Estimate FLOPs Ratio (attn+ffn approx) per example, then average
        try:
            num_layers = getattr(model.model.config, 'num_hidden_layers', None)
            if num_layers is None:
                num_layers = len(getattr(model.model, 'layers', []))
            # Compute text token length in the prompt (excluding image placeholder and system tokens)
            text_len = int(input_ids.shape[1]) - 1 - int(args.sys_length)
            base_len = int(args.sys_length) + int(args.img_length) + max(text_len, 0)

            if args.use_hmap_v and all(
                v is not None for v in [args.hmap_v_attn_txt_layer, args.hmap_v_attn_img_layer, args.hmap_v_attn_txt_rank, args.hmap_v_attn_img_rank]
            ):
                n_before = base_len
                n_after_txt = int(args.sys_length) + int(args.hmap_v_attn_txt_rank) + max(text_len, 0)
                n_after_img = int(args.sys_length) + int(args.hmap_v_attn_img_rank) + max(text_len, 0)

                L_txt = max(min(int(args.hmap_v_attn_txt_layer), num_layers), 0)
                L_img = max(min(int(args.hmap_v_attn_img_layer), num_layers), L_txt)
                L_total = num_layers

                def cost(n: int) -> int:
                    # attn (n^2) + ffn (n) normalized later
                    return n * n + n

                sum_cost = (
                    L_txt * cost(n_before) +
                    max(L_img - L_txt, 0) * cost(n_after_txt) +
                    max(L_total - L_img, 0) * cost(n_after_img)
                )
                base_cost = L_total * cost(n_before)
                ratio = float(sum_cost) / float(base_cost) if base_cost > 0 else 1.0
            else:
                ratio = 1.0
        except Exception:
            ratio = 1.0

        total_flops_ratio_attn_ffn += ratio
    
    # Report metrics
    avg_latency = total_latency / max(num_sample, 1)
    avg_flops_ratio = total_flops_ratio_attn_ffn / max(num_sample, 1)
    accuracy = corr_sample/num_sample
    print(corr_sample/num_sample)
    print(f'Avg Latency/Example: {avg_latency:.6f}s')
    print(f'FLOPs Ratio (attn+ffn approx): {avg_flops_ratio*100:.2f}%')

     # 保存结果到文件
    results = {
        'accuracy': accuracy,
        'avg_latency': avg_latency,
        'total_samples': num_sample,
        'correct_samples': corr_sample,
        'flops_info': avg_flops_ratio,
        'model_config': {
            'use_himap': args.use_hmap_v,
            'use_fastv': args.use_fast_v,
            'use_jsd_entropy_prune': args.use_jsd_entropy_prune,
            'sys_length': args.sys_length,
            'img_length': args.img_length,
            'txt_layer': args.hmap_v_attn_txt_layer,
            'img_layer': args.hmap_v_attn_img_layer,
            'txt_rank': args.hmap_v_attn_txt_rank,
            'img_rank': args.hmap_v_attn_img_rank,
            # FastV Advanced config
            'fast_v_sys_length': args.fast_v_sys_length if args.use_fast_v else None,
            'fast_v_image_token_length': args.fast_v_image_token_length if args.use_fast_v else None,
            'fast_v_attention_rank': args.fast_v_attention_rank if args.use_fast_v else None,
            'fast_v_agg_layer': args.fast_v_agg_layer if args.use_fast_v else None,
            'fast_v_token_selection_method': args.fast_v_token_selection_method if args.use_fast_v else None,
            'fast_v_weighted_alpha': args.fast_v_weighted_alpha if args.use_fast_v else None,
            # JSD+Entropy adaptive config
            'jsd_entropy_sys_length': args.jsd_entropy_sys_length if args.use_jsd_entropy_prune else None,
            'jsd_entropy_img_length': args.jsd_entropy_img_length if args.use_jsd_entropy_prune else None,
            'jsd_entropy_topk_percent': args.jsd_entropy_topk_percent if args.use_jsd_entropy_prune else None,
            'jsd_entropy_topk_attention_mode': args.jsd_entropy_topk_attention_mode if args.use_jsd_entropy_prune else None,
            'jsd_entropy_stage_ranges': parse_stage_ranges(args.jsd_entropy_stage_ranges) if args.use_jsd_entropy_prune else None,
            'jsd_entropy_stage_prune_ratios': parse_stage_prune_ratios(args.jsd_entropy_stage_prune_ratios) if args.use_jsd_entropy_prune else None,
            'jsd_entropy_target_tokens': args.jsd_entropy_target_tokens if args.use_jsd_entropy_prune else None,
            'jsd_entropy_n0': args.jsd_entropy_n0 if args.use_jsd_entropy_prune else None,
            'jsd_entropy_phase1_prune_layer': args.jsd_entropy_phase1_prune_layer if args.use_jsd_entropy_prune else None,
            'jsd_entropy_phase2_prune_layer': args.jsd_entropy_phase2_prune_layer if args.use_jsd_entropy_prune else None,
            'jsd_entropy_phase3_prune_layer': args.jsd_entropy_phase3_prune_layer if args.use_jsd_entropy_prune else None,
            'jsd_entropy_n_base_192': args.jsd_entropy_n_base_192 if args.use_jsd_entropy_prune else None,
            'jsd_entropy_n_base_128': args.jsd_entropy_n_base_128 if args.use_jsd_entropy_prune else None,
            'jsd_entropy_n_base_64': args.jsd_entropy_n_base_64 if args.use_jsd_entropy_prune else None,
            'jsd_entropy_use_only_prompt2image_scoring': args.jsd_entropy_use_only_prompt2image_scoring if args.use_jsd_entropy_prune else None,
            'jsd_entropy_use_adaptive_keep_ratio': args.jsd_entropy_use_adaptive_keep_ratio if args.use_jsd_entropy_prune else None,
            'jsd_entropy_mu_h': args.jsd_entropy_mu_h if args.use_jsd_entropy_prune else None,
            'jsd_entropy_sigma_h': args.jsd_entropy_sigma_h if args.use_jsd_entropy_prune else None,
            'jsd_entropy_mu_w': args.jsd_entropy_mu_w if args.use_jsd_entropy_prune else None,
            'jsd_entropy_sigma_w': args.jsd_entropy_sigma_w if args.use_jsd_entropy_prune else None,
            'jsd_entropy_alpha': args.jsd_entropy_alpha if args.use_jsd_entropy_prune else None,
            'jsd_entropy_beta': args.jsd_entropy_beta if args.use_jsd_entropy_prune else None,
        }
    }
    
    # 保存结果
    if args.use_hmap_v:
        output_file = "scienceqa_results_himap.json"
    elif args.use_fast_v:
        method_name = args.fast_v_token_selection_method
        if method_name == 'weighted_combination':
            output_file = f"scienceqa_results_fastv_{method_name}_alpha{args.fast_v_weighted_alpha}.json"
        else:
            output_file = f"scienceqa_results_fastv_{method_name}.json"
    elif args.use_jsd_entropy_prune:
        output_file = "scienceqa_results_jsd_entropy.json"
    else:
        output_file = "scienceqa_results_baseline.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")

    if args.save_attn_diagnostics:
        if args.use_hmap_v:
            attn_diag_file = "scienceqa_attention_diagnostics_himap.json"
        elif args.use_fast_v:
            method_name = args.fast_v_token_selection_method
            if method_name == 'weighted_combination':
                attn_diag_file = f"scienceqa_attention_diagnostics_fastv_{method_name}_alpha{args.fast_v_weighted_alpha}.json"
            else:
                attn_diag_file = f"scienceqa_attention_diagnostics_fastv_{method_name}.json"
        elif args.use_jsd_entropy_prune:
            attn_diag_file = "scienceqa_attention_diagnostics_jsd_entropy.json"
        else:
            attn_diag_file = "scienceqa_attention_diagnostics_baseline.json"

        with open(attn_diag_file, 'w', encoding='utf-8') as f:
            json.dump(attention_diagnostics_by_sample, f, indent=2, ensure_ascii=False)
        print(f"注意力诊断结果已保存到: {attn_diag_file}")