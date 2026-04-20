import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import math
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

def calculate_mme_scores(results):
    # Define categories
    perception_cats = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"]
    cognition_cats = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
    
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
        # MME usually has 2 questions per image. 
        # We group by question_id (which seems to be the original filename)
        img_groups = {}
        for x in items:
            qid = x['question_id']
            if qid not in img_groups:
                img_groups[qid] = []
            # Check if prediction matches ground truth
            img_groups[qid].append(x['pred'].lower() == x['gt'].lower())
        
        # A pair is correct only if ALL questions for that image are correct
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
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_stage_ranges(range_text):
    if range_text is None:
        return [(2, 5), (6, 16), (17, 31)]
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
    # adaptive JSD+Entropy stage pruning config
    parser.add_argument('--use-jsd-entropy-prune', default=False, action='store_true', help='whether to use adaptive top-k JSD+entropy 3-stage pruning')
    parser.add_argument('--jsd-entropy-sys-length', type=int, required=False, help='system token length for adaptive pruning')
    parser.add_argument('--jsd-entropy-img-length', type=int, required=False, help='image token length for adaptive pruning')
    parser.add_argument('--jsd-entropy-topk-attention-mode', '--jsd_entropy_topk_attention_mode', dest='jsd_entropy_topk_attention_mode', type=str, default="prompt_image", choices=["prompt_image", "global"], help='whether to calculate JSD+entropy based top-k attention scores using only prompt+image tokens or all tokens')
    parser.add_argument('--jsd-entropy-topk-percent', type=float, default=10.0, help='top-k percent used for JSD+entropy stage detection')
    parser.add_argument('--jsd-entropy-stage-ranges', type=str, default='0-4,5-14,15-20', help='layer ranges to pick 3 stage nodes, e.g. 2-8,9-20,21-31')
    parser.add_argument('--jsd-entropy-use-dynamic-boundaries', action='store_true', help='whether to use dynamic boundaries')
    parser.add_argument('--jsd-entropy-stage-prune-ratios', type=str, default='0.2,0.3,0.5', help='incremental prune ratios per stage, default 20%,30%,50%')
    parser.add_argument('--jsd-entropy-use-only-prompt2image-scoring', type=str, default='True', help='whether to compute score based only on prompt2image attention')
    parser.add_argument('--jsd-entropy-use-adaptive-keep-ratio', type=str, default='True', help='whether to use adaptive keep ratio')
    parser.add_argument('--jsd-entropy-target-tokens', type=int, required=False, help='target number of image tokens for JSD entropy pruning')
    parser.add_argument('--jsd-entropy-n0', type=int, required=False, help='initial image token length n0 used in stage budgeting')
    parser.add_argument('--jsd-entropy-phase1-prune-layer', type=int, required=False, help='1-based layer index to apply phase-1 pruning')
    parser.add_argument('--jsd-entropy-phase2-prune-layer', type=int, required=False, help='1-based layer index to apply phase-2 pruning')
    parser.add_argument('--jsd-entropy-phase3-prune-layer', type=int, required=False, help='1-based layer index to drop all remaining image tokens')
    parser.add_argument('--jsd-entropy-mu-h', type=float, required=False, help='mean of H metric for z-score normalization')
    parser.add_argument('--jsd-entropy-sigma-h', type=float, required=False, help='std of H metric for z-score normalization')
    parser.add_argument('--jsd-entropy-mu-w', type=float, required=False, help='mean of W metric for z-score normalization')
    parser.add_argument('--jsd-entropy-sigma-w', type=float, required=False, help='std of W metric for z-score normalization')
    parser.add_argument('--jsd-entropy-alpha', type=float, required=False, help='alpha weight for H z-score term in phase-1 keep budget')
    parser.add_argument('--jsd-entropy-beta', type=float, required=False, help='beta weight for W z-score term in phase-1 keep budget')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-detailed-results-file", type=str, default=None,
                        help='optional path to save per-sample predictions for external evaluators')
    args = parser.parse_args()
    

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
        print('FASTV TECHNIQUE WILL BE USED ------')
        model.model.reset_fastv()
    elif args.use_jsd_entropy_prune == True:
        adaptive_sys_len = args.jsd_entropy_sys_length if args.jsd_entropy_sys_length is not None else args.sys_length
        adaptive_img_len = args.jsd_entropy_img_length if args.jsd_entropy_img_length is not None else args.img_length
        if adaptive_sys_len is None or adaptive_img_len is None:
            raise ValueError('Adaptive pruning requires --jsd-entropy-sys-length/--jsd-entropy-img-length (or fallback --sys-length/--img-length).')

        model.config.use_hmap_v = False
        model.config.use_fast_v = False
        model.config.use_jsd_entropy_pruning = True
        # Enable per-sample dynamic stage boundary detection in JSDEntropy model.
        model.config.jsd_entropy_use_dynamic_boundaries = True
        model.config.jsd_entropy_sys_length = adaptive_sys_len
        model.config.jsd_entropy_image_token_length = adaptive_img_len
        model.config.jsd_entropy_topk_percent = args.jsd_entropy_topk_percent
        model.config.jsd_entropy_topk_attention_mode = args.jsd_entropy_topk_attention_mode
        model.config.jsd_entropy_stage_ranges = parse_stage_ranges(args.jsd_entropy_stage_ranges)
        model.config.jsd_entropy_abrupt_stage_ranges = args.jsd_entropy_stage_ranges
        model.config.jsd_entropy_stage_prune_ratios = parse_stage_prune_ratios(args.jsd_entropy_stage_prune_ratios)
        
        # Parse boolean parameters
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

    num_sample = len(questions)
    total_latency = 0.0
    total_flops_ratio_attn_ffn = 0.0
    
    results = []

    # Force garbage collection before loop
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    for i, line in enumerate(tqdm(questions)):
        
        # MME specific fields
        idx = line.get("question_id")
        qs = line["question"]
        label = line["answer"]
        category = line["category"]
        image_file = line["image_file"]
        
        cur_prompt = qs

        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path)
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

        # MME prompt addition
        qs = qs + '\n' + "Answer the question using a single word or phrase."
        cur_prompt = cur_prompt + '\n' + "Answer the question using a single word or phrase."

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
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else 0.0,
                max_new_tokens=1024,
                use_cache=False,
                stopping_criteria=stopping_criteria,
                output_attentions=True,
                return_dict_in_generate=True,
            )
            
        torch.cuda.synchronize()  # 确保GPU操作完成
        end_time = time.time()
        inference_latency = end_time - t0
        total_latency += inference_latency

        input_token_len = input_ids.shape[1]
        sequences = output_ids['sequences']
        seq_token_len = sequences.shape[1]

        # Some model implementations return only newly generated tokens,
        # while others return prompt+generated tokens.
        if seq_token_len >= input_token_len:
            n_diff_input_output = (input_ids != sequences[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            decode_ids = sequences[:, input_token_len:]
        else:
            decode_ids = sequences

        outputs = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        # Clean up prediction for MME
        pred = outputs
        # Remove punctuation
        if pred.endswith('.'):
            pred = pred[:-1]
            
        results.append({
            'question_id': idx,
            'category': category,
            'question': line.get('question', ''),
            'image_file': image_file,
            'pred': pred,
            'gt': label
        })

        # Estimate FLOPs Ratio (attn+ffn approx) per example, then average
        try:
            num_layers = getattr(model.model.config, 'num_hidden_layers', None)
            if num_layers is None:
                num_layers = len(getattr(model.model, 'layers', []))
            # Compute text token length in the prompt (excluding image placeholder and system tokens)
            text_len = int(input_ids.shape[1]) - 1 - int(args.sys_length)
            base_len = int(args.sys_length) + int(args.img_length) + max(text_len, 0)
            # For HiMAP
            if args.use_hmap_v:
                # Calculate theoretical FLOPs reduction
                # This is an approximation based on the paper's logic
                # Standard LLaVA: 32 layers, 576 image tokens
                # HiMAP: 
                # - Layers 0 to txt_layer-1: 576 tokens
                # - Layers txt_layer to img_layer-1: txt_rank tokens
                # - Layers img_layer to 32: img_rank tokens
                # - If cut_off_layer is set, tokens are 0 after that layer
                
                if args.use_hmap_v and all(
                    v is not None for v in [args.hmap_v_attn_txt_layer, args.hmap_v_attn_img_layer, args.hmap_v_attn_txt_rank, args.hmap_v_attn_img_rank]
                ):
                    n_before = base_len
                    n_after_txt = int(args.sys_length) + int(args.hmap_v_attn_txt_rank) + max(text_len, 0)
                    n_after_img = int(args.sys_length) + int(args.hmap_v_attn_img_rank) + max(text_len, 0)

                    L_txt = max(min(int(args.hmap_v_attn_txt_layer), num_layers), 0)
                    L_img = max(min(int(args.hmap_v_attn_img_layer), num_layers), L_txt)
                    L_total = num_layers
                

                
                # baseline_flops = total_layers * base_tokens
                # ratio = flops_sum / baseline_flops
                
            elif args.use_fast_v:
                # FastV:
                # - Layers 0 to agg_layer-1: 576 tokens
                # - Layers agg_layer to 32: attention_rank tokens
                
                total_layers = 32
                base_tokens = 576
                
                flops_sum = 0
                for l in range(total_layers):
                    if l < args.fast_v_agg_layer:
                        current_tokens = base_tokens
                    else:
                        current_tokens = args.fast_v_attention_rank
                        
                baseline_flops = total_layers * base_tokens
                ratio = flops_sum / baseline_flops
            else:
                ratio = 1.0
                
        except Exception:
            ratio = 1.0

        # total_flops_ratio_attn_ffn += ratio
    
    # Calculate MME scores
    scores, perception_score, cognition_score = calculate_mme_scores(results)
    
    # Report metrics
    # avg_latency = total_latency / max(num_sample, 1)
    # avg_flops_ratio = total_flops_ratio_attn_ffn / max(num_sample, 1)
    
    # print(f'Avg Latency/Example: {avg_latency:.6f}s')
    # print(f'FLOPs Ratio (approx): {avg_flops_ratio*100:.2f}%')

     # 保存结果到文件
    final_results = {
        'scores': scores,
        'perception_score': perception_score,
        'cognition_score': cognition_score,
        'total_score': perception_score + cognition_score,
        'total_samples': num_sample,
        # 'avg_latency': avg_latency,
        # 'flops_info': avg_flops_ratio,
        'model_config': {
            'use_himap': args.use_hmap_v,
            'sys_length': args.sys_length,
            'img_length': args.img_length,
            'txt_layer': args.hmap_v_attn_txt_layer,
            'img_layer': args.hmap_v_attn_img_layer,
            'txt_rank': args.hmap_v_attn_txt_rank,
            'img_rank': args.hmap_v_attn_img_rank,
            'cut_off_layer': args.cut_off_layer
        } if args.use_hmap_v else ({
            'use_jsd_entropy_pruning': args.use_jsd_entropy_prune,
            'sys_length': args.jsd_entropy_sys_length if args.jsd_entropy_sys_length is not None else args.sys_length,
            'img_length': args.jsd_entropy_img_length if args.jsd_entropy_img_length is not None else args.img_length,
            'topk_percent': args.jsd_entropy_topk_percent,
            'stage_ranges': parse_stage_ranges(args.jsd_entropy_stage_ranges),
            'stage_prune_ratios': parse_stage_prune_ratios(args.jsd_entropy_stage_prune_ratios)
        } if args.use_jsd_entropy_prune else {
            'use_fast_v': args.use_fast_v,
            'sys_length': args.fast_v_sys_length,
            'img_length': args.fast_v_image_token_length,
            'attn_rank': args.fast_v_attention_rank,
            'agg_layer': args.fast_v_agg_layer
        })
    }
    
    # 保存结果
    output_file = f"mme_results_{'himap' if args.use_hmap_v else ('jsd_entropy' if args.use_jsd_entropy_prune else ('fastv' if args.use_fast_v else 'baseline'))}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")

    if args.save_detailed_results_file:
        detailed_path = os.path.expanduser(args.save_detailed_results_file)
        detailed_dir = os.path.dirname(detailed_path)
        if detailed_dir:
            os.makedirs(detailed_dir, exist_ok=True)
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"逐样本结果已保存到: {detailed_path}")
