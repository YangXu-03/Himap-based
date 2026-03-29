import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import math
import time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

def calculate_mme_scores(results):
    """Calculate MME scores with perception and cognition breakdown"""
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
    
    print(f"\n{'Category':<30} {'Acc':<10} {'Acc+':<10} {'Score':<10}")
    print("-" * 70)

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
        scores[cat] = {
            'accuracy': acc,
            'accuracy_plus': acc_plus,
            'score': score,
            'num_samples': len(items),
            'num_pairs': len(img_groups)
        }
        
        print(f"{cat:<30} {acc:<10.2f} {acc_plus:<10.2f} {score:<10.2f}")

        if cat in perception_cats:
            perception_score += score
        elif cat in cognition_cats:
            cognition_score += score
            
    print("-" * 70)
    print(f"Perception Score: {perception_score:.2f}")
    print(f"Cognition Score: {cognition_score:.2f}")
    print(f"Total MME Score: {perception_score + cognition_score:.2f}")
    
    return scores, perception_score, cognition_score

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=None, help="number of samples to test (for quick testing)")
    
    # FastV Advanced parameters
    parser.add_argument('--use-fast-v', default=False, action='store_true', help='whether to use fast-v')
    parser.add_argument('--fast-v-sys-length', type=int, default=35, help='the length of system prompt for fast-v')
    parser.add_argument('--fast-v-image-token-length', type=int, default=576, help='the length of image token for fast-v')
    parser.add_argument('--fast-v-attention-rank', type=int, default=288, help='the rank of attention for fast-v')
    parser.add_argument('--fast-v-agg-layer', type=int, default=12, help='the aggregation layer for fast-v')
    parser.add_argument('--fast-v-token-selection-method', type=str, default='avg_all_heads', 
                        choices=['max_head', 'avg_all_heads', 'weighted_combination', 'text_weighted', 'text_weighted_max_head'],
                        help='token selection method')
    parser.add_argument('--fast-v-weighted-alpha', type=float, default=0.5, 
                        help='alpha weight for weighted_combination method')
    
    # Output file
    parser.add_argument('--output-file', type=str, default='mme_results.json', help='output file path')
    
    args = parser.parse_args()
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    # 检查是否为本地路径
    if os.path.exists(model_path):
        model_name = get_model_name_from_path(model_path)
    else:
        model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Set FastV Advanced config
    if args.use_fast_v:
        model.config.use_fast_v = True
        model.config.fast_v_sys_length = args.fast_v_sys_length
        model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.config.fast_v_agg_layer = args.fast_v_agg_layer
        model.config.fast_v_token_selection_method = args.fast_v_token_selection_method
        model.config.fast_v_weighted_alpha = args.fast_v_weighted_alpha
        
        print('========================================')
        print('FASTV ADVANCED TECHNIQUE WILL BE USED')
        print(f'  Token Selection Method: {args.fast_v_token_selection_method}')
        print(f'  System Length: {args.fast_v_sys_length}')
        print(f'  Image Token Length: {args.fast_v_image_token_length}')
        print(f'  Attention Rank: {args.fast_v_attention_rank}')
        print(f'  Aggregation Layer: {args.fast_v_agg_layer}')
        if args.fast_v_token_selection_method == 'weighted_combination':
            print(f'  Alpha Weight: {args.fast_v_weighted_alpha}')
        print('========================================')
        
        model.model.reset_fastv()
        
        # 验证配置是否正确设置
        print('\n验证模型内部配置:')
        print(f'  model.model.token_selection_method: {model.model.token_selection_method}')
        print(f'  model.model.weighted_alpha: {model.model.weighted_alpha}')
        print(f'  model.model.use_fast_v: {model.model.use_fast_v}')
        print(f'  model.model.fast_v_attention_rank: {model.model.fast_v_attention_rank}')
        print('========================================\n')
    else:
        model.config.use_fast_v = False
        print('NO TOKEN PRUNING TECHNIQUE WILL BE USED (BASELINE)')

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # Limit number of samples if specified (for quick testing)
    if args.num_samples is not None and args.num_samples > 0:
        questions = questions[:args.num_samples]
        print(f"限制样本数量为: {args.num_samples}")

    num_sample = len(questions)
    total_latency = 0.0
    
    results = []
    
    # 在第一个样本前再次打印确认配置
    if args.use_fast_v:
        print(f"\n开始推理，再次确认配置:")
        print(f"  model.model.token_selection_method: {model.model.token_selection_method}")
        print(f"  model.model.use_fast_v: {model.model.use_fast_v}")
        print(f"  model.model.fast_v_attention_rank: {model.model.fast_v_attention_rank}\n")

    for i, line in enumerate(tqdm(questions, desc="Processing MME")):
        
        # MME specific fields
        idx = line.get("question_id")
        qs = line["question"]
        label = line["answer"]
        category = line["category"]
        image_file = line["image_file"]
        
        cur_prompt = qs

        image_path = os.path.join(args.image_folder, image_file)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
            
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
            )
            torch.cuda.synchronize()
            inference_latency = time.time() - t0
            total_latency += inference_latency

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        # Clean up prediction for MME
        pred = outputs
        if pred.endswith('.'):
            pred = pred[:-1]
            
        results.append({
            'question_id': idx,
            'category': category,
            'pred': pred,
            'gt': label
        })

    # Calculate MME scores
    scores, perception_score, cognition_score = calculate_mme_scores(results)
    
    avg_latency = total_latency / max(num_sample, 1)
    
    print(f'\nAvg Latency/Example: {avg_latency:.4f}s')

    # Save results
    final_results = {
        'total_score': perception_score + cognition_score,
        'perception_score': perception_score,
        'cognition_score': cognition_score,
        'category_scores': scores,
        'total_samples': num_sample,
        'avg_latency': avg_latency,
        'model_config': {
            'use_fast_v': args.use_fast_v,
            'fast_v_sys_length': args.fast_v_sys_length,
            'fast_v_image_token_length': args.fast_v_image_token_length,
            'fast_v_attention_rank': args.fast_v_attention_rank,
            'fast_v_agg_layer': args.fast_v_agg_layer,
            'fast_v_token_selection_method': args.fast_v_token_selection_method,
            'fast_v_weighted_alpha': args.fast_v_weighted_alpha,
        } if args.use_fast_v else {
            'baseline': True
        },
        'predictions': results
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {args.output_file}")
