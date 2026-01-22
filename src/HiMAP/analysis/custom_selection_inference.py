import argparse
import torch
import os
import json
from tqdm import tqdm
import time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Token Selection Inference")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument('--baseline', action='store_true',
                        help="Run in baseline mode without token pruning (full 576 tokens)")
    
    # Custom selection parameters
    parser.add_argument('--custom-sys-length', type=int, default=35,
                        help="System prompt length (tokens before image)")
    parser.add_argument('--custom-image-token-length', type=int, default=576,
                        help="Number of image tokens")
    parser.add_argument('--custom-kept-tokens', type=int, default=8,
                        help="Number of image tokens to keep (default: 8)")
    parser.add_argument('--custom-agg-layer', type=int, default=12,
                        help="Layer to apply token selection (default: 12)")
    parser.add_argument('--custom-selection-method', type=str, default='fps', 
                        choices=['fps', 'tome'],
                        help="Token selection method: 'fps' or 'tome'")
    parser.add_argument('--custom-temperature', type=float, default=0.1,
                        help="Temperature for cross-attention softmax")
    
    # Output
    parser.add_argument('--output-file', type=str, default=None,
                        help="Output JSON file for results")
    
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

    # set model custom selection config
    # In baseline mode, disable all pruning techniques
    # Only set custom selection in non-baseline mode
    if not args.baseline:
        model.config.use_custom_selection = True
        model.config.custom_sys_length = args.custom_sys_length
        model.config.custom_image_token_length = args.custom_image_token_length
        model.config.custom_kept_tokens = args.custom_kept_tokens
        model.config.custom_agg_layer = args.custom_agg_layer
        model.config.custom_selection_method = args.custom_selection_method
        model.config.custom_temperature = args.custom_temperature
        print('CUSTOM TOKEN SELECTION TECHNIQUE WILL BE USED ------')
        
        # Reset custom selection if the model supports it
        if hasattr(model.model, 'reset_custom_selection'):
            model.model.reset_custom_selection()
    else:
        # Baseline mode: disable HiMAP to avoid memory overhead
        # DO NOT call reset_hmapv() - just set the flag to False
        model.config.use_hmap_v = False
        print('NO TOKEN PRUNING TECHNIQUE WILL BE USED ------')

    
    # 加载问题
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.num_samples > 0:
        questions = questions[:args.num_samples]
    
    num_sample = len(questions)
    corr_sample = 0
    total_latency = 0.0

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
        n_diff_input_output = (input_ids != output_ids['sequences'][:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids['sequences'][:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        if outputs == label:
            corr_sample += 1
    
    # Report metrics
    avg_latency = total_latency / max(num_sample, 1)
    accuracy = corr_sample/num_sample
    print(corr_sample/num_sample)
    print(f'Avg Latency/Example: {avg_latency:.6f}s')

     # 保存结果到文件
    results = {
        'accuracy': accuracy,
        'avg_latency': avg_latency,
        'total_samples': num_sample,
        'correct_samples': corr_sample,
        'model_config': {
            'use_custom_selection': not args.baseline,
            'sys_length': args.custom_sys_length,
            'img_length': args.custom_image_token_length,
            'kept_tokens': args.custom_kept_tokens,
            'agg_layer': args.custom_agg_layer,
            'selection_method': args.custom_selection_method,
            'temperature': args.custom_temperature
        }
    }
    
    # 保存结果
    output_file = args.output_file if args.output_file else f"custom_selection_results_{'custom' if not args.baseline else 'baseline'}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")
