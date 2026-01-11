import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import time
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


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
    
    # AdaptInfer hyperparameters
    parser.add_argument('--use-adaptinfer', default=False, action='store_true', help='whether to use AdaptInfer pruning')
    parser.add_argument('--adaptinfer-sys-length', type=int, required=False, help='the length of system prompt')
    parser.add_argument('--adaptinfer-img-length', type=int, required=False, help='the length of image token')
    parser.add_argument('--adaptinfer-pruning-layers', type=int, nargs='+', required=False, help='the layers for pruning (e.g., 8 16 24)')
    parser.add_argument('--adaptinfer-keep-ratio', type=float, required=False, help='the ratio of visual tokens to keep (e.g., 0.5 for 50%)')
    parser.add_argument('--adaptinfer-keep-tokens', type=int, required=False, help='the number of tokens to keep (e.g., 128 for 128 tokens)')
    
    args = parser.parse_args()

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # set model AdaptInfer config
    if args.use_adaptinfer == True:
        model.config.use_adaptinfer = True
        model.config.adaptinfer_sys_length = args.adaptinfer_sys_length
        model.config.adaptinfer_img_length = args.adaptinfer_img_length
        model.config.adaptinfer_pruning_layers = args.adaptinfer_pruning_layers
        model.config.adaptinfer_keep_ratio = args.adaptinfer_keep_ratio
        model.config.adaptinfer_keep_tokens = args.adaptinfer_keep_tokens
        print('ADAPTINFER TECHNIQUE WILL BE USED ------')
        print(f'Pruning layers: {args.adaptinfer_pruning_layers}')
        print(f'Keep tokens: {args.adaptinfer_keep_tokens}')
        # print(f'Keep ratio: {args.adaptinfer_keep_ratio}')
    else:
        model.config.use_adaptinfer = False
        print('NO TOKEN PRUNING TECHNIQUE WILL BE USED ------')

    # Reset AdaptInfer parameters
    if hasattr(model.model, 'reset_adaptinfer'):
        model.model.reset_adaptinfer()

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    num_sample = len(questions)
    corr_sample = 0
    total_latency = 0.0
    total_flops_ratio_attn_ffn = 0.0


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

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
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
    
    print(f'Accuracy: {corr_sample/num_sample:.4f}')
    print(f'Correct samples: {corr_sample}/{num_sample}')

# 计算并输出结果
    accuracy = corr_sample / num_sample
    avg_latency = total_latency / num_sample
    
    print(f"\n=== ScienceQA 评估结果 ===")
    print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"平均延迟 (Average Latency): {avg_latency:.4f} 秒/样本")
    print(f"总样本数: {num_sample}")
    print(f"正确样本数: {corr_sample}")
    
    # # 输出FLOPs信息
    # print(f"\n=== FLOPs 分析 ===")
    # print(f"原始模型FLOPs: {flops_info['vanilla_flops']:.2e}")
    # if 'adaptinfer_flops' in flops_info:
    #     print(f"AdaptInfer模型FLOPs: {flops_info['adaptinfer_flops']:.2e}")
    #     print(f"FLOPs Ratio: {flops_info['flops_ratio']:.1f}%")
    #     print(f"FLOPs减少: {flops_info['reduction']:.1f}%")
    # else:
    #     print(f"FLOPs Ratio: {flops_info['flops_ratio']:.1f}% (基线模型)")
    
    # 输出AdaptInfer技术参数
    if args.use_adaptinfer:
        print(f"\n=== AdaptInfer 技术参数 ===")
        print(f"系统提示长度: {args.adaptinfer_sys_length}")
        print(f"图像token长度: {args.adaptinfer_img_length}")
        print(f"剪枝层: {args.adaptinfer_pruning_layers}")
        print(f"保留token数: {args.adaptinfer_keep_tokens}")
    else:
        print("\n=== 基线模型 (无剪枝) ===")
        print("注意: 这是基线模型的性能")
    
    # 保存结果到文件
    results = {
        'accuracy': accuracy,
        'avg_latency': avg_latency,
        'total_samples': num_sample,
        'correct_samples': corr_sample,
        'model_config': {
            'use_adaptinfer': args.use_adaptinfer,
            'sys_length': args.adaptinfer_sys_length,
            'img_length': args.adaptinfer_img_length,
            'pruning_layers': args.adaptinfer_pruning_layers,
            'keep_tokens': args.adaptinfer_keep_tokens
        }
    }
    
    # 保存结果
    output_file = f"scienceqa_results_{'adaptinfer' if args.use_adaptinfer else 'baseline'}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")