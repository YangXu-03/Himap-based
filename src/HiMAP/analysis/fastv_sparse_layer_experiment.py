import argparse
import torch
import os
import json
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def run_experiment(args):
    """
    实验目标：测试从第2层到最后一层，在每一层开始只保留指定数量（默认8个）图像token的准确率
    与 fastv_layer_cutoff_experiment.py 的区别：
    - fastv_layer_cutoff_experiment.py: 完全剪除图像token (rank=0)
    - 本实验: 保留少量图像token (rank=kept_tokens，默认8)
    """
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # 加载问题
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    
    # 限制样本数量以加快实验
    if args.num_samples > 0:
        questions = questions[:args.num_samples]
    
    print(f"Total samples for experiment: {len(questions)}")
    print(f"Keeping {args.kept_tokens} image tokens per layer")
    
    # 获取模型层数
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")
    
    # 实验参数：测试从第2层到最后一层
    layers_to_test = list(range(2, num_layers + 1))  # 从第2层开始（索引从1开始计数）
    
    results = {
        'layers': [],
        'accuracies': [],
        'correct_counts': [],
        'total_samples': len(questions),
        'kept_tokens': args.kept_tokens
    }
    
    # 先做一次不剪枝的基线实验（no pruning baseline）
    print("Running no-pruning baseline (this may take a while)...")
    # 确保 FastV 关闭以获得完全的基线
    model.config.use_fast_v = False
    if hasattr(model.model, 'reset_fastv'):
        model.model.reset_fastv()

    baseline_corr = 0
    for line in tqdm(questions, desc="Baseline evaluation"):
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        label = line['conversations'][1]['value']

        image_file = line['image']
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

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

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
            output = model.generate(
                input_ids,
                images=images,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        if outputs == label:
            baseline_corr += 1

    baseline_acc = baseline_corr / len(questions) if len(questions) > 0 else 0.0
    results['baseline_acc'] = baseline_acc
    results['baseline_correct'] = baseline_corr
    print(f"Baseline (no pruning) Accuracy = {baseline_acc:.4f} ({baseline_corr}/{len(questions)})")

    # 重新启用 FastV 并设置参数
    
    # 启用 FastV
    model.config.use_fast_v = True
    model.config.fast_v_sys_length = args.fast_v_sys_length
    model.config.fast_v_image_token_length = args.fast_v_image_token_length
    # 保留指定数量的token：设置 rank 为 kept_tokens
    model.config.fast_v_attention_rank = args.kept_tokens
    
    print(f"Starting layer pruning experiment (keeping {args.kept_tokens} image tokens)")
    print(f"Testing layers from 2 to {num_layers}")
    
    for layer_idx in tqdm(layers_to_test, desc="Testing Layers"):
        # 设置从哪一层开始剪除
        model.config.fast_v_agg_layer = layer_idx
        model.model.reset_fastv()  # 确保参数更新
        
        corr_sample = 0
        
        for line in questions:
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
            
            if args.single_pred_prompt:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

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
                output = model.generate(
                    input_ids,
                    images=images,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )
            
            # 计算准确率
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            
            if outputs == label:
                corr_sample += 1
        
        # 记录结果
        acc = corr_sample / len(questions)
        results['layers'].append(layer_idx)
        results['accuracies'].append(acc)
        results['correct_counts'].append(corr_sample)
        
        print(f"Layer {layer_idx}: Accuracy = {acc:.4f} ({corr_sample}/{len(questions)})")
    
    # 保存结果
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
    
    # 绘图
    plot_results(results, args.output_plot)


def plot_results(results, output_plot):
    """绘制准确率随层数变化的曲线"""
    layers = results['layers']
    accs = np.array(results['accuracies'])
    kept_tokens = results['kept_tokens']
    
    plt.figure(figsize=(12, 6))
    
    # 绘制准确率曲线
    plt.plot(layers, accs, marker='o', linewidth=2, markersize=6, color='tomato', label='Accuracy')
    # 如果有 baseline，则绘制基线虚线
    baseline_acc = results.get('baseline_acc', None)
    if baseline_acc is not None:
        plt.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, label='No Pruning Baseline')
    
    plt.xlabel('Layer Index (Pruning Start Layer)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'FastV: Impact of Keeping {kept_tokens} Image Tokens at Different Layers', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 添加数值标注
    for i, (layer, acc) in enumerate(zip(layers, accs)):
        if i % 3 == 0:  # 每隔3个点标注一次，避免拥挤
            plt.annotate(f'{acc:.3f}', 
                        xy=(layer, acc), 
                        xytext=(0, 8),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_plot}")
    
    # 额外绘制一个对比图：显示相对于 baseline 的性能下降
    if len(accs) > 0:
        baseline_for_drop = results.get('baseline_acc', None)
        if baseline_for_drop is None:
            baseline_for_drop = accs[0]
        if baseline_for_drop > 0:
            plt.figure(figsize=(12, 6))
            relative_drop = (baseline_for_drop - accs) / baseline_for_drop * 100  # 百分比下降
            plt.plot(layers, relative_drop, marker='s', linewidth=2, markersize=6, color='dodgerblue', label='Relative Performance Drop (%)')
            plt.xlabel('Layer Index (Pruning Start Layer)', fontsize=12)
            plt.ylabel('Performance Drop (%)', fontsize=12)
            plt.title(f'FastV: Relative Performance Drop (Keeping {kept_tokens} tokens, Baseline Acc={baseline_for_drop:.4f})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.tight_layout()
            relative_plot = output_plot.replace('.png', '_relative_drop.png')
            plt.savefig(relative_plot, dpi=300, bbox_inches='tight')
            print(f"Relative drop plot saved to {relative_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastV Sparse Pruning Experiment: Keep Limited Image Tokens at Different Layers")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--num-samples", type=int, default=-1, 
                        help="Number of samples to test (-1 for all samples)")
    
    # FastV parameters
    parser.add_argument('--fast-v-sys-length', type=int, default=35,
                        help="System prompt length (tokens before image)")
    parser.add_argument('--fast-v-image-token-length', type=int, default=576,
                        help="Number of image tokens")
    parser.add_argument('--kept-tokens', type=int, default=8,
                        help="Number of image tokens to keep after pruning (default: 8)")
    
    # Output files
    parser.add_argument('--output-file', type=str, default='fastv_sparse_layer_results.json',
                        help="Output JSON file for results")
    parser.add_argument('--output-plot', type=str, default='fastv_sparse_layer_plot.png',
                        help="Output plot file")
    
    args = parser.parse_args()
    run_experiment(args)
