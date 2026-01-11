import argparse
import torch
import torch.nn.functional as F
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


def compute_cosine_similarity_matrix(tokens):
    """
    计算token之间的余弦相似度矩阵
    Args:
        tokens: (N, D) tensor of N tokens with D dimensions
    Returns:
        similarity_matrix: (N, N) tensor of cosine similarities
    """
    # 归一化
    tokens_norm = F.normalize(tokens, p=2, dim=1)
    # 计算余弦相似度
    similarity_matrix = torch.mm(tokens_norm, tokens_norm.t())
    return similarity_matrix


def farthest_point_sampling(tokens, num_samples):
    """
    最远点采样算法（FPS）
    Args:
        tokens: (N, D) tensor
        num_samples: M, 要选择的token数量
    Returns:
        selected_indices: (M,) tensor of selected token indices
    """
    N = tokens.shape[0]
    if num_samples >= N:
        return torch.arange(N, device=tokens.device)
    
    # 计算相似度矩阵（使用距离，1 - cosine similarity）
    similarity = compute_cosine_similarity_matrix(tokens)
    distance = 1 - similarity
    
    selected_indices = []
    
    # 随机选择第一个点
    first_idx = torch.randint(0, N, (1,), device=tokens.device).item()
    selected_indices.append(first_idx)
    
    # 维护每个点到已选点集的最小距离
    min_distances = distance[first_idx].clone()
    
    # 迭代选择剩余的点
    for _ in range(num_samples - 1):
        # 选择距离已选点集最远的点
        farthest_idx = torch.argmax(min_distances).item()
        selected_indices.append(farthest_idx)
        
        # 更新最小距离
        new_distances = distance[farthest_idx]
        min_distances = torch.min(min_distances, new_distances)
    
    return torch.tensor(selected_indices, device=tokens.device)


def tome_clustering_selection(tokens, num_samples):
    """
    ToMe简化版：基于连通性的聚类选择
    Args:
        tokens: (N, D) tensor
        num_samples: M, 要选择的token数量
    Returns:
        selected_indices: (M,) tensor of selected token indices
    """
    N = tokens.shape[0]
    if num_samples >= N:
        return torch.arange(N, device=tokens.device)
    
    # 计算相似度矩阵
    similarity = compute_cosine_similarity_matrix(tokens)
    
    # 计算每个token的度（与其他token的相似度之和）
    degree = similarity.sum(dim=1)
    
    # 使用简化的k-means风格聚类
    # 初始化：选择度最大的M个token作为初始质心
    _, initial_centers = torch.topk(degree, num_samples)
    centers = initial_centers.clone()
    
    # 迭代优化（简化版，只做几轮）
    for iteration in range(3):
        # 为每个token分配到最近的中心
        center_tokens = tokens[centers]  # (M, D)
        center_sim = torch.mm(F.normalize(tokens, p=2, dim=1), 
                             F.normalize(center_tokens, p=2, dim=1).t())  # (N, M)
        assignments = torch.argmax(center_sim, dim=1)  # (N,)
        
        # 更新中心：每个簇选择度最大的token
        new_centers = []
        for cluster_id in range(num_samples):
            cluster_mask = (assignments == cluster_id)
            if cluster_mask.sum() > 0:
                cluster_indices = torch.where(cluster_mask)[0]
                cluster_degrees = degree[cluster_indices]
                best_in_cluster = cluster_indices[torch.argmax(cluster_degrees)]
                new_centers.append(best_in_cluster.item())
            else:
                # 如果簇为空，保持原中心
                new_centers.append(centers[cluster_id].item())
        
        centers = torch.tensor(new_centers, device=tokens.device)
    
    return centers


def cross_attention_aggregation(query_tokens, key_value_tokens, num_heads=8):
    """
    使用Cross-Attention进行信息聚合
    Args:
        query_tokens: (M, D) tensor, 质心tokens
        key_value_tokens: (N, D) tensor, 原始所有tokens
        num_heads: 多头注意力的头数
    Returns:
        aggregated_tokens: (M, D) tensor, 聚合后的tokens
    """
    M, D = query_tokens.shape
    N = key_value_tokens.shape[0]
    
    head_dim = D // num_heads
    assert D % num_heads == 0, "Embedding dimension must be divisible by num_heads"
    
    # 简化版：使用单头注意力或直接基于余弦相似度的加权聚合
    # 计算query和key之间的相似度
    query_norm = F.normalize(query_tokens, p=2, dim=1)
    key_norm = F.normalize(key_value_tokens, p=2, dim=1)
    
    # attention scores: (M, N)
    attention_scores = torch.mm(query_norm, key_norm.t())
    
    # 使用softmax得到attention权重
    attention_weights = F.softmax(attention_scores / 0.1, dim=1)  # temperature=0.1
    
    # 加权聚合value (这里key=value)
    aggregated_tokens = torch.mm(attention_weights, key_value_tokens)
    
    return aggregated_tokens


class TokenSelectionHook:
    """
    用于在模型forward过程中hook并修改视觉tokens
    保持序列长度不变，将聚合后的tokens放在前M个位置，其余位置置零
    """
    def __init__(self, model, sys_length, img_length, num_kept_tokens, selection_method='fps'):
        self.model = model
        self.sys_length = sys_length
        self.img_length = img_length
        self.num_kept_tokens = num_kept_tokens
        self.selection_method = selection_method
        self.hook_handle = None
        self.applied = False  # 标记是否已经应用过（每次forward只应用一次）
        
    def hook_fn(self, module, input, output):
        """
        Hook function to modify hidden states
        保持序列长度不变，替换图像token区域
        """
        hidden_states = output[0] if isinstance(output, tuple) else output
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 提取图像tokens
        img_start = self.sys_length
        img_end = self.sys_length + self.img_length
        
        if seq_len < img_end:
            return output
        
        # 提取视觉tokens
        image_tokens = hidden_states[:, img_start:img_end, :]  # (batch, N, D)
        
        # 处理每个batch（通常batch_size=1）
        for b in range(batch_size):
            tokens = image_tokens[b]  # (N, D)
            
            # 选择质心tokens
            if self.selection_method == 'fps':
                selected_indices = farthest_point_sampling(tokens, self.num_kept_tokens)
            elif self.selection_method == 'tome':
                selected_indices = tome_clustering_selection(tokens, self.num_kept_tokens)
            else:
                raise ValueError(f"Unknown selection method: {self.selection_method}")
            
            # 提取选中的质心tokens
            centroid_tokens = tokens[selected_indices]  # (M, D)
            
            # 使用Cross-Attention聚合信息
            aggregated_tokens = cross_attention_aggregation(centroid_tokens, tokens)  # (M, D)
            
            # 替换策略：将前M个位置替换为聚合后的tokens，其余位置置零
            # 创建新的image token区域
            new_img_tokens = torch.zeros_like(image_tokens[b])  # (N, D)
            new_img_tokens[:self.num_kept_tokens] = aggregated_tokens
            
            # 更新hidden_states中的图像token部分
            hidden_states[b, img_start:img_end, :] = new_img_tokens
        
        # 返回修改后的output
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states
    
    def register(self, layer_idx):
        """注册hook到指定层"""
        target_layer = self.model.model.layers[layer_idx]
        self.hook_handle = target_layer.register_forward_hook(self.hook_fn)
        
    def reset(self):
        """重置applied标记，用于新的forward调用"""
        self.applied = False
        
    def remove(self):
        """移除hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def run_experiment(args):
    """
    实验目标：测试使用自定义token选择算法（FPS/ToMe）+ Cross-Attention聚合
    在不同层开始处理时的准确率变化
    """
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # 加载问题
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    
    # 限制样本数量
    if args.num_samples > 0:
        questions = questions[:args.num_samples]
    
    print(f"Total samples for experiment: {len(questions)}")
    print(f"Token selection method: {args.selection_method}")
    print(f"Keeping {args.kept_tokens} image tokens")
    
    # 获取模型层数
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")
    
    # 实验参数：测试从第2层到最后一层
    layers_to_test = list(range(2, num_layers + 1))
    
    results = {
        'layers': [],
        'accuracies': [],
        'correct_counts': [],
        'total_samples': len(questions),
        'kept_tokens': args.kept_tokens,
        'selection_method': args.selection_method
    }
    
    # 基线实验（不做任何处理）
    print("Running no-pruning baseline...")
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
    print(f"Baseline Accuracy = {baseline_acc:.4f} ({baseline_corr}/{len(questions)})")

    # 开始实验
    print(f"\nStarting custom token selection experiment...")
    print(f"Testing layers from 2 to {num_layers}")
    
    for layer_idx in tqdm(layers_to_test, desc="Testing Layers"):
        # 创建并注册hook
        hook = TokenSelectionHook(
            model=model,
            sys_length=args.fast_v_sys_length,
            img_length=args.fast_v_image_token_length,
            num_kept_tokens=args.kept_tokens,
            selection_method=args.selection_method
        )
        hook.register(layer_idx - 1)  # 转换为0-based index
        
        corr_sample = 0
        
        try:
            for line in questions:
                idx = line["id"]
                question = line['conversations'][0]
                qs = question['value'].replace('<image>', '').strip()
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
        
        finally:
            # 确保移除hook
            hook.remove()
        
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
    print(f"\nResults saved to {output_file}")
    
    # 绘图
    plot_results(results, args.output_plot)


def plot_results(results, output_plot):
    """绘制准确率随层数变化的曲线"""
    layers = results['layers']
    accs = np.array(results['accuracies'])
    kept_tokens = results['kept_tokens']
    method = results['selection_method']
    
    plt.figure(figsize=(12, 6))
    
    # 绘制准确率曲线
    plt.plot(layers, accs, marker='o', linewidth=2, markersize=6, color='tomato', label='Accuracy')
    
    # 绘制基线
    baseline_acc = results.get('baseline_acc', None)
    if baseline_acc is not None:
        plt.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, label='No Pruning Baseline')
    
    plt.xlabel('Layer Index (Processing Start Layer)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Custom Token Selection ({method.upper()}): Keeping {kept_tokens} Tokens + Cross-Attention', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 添加数值标注
    for i, (layer, acc) in enumerate(zip(layers, accs)):
        if i % 3 == 0:
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
    
    # 性能下降图
    if len(accs) > 0 and baseline_acc is not None and baseline_acc > 0:
        plt.figure(figsize=(12, 6))
        relative_drop = (baseline_acc - accs) / baseline_acc * 100
        plt.plot(layers, relative_drop, marker='s', linewidth=2, markersize=6, 
                color='dodgerblue', label='Relative Performance Drop (%)')
        plt.xlabel('Layer Index (Processing Start Layer)', fontsize=12)
        plt.ylabel('Performance Drop (%)', fontsize=12)
        plt.title(f'Custom Selection ({method.upper()}): Performance Drop (Baseline={baseline_acc:.4f})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        relative_plot = output_plot.replace('.png', '_relative_drop.png')
        plt.savefig(relative_plot, dpi=300, bbox_inches='tight')
        print(f"Relative drop plot saved to {relative_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Token Selection Experiment with FPS/ToMe + Cross-Attention")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--num-samples", type=int, default=-1, 
                        help="Number of samples to test (-1 for all samples)")
    
    # Token selection parameters
    parser.add_argument('--fast-v-sys-length', type=int, default=35,
                        help="System prompt length (tokens before image)")
    parser.add_argument('--fast-v-image-token-length', type=int, default=576,
                        help="Number of image tokens")
    parser.add_argument('--kept-tokens', type=int, default=8,
                        help="Number of image tokens to keep (default: 8)")
    parser.add_argument('--selection-method', type=str, default='fps', choices=['fps', 'tome'],
                        help="Token selection method: 'fps' (Farthest Point Sampling) or 'tome' (ToMe-style clustering)")
    
    # Output files
    parser.add_argument('--output-file', type=str, default='custom_selection_results.json',
                        help="Output JSON file for results")
    parser.add_argument('--output-plot', type=str, default='custom_selection_plot.png',
                        help="Output plot file")
    
    args = parser.parse_args()
    run_experiment(args)
