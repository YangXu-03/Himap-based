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
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def calculate_rank(attention_matrix):
    """
    计算注意力矩阵的平均秩
    attention_matrix shape: (batch_size, num_heads, seq_len, seq_len)
    我们只关心图像 token 部分的秩
    """
    # 假设 batch_size=1
    # attention_matrix: (num_heads, seq_len, seq_len)
    if attention_matrix.dim() == 4:
        attention_matrix = attention_matrix.squeeze(0)
    
    num_heads = attention_matrix.shape[0]
    ranks = []
    
    for h in range(num_heads):
        # 使用 torch.linalg.matrix_rank 计算秩
        # 注意：需要 float32 或 float64
        head_attn = attention_matrix[h].float()
        try:
            # 使用基于奇异值的数值秩判定,滤除接近 0 的奇异值
            s = torch.linalg.svdvals(head_attn)
            if s.numel() == 0:
                ranks.append(0)
            else:
                tol = 1e-6 * s[0].item()
                rank = (s > tol).sum().item()
                ranks.append(rank)
        except Exception as e:
            print(f"Rank calculation failed: {e}")
            continue
            
    return np.mean(ranks) if ranks else 0

def calculate_effective_rank(attention_matrix):
    """
    计算有效秩（基于香农熵）
    定义：effective_rank = exp(H(p))
    其中 p_i = s_i^2 / sum(s_j^2) 是归一化的奇异值平方
    H(p) = -sum(p_i * log(p_i)) 是香农熵
    
    attention_matrix shape: (batch_size, num_heads, seq_len, seq_len) 或 (num_heads, seq_len, seq_len)
    返回 heads 上的均值
    """
    if attention_matrix.dim() == 4:
        attention_matrix = attention_matrix.squeeze(0)

    num_heads = attention_matrix.shape[0]
    effs = []
    for h in range(num_heads):
        head_attn = attention_matrix[h].float()
        try:
            s = torch.linalg.svdvals(head_attn)
            if s.numel() == 0:
                effs.append(0.0)
            else:
                # 计算归一化的奇异值平方（作为概率分布）
                s_sq = s ** 2
                total_energy = s_sq.sum()
                
                if total_energy == 0.0:
                    effs.append(0.0)
                else:
                    # 归一化得到概率分布
                    p = s_sq / total_energy
                    
                    # 计算香农熵 H = -sum(p_i * log(p_i))
                    # 过滤掉接近 0 的值以避免 log(0)
                    p_positive = p[p > 1e-12]
                    if p_positive.numel() == 0:
                        effs.append(0.0)
                    else:
                        entropy = -(p_positive * torch.log(p_positive)).sum().item()
                        # 有效秩 = exp(熵)
                        eff_rank = np.exp(entropy)
                        effs.append(eff_rank)
        except Exception as e:
            print(f"Effective rank calc failed: {e}")
            continue
    return np.mean(effs) if effs else 0.0


def normalize_data(data):
    """最大最小值归一化"""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)


def compute_text_weighted_image_scores(attention, sys_len, img_len, text_start_idx):
    """
    新的剪枝策略：使用文本token的重要性加权文本到图像的注意力
    
    Args:
        attention: (num_heads, seq_len, seq_len) 注意力矩阵
        sys_len: 系统提示长度
        img_len: 图像token长度
        text_start_idx: 文本token的起始位置（在图像token之后）
    
    Returns:
        weighted_scores: (img_len,) 加权后的图像token分数
    """
    # 1. 计算每个文本token从其他文本token收到的注意力（文本重要性）
    # attention[:, text_tokens, text_tokens] -> 文本到文本的注意力
    
    # 先对heads求平均
    attn_avg = torch.mean(attention, dim=0)  # (seq_len, seq_len)
    
    # 提取文本区域（图像之后的所有token）
    text_to_text = attn_avg[text_start_idx:, text_start_idx:]  # (num_text, num_text)
    
    # 计算每个文本token收到的总注意力（列求和，表示被关注的程度）
    # 这反映了每个文本token的重要性
    text_importance = text_to_text.sum(dim=0)  # (num_text,)
    
    # 归一化文本重要性为权重
    if text_importance.sum() > 0:
        text_weights = text_importance / text_importance.sum()
    else:
        # 如果全为0，则均匀分配
        text_weights = torch.ones_like(text_importance) / text_importance.numel()
    
    # 2. 提取文本到图像的注意力
    # attn_avg[text_tokens, image_tokens]
    text_to_image = attn_avg[text_start_idx:, sys_len:sys_len+img_len]  # (num_text, img_len)
    
    # 3. 使用文本重要性加权，得到每个图像token的加权分数
    # weighted_scores[j] = sum_i(text_weights[i] * text_to_image[i, j])
    weighted_scores = torch.matmul(text_weights.unsqueeze(0), text_to_image).squeeze(0)  # (img_len,)
    
    return weighted_scores


def run_experiment(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # 加载问题
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    # 为了快速实验，只取前 N 个样本
    if args.num_samples > 0:
        questions = questions[:args.num_samples]

    print(f"Total samples for experiment: {len(questions)}")

    # 实验参数设置
    ranks_to_test = list(range(576, 15, -10))  # 576 down to 16, step -10
    
    results = {
        'tokens': [],
        'accuracies': [],
        'layer31_numeric_ranks': [],
        'layer31_effective_ranks': []
    }

    # 禁用FastV的内置剪枝，我们将手动实现
    model.config.use_fast_v = False
    
    print(f"Starting text-weighted pruning experiment at layer {args.pruning_layer}")
    print(f"Testing {len(ranks_to_test)} different token retention configurations")

    for rank in tqdm(ranks_to_test, desc="Testing Ranks"):
        
        corr_sample = 0
        current_layer31_ranks = []
        current_layer31_effective_ranks = []
        
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
                    use_cache=False,
                    stopping_criteria=stopping_criteria,
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            
            # 计算准确率
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output['sequences'][:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            if outputs == label:
                corr_sample += 1
            
            # 分析注意力并计算秩
            if output.attentions:
                prefill_attentions = output.attentions[0]
                
                sys_len = args.fast_v_sys_length
                img_len = args.fast_v_image_token_length
                
                # 在指定层应用我们的新剪枝策略来分析
                if args.pruning_layer < len(prefill_attentions):
                    pruning_layer_attn = prefill_attentions[args.pruning_layer]  # (batch, heads, seq, seq)
                    
                    if pruning_layer_attn.dim() == 4:
                        pruning_layer_attn = pruning_layer_attn.squeeze(0)  # (heads, seq, seq)
                    
                    # 文本token的起始位置（在图像token之后）
                    text_start_idx = sys_len + img_len
                    
                    # 使用新策略计算图像token的加权分数
                    weighted_scores = compute_text_weighted_image_scores(
                        pruning_layer_attn, sys_len, img_len, text_start_idx
                    )
                    
                    # 根据加权分数选择top-k图像token（模拟剪枝效果）
                    if rank > 0:
                        top_k_indices = weighted_scores.topk(min(rank, img_len)).indices
                    else:
                        top_k_indices = torch.tensor([], dtype=torch.long, device=weighted_scores.device)
                
                # Layer 31 秩分析
                layer31_attn = prefill_attentions[-1]
                img_attn_31 = layer31_attn[:, :, sys_len:sys_len+img_len, sys_len:sys_len+img_len]

                # 如果有选中的token，只在这些token的子矩阵上计算秩
                if len(top_k_indices) > 0:
                    top_k_indices_sorted = top_k_indices.sort().values
                    try:
                        sub = img_attn_31.index_select(2, top_k_indices_sorted).index_select(3, top_k_indices_sorted)
                        rank_val = calculate_rank(sub)
                        eff_val = calculate_effective_rank(sub)
                        current_layer31_ranks.append(rank_val)
                        current_layer31_effective_ranks.append(eff_val)
                    except Exception:
                        rank_val = calculate_rank(img_attn_31)
                        eff_val = calculate_effective_rank(img_attn_31)
                        current_layer31_ranks.append(rank_val)
                        current_layer31_effective_ranks.append(eff_val)
                else:
                    # rank=0的情况
                    current_layer31_ranks.append(0.0)
                    current_layer31_effective_ranks.append(0.0)

        # 记录本轮实验结果
        acc = corr_sample / len(questions)
        avg_layer31_rank = np.mean(current_layer31_ranks) if current_layer31_ranks else 0
        avg_layer31_eff = np.mean(current_layer31_effective_ranks) if current_layer31_effective_ranks else 0
        
        results['tokens'].append(rank)
        results['accuracies'].append(acc)
        results['layer31_numeric_ranks'].append(avg_layer31_rank)
        results['layer31_effective_ranks'].append(avg_layer31_eff)
        
        print(f"Tokens: {rank}, Acc: {acc:.4f}, L31 Numeric Rank: {avg_layer31_rank:.2f}, L31 Effective Rank: {avg_layer31_eff:.2f}")

    # 保存原始数据
    output_file = f'fastv_textweighted_layer{args.pruning_layer}_experiment.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # 绘图
    plot_results(results, args.pruning_layer)

def plot_results(results, pruning_layer):
    tokens = results['tokens']
    accs = np.array(results['accuracies'])
    l31_ranks = np.array(results.get('layer31_numeric_ranks', []))
    l31_effective = np.array(results.get('layer31_effective_ranks', []))
    
    # 归一化
    norm_accs = normalize_data(accs)
    norm_l31 = normalize_data(l31_ranks) if l31_ranks.size else np.zeros_like(norm_accs)
    norm_l31_eff = normalize_data(l31_effective) if l31_effective.size else np.zeros_like(norm_accs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(tokens, norm_accs, label='Accuracy (Normalized)', color='tomato')
    plt.plot(tokens, norm_l31, label='Layer 31 Rank (Normalized)', color='forestgreen')
    plt.plot(tokens, norm_l31_eff, label='Layer 31 Effective Rank (Normalized)', color='dodgerblue')
    
    plt.xlabel('Retained Image Tokens')
    plt.ylabel('Normalized Value')
    plt.title(f'Text-Weighted Pruning Experiment (Layer {pruning_layer})')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()  # X轴从大到小
    
    output_plot = f'fastv_textweighted_layer{pruning_layer}_experiment.png'
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to test per rank")
    
    # FastV defaults
    parser.add_argument('--fast-v-sys-length', type=int, default=35)
    parser.add_argument('--fast-v-image-token-length', type=int, default=576)
    
    # 新增：指定在哪一层进行剪枝分析
    parser.add_argument('--pruning-layer', type=int, default=2, help="Layer index to apply text-weighted pruning")
    
    args = parser.parse_args()
    run_experiment(args)
