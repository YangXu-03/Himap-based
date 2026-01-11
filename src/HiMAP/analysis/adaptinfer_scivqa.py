import argparse
import torch
import os
import json
from tqdm import tqdm
import math
import numpy as np # 新增导入
import matplotlib.pyplot as plt # 新增导入
from typing import List, Dict, Tuple

# 假设所有必要的 LLaVA 导入都已完成
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image
# from atten_adapter import AttnAdapter, get_props # 已移除
import torch.nn.functional as F 

# --- 辅助函数：与您的代码保持一致 ---
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
# ----------------------------------------

# --- 核心函数：计算注意力总和 (与之前一致) ---
def compute_attention_sums(
    attention_weights_by_layer: List[torch.Tensor], 
    token_type_map: Dict[str, List[int]]    #有写明token的type吗？
) -> Dict[str, Dict[str, List[float]]]:
    """
    计算模型每层中，不同类型 token 接收到 (Received) 和产生 (Produced) 的总注意力之和。
    """
    results = {
        'Received': {'System': [], 'Image': [], 'Instruction': []},
        'Produced': {'System': [], 'Image': [], 'Instruction': []}
    }
    
    for att_weights in attention_weights_by_layer:
        # 步骤 1: 将所有 Head 的注意力权重求平均
        # (N_Heads, Query, Key) -> (Query, Key)
        att_matrix = att_weights.mean(dim=0)
        
        for token_type, indices in token_type_map.items():
            if not indices:
                results['Received'][token_type].append(0.0)
                results['Produced'][token_type].append(0.0)
                continue
            
            # Received Attention: 目标 token (Query) 是该类型。Q 在维度 0。
            received_sum = att_matrix[indices, :].sum().item()
            results['Received'][token_type].append(received_sum)

            # Produced Attention: 源 token (Key) 是该类型。K 在维度 1。
            produced_sum = att_matrix[:, indices].sum().item()
            results['Produced'][token_type].append(produced_sum)
            
    return results

# --- 新增函数：绘制注意力总和堆叠柱状图 ---
def plot_attention_sums(attention_data: Dict[str, Dict[str, List[float]]], title_prefix: str):
    """
    绘制模型每层中不同类型 token 的总注意力之和的堆叠柱状图。
    """
    token_types = ['System', 'Image', 'Instruction']
    colors = {'System': '#4CAF50', 'Image': '#FF9800', 'Instruction': '#2196F3'}
    
    # 分别绘制 Received 和 Produced 两个图
    for att_type in ['Received', 'Produced']:
        data = attention_data[att_type]
        
        # 检查数据是否为空
        if not data or not data[token_types[0]]:
            print(f"⚠️ {att_type} 维度数据为空，跳过绘图。")
            continue

        num_layers = len(data[token_types[0]])
        labels = [f'L{i+1}' for i in range(num_layers)]
        
        fig, ax = plt.subplots(figsize=(16, 7))
        
        # 初始化堆叠的底部值
        bottom_values = np.zeros(num_layers)

        print(f"Drawing chart for: {att_type} Attention Sum...")

        for token_type in token_types:
            # 获取当前类型 token 在所有层上的注意力总和列表
            values = np.array(data.get(token_type, [0] * num_layers))
            
            # 绘制当前类型 token 的柱子，底部是前面积累的值
            ax.bar(
                labels, 
                values, 
                bottom=bottom_values, 
                label=f'{token_type} Tokens', 
                color=colors[token_type],
                edgecolor='white'
            )
            
            # 更新底部值，用于下一个堆叠
            bottom_values += values

        # 添加标签和标题
        ax.set_ylabel(f'Total Attention Sum ({att_type} - per Layer)', fontsize=12)
        ax.set_xlabel('Model Layer', fontsize=12)
        ax.set_title(f'{title_prefix} - Total {att_type} Attention', pad=15, fontsize=14)
        ax.legend(title="Source/Target Tokens", loc='upper left')
        
        plt.xticks(rotation=0, ha='center')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# --- 主执行逻辑 (修改了最后的聚合和保存部分) ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--plot-results", action="store_true", help="Whether to plot the results.") # 新增参数
    args = parser.parse_args()

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base, 
        model_name,
        device='cuda',
        attn_implementation='eager' 
    )

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    all_attention_results = [] 

    for i, line in enumerate(tqdm(questions)):
        # ... (数据准备、prompt 构建、tokenization 逻辑省略) ...
        
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs
        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        images = image_tensor.unsqueeze(0).half().cuda()
        
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
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # 禁用梯度计算，只进行前向传播
        with torch.no_grad():
            output = model.forward(
                input_ids,
                images=images,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )

        attention_weights_by_layer = output['attentions']
        att_weights_cpu = [att.squeeze(0).cpu() for att in attention_weights_by_layer]

        # --- 确定 Token 索引 ---
        image_token_indices = (input_ids.squeeze(0) == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].tolist()
        
        system_indices = []
        instruction_indices = []
        
        if image_token_indices:
            first_image_idx = image_token_indices[0]
            last_image_idx = image_token_indices[-1]

            system_indices = list(range(first_image_idx)) 
            instruction_start_idx = last_image_idx + 1
            instruction_indices = list(range(instruction_start_idx, input_ids.shape[1]))
            
        token_map = {
            'System': system_indices,
            'Image': image_token_indices,
            'Instruction': instruction_indices
        }
        
        # 3. 计算注意力总和
        attention_results = compute_attention_sums(
            attention_weights_by_layer=att_weights_cpu,
            token_type_map=token_map
        )
        
        all_attention_results.append(attention_results)

    # --- 结果聚合、保存与绘图 ---
    
    if all_attention_results:
        # 步骤 1: 聚合结果 (计算平均值)
        avg_results = {
            'Received': {'System': [], 'Image': [], 'Instruction': []},
            'Produced': {'System': [], 'Image': [], 'Instruction': []}
        }
        
        num_layers = len(all_attention_results[0]['Received']['System'])
        num_samples = len(all_attention_results)

        for att_type in ['Received', 'Produced']:
            for token_type in ['System', 'Image', 'Instruction']:
                stacked_values = np.stack([
                    res[att_type][token_type] for res in all_attention_results
                ])
                avg_results[att_type][token_type] = stacked_values.mean(axis=0).tolist()
                
        # 步骤 2: 保存结果
        with open(args.result_file, 'w') as f:
            json.dump(avg_results, f, indent=4)
        
        print(f"✅ 平均注意力总和结果已保存到: {args.result_file}")
        
        # 步骤 3: 绘制图表
        if args.plot_results:
            plot_attention_sums(
                avg_results,
                title_prefix=f'Average Attention across {num_samples} Samples'
            )
    else:
        print("⚠️ 未处理任何样本，结果为空。")