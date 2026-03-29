#!/usr/bin/env python3
"""
测试不同的 token 选择策略是否产生不同的结果
"""
import torch
import sys
sys.path.insert(0, './src/LLaVA')

from llava.model.language_model.fastv_advanced import FastvAdvanced_LlamaModel
from transformers import LlamaConfig

def create_test_attention():
    """创建测试用的注意力张量"""
    batch_size = 1
    num_heads = 32
    seq_len = 612  # 36 (system) + 576 (image)
    
    # 创建一个随机但固定的注意力矩阵
    torch.manual_seed(42)
    attention = torch.randn(batch_size, num_heads, seq_len, seq_len)
    attention = torch.nn.functional.softmax(attention, dim=-1)
    
    return attention

def test_strategy(strategy_name, alpha=0.5):
    """测试单个策略"""
    print(f"\n{'='*60}")
    print(f"测试策略: {strategy_name}")
    if strategy_name == 'weighted_combination':
        print(f"Alpha: {alpha}")
    print('='*60)
    
    # 创建配置
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        use_fast_v=True,
        fast_v_sys_length=36,
        fast_v_image_token_length=576,
        fast_v_attention_rank=288,
        fast_v_agg_layer=12,
        fast_v_token_selection_method=strategy_name,
        fast_v_weighted_alpha=alpha,
    )
    
    # 创建模型
    model = FastvAdvanced_LlamaModel(config)
    
    # 验证配置
    print(f"Config: token_selection_method = {config.fast_v_token_selection_method}")
    print(f"Model:  token_selection_method = {model.token_selection_method}")
    print(f"Model:  weighted_alpha = {model.weighted_alpha}")
    
    # 创建测试数据
    attention = create_test_attention()
    
    # 选择 tokens
    if strategy_name == 'max_head':
        top_indices = model._select_tokens_max_head(
            attention, 36, 576, 288
        )
    elif strategy_name == 'avg_all_heads':
        top_indices = model._select_tokens_avg_all_heads(
            attention, 36, 576, 288
        )
    elif strategy_name == 'weighted_combination':
        top_indices = model._select_tokens_weighted_combination(
            attention, 36, 576, 288
        )
    elif strategy_name == 'text_weighted':
        top_indices = model._select_tokens_text_weighted(
            attention, 36, 576, 288
        )
    elif strategy_name == 'text_weighted_max_head':
        top_indices = model._select_tokens_text_weighted_max_head(
            attention, 36, 576, 288
        )
    
    # 输出前10个选中的 token 索引
    indices = sorted(top_indices.cpu().numpy().tolist())
    print(f"\n选中的 token 数量: {len(indices)}")
    print(f"前10个索引: {indices[:10]}")
    print(f"后10个索引: {indices[-10:]}")
    
    # 返回索引用于比较
    return set(indices)

def main():
    print("="*60)
    print("Token 选择策略差异测试")
    print("="*60)
    
    # 测试所有策略
    results = {}
    
    strategies = [
        ('max_head', None),
        ('avg_all_heads', None),
        ('weighted_combination', 0.3),
        ('weighted_combination', 0.5),
        ('weighted_combination', 0.7),
        ('text_weighted', None),
        ('text_weighted_max_head', None),
    ]
    
    for strategy, alpha in strategies:
        key = f"{strategy}_{alpha}" if alpha is not None else strategy
        if alpha is not None:
            results[key] = test_strategy(strategy, alpha)
        else:
            results[key] = test_strategy(strategy)
    
    # 比较策略之间的差异
    print("\n" + "="*60)
    print("策略差异比较")
    print("="*60)
    
    strategy_names = list(results.keys())
    for i in range(len(strategy_names)):
        for j in range(i+1, len(strategy_names)):
            name1 = strategy_names[i]
            name2 = strategy_names[j]
            set1 = results[name1]
            set2 = results[name2]
            
            # 计算重叠率
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            overlap = intersection / union * 100 if union > 0 else 0
            
            print(f"{name1:<30} vs {name2:<30}: {overlap:.1f}% 重叠")
    
    print("\n" + "="*60)
    print("测试结论:")
    print("如果不同策略的重叠率都是 100%，说明策略没有生效！")
    print("如果不同策略的重叠率不同，说明策略正常工作。")
    print("="*60)

if __name__ == "__main__":
    main()
