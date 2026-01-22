"""
测试 FastV Advanced 的三种 token 选择策略

此脚本演示如何使用和比较三种不同的 token 选择策略：
1. max_head - 最大注意力头策略
2. avg_all_heads - 全头平均策略（原始 FastV）
3. weighted_combination - 加权组合策略
"""

import torch
import numpy as np
from llava.model.language_model.fastv_advanced import FastvAdvanced_LlamaModel
from llava.model.language_model.himap_configuration_llama import LlamaConfig


def create_test_attention(batch_size=1, num_heads=32, seq_len=612, image_start=36, image_len=576):
    """
    创建测试用的注意力权重
    模拟一个场景：某些头对特定图像 tokens 有强注意力
    """
    attention = torch.rand(batch_size, num_heads, seq_len, seq_len)
    
    # 让第 5 个头对图像区域有特别强的注意力
    attention[:, 5, -1, image_start:image_start+image_len] *= 3.0
    
    # 让第 10 个头对不同的图像区域有强注意力
    attention[:, 10, -1, image_start:image_start+image_len] *= 2.0
    
    # 归一化（softmax）
    attention = torch.softmax(attention, dim=-1)
    
    return attention


def test_strategy(model, strategy_name, alpha=None):
    """测试单个策略"""
    print(f"\n{'='*60}")
    print(f"测试策略: {strategy_name}")
    if alpha is not None:
        print(f"Alpha 参数: {alpha}")
    print('='*60)
    
    # 设置策略
    model.token_selection_method = strategy_name
    if alpha is not None:
        model.weighted_alpha = alpha
    
    # 创建测试输入
    batch_size = 1
    seq_length = 612  # 36 (system) + 576 (image) = 612
    hidden_size = 4096
    
    # 模拟 inputs_embeds
    inputs_embeds = torch.randn(batch_size, seq_length, hidden_size)
    
    # 模拟 layer_outputs（包含注意力权重）
    attention_weights = create_test_attention(
        batch_size=batch_size,
        num_heads=32,
        seq_len=seq_length,
        image_start=36,
        image_len=576
    )
    
    # 使用内部方法测试 token 选择
    top_indices = None
    if strategy_name == 'max_head':
        top_indices = model._select_tokens_max_head(
            attention_weights, 
            sys_length=36,
            image_token_length=576, 
            attention_rank=288
        )
    elif strategy_name == 'avg_all_heads':
        top_indices = model._select_tokens_avg_all_heads(
            attention_weights,
            sys_length=36,
            image_token_length=576,
            attention_rank=288
        )
    elif strategy_name == 'weighted_combination':
        top_indices = model._select_tokens_weighted_combination(
            attention_weights,
            sys_length=36,
            image_token_length=576,
            attention_rank=288
        )
    
    # 分析结果
    print(f"\n选择的 token 数量: {len(top_indices)}")
    print(f"Token 索引范围: [{top_indices.min().item()}, {top_indices.max().item()}]")
    print(f"Token 索引（前10个）: {top_indices[:10].cpu().numpy()}")
    
    # 显示元数据
    metadata = model.last_selection_metadata
    print(f"\n策略元数据:")
    for key, value in metadata.items():
        if key == 'head_importance':
            # 显示每个头的重要性
            importance = value
            top_3_heads = np.argsort(importance)[-3:][::-1]
            print(f"  - 最重要的3个头: {top_3_heads} (重要性: {importance[top_3_heads]})")
        else:
            print(f"  - {key}: {value}")
    
    return top_indices, metadata


def compare_strategies(model):
    """比较不同策略选择的 tokens 的重叠情况"""
    print(f"\n{'='*60}")
    print("策略比较分析")
    print('='*60)
    
    # 测试所有策略
    results = {}
    
    # 1. max_head
    tokens_max_head, _ = test_strategy(model, 'max_head')
    results['max_head'] = set(tokens_max_head.cpu().numpy())
    
    # 2. avg_all_heads
    tokens_avg, _ = test_strategy(model, 'avg_all_heads')
    results['avg_all_heads'] = set(tokens_avg.cpu().numpy())
    
    # 3. weighted_combination (不同 alpha 值)
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        tokens_weighted, _ = test_strategy(model, 'weighted_combination', alpha=alpha)
        results[f'weighted_{alpha}'] = set(tokens_weighted.cpu().numpy())
    
    # 计算重叠
    print(f"\n{'='*60}")
    print("Token 选择重叠分析")
    print('='*60)
    
    strategies = list(results.keys())
    for i, strategy1 in enumerate(strategies):
        for strategy2 in strategies[i+1:]:
            overlap = results[strategy1] & results[strategy2]
            overlap_ratio = len(overlap) / len(results[strategy1])
            print(f"\n{strategy1} vs {strategy2}:")
            print(f"  重叠 tokens: {len(overlap)}/288 ({overlap_ratio*100:.1f}%)")
            print(f"  仅在 {strategy1}: {len(results[strategy1] - results[strategy2])}")
            print(f"  仅在 {strategy2}: {len(results[strategy2] - results[strategy1])}")


def test_edge_cases(model):
    """测试边界情况"""
    print(f"\n{'='*60}")
    print("边界情况测试")
    print('='*60)
    
    # 测试 attention_rank = 0（完全移除所有图像 tokens）
    print("\n测试: attention_rank = 0")
    model.fast_v_attention_rank = 0
    
    batch_size = 1
    seq_length = 612
    hidden_size = 4096
    inputs_embeds = torch.randn(batch_size, seq_length, hidden_size)
    attention_weights = create_test_attention()
    
    top_indices = model._select_tokens_max_head(
        attention_weights,
        sys_length=36,
        image_token_length=576,
        attention_rank=0
    )
    print(f"选择的 token 数量: {len(top_indices)} (预期: 0)")
    
    # 恢复
    model.fast_v_attention_rank = 288
    
    # 测试 attention_rank = 576（保留所有图像 tokens）
    print("\n测试: attention_rank = 576")
    top_indices = model._select_tokens_max_head(
        attention_weights,
        sys_length=36,
        image_token_length=576,
        attention_rank=576
    )
    print(f"选择的 token 数量: {len(top_indices)} (预期: 576)")


def main():
    """主测试函数"""
    print("="*60)
    print("FastV Advanced Token Selection Strategy Test")
    print("="*60)
    
    # 创建配置
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        max_position_embeddings=2048,
        # FastV 参数
        use_fast_v=True,
        fast_v_sys_length=36,
        fast_v_image_token_length=576,
        fast_v_attention_rank=288,
        fast_v_agg_layer=2,
        # Advanced FastV 参数
        fast_v_token_selection_method='avg_all_heads',
        fast_v_weighted_alpha=0.5,
    )
    
    # 创建模型（仅用于测试 token 选择逻辑）
    print("\n创建 FastvAdvanced_LlamaModel...")
    print(f"配置:")
    print(f"  - System length: {config.fast_v_sys_length}")
    print(f"  - Image token length: {config.fast_v_image_token_length}")
    print(f"  - Attention rank (保留的 tokens): {config.fast_v_attention_rank}")
    print(f"  - Aggregation layer: {config.fast_v_agg_layer}")
    
    try:
        model = FastvAdvanced_LlamaModel(config)
        print("✓ 模型创建成功")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return
    
    # 测试各个策略
    try:
        # 1. 测试单个策略
        test_strategy(model, 'max_head')
        test_strategy(model, 'avg_all_heads')
        test_strategy(model, 'weighted_combination', alpha=0.5)
        test_strategy(model, 'weighted_combination', alpha=0.7)
        test_strategy(model, 'weighted_combination', alpha=0.3)
        
        # 2. 比较策略
        compare_strategies(model)
        
        # 3. 测试边界情况
        test_edge_cases(model)
        
        print(f"\n{'='*60}")
        print("✓ 所有测试完成")
        print('='*60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
