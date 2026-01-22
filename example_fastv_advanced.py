"""
FastV Advanced 使用示例

演示如何在实际应用中使用三种不同的 token 选择策略
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/LLaVA'))

from llava.model.language_model.fastv_advanced import FastvAdvanced_LlamaModel
from llava.model.language_model.himap_configuration_llama import LlamaConfig


def example_1_basic_usage():
    """示例 1: 基本使用 - 使用不同策略"""
    print("="*60)
    print("示例 1: 基本使用")
    print("="*60)
    
    # 创建配置 - 使用 max_head 策略
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        use_fast_v=True,
        fast_v_sys_length=36,
        fast_v_image_token_length=576,
        fast_v_attention_rank=288,  # 保留 50% 的图像 tokens
        fast_v_agg_layer=2,
        fast_v_token_selection_method='max_head',  # 使用最大注意力头策略
    )
    
    model = FastvAdvanced_LlamaModel(config)
    print(f"\n✓ 创建模型，使用策略: {model.token_selection_method}")
    print(f"  保留 tokens: {model.fast_v_attention_rank}/{model.fast_v_image_token_length}")
    print(f"  裁剪比例: {(1 - model.fast_v_attention_rank/model.fast_v_image_token_length)*100:.1f}%")


def example_2_switch_strategy():
    """示例 2: 动态切换策略"""
    print("\n" + "="*60)
    print("示例 2: 动态切换策略")
    print("="*60)
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        use_fast_v=True,
        fast_v_sys_length=36,
        fast_v_image_token_length=576,
        fast_v_attention_rank=288,
        fast_v_agg_layer=2,
    )
    
    model = FastvAdvanced_LlamaModel(config)
    
    # 策略 1: max_head
    model.token_selection_method = 'max_head'
    print(f"\n策略 1: {model.token_selection_method}")
    # ... 进行推理 ...
    
    # 策略 2: avg_all_heads
    model.token_selection_method = 'avg_all_heads'
    print(f"策略 2: {model.token_selection_method}")
    # ... 进行推理 ...
    
    # 策略 3: weighted_combination with alpha=0.7
    model.token_selection_method = 'weighted_combination'
    model.weighted_alpha = 0.7
    print(f"策略 3: {model.token_selection_method} (alpha={model.weighted_alpha})")
    # ... 进行推理 ...


def example_3_weighted_alpha_tuning():
    """示例 3: 调整加权组合的 alpha 参数"""
    print("\n" + "="*60)
    print("示例 3: 调整 alpha 参数（weighted_combination 策略）")
    print("="*60)
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        use_fast_v=True,
        fast_v_sys_length=36,
        fast_v_image_token_length=576,
        fast_v_attention_rank=288,
        fast_v_agg_layer=2,
        fast_v_token_selection_method='weighted_combination',
    )
    
    model = FastvAdvanced_LlamaModel(config)
    
    print("\nalpha 参数的含义:")
    print("  score = max_head_attention × α + avg_other_heads_attention × (1-α)")
    print("  - α → 1.0: 更依赖最大注意力头")
    print("  - α → 0.0: 更依赖其他头的平均")
    print("  - α = 0.5: 平衡两者")
    
    # 测试不同的 alpha 值
    alpha_values = [0.2, 0.5, 0.8]
    for alpha in alpha_values:
        model.weighted_alpha = alpha
        print(f"\n  α = {alpha}:")
        if alpha > 0.7:
            print(f"    → 强调最大注意力头 (max_head 占 {alpha*100:.0f}%)")
        elif alpha < 0.3:
            print(f"    → 强调其他头平均 (其他头占 {(1-alpha)*100:.0f}%)")
        else:
            print(f"    → 平衡策略")


def example_4_different_pruning_ratios():
    """示例 4: 不同的剪枝比例"""
    print("\n" + "="*60)
    print("示例 4: 不同的剪枝比例")
    print("="*60)
    
    image_token_length = 576
    pruning_ratios = [0.25, 0.50, 0.75]  # 剪枝 25%, 50%, 75%
    
    for ratio in pruning_ratios:
        attention_rank = int(image_token_length * (1 - ratio))
        
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            use_fast_v=True,
            fast_v_sys_length=36,
            fast_v_image_token_length=image_token_length,
            fast_v_attention_rank=attention_rank,
            fast_v_agg_layer=2,
            fast_v_token_selection_method='max_head',
        )
        
        model = FastvAdvanced_LlamaModel(config)
        print(f"\n剪枝比例: {ratio*100:.0f}%")
        print(f"  保留 tokens: {attention_rank}/{image_token_length}")
        print(f"  剪掉 tokens: {image_token_length - attention_rank}")


def example_5_aggregation_layer():
    """示例 5: 不同的聚合层"""
    print("\n" + "="*60)
    print("示例 5: 不同的聚合层 (从哪一层开始剪枝)")
    print("="*60)
    
    print("\n聚合层 (fast_v_agg_layer) 的作用:")
    print("  - 在该层计算注意力并决定保留哪些 tokens")
    print("  - 较早的层 (如 layer 1): 可能更关注低级视觉特征")
    print("  - 较晚的层 (如 layer 4): 可能更关注高级语义信息")
    print("  - 默认 layer 2: FastV 的推荐值")
    
    agg_layers = [1, 2, 3, 4]
    
    for agg_layer in agg_layers:
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            use_fast_v=True,
            fast_v_sys_length=36,
            fast_v_image_token_length=576,
            fast_v_attention_rank=288,
            fast_v_agg_layer=agg_layer,
            fast_v_token_selection_method='max_head',
        )
        
        model = FastvAdvanced_LlamaModel(config)
        print(f"\n聚合层: Layer {agg_layer}")
        print(f"  - 前 {agg_layer-1} 层: 使用完整的图像 tokens")
        print(f"  - 第 {agg_layer} 层: 计算注意力并选择 tokens")
        print(f"  - 第 {agg_layer+1}-32 层: 使用剪枝后的 tokens")


def example_6_strategy_comparison():
    """示例 6: 策略对比建议"""
    print("\n" + "="*60)
    print("示例 6: 策略选择建议")
    print("="*60)
    
    print("\n何时使用各种策略：")
    
    print("\n1. max_head (最大注意力头)")
    print("   适用场景:")
    print("   - 模型中某个特定头明显负责视觉信息处理")
    print("   - 需要保留最显著的视觉特征")
    print("   - 追求最快的推理速度")
    print("   优点: 决策清晰，保留最重要特征")
    print("   缺点: 可能忽略其他头的信息")
    
    print("\n2. avg_all_heads (全头平均) [原始 FastV]")
    print("   适用场景:")
    print("   - 通用场景，作为基线方法")
    print("   - 不确定哪个头最重要时")
    print("   - 需要稳定的性能")
    print("   优点: 综合所有信息，稳定可靠")
    print("   缺点: 可能平滑掉重要特征")
    
    print("\n3. weighted_combination (加权组合)")
    print("   适用场景:")
    print("   - 需要精细控制的场景")
    print("   - 希望平衡单头决策和多头共识")
    print("   - 可以根据任务调整 alpha")
    print("   优点: 灵活可调，兼顾两者优势")
    print("   缺点: 需要调参，略复杂")
    
    print("\n建议的测试流程:")
    print("   1. 先用 avg_all_heads 建立基线")
    print("   2. 尝试 max_head 看是否提升")
    print("   3. 如果两者差异大，用 weighted_combination 微调")
    print("   4. 通过实验确定最佳 alpha 值")


def example_7_inspection():
    """示例 7: 检查选择结果"""
    print("\n" + "="*60)
    print("示例 7: 检查和分析 token 选择结果")
    print("="*60)
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        use_fast_v=True,
        fast_v_sys_length=36,
        fast_v_image_token_length=576,
        fast_v_attention_rank=288,
        fast_v_agg_layer=2,
        fast_v_token_selection_method='weighted_combination',
        fast_v_weighted_alpha=0.7,
    )
    
    model = FastvAdvanced_LlamaModel(config)
    
    print("\n模型提供的检查接口:")
    print(f"\n1. 选择的 token 索引:")
    print(f"   model.last_gen_kept_indices")
    print(f"   → 返回保留的 token 在序列中的全局索引")
    
    print(f"\n2. 生成的注意力 mask:")
    print(f"   model.last_gen_attention_mask")
    print(f"   → 返回 bool tensor，True 表示保留，False 表示剪掉")
    
    print(f"\n3. 选择元数据:")
    print(f"   model.last_selection_metadata")
    print(f"   → 包含选择方法、最大头索引、各头重要性等信息")
    
    print(f"\n使用示例:")
    print(f"   # 推理后")
    print(f"   outputs = model(**inputs)")
    print(f"   ")
    print(f"   # 查看选择的 tokens")
    print(f"   kept_indices = model.last_gen_kept_indices")
    print(f"   print(f'保留了 {{len(kept_indices)}} 个 tokens')")
    print(f"   ")
    print(f"   # 查看元数据")
    print(f"   metadata = model.last_selection_metadata")
    print(f"   if 'max_head_idx' in metadata:")
    print(f"       print(f'最重要的注意力头: {{metadata[\"max_head_idx\"]}}')")


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print(" " * 20 + "FastV Advanced 使用示例")
    print("="*80)
    
    try:
        example_1_basic_usage()
        example_2_switch_strategy()
        example_3_weighted_alpha_tuning()
        example_4_different_pruning_ratios()
        example_5_aggregation_layer()
        example_6_strategy_comparison()
        example_7_inspection()
        
        print("\n" + "="*80)
        print("✓ 所有示例运行完成")
        print("="*80)
        print("\n提示:")
        print("  - 查看 FASTV_ADVANCED_README.md 了解详细文档")
        print("  - 运行 test_fastv_advanced.py 进行功能测试")
        print("  - 在您的数据集上测试不同策略以找到最佳配置")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
