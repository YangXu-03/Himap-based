#!/usr/bin/env python3
"""
演示为什么对所有文本tokens求平均会导致不同策略产生相同结果
"""
import torch
import numpy as np

def demonstrate_averaging_effect():
    """演示平均操作如何抹平差异"""
    
    # 模拟参数
    batch_size = 1
    num_heads = 32
    num_text_tokens = 35
    num_image_tokens = 576
    
    # 创建一个有明显head差异的attention矩阵
    torch.manual_seed(42)
    attention = torch.randn(batch_size, num_heads, num_text_tokens, num_image_tokens)
    attention = torch.softmax(attention, dim=-1)
    
    print("="*80)
    print("实验：对比使用最后token vs 平均所有tokens的差异")
    print("="*80)
    
    # ========== 方法1: 只使用最后一个token (v1版本) ==========
    print("\n【方法1】只使用最后一个token的attention")
    print("-"*80)
    
    # 提取最后一个token
    last_token_attn = attention[0, :, -1, :]  # [num_heads, num_image_tokens]
    
    # max_head策略
    head_importance_last = last_token_attn.sum(dim=-1)  # [num_heads]
    max_head_idx_last = head_importance_last.argmax()
    max_head_score_last = last_token_attn[max_head_idx_last]  # [num_image_tokens]
    
    # avg_all_heads策略
    avg_score_last = last_token_attn.mean(dim=0)  # [num_image_tokens]
    
    # 计算相关性
    correlation_last = torch.corrcoef(torch.stack([max_head_score_last, avg_score_last]))[0,1]
    
    # Top-10 tokens
    top10_max_last = max_head_score_last.topk(10).indices
    top10_avg_last = avg_score_last.topk(10).indices
    overlap_last = len(set(top10_max_last.tolist()) & set(top10_avg_last.tolist()))
    
    print(f"选中的最大head: {max_head_idx_last.item()}")
    print(f"max_head vs avg_all_heads 分数相关性: {correlation_last:.4f}")
    print(f"Top-10 tokens重叠数: {overlap_last}/10")
    print(f"Top-10 max_head: {sorted(top10_max_last.tolist()[:5])[:5]}")
    print(f"Top-10 avg_all: {sorted(top10_avg_last.tolist()[:5])[:5]}")
    
    # ========== 方法2: 平均所有文本tokens (当前版本) ==========
    print("\n【方法2】平均所有文本tokens的attention")
    print("-"*80)
    
    # 对所有文本tokens求平均
    mean_over_text = attention[0].mean(dim=1)  # [num_heads, num_image_tokens]
    
    # max_head策略
    head_importance_mean = mean_over_text.sum(dim=-1)  # [num_heads]
    max_head_idx_mean = head_importance_mean.argmax()
    max_head_score_mean = mean_over_text[max_head_idx_mean]  # [num_image_tokens]
    
    # avg_all_heads策略
    avg_score_mean = mean_over_text.mean(dim=0)  # [num_image_tokens]
    
    # 计算相关性
    correlation_mean = torch.corrcoef(torch.stack([max_head_score_mean, avg_score_mean]))[0,1]
    
    # Top-10 tokens
    top10_max_mean = max_head_score_mean.topk(10).indices
    top10_avg_mean = avg_score_mean.topk(10).indices
    overlap_mean = len(set(top10_max_mean.tolist()) & set(top10_avg_mean.tolist()))
    
    print(f"选中的最大head: {max_head_idx_mean.item()}")
    print(f"max_head vs avg_all_heads 分数相关性: {correlation_mean:.4f}")
    print(f"Top-10 tokens重叠数: {overlap_mean}/10")
    print(f"Top-10 max_head: {sorted(top10_max_mean.tolist()[:5])[:5]}")
    print(f"Top-10 avg_all: {sorted(top10_avg_mean.tolist()[:5])[:5]}")
    
    # ========== 对比分析 ==========
    print("\n" + "="*80)
    print("对比分析")
    print("="*80)
    print(f"\n方法1（最后token）:")
    print(f"  - 相关性: {correlation_last:.4f}")
    print(f"  - Top-10重叠: {overlap_last}/10")
    print(f"  - 结论: 相关性较{'低' if correlation_last < 0.9 else '高'}，策略有{'明显' if overlap_last < 8 else '较小'}差异")
    
    print(f"\n方法2（平均所有tokens）:")
    print(f"  - 相关性: {correlation_mean:.4f}")
    print(f"  - Top-10重叠: {overlap_mean}/10")
    print(f"  - 结论: 相关性{'非常高' if correlation_mean > 0.95 else '较高'}，策略{'几乎相同' if overlap_mean > 8 else '有差异'}")
    
    # ========== 数学解释 ==========
    print("\n" + "="*80)
    print("数学解释：为什么平均操作导致策略趋同")
    print("="*80)
    
    # 计算head之间的方差
    print("\n1. Head之间的差异性:")
    
    # 对最后一个token
    std_across_heads_last = last_token_attn.std(dim=0).mean()
    print(f"   最后token - heads间标准差的平均: {std_across_heads_last:.6f}")
    
    # 对平均后的
    std_across_heads_mean = mean_over_text.std(dim=0).mean()
    print(f"   平均后 - heads间标准差的平均: {std_across_heads_mean:.6f}")
    print(f"   标准差减少了: {(1 - std_across_heads_mean/std_across_heads_last)*100:.1f}%")
    
    print("\n2. 不同heads选出的image token重要性排序相似度:")
    
    # 对最后token：不同heads的top tokens
    def kendall_tau_distance(rank1, rank2):
        """计算Kendall tau距离（排序相似度）"""
        n = len(rank1)
        disagreements = 0
        for i in range(n):
            for j in range(i+1, n):
                if (rank1[i] < rank1[j]) != (rank2[i] < rank2[j]):
                    disagreements += 1
        max_disagreements = n * (n - 1) / 2
        return 1 - (disagreements / max_disagreements)
    
    # 随机选5个heads对比
    heads_to_compare = [0, 8, 16, 24, 31]
    
    print("\n   最后token - 不同heads的排序相似度:")
    similarities_last = []
    for i in range(len(heads_to_compare)):
        for j in range(i+1, len(heads_to_compare)):
            h1, h2 = heads_to_compare[i], heads_to_compare[j]
            rank1 = last_token_attn[h1].argsort(descending=True)[:20]
            rank2 = last_token_attn[h2].argsort(descending=True)[:20]
            overlap = len(set(rank1.tolist()) & set(rank2.tolist()))
            similarities_last.append(overlap / 20)
            if i == 0 and j == 1:  # 只打印第一对
                print(f"     Head {h1} vs Head {h2}: Top-20重叠 {overlap}/20 ({overlap/20*100:.0f}%)")
    
    print(f"   平均相似度: {np.mean(similarities_last)*100:.1f}%")
    
    print("\n   平均后 - 不同heads的排序相似度:")
    similarities_mean = []
    for i in range(len(heads_to_compare)):
        for j in range(i+1, len(heads_to_compare)):
            h1, h2 = heads_to_compare[i], heads_to_compare[j]
            rank1 = mean_over_text[h1].argsort(descending=True)[:20]
            rank2 = mean_over_text[h2].argsort(descending=True)[:20]
            overlap = len(set(rank1.tolist()) & set(rank2.tolist()))
            similarities_mean.append(overlap / 20)
            if i == 0 and j == 1:
                print(f"     Head {h1} vs Head {h2}: Top-20重叠 {overlap}/20 ({overlap/20*100:.0f}%)")
    
    print(f"   平均相似度: {np.mean(similarities_mean)*100:.1f}%")
    
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    print("""
对所有文本tokens求平均后，不同heads的attention分布变得高度相似！

原因：
1. 平均操作是一个强平滑器 - 当平均35个tokens时，随机波动被大幅抵消
2. 所有heads都在看相同的文本→图像映射关系，只是关注点略有不同
3. 平均后，这些"略有不同"的关注点被模糊成了相似的分布

因此：
- max_head选出的head和其他heads的平均几乎一样
- 不同策略选出的top-k tokens高度重叠
- 实验结果相同！

解决方案：使用最后一个token（当前生成位置），保留heads的差异性
""")

if __name__ == "__main__":
    demonstrate_averaging_effect()
