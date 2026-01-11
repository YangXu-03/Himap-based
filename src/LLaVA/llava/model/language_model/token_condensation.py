"""
Token Condensation Module for Vision-Language Models
代替 Hard Pruning，通过筛选代表性 Token 并进行信息聚合来保留上下文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TokenCondensation(nn.Module):
    """
    Token 浓缩模块：从 N 个 Token 中筛选 M 个代表性 Token（质心），
    并通过交叉注意力机制聚合全局信息。
    
    Args:
        hidden_size (int): Token 的特征维度
        num_heads (int): 多头注意力的头数
        selection_strategy (str): 选择策略 ['fps', 'connectivity']
            - 'fps': 最远点采样 (Furthest Point Sampling)
            - 'connectivity': 基于连通性的密度筛选
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        selection_strategy: str = 'fps',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.selection_strategy = selection_strategy
        
        # Cross-Attention 的投影层
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def compute_similarity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算 Token 之间的余弦相似度矩阵
        
        Args:
            features: [batch_size, num_tokens, hidden_size]
            
        Returns:
            similarity_matrix: [batch_size, num_tokens, num_tokens]
        """
        # L2 归一化
        features_norm = F.normalize(features, p=2, dim=-1)  # [B, N, D]
        
        # 计算余弦相似度矩阵
        similarity = torch.bmm(features_norm, features_norm.transpose(1, 2))  # [B, N, N]
        
        return similarity
    
    def select_centroids_fps(
        self,
        features: torch.Tensor,
        similarity_matrix: torch.Tensor,
        num_centroids: int
    ) -> torch.Tensor:
        """
        最远点采样 (Furthest Point Sampling) 策略选择代表性 Token
        
        Args:
            features: [batch_size, num_tokens, hidden_size]
            similarity_matrix: [batch_size, num_tokens, num_tokens]
            num_centroids: 要选择的质心数量 M
            
        Returns:
            centroid_indices: [batch_size, num_centroids] 选中的索引
        """
        batch_size, num_tokens, _ = features.shape
        device = features.device
        
        # 将相似度转换为距离（1 - similarity）
        distance_matrix = 1 - similarity_matrix  # [B, N, N]
        
        centroid_indices = []
        
        for b in range(batch_size):
            # 随机选择第一个点作为起始
            first_idx = torch.randint(0, num_tokens, (1,), device=device)
            selected = [first_idx.item()]
            
            # 初始化距离：每个点到已选集合的最小距离
            min_distances = distance_matrix[b, first_idx, :].squeeze(0)  # [N]
            
            # 迭代选择剩余的点
            for _ in range(num_centroids - 1):
                # 选择距离最大的点
                farthest_idx = torch.argmax(min_distances).item()
                selected.append(farthest_idx)
                
                # 更新距离：每个点到新选点的距离
                new_distances = distance_matrix[b, farthest_idx, :]
                min_distances = torch.min(min_distances, new_distances)
            
            centroid_indices.append(torch.tensor(selected, device=device))
        
        centroid_indices = torch.stack(centroid_indices, dim=0)  # [B, M]
        return centroid_indices
    
    def select_centroids_connectivity(
        self,
        similarity_matrix: torch.Tensor,
        num_centroids: int
    ) -> torch.Tensor:
        """
        基于连通性（Degree）的密度筛选策略
        选择与其他 Token 相似度总和最大的 M 个 Token
        
        Args:
            similarity_matrix: [batch_size, num_tokens, num_tokens]
            num_centroids: 要选择的质心数量 M
            
        Returns:
            centroid_indices: [batch_size, num_centroids] 选中的索引
        """
        # 计算每个 Token 的度（与所有其他 Token 的相似度之和）
        degree = similarity_matrix.sum(dim=-1)  # [B, N]
        
        # 选择度最大的 M 个 Token
        _, centroid_indices = torch.topk(degree, k=num_centroids, dim=-1)  # [B, M]
        
        return centroid_indices
    
    def cross_attention_aggregation(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        value_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        通过多头交叉注意力聚合信息
        
        Args:
            query_features: [batch_size, num_centroids, hidden_size] - 质心作为 Query
            key_features: [batch_size, num_tokens, hidden_size] - 所有 Token 作为 Key
            value_features: [batch_size, num_tokens, hidden_size] - 所有 Token 作为 Value
            
        Returns:
            aggregated_features: [batch_size, num_centroids, hidden_size]
        """
        batch_size, num_centroids, _ = query_features.shape
        num_tokens = key_features.shape[1]
        
        # 投影到 Q, K, V
        Q = self.q_proj(query_features)  # [B, M, D]
        K = self.k_proj(key_features)    # [B, N, D]
        V = self.v_proj(value_features)  # [B, N, D]
        
        # 重塑为多头形式
        Q = Q.view(batch_size, num_centroids, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, M, D_h]
        K = K.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)     # [B, H, N, D_h]
        V = V.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)     # [B, H, N, D_h]
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, M, N]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 加权聚合
        aggregated = torch.matmul(attention_weights, V)  # [B, H, M, D_h]
        
        # 合并多头
        aggregated = aggregated.transpose(1, 2).contiguous().view(
            batch_size, num_centroids, self.hidden_size
        )  # [B, M, D]
        
        # 输出投影
        output = self.o_proj(aggregated)  # [B, M, D]
        
        return output
    
    def forward(
        self,
        features: torch.Tensor,
        num_centroids: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Token Condensation 主流程
        
        Args:
            features: [batch_size, num_tokens, hidden_size] 输入特征
            num_centroids: 要保留的质心数量 M
            
        Returns:
            condensed_features: [batch_size, num_centroids, hidden_size] 浓缩后的特征
            centroid_indices: [batch_size, num_centroids] 选中的索引
        """
        batch_size, num_tokens, hidden_size = features.shape
        
        # 如果要求的质心数量 >= 输入 Token 数量，直接返回原始特征
        if num_centroids >= num_tokens:
            return features, torch.arange(num_tokens, device=features.device).unsqueeze(0).expand(batch_size, -1)
        
        # Step 1: 计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix(features)  # [B, N, N]
        
        # Step 2: 选择代表性 Token（质心）
        if self.selection_strategy == 'fps':
            centroid_indices = self.select_centroids_fps(
                features, similarity_matrix, num_centroids
            )  # [B, M]
        elif self.selection_strategy == 'connectivity':
            centroid_indices = self.select_centroids_connectivity(
                similarity_matrix, num_centroids
            )  # [B, M]
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        
        # 提取质心特征
        # 使用 gather 操作：需要扩展 indices 到 [B, M, D]
        centroid_indices_expanded = centroid_indices.unsqueeze(-1).expand(-1, -1, hidden_size)  # [B, M, D]
        centroid_features = torch.gather(features, dim=1, index=centroid_indices_expanded)  # [B, M, D]
        
        # Step 3: 通过交叉注意力聚合全局信息
        condensed_features = self.cross_attention_aggregation(
            query_features=centroid_features,
            key_features=features,
            value_features=features,
        )  # [B, M, D]
        
        return condensed_features, centroid_indices


class TokenCondensationIntegrated(nn.Module):
    """
    集成到 LLaVA 模型中的 Token Condensation 包装器
    用于替代 HiMAP 中的硬剪枝逻辑
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        selection_strategy: str = 'connectivity',  # 默认使用连通性策略，速度更快
    ):
        super().__init__()
        self.condensation_module = TokenCondensation(
            hidden_size=hidden_size,
            num_heads=num_heads,
            selection_strategy=selection_strategy,
        )
        
    def condense_vision_tokens(
        self,
        hidden_states: torch.Tensor,
        vision_token_start: int,
        vision_token_end: int,
        target_num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对视觉 Token 区域进行浓缩
        
        Args:
            hidden_states: [batch_size, seq_length, hidden_size] 完整序列
            vision_token_start: 视觉 Token 起始索引
            vision_token_end: 视觉 Token 结束索引
            target_num_tokens: 目标保留的视觉 Token 数量
            
        Returns:
            new_hidden_states: [batch_size, new_seq_length, hidden_size] 浓缩后的序列
            kept_indices: [batch_size, new_seq_length] 保留的全局索引
        """
        batch_size, seq_length, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # 分割序列：前缀（系统提示） + 视觉 Token + 后缀（文本 Token）
        prefix = hidden_states[:, :vision_token_start, :]  # [B, prefix_len, D]
        vision_tokens = hidden_states[:, vision_token_start:vision_token_end, :]  # [B, vision_len, D]
        suffix = hidden_states[:, vision_token_end:, :]  # [B, suffix_len, D]
        
        # 对视觉 Token 进行浓缩
        condensed_vision, vision_indices = self.condensation_module(
            vision_tokens, num_centroids=target_num_tokens
        )  # [B, M, D], [B, M]
        
        # 重新拼接序列
        new_hidden_states = torch.cat([prefix, condensed_vision, suffix], dim=1)  # [B, new_seq_len, D]
        
        # 构建完整的索引映射
        prefix_len = prefix.shape[1]
        suffix_len = suffix.shape[1]
        
        prefix_indices = torch.arange(prefix_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, prefix_len]
        vision_global_indices = vision_indices + vision_token_start  # 转换为全局索引
        suffix_indices = torch.arange(
            vision_token_end, seq_length, device=device
        ).unsqueeze(0).expand(batch_size, -1)  # [B, suffix_len]
        
        kept_indices = torch.cat([prefix_indices, vision_global_indices, suffix_indices], dim=1)  # [B, new_seq_len]
        
        return new_hidden_states, kept_indices
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        vision_token_start: int,
        vision_token_end: int,
        target_num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """便捷的 forward 接口"""
        return self.condense_vision_tokens(
            hidden_states, vision_token_start, vision_token_end, target_num_tokens
        )
