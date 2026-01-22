import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union
from transformers.utils import logging
from .himap_modeling_llama import LlamaModel
from .himap_configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

logger = logging.get_logger(__name__)


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
    
    return torch.tensor(selected_indices, device=tokens.device, dtype=torch.long)


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
    
    # 迭代优化（简化版，只做3轮）
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
        
        centers = torch.tensor(new_centers, device=tokens.device, dtype=torch.long)
    
    return centers


def cross_attention_aggregation(query_tokens, key_value_tokens, temperature=0.1):
    """
    使用Cross-Attention进行信息聚合
    Args:
        query_tokens: (M, D) tensor, 质心tokens
        key_value_tokens: (N, D) tensor, 原始所有tokens
        temperature: softmax temperature for attention
    Returns:
        aggregated_tokens: (M, D) tensor, 聚合后的tokens
    """
    M, D = query_tokens.shape
    N = key_value_tokens.shape[0]
    
    # 计算query和key之间的相似度
    query_norm = F.normalize(query_tokens, p=2, dim=1)
    key_norm = F.normalize(key_value_tokens, p=2, dim=1)
    
    # attention scores: (M, N)
    attention_scores = torch.mm(query_norm, key_norm.t())
    
    # 使用softmax得到attention权重
    attention_weights = F.softmax(attention_scores / temperature, dim=1)
    
    # 加权聚合value (这里key=value)
    aggregated_tokens = torch.mm(attention_weights, key_value_tokens)
    
    return aggregated_tokens


class CustomTokenSelection_LlamaModel(LlamaModel):
    """
    自定义Token选择模型：使用FPS/ToMe + Cross-Attention聚合
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.config = config
        
        # Custom token selection hyperparameters
        # Use getattr with defaults for robustness when config doesn't have these attributes
        self.use_custom_selection = getattr(config, 'use_custom_selection', False)
        self.custom_sys_length = getattr(config, 'custom_sys_length', 35)
        self.custom_image_token_length = getattr(config, 'custom_image_token_length', 576)
        self.custom_kept_tokens = getattr(config, 'custom_kept_tokens', 8)
        self.custom_agg_layer = getattr(config, 'custom_agg_layer', 12)
        self.custom_selection_method = getattr(config, 'custom_selection_method', 'fps')
        self.custom_temperature = getattr(config, 'custom_temperature', 0.1)
        
        # 保存最近一次生成的kept indices（供外部读取）
        self.last_gen_kept_indices = None
        self.last_selected_positions = None

    def reset_custom_selection(self):
        """重置自定义选择参数"""
        self.use_custom_selection = getattr(self.config, 'use_custom_selection', False)
        self.custom_sys_length = getattr(self.config, 'custom_sys_length', 35)
        self.custom_image_token_length = getattr(self.config, 'custom_image_token_length', 576)
        self.custom_kept_tokens = getattr(self.config, 'custom_kept_tokens', 8)
        self.custom_agg_layer = getattr(self.config, 'custom_agg_layer', 12)
        self.custom_selection_method = getattr(self.config, 'custom_selection_method', 'fps')
        self.custom_temperature = getattr(self.config, 'custom_temperature', 0.1)
        
        # reset dynamic outputs
        self.last_gen_kept_indices = None
        self.last_selected_positions = None

    def _apply_custom_token_selection(self, hidden_states, batch_idx=0):
        """
        应用自定义token选择和聚合
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            batch_idx: 处理哪个batch
        Returns:
            modified hidden_states, attention_mask (updated)
        """
        def _nz(val, default=0):
            """Gracefully handle None hyperparameters"""
            return val if val is not None else default
        
        SYS_LENGTH = _nz(self.custom_sys_length)
        IMAGE_TOKEN_LENGTH = _nz(self.custom_image_token_length)
        KEPT_TOKENS = _nz(self.custom_kept_tokens)
        METHOD = _nz(self.custom_selection_method, 'fps')
        TEMPERATURE = _nz(self.custom_temperature, 0.1)
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 提取图像token区域
        img_start = SYS_LENGTH
        img_end = SYS_LENGTH + IMAGE_TOKEN_LENGTH
        
        if seq_len < img_end or KEPT_TOKENS >= IMAGE_TOKEN_LENGTH:
            return hidden_states
        
        # 处理指定的batch
        image_tokens = hidden_states[batch_idx, img_start:img_end, :]  # (N, D)
        
        # 选择质心tokens
        if METHOD == 'fps':
            selected_indices = farthest_point_sampling(image_tokens, KEPT_TOKENS)
        elif METHOD == 'tome':
            selected_indices = tome_clustering_selection(image_tokens, KEPT_TOKENS)
        else:
            logger.warning(f"Unknown selection method: {METHOD}, using FPS as fallback")
            selected_indices = farthest_point_sampling(image_tokens, KEPT_TOKENS)
        
        # 提取选中的质心tokens
        centroid_tokens = image_tokens[selected_indices]  # (M, D)
        
        # 使用Cross-Attention聚合信息
        aggregated_tokens = cross_attention_aggregation(
            centroid_tokens, image_tokens, temperature=TEMPERATURE
        )  # (M, D)
        
        # 保存选择的索引（相对于图像token区域）
        try:
            self.last_selected_positions = selected_indices.cpu().numpy()
            # 全局索引
            global_indices = selected_indices + SYS_LENGTH
            self.last_gen_kept_indices = global_indices.cpu().numpy()
        except Exception:
            self.last_selected_positions = None
            self.last_gen_kept_indices = None
        
        # 替换策略：将前M个位置替换为聚合后的tokens，其余位置置零
        # 这样保持序列长度不变，避免position_ids不匹配
        new_img_tokens = torch.zeros(
            IMAGE_TOKEN_LENGTH, hidden_dim,
            dtype=image_tokens.dtype,
            device=image_tokens.device
        )  # (N, D)
        # 确保 aggregated_tokens 的 dtype 和 device 与 new_img_tokens 一致
        aggregated_tokens = aggregated_tokens.to(dtype=new_img_tokens.dtype, device=new_img_tokens.device)
        new_img_tokens[:KEPT_TOKENS] = aggregated_tokens
        
        # 更新hidden_states中的图像token部分
        hidden_states[batch_idx, img_start:img_end, :] = new_img_tokens
        
        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Gracefully handle None hyperparameters (treat as 0 / disabled)
        def _nz(val, default=0):
            return val if val is not None else default

        USE_CUSTOM = bool(self.use_custom_selection)
        AGG_LAYER = _nz(self.custom_agg_layer)
        custom_selection_applied = False

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)
                    return custom_forward
                    
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                # CUSTOM TOKEN SELECTION START --------------------------------------------------
                # 只有在非baseline模式（USE_CUSTOM=True）时才执行token选择和聚合
                if USE_CUSTOM and idx == AGG_LAYER and not custom_selection_applied:
                    # 在进入该层之前，对hidden_states进行处理
                    hidden_states = self._apply_custom_token_selection(hidden_states)
                    custom_selection_applied = True
                # CUSTOM TOKEN SELECTION END --------------------------------------------------

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
