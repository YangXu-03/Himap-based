import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaModel, _prepare_4d_causal_attention_mask, Cache, DynamicCache
from transformers.models.llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

logger = logging.get_logger(__name__)

# 为 AdaptInfer 实现 LlamaModel，确保 B=1 并实现动态剪枝
class AdaptInfer_LlamaModel(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        
        # 假设配置已在 LlamaConfig 中设置
        self.adaptinfer_sys_length = config.adaptinfer_sys_length # 通常为 1
        self.adaptinfer_img_length = config.adaptinfer_img_length # 例如 576
        self.adaptinfer_pruning_layers = set(config.adaptinfer_pruning_layers) if hasattr(config, 'adaptinfer_pruning_layers') else set()
        
        # 用于动态剪枝的阈值或控制参数 (例如：基于分数阈值或动态比例)
        self.adaptinfer_threshold = getattr(config, 'adaptinfer_threshold', 0.1) # 动态阈值，用于决定保留数量
        self.adaptinfer_min_keep = getattr(config, 'adaptinfer_min_keep', 32) # 最少保留的视觉 Token 数量
        self.use_adaptinfer = config.use_adaptinfer

    def reset_adaptinfer(self):
        self.adaptinfer_sys_length = self.config.adaptinfer_sys_length
        self.adaptinfer_img_length = self.config.adaptinfer_img_length
        self.adaptinfer_pruning_layers = self.config.adaptinfer_pruning_layers
        # 改成adaptinfer_threshold和adaptinfer_min_keep的重置
        # self.adaptinfer_keep_ratio = self.config.adaptinfer_keep_ratio
        self.use_adaptinfer = self.config.use_adaptinfer
    
    
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
    # --- 1. 标准 Llama 模型输入处理 (省略细节) ---
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
        
        # 检查 Batch Size (强制 B=1 进行动态剪枝)
        if self.use_adaptinfer and batch_size > 1:
             raise ValueError(
                 "AdaptInfer with dynamic pruning must use a batch size of 1 to allow for variable sequence lengths."
             )

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

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Token 索引定义 (假定图像Token紧跟系统Prompt，文本Token紧随图像Token)
        SYS_LENGTH = self.adaptinfer_sys_length
        IMG_LENGTH = self.adaptinfer_img_length
        VISUAL_START_IDX = SYS_LENGTH
        VISUAL_END_IDX = SYS_LENGTH + IMG_LENGTH
        TEXT_START_IDX = VISUAL_END_IDX
        
        # --- 2. 核心迭代和剪枝逻辑 ---
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            # 梯度检查点是一种以计算时间换取内存的技术，通过不存储部分中间激活值（仅在反向传播时重新计算）
            # 来减少训练时的内存占用，常用于大模型或长序列场景。
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
                # --- ADAPTINFER 剪枝逻辑 ---
                if self.use_adaptinfer and idx in self.adaptinfer_pruning_layers:
                    
                    # 确保当前序列长度大于视觉 Token 结束位置，并且 Attention Map 存在
                    if seq_length > VISUAL_END_IDX and output_attentions:
                        
                        # layer_outputs[1] 是 Attention Weights [B, H, L, L]
                        attention_weights = layer_outputs[1] 
                        
                        # (由于 Batch Size 强制为 1，可以直接使用 [0] 索引)
                        
                        # 1. 计算文本Token重要性 (Text-to-Text Attention)
                        # 文本 Query 对文本 Key 的注意力 (Query范围: [TEXT_START_IDX:], Key范围: [TEXT_START_IDX:])
                        # 注意：文本 Query 范围是当前隐藏状态的长度
                        current_seq_length = hidden_states.shape[1]
                        t2t_attn = attention_weights[0, :, TEXT_START_IDX:current_seq_length, TEXT_START_IDX:current_seq_length]
                        
                        # 对 Head 求平均 [L_text, L_text]
                        t2t_attn_avg = torch.mean(t2t_attn, dim=0) 
                        
                        # 文本Token重要性 (Soft Prior) 
                        # 论文中是 sum(A_t2t)
                        text_importance = torch.sum(t2t_attn_avg, dim=-1) # [L_text]
                        text_importance_soft = F.softmax(text_importance, dim=-1) # [L_text]
                        
                        # 2. 计算视觉Token分数 (Text-to-Vision 加权)
                        # 文本 Query 对视觉 Key 的注意力 (Query范围: [TEXT_START_IDX:], Key范围: [VISUAL_START_IDX:VISUAL_END_IDX])
                        # 注意：由于剪枝可能发生在多个层，VISUAL_END_IDX 必须根据初始长度来计算
                        # L_vis 是初始图像 Token 数量，但如果前面层已剪枝，实际 L_vis' < L_vis
                        # 这里必须使用实际的索引！
                        
                        # 为了简化，假设所有非文本Token都是视觉/系统Token，并且它们在前面未被移动或删除
                        # 否则，我们需要维护一个全局的索引映射。
                        
                        # 假设当前视觉/系统Token位于 [0:TEXT_START_IDX]
                        visual_end_idx_current = TEXT_START_IDX # 当前序列中视觉/系统 Token 的结束索引
                        
                        # A_t2v [H, L_text, L_vis']
                        t2v_attn = attention_weights[0, :, TEXT_START_IDX:current_seq_length, VISUAL_START_IDX:visual_end_idx_current]
                        t2v_attn_avg = torch.mean(t2v_attn, dim=0) # [L_text, L_vis']

                        # 使用文本重要性加权 t2v 注意力 (s = ω^T · A_t2v)
                        # [L_text] x [L_text, L_vis'] -> [L_vis']
                        visual_scores = torch.matmul(
                            text_importance_soft.unsqueeze(0), 
                            t2v_attn_avg
                        ).squeeze(0) # [L_vis'] (当前视觉 Token 的分数)
                        
                        # 3. 动态阈值或Top-K选择 (实现数量动态化)
                        
                        # 动态数量剪枝策略：基于阈值 (假设分数越高越重要)
                        # 找到分数大于阈值的 Token
                        important_indices_relative = torch.where(visual_scores > self.adaptinfer_threshold)[0]
                        
                        # 确保最少保留数量 (防止过度剪枝)
                        if important_indices_relative.shape[0] < self.adaptinfer_min_keep:
                            # 如果少于最少保留数，则保留 Top-K 个
                            K = self.adaptinfer_min_keep
                            _, topk_indices = visual_scores.topk(K, dim=0)
                            important_indices_relative = topk_indices
                            
                        # 转换为全局索引
                        global_important_indices = important_indices_relative + VISUAL_START_IDX

                        # 4. 构建完整的保留索引 (系统Prompt + 选中的视觉Token + 文本Token)
                        # 我们保留所有系统Prompt (0:SYS_LENGTH)
                        # 我们保留所有文本Token (TEXT_START_IDX:current_seq_length)
                        
                        text_indices = torch.arange(TEXT_START_IDX, current_seq_length, device=device)
                        system_indices = torch.arange(SYS_LENGTH, device=device)
                        
                        keep_indices = torch.cat(
                            (
                                system_indices,
                                global_important_indices, 
                                text_indices
                            )
                        ).unique().sort().values # 使用 unique().sort().values 来保证索引不重复且有序
                        
                        # 5. 更新状态 (Hidden States, Position IDs)
                        
                        # 使用当前层输出的 H 状态进行切片 (layer_outputs[0] 是 [B, L, D])
                        # 由于 B=1，切片后 shape 仍然是 [1, L', D]
                        hidden_states = layer_outputs[0][:, keep_indices, :] 
                        
                        # Position IDs 必须保持全局索引，但要匹配新的 B=1 形状
                        # position_ids 应该在进入循环前被初始化为 [1, L]
                        position_ids = position_ids[:, keep_indices] 
                        
                        # 6. 重新生成 Attention Mask (基于新的短序列长度)
                        pruned_seq_length = hidden_states.shape[1]
                        
                        # 重新生成因果 Attention Mask (假设没有 past_key_values, 即处于 prefill 阶段)
                        # 注意：如果处于 decoding 阶段 (seq_length=1)，则不应进行剪枝。
                        new_attention_mask = self._prepare_decoder_attention_mask(
                            None, (batch_size, img_seq_length), inputs_embeds, 0
                        ) 
                        """
                        attention_mask = _prepare_4d_causal_attention_mask(
                            None, (batch_size, pruned_seq_length), hidden_states, 0
                        )
                        """

                else: 
                    new_attention_mask = attention_mask
                    
                # print(idx, hidden_states.shape, new_attention_mask.shape, position_ids.shape)
                # 使用剪枝后/未剪枝的隐藏状态、注意力掩码和位置编码，执行当前层的前向传播
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=new_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                    
            # 更新 hidden_states 和 cache
            hidden_states = layer_outputs[0]

            if use_cache:
                # 检查 layer_outputs[2] 是否存在
                if len(layer_outputs) > 2:
                    next_decoder_cache = layer_outputs[2]
                else: # 否则在 layer_outputs[1]
                    next_decoder_cache = layer_outputs[1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        # --- 3. 标准 Llama 模型输出处理 (省略细节) ---
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        
        # 最后的返回部分也需要与 LlamaModel 兼容
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

# 辅助函数，用于适配 _prepare_4d_causal_attention_mask
# 注意：在实际项目中，您需要确保 AdaptInfer_LlamaModel 继承自 LlamaModel 
# 并且能够访问到 LlamaModel 的私有方法，或者将 LlamaModel 的逻辑复制过来。
def _prepare_4d_causal_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # 这里只是一个占位符，模拟 Llama 的 Attention Mask 准备逻辑
    # 实际应用中，应从 transformers.models.llama.modeling_llama 导入
    
    # 获取序列长度
    seq_length = input_shape[-1]
    
    # 创建因果 mask (L x L)
    mask = torch.full((seq_length, seq_length), torch.finfo(inputs_embeds.dtype).min, device=inputs_embeds.device)
    mask_cond = torch.arange(mask.size(-1), device=inputs_embeds.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(inputs_embeds.dtype)

    # 扩展 mask 以适应 Batch 和 Head (B, 1, L, L)
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    # 合并用户提供的 attention_mask (如果存在)
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        # 将用户的 mask (0/1) 转换为 (0 / min_value)
        attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min
        mask = mask + attention_mask

    return mask
