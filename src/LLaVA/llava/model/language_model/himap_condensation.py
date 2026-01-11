"""
HiMAP with Token Condensation
替代硬剪枝，使用 Token 浓缩机制保留更多上下文信息
"""

import torch
from typing import List, Optional, Tuple, Union
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast

from .himap_modeling_llama import LlamaModel
from .himap_configuration_llama import LlamaConfig
from .token_condensation import TokenCondensationIntegrated

logger = logging.get_logger(__name__)


class HimapCondensation_LlamaModel(LlamaModel):
    """
    HiMAP 模型 + Token Condensation
    在指定层对视觉 Token 进行浓缩而非硬剪枝
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.config = config
        
        # HiMAP 参数
        self.hmap_v_sys_length = config.hmap_v_sys_length
        self.hmap_v_img_length = config.hmap_v_img_length
        self.hmap_v_attn_txt_layer = config.hmap_v_attn_txt_layer
        self.hmap_v_attn_txt_rank = config.hmap_v_attn_txt_rank
        self.hmap_v_attn_img_layer = config.hmap_v_attn_img_layer
        self.hmap_v_attn_img_rank = config.hmap_v_attn_img_rank
        self.use_hmap_v = config.use_hmap_v
        self.cut_off_layer = config.cut_off_layer
        
        # Token Condensation 参数
        self.use_token_condensation = getattr(config, 'use_token_condensation', False)
        self.condensation_strategy = getattr(config, 'condensation_strategy', 'connectivity')
        self.condensation_num_heads = getattr(config, 'condensation_num_heads', 8)
        
        # 初始化 Token Condensation 模块
        if self.use_token_condensation:
            self.token_condensation = TokenCondensationIntegrated(
                hidden_size=config.hidden_size,
                num_heads=self.condensation_num_heads,
                selection_strategy=self.condensation_strategy,
            )
        else:
            self.token_condensation = None

    def reset_hmapv(self):
        """重置 HiMAP 参数"""
        self.hmap_v_sys_length = self.config.hmap_v_sys_length
        self.hmap_v_img_length = self.config.hmap_v_img_length
        self.hmap_v_attn_txt_layer = self.config.hmap_v_attn_txt_layer
        self.hmap_v_attn_txt_rank = self.config.hmap_v_attn_txt_rank
        self.hmap_v_attn_img_layer = self.config.hmap_v_attn_img_layer
        self.hmap_v_attn_img_rank = self.config.hmap_v_attn_img_rank
        self.use_hmap_v = self.config.use_hmap_v
        self.cut_off_layer = self.config.cut_off_layer
        self.use_token_condensation = getattr(self.config, 'use_token_condensation', False)

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

        # 输入处理
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

        # 初始化输出
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # 获取配置参数
        USE_HMAP_V = self.use_hmap_v
        SYS_LENGTH = self.hmap_v_sys_length
        IMG_LENGTH = self.hmap_v_img_length
        TXT_LAYER = self.hmap_v_attn_txt_layer
        TXT_ATTN_RANK = self.hmap_v_attn_txt_rank
        IMG_LAYER = self.hmap_v_attn_img_layer
        IMG_ATTN_RANK = self.hmap_v_attn_img_rank
        CUT_OFF_LAYER = self.cut_off_layer
        USE_CONDENSATION = self.use_token_condensation
        
        device = hidden_states.device
        num_text_tokens = seq_length - SYS_LENGTH - IMG_LENGTH

        # 遍历所有层
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing 逻辑
                def create_custom_forward(module):
                    def custom_forward(*inputs):
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
                # === Token 处理逻辑：选择使用 Condensation 还是 Hard Pruning ===
                if USE_HMAP_V:
                    # 参数验证
                    if TXT_LAYER:
                        assert TXT_LAYER > 0, "txt attn layer should be larger than 0"
                    if IMG_LAYER:
                        assert IMG_LAYER > TXT_LAYER, "img attn layer should be larger than txt attn layer"
                    if TXT_ATTN_RANK and IMG_ATTN_RANK:
                        assert TXT_ATTN_RANK >= IMG_ATTN_RANK, "txt attn rank should be larger than img attn rank"

                    # ========== 第一阶段：TXT_LAYER - 基于 img2txt 注意力的处理 ==========
                    if idx < TXT_LAYER:
                        new_attention_mask = attention_mask
                        current_keep_indexs = None

                    elif idx == TXT_LAYER:
                        # 计算 img2txt 注意力分数
                        txt_layer_attn = layer_outputs[1]
                        if isinstance(txt_layer_attn, tuple):
                            txt_layer_attn = txt_layer_attn[0]
                        txt_layer_attn_avg = torch.mean(txt_layer_attn, dim=1)[0]
                        
                        img2txt_attn = torch.sum(
                            txt_layer_attn_avg[SYS_LENGTH+IMG_LENGTH:, SYS_LENGTH:SYS_LENGTH+IMG_LENGTH], dim=0
                        )
                        
                        # 使用 Token Condensation 或传统 Top-K 选择
                        if USE_CONDENSATION and self.token_condensation is not None:
                            # Token Condensation: 浓缩视觉 Token
                            hidden_states, kept_indices = self.token_condensation(
                                hidden_states=hidden_states,
                                vision_token_start=SYS_LENGTH,
                                vision_token_end=SYS_LENGTH + IMG_LENGTH,
                                target_num_tokens=TXT_ATTN_RANK,
                            )
                            txt_seq_length = hidden_states.shape[1]
                            position_ids = kept_indices.unsqueeze(0) if kept_indices.dim() == 1 else kept_indices
                            
                        else:
                            # 传统 Hard Pruning: Top-K 选择
                            img2txt_attn_len = img2txt_attn.shape[0]
                            topk_rank = min(TXT_ATTN_RANK, img2txt_attn_len)
                            img2txt_attn_topk_index = img2txt_attn.topk(topk_rank).indices + SYS_LENGTH
                            
                            txt_keep_indexs = torch.cat(
                                (
                                    torch.arange(SYS_LENGTH, device=device),
                                    img2txt_attn_topk_index,
                                    torch.arange(SYS_LENGTH+IMG_LENGTH, seq_length, device=device)
                                )
                            )
                            txt_keep_indexs = txt_keep_indexs.sort().values
                            current_keep_indexs = txt_keep_indexs
                            
                            txt_seq_length = txt_keep_indexs.shape[0]
                            hidden_states = hidden_states[:, txt_keep_indexs, :]
                            position_ids = txt_keep_indexs.unsqueeze(0)
                        
                        new_attention_mask = self._prepare_decoder_attention_mask(
                            None, (batch_size, txt_seq_length), inputs_embeds, 0
                        )

                    # ========== 第二阶段：IMG_LAYER - 基于 img2img 注意力的处理 ==========
                    elif idx == IMG_LAYER:
                        # 计算 img2img 注意力分数
                        img_layer_attn = layer_outputs[1]
                        if isinstance(img_layer_attn, tuple):
                            img_layer_attn = img_layer_attn[0]
                        img_layer_attn_avg = torch.mean(img_layer_attn, dim=1)[0]

                        current_seq_length = hidden_states.shape[1]
                        current_img_len = current_seq_length - SYS_LENGTH - num_text_tokens
                        if current_img_len < 0:
                            current_img_len = 0

                        img2img_attn = torch.sum(
                            img_layer_attn_avg[
                                SYS_LENGTH:SYS_LENGTH + current_img_len,
                                SYS_LENGTH:SYS_LENGTH + current_img_len,
                            ],
                            dim=0,
                        )
                        
                        # 使用 Token Condensation 或传统 Top-K 选择
                        if USE_CONDENSATION and self.token_condensation is not None and current_img_len > 0:
                            # Token Condensation: 进一步浓缩视觉 Token
                            vision_end_idx = SYS_LENGTH + current_img_len
                            hidden_states, kept_indices = self.token_condensation(
                                hidden_states=hidden_states,
                                vision_token_start=SYS_LENGTH,
                                vision_token_end=vision_end_idx,
                                target_num_tokens=IMG_ATTN_RANK,
                            )
                            img_seq_length = hidden_states.shape[1]
                            position_ids = kept_indices.unsqueeze(0) if kept_indices.dim() == 1 else kept_indices
                            
                        else:
                            # 传统 Hard Pruning
                            img2img_attn_len = img2img_attn.shape[0]
                            topk_rank = min(IMG_ATTN_RANK, img2img_attn_len)
                            if topk_rank > 0:
                                img2img_attn_topk_index = img2img_attn.topk(topk_rank).indices + SYS_LENGTH
                            else:
                                img2img_attn_topk_index = torch.tensor([], dtype=torch.long, device=device)

                            rest_start = SYS_LENGTH + current_img_len
                            rest_indices = (
                                torch.arange(rest_start, txt_seq_length, device=device)
                                if (rest_start < txt_seq_length)
                                else torch.tensor([], dtype=torch.long, device=device)
                            )
                            img_keep_indexs = torch.cat(
                                (
                                    torch.arange(SYS_LENGTH, device=device),
                                    img2img_attn_topk_index,
                                    rest_indices
                                )
                            )
                            img_keep_indexs = img_keep_indexs.sort().values
                            current_keep_indexs = img_keep_indexs
                            
                            img_seq_length = img_keep_indexs.shape[0]
                            hidden_states = hidden_states[:, img_keep_indexs, :]
                            position_ids = txt_keep_indexs[img_keep_indexs].unsqueeze(0)
                        
                        new_attention_mask = self._prepare_decoder_attention_mask(
                            None, (batch_size, img_seq_length), inputs_embeds, 0
                        )

                    # ========== 第三阶段：CUT_OFF_LAYER - 完全移除视觉 Token ==========
                    elif (CUT_OFF_LAYER > 0) and (idx == CUT_OFF_LAYER):
                        current_seq_length = hidden_states.shape[1]
                        num_current_img_tokens = current_seq_length - SYS_LENGTH - num_text_tokens

                        if num_current_img_tokens > 0:
                            keep_indices = torch.cat(
                                (
                                    torch.arange(SYS_LENGTH, device=device),
                                    torch.arange(SYS_LENGTH + num_current_img_tokens, current_seq_length, device=device)
                                )
                            )
                            target_device = hidden_states.device
                            keep_indices = keep_indices.to(target_device)
                            
                            new_seq_length = keep_indices.shape[0]
                            hidden_states = hidden_states[:, keep_indices, :]
                            
                            current_position_ids = position_ids.squeeze(0)
                            current_position_ids = current_position_ids.to(target_device)
                            position_ids = current_position_ids[keep_indices].unsqueeze(0)
                            
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, new_seq_length), inputs_embeds, 0
                            )
                        else:
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, current_seq_length), inputs_embeds, 0
                            )

                    # ========== 其他层：保持当前序列长度 ==========
                    else:
                        current_pruned_seq_length = hidden_states.shape[1]
                        if current_pruned_seq_length == seq_length_with_past:
                            new_attention_mask = attention_mask
                        else:
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, current_pruned_seq_length), inputs_embeds, 0
                            )

                else:
                    new_attention_mask = attention_mask

                # 执行 Transformer 层
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=new_attention_mask,
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

        # 最终归一化
        hidden_states = self.norm(hidden_states)

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
