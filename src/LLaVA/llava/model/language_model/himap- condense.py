import torch
from typing import Tuple
from transformers.utils import logging
from .himap_modeling_llama import LlamaModel
from .himap_configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union

logger = logging.get_logger(__name__)

class Himap_LlamaModel(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.config = config
        # hmapv hyperparameter
        self.hmap_v_sys_length = config.hmap_v_sys_length
        self.hmap_v_img_length = config.hmap_v_img_length
        self.hmap_v_attn_txt_layer = config.hmap_v_attn_txt_layer
        self.hmap_v_attn_txt_rank = config.hmap_v_attn_txt_rank
        self.hmap_v_attn_img_layer = config.hmap_v_attn_img_layer
        self.hmap_v_attn_img_rank = config.hmap_v_attn_img_rank
        self.use_hmap_v = config.use_hmap_v 
        self.cut_off_layer = config.cut_off_layer
        
        # Token Condensation parameters
        self.use_token_condensation = getattr(config, 'use_token_condensation', False)
        self.condensation_strategy = getattr(config, 'condensation_strategy', 'connectivity')
        self.condensation_num_heads = getattr(config, 'condensation_num_heads', 8)
    def reset_hmapv(self):
        self.hmap_v_sys_length = self.config.hmap_v_sys_length
        self.hmap_v_img_length = self.config.hmap_v_img_length
        self.hmap_v_attn_txt_layer = self.config.hmap_v_attn_txt_layer
        self.hmap_v_attn_txt_rank = self.config.hmap_v_attn_txt_rank
        self.hmap_v_attn_img_layer = self.config.hmap_v_attn_img_layer
        self.hmap_v_attn_img_rank = self.config.hmap_v_attn_img_rank
        self.use_hmap_v = self.config.use_hmap_v  
        self.cut_off_layer = self.config.cut_off_layer
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

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Track current keep indices for cut-off layer pruning
        current_keep_indexs = None
        txt_keep_indexs = None
        img_keep_indexs = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Default None to 0 to avoid TypeError when HiMAP params are unset
            sys_len = self.hmap_v_sys_length if self.hmap_v_sys_length is not None else 0
            img_len = self.hmap_v_img_length if self.hmap_v_img_length is not None else 0
            num_text_tokens = seq_length_with_past - sys_len - img_len

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
                # Gracefully handle None hyperparameters (treat as 0 / disabled)
                def _nz(val, default=0):
                    return val if val is not None else default

                USE_HMAP_V = bool(self.use_hmap_v)
                SYS_LENGTH = _nz(self.hmap_v_sys_length)
                IMG_LENGTH = _nz(self.hmap_v_img_length)
                TXT_LAYER = _nz(self.hmap_v_attn_txt_layer)
                TXT_ATTN_RANK = _nz(self.hmap_v_attn_txt_rank)
                IMG_LAYER = _nz(self.hmap_v_attn_img_layer)
                IMG_ATTN_RANK = _nz(self.hmap_v_attn_img_rank)
                CUT_OFF_LAYER = _nz(self.cut_off_layer)

                if TXT_LAYER:
                    assert TXT_LAYER > 0, "txt attn layer should be larger than 0"
                if IMG_LAYER:
                    assert IMG_LAYER > TXT_LAYER, "img attn layer should be larger than txt attn layer"
                if TXT_ATTN_RANK and IMG_ATTN_RANK:
                    assert TXT_ATTN_RANK >= IMG_ATTN_RANK, "txt attn rank should be larger than img attn rank"

                # IMAGE TOKEN PRUNING BEGIN --HiMAP TECHNIQUE 
                # Token Condensation 在 decoder_layer 之前执行
                use_condensation = getattr(self, 'use_token_condensation', False)
                
                if USE_HMAP_V and use_condensation:
                    # ===== Token Condensation Mode: 在 decoder_layer 调用前处理 =====
                    if idx == TXT_LAYER:
                        # Step 1: 切分 Tensor
                        sys_tokens = hidden_states[:, :SYS_LENGTH, :]
                        visual_tokens = hidden_states[:, SYS_LENGTH:SYS_LENGTH+IMG_LENGTH, :]
                        text_tokens = hidden_states[:, SYS_LENGTH+IMG_LENGTH:, :]
                        
                        # Step 2: 执行 Token Condensation
                        condensed_visual = self._condense_visual_tokens(
                            visual_tokens, 
                            target_num=TXT_ATTN_RANK,
                            strategy=getattr(self, 'condensation_strategy', 'connectivity'),
                            num_heads=getattr(self, 'condensation_num_heads', 8)
                        )
                        
                        # Step 3: 拼接
                        hidden_states = torch.cat([sys_tokens, condensed_visual, text_tokens], dim=1)
                        
                        # Step 4: 修正元数据
                        txt_seq_length = hidden_states.shape[1]
                        position_ids = torch.arange(txt_seq_length, device=device, dtype=torch.long).unsqueeze(0)
                        new_attention_mask = self._prepare_decoder_attention_mask(
                            None, (batch_size, txt_seq_length), inputs_embeds, 0
                        )
                    
                    elif idx == IMG_LAYER:
                        # 二次浓缩
                        current_seq_length = hidden_states.shape[1]
                        num_text_tokens = seq_length_with_past - SYS_LENGTH - IMG_LENGTH
                        current_visual_len = current_seq_length - SYS_LENGTH - num_text_tokens
                        
                        if current_visual_len > IMG_ATTN_RANK:
                            sys_tokens = hidden_states[:, :SYS_LENGTH, :]
                            visual_tokens = hidden_states[:, SYS_LENGTH:SYS_LENGTH+current_visual_len, :]
                            text_tokens = hidden_states[:, SYS_LENGTH+current_visual_len:, :]
                            
                            condensed_visual = self._condense_visual_tokens(
                                visual_tokens,
                                target_num=IMG_ATTN_RANK,
                                strategy=getattr(self, 'condensation_strategy', 'connectivity'),
                                num_heads=getattr(self, 'condensation_num_heads', 8)
                            )
                            
                            hidden_states = torch.cat([sys_tokens, condensed_visual, text_tokens], dim=1)
                            img_seq_length = hidden_states.shape[1]
                            position_ids = torch.arange(img_seq_length, device=device, dtype=torch.long).unsqueeze(0)
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, img_seq_length), inputs_embeds, 0
                            )
                        else:
                            img_seq_length = current_seq_length
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, img_seq_length), inputs_embeds, 0
                            )
                    
                    elif (CUT_OFF_LAYER > 0) and (idx == CUT_OFF_LAYER):
                        # Cut-off 逻辑
                        current_seq_length = hidden_states.shape[1]
                        num_text_tokens = seq_length_with_past - SYS_LENGTH - IMG_LENGTH
                        num_current_img_tokens = current_seq_length - SYS_LENGTH - num_text_tokens

                        if num_current_img_tokens > 0:
                            keep_indices = torch.cat(
                                (
                                    torch.arange(SYS_LENGTH, device=device),
                                    torch.arange(SYS_LENGTH + num_current_img_tokens, current_seq_length, device=device)
                                )
                            )
                            new_seq_length = keep_indices.shape[0]
                            hidden_states = hidden_states[:, keep_indices, :]
                            position_ids = torch.arange(new_seq_length, device=device, dtype=torch.long).unsqueeze(0)
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, new_seq_length), inputs_embeds, 0
                            )
                        else:
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, current_seq_length), inputs_embeds, 0
                            )
                    else:
                        # 其他层：使用当前序列长度创建 mask
                        current_pruned_seq_length = hidden_states.shape[1]
                        if current_pruned_seq_length == seq_length_with_past:
                            new_attention_mask = attention_mask
                        else:
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, current_pruned_seq_length), inputs_embeds, 0
                            )
                
                elif USE_HMAP_V and not use_condensation:
                    # ===== 原始 HiMAP Hard Pruning: 需要在 decoder_layer 之后处理 =====
                    if idx < TXT_LAYER:
                        new_attention_mask = attention_mask
                        current_keep_indexs = None
                    else:
                        # 对于 TXT_LAYER 及之后的层，先用原始 mask 调用 decoder_layer
                        # 然后在 decoder_layer 调用后再进行剪枝
                        new_attention_mask = attention_mask

                else: 
                    new_attention_mask = attention_mask
                    
                # print(idx, hidden_states.shape, new_attention_mask.shape, position_ids.shape)

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=new_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            
            # ===== 原始 HiMAP 剪枝逻辑：在 decoder_layer 之后执行 =====
            if not self.gradient_checkpointing or not self.training:
                use_condensation = getattr(self, 'use_token_condensation', False)
                USE_HMAP_V = bool(self.use_hmap_v)
                
                if USE_HMAP_V and not use_condensation:
                    def _nz(val, default=0):
                        return val if val is not None else default
                    
                    SYS_LENGTH = _nz(self.hmap_v_sys_length)
                    IMG_LENGTH = _nz(self.hmap_v_img_length)
                    TXT_LAYER = _nz(self.hmap_v_attn_txt_layer)
                    TXT_ATTN_RANK = _nz(self.hmap_v_attn_txt_rank)
                    IMG_LAYER = _nz(self.hmap_v_attn_img_layer)
                    IMG_ATTN_RANK = _nz(self.hmap_v_attn_img_rank)
                    CUT_OFF_LAYER = _nz(self.cut_off_layer)
                    
                    sys_len = self.hmap_v_sys_length if self.hmap_v_sys_length is not None else 0
                    img_len = self.hmap_v_img_length if self.hmap_v_img_length is not None else 0
                    num_text_tokens = seq_length_with_past - sys_len - img_len
                    device = hidden_states.device
                    
                    # TXT_LAYER: 使用注意力分数进行第一次剪枝
                    if idx == TXT_LAYER:
                        txt_layer_attn = layer_outputs[1]
                        if isinstance(txt_layer_attn, tuple):
                            txt_layer_attn = txt_layer_attn[0]
                        txt_layer_attn_avg = torch.mean(txt_layer_attn, dim=1)[0]
                        img2txt_attn = torch.sum(
                            txt_layer_attn_avg[SYS_LENGTH+IMG_LENGTH:, SYS_LENGTH:SYS_LENGTH+IMG_LENGTH], dim=0
                        )
                        
                        img2txt_attn_len = img2txt_attn.shape[0]
                        topk_rank = min(TXT_ATTN_RANK, img2txt_attn_len)
                        img2txt_attn_topk_index = img2txt_attn.topk(topk_rank).indices + SYS_LENGTH
                        
                        txt_keep_indexs = torch.cat(
                            (
                                torch.arange(SYS_LENGTH, device=device),
                                img2txt_attn_topk_index,
                                torch.arange(SYS_LENGTH+IMG_LENGTH, seq_length_with_past, device=device)
                            )
                        )
                        txt_keep_indexs = txt_keep_indexs.sort().values
                        current_keep_indexs = txt_keep_indexs
                        
                        # 更新 hidden_states, position_ids
                        txt_seq_length = txt_keep_indexs.shape[0]
                        hidden_states = hidden_states[:, txt_keep_indexs, :]
                        position_ids = txt_keep_indexs.unsqueeze(0)
                    
                    # IMG_LAYER: 使用注意力分数进行第二次剪枝
                    elif idx == IMG_LAYER:
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
                        
                        # 更新 hidden_states, position_ids
                        img_seq_length = img_keep_indexs.shape[0]
                        hidden_states = hidden_states[:, img_keep_indexs, :]
                        position_ids = txt_keep_indexs[img_keep_indexs].unsqueeze(0)
                    
                    # CUT_OFF_LAYER: 移除所有视觉 token
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
                            hidden_states = hidden_states[:, keep_indices, :]
                            current_position_ids = position_ids.squeeze(0)
                            position_ids = current_position_ids[keep_indices].unsqueeze(0)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            # change the code to make llama model will not save attention scores
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
    
    def _condense_visual_tokens(self, visual_tokens, target_num, strategy='connectivity', num_heads=8):
        """
        Token Condensation: 通过相似度计算选择代表性 Token 并聚合信息
        
        Args:
            visual_tokens: [B, N, C] 原始视觉 Token
            target_num: M, 目标浓缩后的 Token 数量
            strategy: 'fps' 或 'connectivity'
            num_heads: 交叉注意力的头数
            
        Returns:
            condensed_tokens: [B, M, C] 浓缩后的 Token
        """
        B, N, C = visual_tokens.shape
        device = visual_tokens.device
        
        # 边界检查：如果目标数量 >= 当前数量，直接返回
        if target_num >= N:
            return visual_tokens
        
        # Step 1: 归一化并计算相似度矩阵
        normalized_tokens = torch.nn.functional.normalize(visual_tokens, p=2, dim=-1)  # [B, N, C]
        # 计算余弦相似度: [B, N, N]
        similarity_matrix = torch.bmm(normalized_tokens, normalized_tokens.transpose(1, 2))
                                # Step 1: 切分
                                sys_tokens = hidden_states[:, :SYS_LENGTH, :]
                                visual_tokens = hidden_states[:, SYS_LENGTH:SYS_LENGTH+current_visual_len, :]
                                text_tokens = hidden_states[:, SYS_LENGTH+current_visual_len:, :]
                                
                                # Step 2: 二次浓缩
                                condensed_visual = self._condense_visual_tokens(
                                    visual_tokens,
                                    target_num=IMG_ATTN_RANK,
                                    strategy=getattr(self, 'condensation_strategy', 'connectivity'),
                                    num_heads=getattr(self, 'condensation_num_heads', 8)
                                )
                                
                                # Step 3: 拼接
                                hidden_states = torch.cat([sys_tokens, condensed_visual, text_tokens], dim=1)
                                
                                # Step 4: 修正元数据
                                img_seq_length = hidden_states.shape[1]
                                position_ids = torch.arange(img_seq_length, device=device, dtype=torch.long).unsqueeze(0)
                                new_attention_mask = self._prepare_decoder_attention_mask(
                                    None, (batch_size, img_seq_length), inputs_embeds, 0
                                )
                            else:
                                # 当前视觉 Token 数量已经足够少，无需二次浓缩
                                img_seq_length = current_seq_length
                                new_attention_mask = self._prepare_decoder_attention_mask(
                                    None, (batch_size, img_seq_length), inputs_embeds, 0
                                )
                        else:
                            # ===== 原始 HiMAP Hard Pruning 逻辑 =====
                            # compute the img2img attention score
                            img_layer_attn = layer_outputs[1]
                            if isinstance(img_layer_attn, tuple):
                                img_layer_attn = img_layer_attn[0]
                            img_layer_attn_avg = torch.mean(img_layer_attn, dim=1)[0]

                            # After TXT_LAYER pruning, the remaining image-token count can be < TXT_ATTN_RANK.
                            # Compute the current image-token span in the *current* (possibly pruned) sequence.
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
                            
                            # get the indexs of selected image tokens (guard k <= len)
                            img2img_attn_len = img2img_attn.shape[0]
                            topk_rank = min(IMG_ATTN_RANK, img2img_attn_len)
                            if topk_rank > 0:
                                img2img_attn_topk_index = img2img_attn.topk(topk_rank).indices + SYS_LENGTH
                            else:
                                img2img_attn_topk_index = torch.tensor([], dtype=torch.long, device=device)

                            # text-token part starts right after the remaining image tokens in current sequence
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
        
                            # img_keep_indexs = torch.cat(
                            #     (
                            #         torch.arange(SYS_LENGTH, device=device),
                            #         img2img_attn_topk_index,
                            #         torch.arange(SYS_LENGTH+TXT_ATTN_RANK, txt_seq_length, device=device)
                            #     )
                            # ) 
                            img_keep_indexs = img_keep_indexs.sort().values
                            current_keep_indexs = img_keep_indexs
                            # update the hidden states, position ids and attention mask
                            img_seq_length = img_keep_indexs.shape[0]  
                            hidden_states = hidden_states[:, img_keep_indexs, :]
                            position_ids = txt_keep_indexs[img_keep_indexs].unsqueeze(0)
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, img_seq_length), inputs_embeds, 0
                            ) 

                    # cut-off layer: remove all image tokens after this layer
                    # === 新增：第三阶段，Cut-off 剪枝 ===
                    elif (CUT_OFF_LAYER > 0) and (idx == CUT_OFF_LAYER):
                        current_seq_length = hidden_states.shape[1]
                        # 计算当前剩余的 image token 数量
                        num_current_img_tokens = current_seq_length - SYS_LENGTH - num_text_tokens

                        if num_current_img_tokens > 0:
                            # 保留 sys tokens 和 text tokens
                            keep_indices = torch.cat(
                                (
                                    torch.arange(SYS_LENGTH, device=device), # sys tokens
                                    torch.arange(SYS_LENGTH + num_current_img_tokens, current_seq_length, device=device) # text tokens
                                )
                            )
                            target_device = hidden_states.device
                            keep_indices = keep_indices.to(target_device)
                            
                            new_seq_length = keep_indices.shape[0]
                            hidden_states = hidden_states[:, keep_indices, :]
                            
                            # 同样要裁剪 position_ids
                            current_position_ids = position_ids.squeeze(0)
                            current_position_ids = current_position_ids.to(target_device)
                            position_ids = current_position_ids[keep_indices].unsqueeze(0)
                            
                            
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, new_seq_length), inputs_embeds, 0
                            )

                            # 更新状态
                            self.temp_txt_keep_indexs = position_ids.squeeze(0)
                            self.temp_txt_seq_length = new_seq_length
                        else:
                            # image token 已经是 0，无需操作，只需准备 mask
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, current_seq_length), inputs_embeds, 0
                            )

                    # === 修改：处理所有其他层（剪枝前、剪枝间、剪枝后）的 mask ===
                    else:
                        current_pruned_seq_length = hidden_states.shape[1]
                        if current_pruned_seq_length == seq_length_with_past:
                            # 尚未发生剪枝
                            new_attention_mask = attention_mask
                        else:
                            # 已经发生剪枝，为当前较短的序列创建 mask
                            new_attention_mask = self._prepare_decoder_attention_mask(
                                None, (batch_size, current_pruned_seq_length), inputs_embeds, 0
                            )

                else: 
                    new_attention_mask = attention_mask
                    
                # print(idx, hidden_states.shape, new_attention_mask.shape, position_ids.shape)

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

            # change the code to make llama model will not save attention scores
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                # all_self_attns = None

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
    
    def _condense_visual_tokens(self, visual_tokens, target_num, strategy='connectivity', num_heads=8):
        """
        Token Condensation: 通过相似度计算选择代表性 Token 并聚合信息
        
        Args:
            visual_tokens: [B, N, C] 原始视觉 Token
            target_num: M, 目标浓缩后的 Token 数量
            strategy: 'fps' 或 'connectivity'
            num_heads: 交叉注意力的头数
            
        Returns:
            condensed_tokens: [B, M, C] 浓缩后的 Token
        """
        B, N, C = visual_tokens.shape
        device = visual_tokens.device
        
        # 边界检查：如果目标数量 >= 当前数量，直接返回
        if target_num >= N:
            return visual_tokens
        
        # Step 1: 归一化并计算相似度矩阵
        normalized_tokens = torch.nn.functional.normalize(visual_tokens, p=2, dim=-1)  # [B, N, C]
        # 计算余弦相似度: [B, N, N]
        similarity_matrix = torch.bmm(normalized_tokens, normalized_tokens.transpose(1, 2))
        
        # Step 2: 选择代表性 Token
        if strategy == 'fps':
            # 最远点采样 (Furthest Point Sampling)
            indices = self._fps_sampling(similarity_matrix, target_num)
        else:  # 'connectivity'
            # 基于连通性/度的选择
            indices = self._connectivity_sampling(similarity_matrix, target_num)
        
        # Step 3: 提取选中的 Token 作为质心
        # indices: [B, M]
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, target_num)  # [B, M]
        centroid_tokens = visual_tokens[batch_indices, indices]  # [B, M, C]
        
        # Step 4: 通过交叉注意力聚合全局信息
        condensed_tokens = self._cross_attention_aggregation(
            query=centroid_tokens,    # [B, M, C]
            key=visual_tokens,        # [B, N, C]
            value=visual_tokens,      # [B, N, C]
            num_heads=num_heads
        )
        
        return condensed_tokens
    
    def _fps_sampling(self, similarity_matrix, M):
        """
        最远点采样 (Furthest Point Sampling)
        
        Args:
            similarity_matrix: [B, N, N] 相似度矩阵
            M: 目标采样数量
            
        Returns:
            indices: [B, M] 选中的索引
        """
        B, N, _ = similarity_matrix.shape
        device = similarity_matrix.device
        
        # 距离 = 1 - 相似度
        distance_matrix = 1.0 - similarity_matrix  # [B, N, N]
        
        # 初始化
        selected_indices = []
        # 随机选择第一个点
        first_idx = torch.randint(0, N, (B,), device=device)  # [B]
        selected_indices.append(first_idx)
        
        # 到已选点集的最小距离
        min_distances = distance_matrix[torch.arange(B, device=device), first_idx, :]  # [B, N]
        
        # 迭代选择剩余的 M-1 个点
        for _ in range(M - 1):
            # 选择距离最大的点
            next_idx = torch.argmax(min_distances, dim=1)  # [B]
            selected_indices.append(next_idx)
            
            # 更新最小距离
            new_distances = distance_matrix[torch.arange(B, device=device), next_idx, :]  # [B, N]
            min_distances = torch.min(min_distances, new_distances)
        
        # 堆叠为 [B, M]
        indices = torch.stack(selected_indices, dim=1)
        return indices
    
    def _connectivity_sampling(self, similarity_matrix, M):
        """
        基于连通性/度的采样：选择度最大的 M 个 Token
        
        Args:
            similarity_matrix: [B, N, N] 相似度矩阵
            M: 目标采样数量
            
        Returns:
            indices: [B, M] 选中的索引
        """
        B, N, _ = similarity_matrix.shape
        
        # 计算每个 Token 的度（与其他所有 Token 的相似度之和）
        degree = torch.sum(similarity_matrix, dim=2)  # [B, N]
        
        # 选择度最大的 M 个 Token
        _, indices = torch.topk(degree, k=M, dim=1)  # [B, M]
        
        return indices
    
    def _cross_attention_aggregation(self, query, key, value, num_heads=8):
        """
        多头交叉注意力聚合
        
        Args:
            query: [B, M, C] 质心 Token
            key: [B, N, C] 所有 Token
            value: [B, N, C] 所有 Token
            num_heads: 注意力头数
            
        Returns:
            output: [B, M, C] 聚合后的 Token
        """
        B, M, C = query.shape
        N = key.shape[1]
        
        # 确保 C 可以被 num_heads 整除
        head_dim = C // num_heads
        assert C == head_dim * num_heads, f"Embedding dim {C} must be divisible by num_heads {num_heads}"
        
        # Reshape 为多头格式: [B, num_heads, M/N, head_dim]
        Q = query.view(B, M, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, M, head_dim]
        K = key.view(B, N, num_heads, head_dim).transpose(1, 2)    # [B, num_heads, N, head_dim]
        V = value.view(B, N, num_heads, head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        
        # 计算注意力分数: Q @ K^T / sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)  # [B, num_heads, M, N]
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # 加权求和: attn_weights @ V
        output = torch.matmul(attn_weights, V)  # [B, num_heads, M, head_dim]
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(B, M, C)  # [B, M, C]
        
        return output

