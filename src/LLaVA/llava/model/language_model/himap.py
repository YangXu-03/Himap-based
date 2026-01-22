import random
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
        
        # FastV hyperparameters
        self.fast_v_sys_length = config.fast_v_sys_length
        self.fast_v_image_token_length = config.fast_v_image_token_length
        self.fast_v_attention_rank = config.fast_v_attention_rank
        self.fast_v_agg_layer = config.fast_v_agg_layer
        self.use_fast_v = config.use_fast_v
        
        # FastV Advanced parameters
        self.token_selection_method = getattr(config, 'fast_v_token_selection_method', 'avg_all_heads')
        self.weighted_alpha = getattr(config, 'fast_v_weighted_alpha', 0.5)
        
        # Storage for last generated mask and indices
        self.last_gen_attention_mask = None
        self.last_gen_kept_indices = None
        self.last_selection_metadata = {}
        
    def reset_hmapv(self):
        self.hmap_v_sys_length = self.config.hmap_v_sys_length
        self.hmap_v_img_length = self.config.hmap_v_img_length
        self.hmap_v_attn_txt_layer = self.config.hmap_v_attn_txt_layer
        self.hmap_v_attn_txt_rank = self.config.hmap_v_attn_txt_rank
        self.hmap_v_attn_img_layer = self.config.hmap_v_attn_img_layer
        self.hmap_v_attn_img_rank = self.config.hmap_v_attn_img_rank
        self.use_hmap_v = self.config.use_hmap_v  
        self.cut_off_layer = self.config.cut_off_layer
        
    def reset_fastv(self):
        """Reset FastV parameters to config defaults"""
        self.fast_v_sys_length = self.config.fast_v_sys_length
        self.fast_v_image_token_length = self.config.fast_v_image_token_length
        self.fast_v_attention_rank = self.config.fast_v_attention_rank
        self.fast_v_agg_layer = self.config.fast_v_agg_layer
        self.use_fast_v = self.config.use_fast_v
        self.token_selection_method = getattr(self.config, 'fast_v_token_selection_method', 'avg_all_heads')
        self.weighted_alpha = getattr(self.config, 'fast_v_weighted_alpha', 0.5)
        # Reset dynamic outputs
        self.last_gen_attention_mask = None
        self.last_gen_kept_indices = None
        self.last_selection_metadata = {}
        
    def _select_tokens_fastv(self, attention_weights, sys_length, image_token_length, attention_rank):
        """Select tokens based on the configured strategy"""
        if self.token_selection_method == 'max_head':
            return self._select_tokens_max_head(attention_weights, sys_length, image_token_length, attention_rank)
        elif self.token_selection_method == 'weighted_combination':
            return self._select_tokens_weighted_combination(attention_weights, sys_length, image_token_length, attention_rank)
        else:  # 'avg_all_heads' or default
            return self._select_tokens_avg_all_heads(attention_weights, sys_length, image_token_length, attention_rank)
    
    def _select_tokens_max_head(self, attention_weights, sys_length, image_token_length, attention_rank):
        """Strategy 1: Select tokens based on the attention head with maximum text-to-vision attention"""
        last_token_attention = attention_weights[:, :, -1, :]
        image_attention = last_token_attention[:, :, sys_length:sys_length+image_token_length]
        head_importance = image_attention.sum(dim=-1)
        max_head_idx = head_importance.argmax(dim=1)
        batch_size = attention_weights.shape[0]
        max_head_attention = torch.stack([image_attention[b, max_head_idx[b], :] for b in range(batch_size)])
        
        if attention_rank > 0:
            top_indices = max_head_attention[0].topk(attention_rank).indices
        else:
            top_indices = torch.tensor([], dtype=torch.long, device=attention_weights.device)
        
        self.last_selection_metadata = {
            'method': 'max_head',
            'max_head_idx': max_head_idx[0].item(),
            'head_importance': head_importance[0].cpu().numpy(),
        }
        return top_indices

    def _select_tokens_avg_all_heads(self, attention_weights, sys_length, image_token_length, attention_rank):
        """Strategy 2: Select tokens based on average attention across all heads (original FastV)"""
        avg_attention = torch.mean(attention_weights, dim=1)
        last_token_image_attention = avg_attention[0, -1, sys_length:sys_length+image_token_length]
        
        if attention_rank > 0:
            top_indices = last_token_image_attention.topk(attention_rank).indices
        else:
            top_indices = torch.tensor([], dtype=torch.long, device=attention_weights.device)
        
        self.last_selection_metadata = {'method': 'avg_all_heads'}
        return top_indices

    def _select_tokens_weighted_combination(self, attention_weights, sys_length, image_token_length, attention_rank):
        """Strategy 3: Weighted combination of max head and average of other heads"""
        last_token_attention = attention_weights[:, :, -1, :]
        image_attention = last_token_attention[:, :, sys_length:sys_length+image_token_length]
        head_importance = image_attention.sum(dim=-1)
        max_head_idx = head_importance.argmax(dim=1)
        batch_size = attention_weights.shape[0]
        max_head_image_attention = torch.stack([image_attention[b, max_head_idx[b], :] for b in range(batch_size)])
        
        num_heads = attention_weights.shape[1]
        other_heads_mask = torch.ones(num_heads, dtype=torch.bool, device=attention_weights.device)
        other_heads_mask[max_head_idx[0]] = False
        
        if other_heads_mask.sum() > 0:
            other_heads_attention = image_attention[:, other_heads_mask, :].mean(dim=1)
        else:
            other_heads_attention = torch.zeros_like(max_head_image_attention)
        
        alpha = self.weighted_alpha
        combined_attention = alpha * max_head_image_attention + (1 - alpha) * other_heads_attention
        
        if attention_rank > 0:
            top_indices = combined_attention[0].topk(attention_rank).indices
        else:
            top_indices = torch.tensor([], dtype=torch.long, device=attention_weights.device)
        
        self.last_selection_metadata = {
            'method': 'weighted_combination',
            'alpha': alpha,
            'max_head_idx': max_head_idx[0].item(),
            'head_importance': head_importance[0].cpu().numpy(),
        }
        return top_indices
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
                if USE_HMAP_V:
                    
                    # Before image tokens pruning
                    if idx < TXT_LAYER:
                        new_attention_mask = attention_mask
                        current_keep_indexs = None

                    # image token pruning according to img2txt information
                    elif idx == TXT_LAYER:
                        # compute the img2txt attention score
                        txt_layer_attn = layer_outputs[1]
                        if isinstance(txt_layer_attn, tuple):  #修改1turple错误
                            txt_layer_attn = txt_layer_attn[0]
                        txt_layer_attn_avg = torch.mean(txt_layer_attn, dim=1)[0]                        
                        img2txt_attn = torch.sum(
                            txt_layer_attn_avg[SYS_LENGTH+IMG_LENGTH:, SYS_LENGTH:SYS_LENGTH+IMG_LENGTH], dim=0
                        )
                        # get the indexs of selected image tokens。修改边界取rank最小值
                        img2txt_attn_len = img2txt_attn.shape[0]
                        topk_rank = min(TXT_ATTN_RANK, img2txt_attn_len)
                        img2txt_attn_topk_index = img2txt_attn.topk(topk_rank).indices + SYS_LENGTH
                        #img2txt_attn_topk_index = img2txt_attn.topk(TXT_ATTN_RANK).indices + SYS_LENGTH
                        txt_keep_indexs = torch.cat(
                            (
                                torch.arange(SYS_LENGTH, device=device),
                                img2txt_attn_topk_index,
                                torch.arange(SYS_LENGTH+IMG_LENGTH, seq_length_with_past, device=device)
                            )
                        )
                        txt_keep_indexs = txt_keep_indexs.sort().values
                        current_keep_indexs = txt_keep_indexs
                        # update the hidden states, position ids and attention mask
                        txt_seq_length = txt_keep_indexs.shape[0]
                        hidden_states = hidden_states[:, txt_keep_indexs, :]
                        position_ids = txt_keep_indexs.unsqueeze(0)
                        new_attention_mask = self._prepare_decoder_attention_mask(
                            None, (batch_size, txt_seq_length), inputs_embeds, 0
                        )                        

                    # image token pruning according to img2img information
                    elif idx == IMG_LAYER:
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
                        # img2img_attn = torch.sum(
                        #     img_layer_attn_avg[SYS_LENGTH+TXT_ATTN_RANK:, SYS_LENGTH:SYS_LENGTH+TXT_ATTN_RANK], dim=0
                        # )
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

                # FastV Advanced logic
                elif self.use_fast_v:
                    USE_FAST_V = True
                    SYS_LENGTH_FV = _nz(self.fast_v_sys_length)
                    IMAGE_TOKEN_LENGTH = _nz(self.fast_v_image_token_length)
                    ATTENTION_RANK = _nz(self.fast_v_attention_rank)
                    AGG_LAYER = _nz(self.fast_v_agg_layer)
                    
                    # Track generated attention mask
                    if not hasattr(self, '_fastv_gen_attention_mask'):
                        self._fastv_gen_attention_mask = None
                    
                    if idx < AGG_LAYER:
                        # Before aggregation layer: use full attention
                        new_attention_mask = torch.ones(
                            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                        )
                        new_attention_mask = self._prepare_decoder_attention_mask(
                            new_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                        )
                        
                    elif idx == AGG_LAYER:
                        # At aggregation layer: generate pruned attention mask
                        if idx != 0:
                            # Use attention from previous layer
                            att_out = layer_outputs[1]
                            if isinstance(att_out, (tuple, list)):
                                last_layer_attention = att_out[0] if len(att_out) > 0 else None
                            else:
                                last_layer_attention = att_out
                            
                            if last_layer_attention is None:
                                # Fallback: no pruning
                                gen_attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
                                gen_attention_mask = self._prepare_decoder_attention_mask(
                                    gen_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                                )
                                new_attention_mask = gen_attention_mask
                            else:
                                # Select tokens based on strategy
                                top_indices = self._select_tokens_fastv(
                                    last_layer_attention, SYS_LENGTH_FV, IMAGE_TOKEN_LENGTH, ATTENTION_RANK
                                )
                                
                                # Generate attention mask
                                gen_attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
                                gen_attention_mask[:, SYS_LENGTH_FV:SYS_LENGTH_FV+IMAGE_TOKEN_LENGTH] = False
                                
                                if ATTENTION_RANK > 0:
                                    global_indices = top_indices + SYS_LENGTH_FV
                                    gen_attention_mask[:, global_indices] = True
                                
                                # Save mask and indices
                                try:
                                    self.last_gen_attention_mask = gen_attention_mask.clone().detach().cpu()
                                    kept = gen_attention_mask[0].nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
                                    self.last_gen_kept_indices = kept
                                except Exception:
                                    self.last_gen_attention_mask = None
                                    self.last_gen_kept_indices = None
                                
                                gen_attention_mask = self._prepare_decoder_attention_mask(
                                    gen_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                                )
                                new_attention_mask = gen_attention_mask
                                self._fastv_gen_attention_mask = gen_attention_mask
                        else:
                            # idx == 0: random selection
                            gen_attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
                            
                            if ATTENTION_RANK > 0:
                                rand_image_attention_mask = [1]*ATTENTION_RANK + [0]*(IMAGE_TOKEN_LENGTH-ATTENTION_RANK)
                                random.shuffle(rand_image_attention_mask)
                                gen_attention_mask[:, SYS_LENGTH_FV:SYS_LENGTH_FV+IMAGE_TOKEN_LENGTH] = torch.tensor(
                                    rand_image_attention_mask, dtype=attention_mask.dtype, device=inputs_embeds.device
                                )
                            else:
                                gen_attention_mask[:, SYS_LENGTH_FV:SYS_LENGTH_FV+IMAGE_TOKEN_LENGTH] = False

                            try:
                                self.last_gen_attention_mask = gen_attention_mask.clone().detach().cpu()
                                kept = gen_attention_mask[0].nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
                                self.last_gen_kept_indices = kept
                            except Exception:
                                self.last_gen_attention_mask = None
                                self.last_gen_kept_indices = None

                            gen_attention_mask = self._prepare_decoder_attention_mask(
                                gen_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                            )
                            new_attention_mask = gen_attention_mask
                            self._fastv_gen_attention_mask = gen_attention_mask

                    else:
                        # After aggregation layer: reuse generated mask
                        new_attention_mask = self._fastv_gen_attention_mask
                
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
