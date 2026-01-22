import random
import torch
from typing import Tuple
from transformers.utils import logging
from .himap_modeling_llama import LlamaModel
from .himap_configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union

logger = logging.get_logger(__name__)


class FastvAdvanced_LlamaModel(LlamaModel):
    """
    Advanced FastV implementation with multiple token selection strategies:
    
    1. 'max_head': Select tokens based on attention from the head with maximum text-to-vision attention
    2. 'avg_all_heads': Select tokens based on average attention across all heads (original FastV)
    3. 'weighted_combination': Weighted combination of max head and average of other heads
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.config = config
        # FastV hyperparameters
        self.fast_v_sys_length = config.fast_v_sys_length
        self.fast_v_image_token_length = config.fast_v_image_token_length
        self.fast_v_attention_rank = config.fast_v_attention_rank
        self.fast_v_agg_layer = config.fast_v_agg_layer
        self.use_fast_v = config.use_fast_v
        
        # Advanced FastV parameters
        # token_selection_method: 'max_head', 'avg_all_heads', 'weighted_combination'
        self.token_selection_method = getattr(config, 'fast_v_token_selection_method', 'avg_all_heads')
        # alpha weight for weighted_combination: max_head * alpha + avg_other_heads * (1 - alpha)
        self.weighted_alpha = getattr(config, 'fast_v_weighted_alpha', 0.5)
        
        # Storage for last generated mask and indices (for external access)
        self.last_gen_attention_mask = None
        self.last_gen_kept_indices = None
        self.last_selection_metadata = {}  # Store additional metadata about selection

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

    def _select_tokens_max_head(self, attention_weights, sys_length, image_token_length, attention_rank):
        """
        Strategy 1: Select tokens based on the attention head with maximum text-to-vision attention
        
        Args:
            attention_weights: [batch, num_heads, seq_len, seq_len]
            sys_length: Starting position of image tokens
            image_token_length: Number of image tokens
            attention_rank: Number of tokens to keep
            
        Returns:
            Indices of selected tokens (relative to image token region)
        """
        # Get attention from last token to all positions: [batch, num_heads, seq_len]
        last_token_attention = attention_weights[:, :, -1, :]
        
        # Get attention to image tokens: [batch, num_heads, image_token_length]
        image_attention = last_token_attention[:, :, sys_length:sys_length+image_token_length]
        
        # Sum attention across image tokens for each head: [batch, num_heads]
        head_importance = image_attention.sum(dim=-1)
        
        # Find the head with maximum attention to vision tokens
        max_head_idx = head_importance.argmax(dim=1)  # [batch]
        
        # Get attention from the max head: [batch, image_token_length]
        batch_size = attention_weights.shape[0]
        max_head_attention = torch.stack([
            image_attention[b, max_head_idx[b], :] for b in range(batch_size)
        ])
        
        # Select top-k tokens
        if attention_rank > 0:
            top_indices = max_head_attention[0].topk(attention_rank).indices
        else:
            top_indices = torch.tensor([], dtype=torch.long, device=attention_weights.device)
        
        # Store metadata
        self.last_selection_metadata = {
            'method': 'max_head',
            'max_head_idx': max_head_idx[0].item(),
            'head_importance': head_importance[0].cpu().numpy(),
        }
        
        return top_indices

    def _select_tokens_avg_all_heads(self, attention_weights, sys_length, image_token_length, attention_rank):
        """
        Strategy 2: Select tokens based on average attention across all heads (original FastV method)
        
        Args:
            attention_weights: [batch, num_heads, seq_len, seq_len]
            sys_length: Starting position of image tokens
            image_token_length: Number of image tokens
            attention_rank: Number of tokens to keep
            
        Returns:
            Indices of selected tokens (relative to image token region)
        """
        # Average attention across all heads: [batch, seq_len, seq_len]
        avg_attention = torch.mean(attention_weights, dim=1)
        
        # Get attention from last token to image tokens: [batch, image_token_length]
        last_token_image_attention = avg_attention[0, -1, sys_length:sys_length+image_token_length]
        
        # Select top-k tokens
        if attention_rank > 0:
            top_indices = last_token_image_attention.topk(attention_rank).indices
        else:
            top_indices = torch.tensor([], dtype=torch.long, device=attention_weights.device)
        
        # Store metadata
        self.last_selection_metadata = {
            'method': 'avg_all_heads',
        }
        
        return top_indices

    def _select_tokens_weighted_combination(self, attention_weights, sys_length, image_token_length, attention_rank):
        """
        Strategy 3: Weighted combination of max head attention and average of other heads
        score = max_head_attention * alpha + avg_other_heads_attention * (1 - alpha)
        
        Args:
            attention_weights: [batch, num_heads, seq_len, seq_len]
            sys_length: Starting position of image tokens
            image_token_length: Number of image tokens
            attention_rank: Number of tokens to keep
            
        Returns:
            Indices of selected tokens (relative to image token region)
        """
        # Get attention from last token to all positions: [batch, num_heads, seq_len]
        last_token_attention = attention_weights[:, :, -1, :]
        
        # Get attention to image tokens: [batch, num_heads, image_token_length]
        image_attention = last_token_attention[:, :, sys_length:sys_length+image_token_length]
        
        # Sum attention across image tokens for each head: [batch, num_heads]
        head_importance = image_attention.sum(dim=-1)
        
        # Find the head with maximum attention
        max_head_idx = head_importance.argmax(dim=1)  # [batch]
        
        # Get attention from max head: [batch, image_token_length]
        batch_size = attention_weights.shape[0]
        max_head_image_attention = torch.stack([
            image_attention[b, max_head_idx[b], :] for b in range(batch_size)
        ])
        
        # Calculate average attention from other heads
        num_heads = attention_weights.shape[1]
        other_heads_mask = torch.ones(num_heads, dtype=torch.bool, device=attention_weights.device)
        other_heads_mask[max_head_idx[0]] = False
        
        if other_heads_mask.sum() > 0:
            # Average of other heads: [batch, image_token_length]
            other_heads_attention = image_attention[:, other_heads_mask, :].mean(dim=1)
        else:
            # If only one head, use zero for other heads
            other_heads_attention = torch.zeros_like(max_head_image_attention)
        
        # Weighted combination
        alpha = self.weighted_alpha
        combined_attention = alpha * max_head_image_attention + (1 - alpha) * other_heads_attention
        
        # Select top-k tokens
        if attention_rank > 0:
            top_indices = combined_attention[0].topk(attention_rank).indices
        else:
            top_indices = torch.tensor([], dtype=torch.long, device=attention_weights.device)
        
        # Store metadata
        self.last_selection_metadata = {
            'method': 'weighted_combination',
            'alpha': alpha,
            'max_head_idx': max_head_idx[0].item(),
            'head_importance': head_importance[0].cpu().numpy(),
        }
        
        return top_indices

    def _generate_attention_mask_with_selection(
        self, 
        layer_outputs, 
        batch_size, 
        seq_length_with_past, 
        seq_length,
        inputs_embeds,
        past_key_values_length,
        sys_length,
        image_token_length,
        attention_rank
    ):
        """Generate attention mask based on selected token pruning strategy"""
        
        # Fetch attention output
        att_out = layer_outputs[1]
        
        # Normalize to a Tensor or None
        if isinstance(att_out, (tuple, list)):
            if len(att_out) == 0:
                last_layer_attention = None
            else:
                last_layer_attention = att_out[0]
        else:
            last_layer_attention = att_out
        
        # If no attention tensor available, fall back to no pruning
        if last_layer_attention is None:
            gen_attention_mask = torch.ones(
                (batch_size, seq_length_with_past), 
                dtype=torch.bool, 
                device=inputs_embeds.device
            )
            gen_attention_mask = self._prepare_decoder_attention_mask(
                gen_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
            return gen_attention_mask
        
        # Select tokens based on the chosen strategy
        if self.token_selection_method == 'max_head':
            top_indices = self._select_tokens_max_head(
                last_layer_attention, sys_length, image_token_length, attention_rank
            )
        elif self.token_selection_method == 'weighted_combination':
            top_indices = self._select_tokens_weighted_combination(
                last_layer_attention, sys_length, image_token_length, attention_rank
            )
        else:  # 'avg_all_heads' or default
            top_indices = self._select_tokens_avg_all_heads(
                last_layer_attention, sys_length, image_token_length, attention_rank
            )
        
        # Generate attention mask
        gen_attention_mask = torch.ones(
            (batch_size, seq_length_with_past), 
            dtype=torch.bool, 
            device=inputs_embeds.device
        )
        gen_attention_mask[:, sys_length:sys_length+image_token_length] = False
        
        if attention_rank > 0:
            # Map selected indices to global positions
            global_indices = top_indices + sys_length
            gen_attention_mask[:, global_indices] = True
        
        # Save mask and indices for external access
        try:
            self.last_gen_attention_mask = gen_attention_mask.clone().detach().cpu()
            kept = gen_attention_mask[0].nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
            self.last_gen_kept_indices = kept
        except Exception:
            self.last_gen_attention_mask = None
            self.last_gen_kept_indices = None
        
        # Prepare decoder attention mask
        gen_attention_mask = self._prepare_decoder_attention_mask(
            gen_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        
        return gen_attention_mask

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

        # Retrieve input_ids and inputs_embeds
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
        
        # Embed positions
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

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Track generated attention mask for FastV
        gen_attention_mask = None

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
                # ==================== ADVANCED FASTV START ====================
                
                USE_FAST_V = self.use_fast_v
                SYS_LENGTH = self.fast_v_sys_length
                IMAGE_TOKEN_LENGTH = self.fast_v_image_token_length
                ATTENTION_RANK = self.fast_v_attention_rank
                AGG_LAYER = self.fast_v_agg_layer

                if USE_FAST_V:
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
                            gen_attention_mask = self._generate_attention_mask_with_selection(
                                layer_outputs,
                                batch_size,
                                seq_length_with_past,
                                seq_length,
                                inputs_embeds,
                                past_key_values_length,
                                SYS_LENGTH,
                                IMAGE_TOKEN_LENGTH,
                                ATTENTION_RANK
                            )
                            new_attention_mask = gen_attention_mask
                        else:
                            # idx == 0: random selection (fallback for first layer)
                            gen_attention_mask = torch.ones(
                                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                            )
                            
                            if ATTENTION_RANK > 0:
                                # Random mask for image tokens
                                rand_image_attention_mask = [1]*ATTENTION_RANK + [0]*(IMAGE_TOKEN_LENGTH-ATTENTION_RANK)
                                random.shuffle(rand_image_attention_mask)
                                gen_attention_mask[:, SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH] = torch.tensor(
                                    rand_image_attention_mask, dtype=attention_mask.dtype, device=inputs_embeds.device
                                )
                            else:
                                # Complete removal: set all image tokens to False
                                gen_attention_mask[:, SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH] = False

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

                    else:
                        # After aggregation layer: reuse generated mask
                        new_attention_mask = gen_attention_mask
                
                else: 
                    # FastV disabled: use original attention mask
                    new_attention_mask = attention_mask

                # ==================== ADVANCED FASTV END ====================

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

        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
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
