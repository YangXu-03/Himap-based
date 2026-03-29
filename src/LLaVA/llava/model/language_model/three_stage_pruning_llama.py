import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from .himap_configuration_llama import LlamaConfig
from .himap_modeling_llama import LlamaModel


logger = logging.get_logger(__name__)


class ThreeStagePruningLlamaModel(LlamaModel):
    """
    Three-stage image token pruning.

    Stage-1 (layer=3):
    - Within a local 3x3 image-grid window, compute cosine similarity between image tokens.
    - Apply RoPE-style distance penalty to similarity.
    - Prune redundant token when penalized similarity > threshold.

    Stage-2 (layer=12):
    - Use text-to-image attention from previous layer to rank image tokens.
    - Keep top-k image tokens.

    Stage-3 (layer=20):
    - Remove all image tokens.

    Note:
    - Layer ids in config are 1-based and converted to 0-based in runtime.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config

        # Global switch.
        self.use_three_stage_pruning = getattr(config, "use_three_stage_pruning", False)

        # Token span settings.
        self.three_stage_sys_length = self._coalesce_int(
            getattr(config, "three_stage_sys_length", None),
            getattr(config, "fast_v_sys_length", None),
            35,
        )
        self.three_stage_image_token_length = self._coalesce_int(
            getattr(config, "three_stage_image_token_length", None),
            getattr(config, "fast_v_image_token_length", None),
            576,
        )

        # 1-based layer ids in config.
        self.stage1_layer = self._coalesce_int(getattr(config, "stage1_layer", None), 3)
        self.stage2_layer = self._coalesce_int(getattr(config, "stage2_layer", None), 12)
        self.stage3_layer = self._coalesce_int(getattr(config, "stage3_layer", None), 20)

        # Stage-1 hyperparameters.
        self.stage1_similarity_threshold = float(getattr(config, "stage1_similarity_threshold", 0.92))
        self.stage1_rope_penalty_alpha = float(getattr(config, "stage1_rope_penalty_alpha", 0.08))
        self.stage1_window_size = self._coalesce_int(getattr(config, "stage1_window_size", None), 3)
        self.stage1_min_keep_tokens = self._coalesce_int(
            getattr(config, "stage1_min_keep_tokens", None),
            getattr(config, "stage2_keep_tokens", None),
            64,
        )

        # Stage-2 hyperparameters.
        self.stage2_keep_tokens = self._coalesce_int(getattr(config, "stage2_keep_tokens", None), 128)

        # Debug outputs for external inspection.
        self.last_gen_attention_mask = None
        self.last_gen_kept_indices = None
        self.last_stage_stats = {}

    @staticmethod
    def _coalesce_int(*values):
        for v in values:
            if v is not None:
                return int(v)
        return 0

    def reset_three_stage_pruning(self):
        self.use_three_stage_pruning = getattr(self.config, "use_three_stage_pruning", False)
        self.three_stage_sys_length = self._coalesce_int(
            getattr(self.config, "three_stage_sys_length", None),
            getattr(self.config, "fast_v_sys_length", None),
            35,
        )
        self.three_stage_image_token_length = self._coalesce_int(
            getattr(self.config, "three_stage_image_token_length", None),
            getattr(self.config, "fast_v_image_token_length", None),
            576,
        )
        self.stage1_layer = self._coalesce_int(getattr(self.config, "stage1_layer", None), 3)
        self.stage2_layer = self._coalesce_int(getattr(self.config, "stage2_layer", None), 12)
        self.stage3_layer = self._coalesce_int(getattr(self.config, "stage3_layer", None), 20)
        self.stage1_similarity_threshold = float(getattr(self.config, "stage1_similarity_threshold", 0.92))
        self.stage1_rope_penalty_alpha = float(getattr(self.config, "stage1_rope_penalty_alpha", 0.08))
        self.stage1_window_size = self._coalesce_int(getattr(self.config, "stage1_window_size", None), 3)
        self.stage1_min_keep_tokens = self._coalesce_int(
            getattr(self.config, "stage1_min_keep_tokens", None),
            getattr(self.config, "stage2_keep_tokens", None),
            64,
        )
        self.stage2_keep_tokens = self._coalesce_int(getattr(self.config, "stage2_keep_tokens", None), 128)
        self.last_gen_attention_mask = None
        self.last_gen_kept_indices = None
        self.last_stage_stats = {}

    def _safe_image_span(self, seq_len: int) -> Tuple[int, int]:
        img_start = self.three_stage_sys_length
        img_end = min(img_start + self.three_stage_image_token_length, seq_len)
        return img_start, img_end

    def _infer_grid_hw(self, n_tokens: int) -> Tuple[int, int]:
        side = int(math.sqrt(n_tokens))
        if side * side == n_tokens:
            return side, side
        return 1, n_tokens

    def _apply_stage1_similarity_pruning(
        self,
        token_keep_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        img_start: int,
        img_end: int,
    ) -> torch.Tensor:
        image_states = hidden_states[0, img_start:img_end, :]
        n_img = image_states.shape[0]
        if n_img <= 1:
            return token_keep_mask

        h, w = self._infer_grid_hw(n_img)
        radius = max(0, self.stage1_window_size // 2)
        norm_states = F.normalize(image_states, p=2, dim=-1)

        local_keep = token_keep_mask[0, img_start:img_end].clone()

        for i in range(n_img):
            if not local_keep[i]:
                continue

            ri = i // w
            ci = i % w

            for j in range(i + 1, n_img):
                if not local_keep[j]:
                    continue

                rj = j // w
                cj = j % w

                if abs(ri - rj) > radius or abs(ci - cj) > radius:
                    continue

                cosine_sim = torch.dot(norm_states[i], norm_states[j])
                dist = math.sqrt((ri - rj) * (ri - rj) + (ci - cj) * (ci - cj))
                rope_penalty = math.exp(-self.stage1_rope_penalty_alpha * dist)
                penalized_sim = cosine_sim * rope_penalty

                if penalized_sim > self.stage1_similarity_threshold:
                    local_keep[j] = False

        keep_count = int(local_keep.sum().item())
        min_keep = min(max(1, self.stage1_min_keep_tokens), n_img)
        if keep_count < min_keep:
            dropped = (~local_keep).nonzero(as_tuple=False).squeeze(-1)
            restore_n = min_keep - keep_count
            if dropped.numel() > 0 and restore_n > 0:
                local_keep[dropped[:restore_n]] = True

        token_keep_mask[:, img_start:img_end] = local_keep.unsqueeze(0)
        self.last_stage_stats["stage1"] = {
            "before": n_img,
            "after": int(local_keep.sum().item()),
            "threshold": self.stage1_similarity_threshold,
            "window_size": self.stage1_window_size,
        }
        return token_keep_mask

    def _apply_stage2_text_attention_topk(
        self,
        token_keep_mask: torch.Tensor,
        previous_layer_attention: Optional[torch.Tensor],
        img_start: int,
        img_end: int,
        seq_length_with_past: int,
    ) -> torch.Tensor:
        n_img = img_end - img_start
        if n_img <= 0:
            return token_keep_mask

        if previous_layer_attention is None:
            return token_keep_mask

        text_start = img_end
        text_end = seq_length_with_past
        if text_start >= text_end:
            text_start = max(0, seq_length_with_past - 1)

        text_to_img = previous_layer_attention[0, :, text_start:text_end, img_start:img_end]
        if text_to_img.numel() == 0:
            return token_keep_mask

        scores = text_to_img.mean(dim=0).mean(dim=0)

        current_local_keep = token_keep_mask[0, img_start:img_end]
        masked_scores = scores.clone()
        masked_scores[~current_local_keep] = -1e9

        keep_k = min(max(0, self.stage2_keep_tokens), n_img)
        if keep_k == 0:
            token_keep_mask[:, img_start:img_end] = False
            self.last_stage_stats["stage2"] = {
                "before": int(current_local_keep.sum().item()),
                "after": 0,
                "topk": 0,
            }
            return token_keep_mask

        top_idx = masked_scores.topk(keep_k).indices
        new_local_keep = torch.zeros_like(current_local_keep)
        new_local_keep[top_idx] = True

        token_keep_mask[:, img_start:img_end] = new_local_keep.unsqueeze(0)
        self.last_stage_stats["stage2"] = {
            "before": int(current_local_keep.sum().item()),
            "after": int(new_local_keep.sum().item()),
            "topk": keep_k,
        }
        return token_keep_mask

    def _prepare_mask(
        self,
        token_keep_mask: torch.Tensor,
        batch_size: int,
        seq_length: int,
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> torch.Tensor:
        return self._prepare_decoder_attention_mask(
            token_keep_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

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
            seq_length_with_past += past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            token_keep_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        else:
            token_keep_mask = attention_mask.to(dtype=torch.bool)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        current_attention_mask = self._prepare_mask(
            token_keep_mask, batch_size, seq_length, inputs_embeds, past_key_values_length
        )

        # Convert 1-based to 0-based.
        stage1_idx = max(0, self.stage1_layer - 1)
        stage2_idx = max(0, self.stage2_layer - 1)
        stage3_idx = max(0, self.stage3_layer - 1)

        stage1_done = False
        stage2_done = False
        stage3_done = False

        prev_layer_attention = None

        can_prune = bool(self.use_three_stage_pruning) and past_key_values is None and batch_size == 1
        img_start, img_end = self._safe_image_span(seq_length_with_past)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if can_prune and img_start < img_end:
                if (idx == stage1_idx) and (not stage1_done):
                    token_keep_mask = self._apply_stage1_similarity_pruning(
                        token_keep_mask, hidden_states, img_start, img_end
                    )
                    current_attention_mask = self._prepare_mask(
                        token_keep_mask, batch_size, seq_length, inputs_embeds, past_key_values_length
                    )
                    stage1_done = True

                if (idx == stage2_idx) and (not stage2_done):
                    token_keep_mask = self._apply_stage2_text_attention_topk(
                        token_keep_mask,
                        prev_layer_attention,
                        img_start,
                        img_end,
                        seq_length_with_past,
                    )
                    current_attention_mask = self._prepare_mask(
                        token_keep_mask, batch_size, seq_length, inputs_embeds, past_key_values_length
                    )
                    stage2_done = True

                if (idx == stage3_idx) and (not stage3_done):
                    before = int(token_keep_mask[:, img_start:img_end].sum().item())
                    token_keep_mask[:, img_start:img_end] = False
                    current_attention_mask = self._prepare_mask(
                        token_keep_mask, batch_size, seq_length, inputs_embeds, past_key_values_length
                    )
                    self.last_stage_stats["stage3"] = {
                        "before": before,
                        "after": 0,
                    }
                    stage3_done = True

            need_attention_for_stage2 = can_prune and (idx == stage2_idx - 1)
            layer_output_attentions = output_attentions or need_attention_for_stage2

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, layer_output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    current_attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=current_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=layer_output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if layer_output_attentions and len(layer_outputs) > 1:
                att_candidate = layer_outputs[1]
                if isinstance(att_candidate, torch.Tensor):
                    prev_layer_attention = att_candidate

            if use_cache:
                cache_idx = 2 if layer_output_attentions else 1
                if len(layer_outputs) > cache_idx:
                    next_decoder_cache += (layer_outputs[cache_idx],)

            if output_attentions and len(layer_outputs) > 1:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        try:
            self.last_gen_attention_mask = token_keep_mask.clone().detach().cpu()
            self.last_gen_kept_indices = (
                token_keep_mask[0].nonzero(as_tuple=False).squeeze(-1).detach().cpu().numpy()
            )
        except Exception:
            self.last_gen_attention_mask = None
            self.last_gen_kept_indices = None

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )