import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from .himap_configuration_llama import LlamaConfig
from .himap_modeling_llama import LlamaModel


logger = logging.get_logger(__name__)


class ThreeStage_LlamaModel(LlamaModel):
	"""LLaMA with configurable three-stage image-token pruning."""

	def __init__(self, config: LlamaConfig):
		super().__init__(config)
		self.config = config
		self.reset_three_stage_pruning()

		# Debug metadata from the latest forward.
		self.last_stage1_keep_indices = None
		self.last_stage2_keep_indices = None
		self.last_stage3_keep_indices = None
		self.last_three_stage_metadata = {}

	def reset_three_stage_pruning(self):
		"""Reset three-stage pruning hyperparameters from config."""
		self.use_three_stage_pruning = getattr(self.config, "use_three_stage_pruning", False)
		self.three_stage_sys_length = getattr(self.config, "three_stage_sys_length", None)
		self.three_stage_image_token_length = getattr(self.config, "three_stage_image_token_length", None)

		# Stage layer ids are configured as 1-based, converted to 0-based for runtime.
		self.stage1_layer = self._to_runtime_layer_idx(getattr(self.config, "stage1_layer", 3))
		self.stage2_layer = self._to_runtime_layer_idx(getattr(self.config, "stage2_layer", 12))
		self.stage3_layer = self._to_runtime_layer_idx(getattr(self.config, "stage3_layer", 20))

		self.stage1_similarity_threshold = getattr(self.config, "stage1_similarity_threshold", 0.92)
		self.stage1_rope_penalty_alpha = getattr(self.config, "stage1_rope_penalty_alpha", 0.08)
		self.stage1_window_size = getattr(self.config, "stage1_window_size", 3)
		self.stage1_min_keep_tokens = getattr(self.config, "stage1_min_keep_tokens", 64)
		self.stage2_keep_tokens = getattr(self.config, "stage2_keep_tokens", 128)

	@staticmethod
	def _to_runtime_layer_idx(layer_idx: Optional[int]) -> int:
		if layer_idx is None:
			return -1
		layer_idx = int(layer_idx)
		if layer_idx <= 0:
			return -1
		return layer_idx - 1

	@staticmethod
	def _prepare_pruned_attention_mask(
		attention_mask_fn,
		batch_size: int,
		seq_length: int,
		inputs_embeds: torch.Tensor,
	) -> torch.Tensor:
		return attention_mask_fn(None, (batch_size, seq_length), inputs_embeds, 0)

	@staticmethod
	def _infer_grid_shape(num_tokens: int) -> Tuple[int, int]:
		if num_tokens <= 0:
			return 0, 0

		side = int(math.sqrt(num_tokens))
		if side * side == num_tokens:
			return side, side

		best_h = side
		while best_h > 1:
			if num_tokens % best_h == 0:
				return best_h, num_tokens // best_h
			best_h -= 1

		h = max(1, side)
		w = int(math.ceil(num_tokens / h))
		return h, w

	def _build_local_window_mask(self, num_tokens: int, device: torch.device) -> torch.Tensor:
		"""Build a 3x3 neighborhood mask over flattened 2D patch tokens."""
		h, w = self._infer_grid_shape(num_tokens)
		if h == 0 or w == 0:
			return torch.zeros((0, 0), dtype=torch.bool, device=device)

		indices = torch.arange(num_tokens, device=device)
		rows = torch.div(indices, w, rounding_mode="floor")
		cols = torch.remainder(indices, w)

		row_dist = (rows[:, None] - rows[None, :]).abs()
		col_dist = (cols[:, None] - cols[None, :]).abs()

		# 3x3 window => Chebyshev distance <= 1.
		mask = (row_dist <= 1) & (col_dist <= 1)
		mask.fill_diagonal_(False)
		return mask

	def _compute_current_image_len(
		self,
		current_seq_len: int,
		sys_len: int,
		fixed_text_tokens: int,
	) -> int:
		cur_img_len = current_seq_len - sys_len - fixed_text_tokens
		return max(cur_img_len, 0)

	def _select_stage1_indices(
		self,
		hidden_states: torch.Tensor,
		sys_len: int,
		img_len: int,
		similarity_threshold: float,
		rope_penalty_alpha: float,
		min_keep_tokens: int,
	) -> torch.Tensor:
		"""
		Stage 1 pruning by cosine similarity + position distance penalty in local 3x3 window.
		Greedy keep-first strategy is used to prune near-duplicate tokens.
		"""
		device = hidden_states.device
		if img_len <= 0:
			return torch.tensor([], dtype=torch.long, device=device)

		img_tokens = hidden_states[:, sys_len : sys_len + img_len, :].mean(dim=0)
		img_tokens = F.normalize(img_tokens, dim=-1, eps=1e-6)

		cosine_sim = torch.matmul(img_tokens, img_tokens.transpose(0, 1))
		ids = torch.arange(img_len, device=device)
		pos_dist = (ids[:, None] - ids[None, :]).abs().float()
		if img_len > 1:
			pos_dist = pos_dist / float(img_len - 1)
		penalized_sim = cosine_sim - rope_penalty_alpha * pos_dist

		local_window_mask = self._build_local_window_mask(img_len, device=device)
		keep_mask = torch.ones(img_len, dtype=torch.bool, device=device)

		for token_idx in range(img_len):
			if not keep_mask[token_idx]:
				continue
			prune_candidates = (
				keep_mask
				& local_window_mask[token_idx]
				& (penalized_sim[token_idx] > similarity_threshold)
			)
			keep_mask[prune_candidates] = False

		min_keep = max(1, min(min_keep_tokens, img_len))
		if int(keep_mask.sum().item()) < min_keep:
			# Recover strongest tokens by norm if pruning is too aggressive.
			scores = hidden_states[:, sys_len : sys_len + img_len, :].mean(dim=0).norm(dim=-1)
			topk_idx = scores.topk(min_keep).indices
			keep_mask[topk_idx] = True

		keep_local_idx = keep_mask.nonzero(as_tuple=False).squeeze(-1)
		return keep_local_idx

	def _apply_keep_indices(
		self,
		hidden_states: torch.Tensor,
		position_ids: torch.Tensor,
		keep_indices: torch.Tensor,
		batch_size: int,
		inputs_embeds: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		keep_indices = keep_indices.sort().values
		hidden_states = hidden_states[:, keep_indices, :]
		position_ids = position_ids[:, keep_indices]
		new_attention_mask = self._prepare_pruned_attention_mask(
			self._prepare_decoder_attention_mask,
			batch_size,
			hidden_states.shape[1],
			inputs_embeds,
		)
		return hidden_states, position_ids, new_attention_mask

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
		if input_ids is not None:
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
			attention_mask = torch.ones(
				(batch_size, seq_length_with_past),
				dtype=torch.bool,
				device=inputs_embeds.device,
			)

		attention_mask = self._prepare_decoder_attention_mask(
			attention_mask,
			(batch_size, seq_length),
			inputs_embeds,
			past_key_values_length,
		)

		hidden_states = inputs_embeds

		if self.gradient_checkpointing and self.training and use_cache:
			logger.warning_once(
				"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
			)
			use_cache = False

		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		next_decoder_cache = () if use_cache else None

		# Run pruning only on prefill pass; generation with KV-cache should stay untouched.
		enable_three_stage = bool(self.use_three_stage_pruning) and (past_key_values is None)

		sys_len = self.three_stage_sys_length if self.three_stage_sys_length is not None else 0
		cfg_img_len = self.three_stage_image_token_length if self.three_stage_image_token_length is not None else 0
		fixed_text_tokens = seq_length_with_past - sys_len - cfg_img_len
		if fixed_text_tokens < 0:
			fixed_text_tokens = 0

		prev_layer_attention = None
		self.last_three_stage_metadata = {}

		for idx, decoder_layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			past_key_value = past_key_values[idx] if past_key_values is not None else None

			current_seq_len = hidden_states.shape[1]
			current_img_len = self._compute_current_image_len(current_seq_len, sys_len, fixed_text_tokens)

			if enable_three_stage and idx == self.stage1_layer and current_img_len > 0:
				stage1_keep_local = self._select_stage1_indices(
					hidden_states=hidden_states,
					sys_len=sys_len,
					img_len=current_img_len,
					similarity_threshold=float(self.stage1_similarity_threshold),
					rope_penalty_alpha=float(self.stage1_rope_penalty_alpha),
					min_keep_tokens=int(self.stage1_min_keep_tokens),
				)
				stage1_keep_global = stage1_keep_local + sys_len
				keep_indices = torch.cat(
					(
						torch.arange(sys_len, device=hidden_states.device),
						stage1_keep_global,
						torch.arange(sys_len + current_img_len, current_seq_len, device=hidden_states.device),
					)
				)
				hidden_states, position_ids, attention_mask = self._apply_keep_indices(
					hidden_states,
					position_ids,
					keep_indices,
					batch_size,
					inputs_embeds,
				)
				self.last_stage1_keep_indices = stage1_keep_global.detach().cpu()
				self.last_three_stage_metadata["stage1_kept_image_tokens"] = int(stage1_keep_local.numel())

				current_seq_len = hidden_states.shape[1]
				current_img_len = self._compute_current_image_len(current_seq_len, sys_len, fixed_text_tokens)

			if enable_three_stage and idx == self.stage2_layer and current_img_len > 0 and prev_layer_attention is not None:
				avg_attn = prev_layer_attention.mean(dim=1)
				text_start = sys_len + current_img_len
				text_to_img = avg_attn[:, text_start:, sys_len : sys_len + current_img_len]
				img_scores = text_to_img.sum(dim=1).mean(dim=0)

				topk = min(int(self.stage2_keep_tokens), current_img_len)
				if topk > 0:
					stage2_keep_local = img_scores.topk(topk).indices
				else:
					stage2_keep_local = torch.tensor([], dtype=torch.long, device=hidden_states.device)

				stage2_keep_global = stage2_keep_local + sys_len
				keep_indices = torch.cat(
					(
						torch.arange(sys_len, device=hidden_states.device),
						stage2_keep_global,
						torch.arange(sys_len + current_img_len, current_seq_len, device=hidden_states.device),
					)
				)
				hidden_states, position_ids, attention_mask = self._apply_keep_indices(
					hidden_states,
					position_ids,
					keep_indices,
					batch_size,
					inputs_embeds,
				)
				self.last_stage2_keep_indices = stage2_keep_global.detach().cpu()
				self.last_three_stage_metadata["stage2_kept_image_tokens"] = int(stage2_keep_local.numel())

				current_seq_len = hidden_states.shape[1]
				current_img_len = self._compute_current_image_len(current_seq_len, sys_len, fixed_text_tokens)

			if enable_three_stage and idx == self.stage3_layer and current_img_len > 0:
				keep_indices = torch.cat(
					(
						torch.arange(sys_len, device=hidden_states.device),
						torch.arange(sys_len + current_img_len, current_seq_len, device=hidden_states.device),
					)
				)
				hidden_states, position_ids, attention_mask = self._apply_keep_indices(
					hidden_states,
					position_ids,
					keep_indices,
					batch_size,
					inputs_embeds,
				)
				self.last_stage3_keep_indices = keep_indices.detach().cpu()
				self.last_three_stage_metadata["stage3_removed_all_image_tokens"] = True

			need_attn_this_layer = output_attentions
			# Stage-2 uses attention from layer (stage2_layer - 1).
			if enable_three_stage and self.stage2_layer > 0 and idx == self.stage2_layer - 1:
				need_attn_this_layer = True

			layer_outputs = decoder_layer(
				hidden_states,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_value=past_key_value,
				output_attentions=need_attn_this_layer,
				use_cache=use_cache,
			)

			hidden_states = layer_outputs[0]
			prev_layer_attention = layer_outputs[1] if need_attn_this_layer else None

			if use_cache:
				next_decoder_cache += (layer_outputs[2 if need_attn_this_layer else 1],)

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

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
