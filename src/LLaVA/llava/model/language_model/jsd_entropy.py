import math
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from .fastv import Fastv_LlamaModel

logger = logging.get_logger(__name__)


class JSDEntropy_LlamaModel(Fastv_LlamaModel):
	"""Adaptive 3-stage image-token pruning with dynamic budgeted Phase-1/2 and full Phase-3 drop."""

	def __init__(self, config):
		super().__init__(config)
		self.use_jsd_entropy_pruning = getattr(config, "use_jsd_entropy_pruning", False)
		self.jsd_entropy_sys_length = getattr(config, "jsd_entropy_sys_length", None)
		self.jsd_entropy_image_token_length = getattr(config, "jsd_entropy_image_token_length", None)

		self.jsd_entropy_n0 = int(getattr(config, "jsd_entropy_n0", 576))
		self.jsd_entropy_target_tokens = int(
			getattr(config, "jsd_entropy_target_tokens", self.jsd_entropy_image_token_length or self.jsd_entropy_n0)
		)

		self.jsd_entropy_phase1_prune_layer = max(0, int(getattr(config, "jsd_entropy_phase1_prune_layer", 3)) - 1)
		self.jsd_entropy_phase2_prune_layer = max(0, int(getattr(config, "jsd_entropy_phase2_prune_layer", 8)) - 1)
		self.jsd_entropy_phase3_prune_layer = max(0, int(getattr(config, "jsd_entropy_phase3_prune_layer", 16)) - 1)

		self.jsd_entropy_alpha = float(getattr(config, "jsd_entropy_alpha", 0.0))
		self.jsd_entropy_beta = float(getattr(config, "jsd_entropy_beta", 0.0))
		self.jsd_entropy_n_base_192 = float(getattr(config, "jsd_entropy_n_base_192", 250.0))
		self.jsd_entropy_n_base_128 = float(getattr(config, "jsd_entropy_n_base_128", 200.0))
		self.jsd_entropy_n_base_64 = float(getattr(config, "jsd_entropy_n_base_64", 60.0))
		self.jsd_entropy_mu_h = float(getattr(config, "jsd_entropy_mu_h", 0.620257))
		self.jsd_entropy_sigma_h = float(getattr(config, "jsd_entropy_sigma_h", 0.030169))
		self.jsd_entropy_mu_w = float(getattr(config, "jsd_entropy_mu_w", 0.667733))
		self.jsd_entropy_sigma_w = float(getattr(config, "jsd_entropy_sigma_w", 0.038618))
		self._apply_phase1_alpha_beta_by_target()

		self.jsd_entropy_w1 = float(getattr(config, "jsd_entropy_w1", 1.0))
		self.jsd_entropy_w2 = float(getattr(config, "jsd_entropy_w2", 1.0))
		self.jsd_entropy_w3 = float(getattr(config, "jsd_entropy_w3", 1.0))

		self.jsd_entropy_lambda1 = float(getattr(config, "jsd_entropy_lambda1", 1.0))
		self.jsd_entropy_lambda2 = float(getattr(config, "jsd_entropy_lambda2", 1.0))
		self.jsd_entropy_lambda3 = float(getattr(config, "jsd_entropy_lambda3", 1.0))

		self.jsd_entropy_grid_h = int(getattr(config, "jsd_entropy_grid_h", 24))
		self.jsd_entropy_grid_w = int(getattr(config, "jsd_entropy_grid_w", 24))

		self.jsd_entropy_topk_attention_mode = self._normalize_attention_mode(
			getattr(config, "jsd_entropy_topk_attention_mode", "prompt_image")
		)
		self.jsd_entropy_use_dynamic_boundaries = bool(
			getattr(config, "jsd_entropy_use_dynamic_boundaries", False)
		)
		self.jsd_entropy_abrupt_stage_ranges = str(
			getattr(config, "jsd_entropy_abrupt_stage_ranges", "2-4,5-15,16-20")
		)

		self.jsd_entropy_use_only_prompt2image_scoring = getattr(config, "jsd_entropy_use_only_prompt2image_scoring", True)
		self.jsd_entropy_use_adaptive_keep_ratio = getattr(config, "jsd_entropy_use_adaptive_keep_ratio", True)

		self.jsd_entropy_phase1_keep = None
		self.jsd_entropy_phase2_keep = None
		self.jsd_entropy_stage_layers = [
			self.jsd_entropy_phase1_prune_layer,
			self.jsd_entropy_phase2_prune_layer,
			self.jsd_entropy_phase3_prune_layer,
		]
		self.jsd_entropy_stage_keep_counts = []
		self.jsd_entropy_stage_scores = []
		self._jsd_entropy_sample_counter = 0
		self._jsd_entropy_stage_layers_locked = False

	def reset_jsd_entropy_pruning(self):
		self.use_jsd_entropy_pruning = getattr(self.config, "use_jsd_entropy_pruning", False)
		self.jsd_entropy_sys_length = getattr(self.config, "jsd_entropy_sys_length", None)
		self.jsd_entropy_image_token_length = getattr(self.config, "jsd_entropy_image_token_length", None)

		self.jsd_entropy_n0 = int(getattr(self.config, "jsd_entropy_n0", 576))
		self.jsd_entropy_target_tokens = int(
			getattr(self.config, "jsd_entropy_target_tokens", self.jsd_entropy_image_token_length or self.jsd_entropy_n0)
		)

		self.jsd_entropy_phase1_prune_layer = max(0, int(getattr(self.config, "jsd_entropy_phase1_prune_layer", 3)) - 1)
		self.jsd_entropy_phase2_prune_layer = max(0, int(getattr(self.config, "jsd_entropy_phase2_prune_layer", 8)) - 1)
		self.jsd_entropy_phase3_prune_layer = max(0, int(getattr(self.config, "jsd_entropy_phase3_prune_layer", 16)) - 1)

		self.jsd_entropy_alpha = float(getattr(self.config, "jsd_entropy_alpha", 0.0))
		self.jsd_entropy_beta = float(getattr(self.config, "jsd_entropy_beta", 0.0))
		self.jsd_entropy_n_base_192 = float(getattr(self.config, "jsd_entropy_n_base_192", 250.0))
		self.jsd_entropy_n_base_128 = float(getattr(self.config, "jsd_entropy_n_base_128", 200.0))
		self.jsd_entropy_n_base_64 = float(getattr(self.config, "jsd_entropy_n_base_64", 60.0))
		self.jsd_entropy_mu_h = float(getattr(self.config, "jsd_entropy_mu_h", 0.620257))
		self.jsd_entropy_sigma_h = float(getattr(self.config, "jsd_entropy_sigma_h", 0.030169))
		self.jsd_entropy_mu_w = float(getattr(self.config, "jsd_entropy_mu_w", 0.667733))
		self.jsd_entropy_sigma_w = float(getattr(self.config, "jsd_entropy_sigma_w", 0.038618))
		self._apply_phase1_alpha_beta_by_target()

		self.jsd_entropy_w1 = float(getattr(self.config, "jsd_entropy_w1", 1.0))
		self.jsd_entropy_w2 = float(getattr(self.config, "jsd_entropy_w2", 1.0))
		self.jsd_entropy_w3 = float(getattr(self.config, "jsd_entropy_w3", 1.0))

		self.jsd_entropy_lambda1 = float(getattr(self.config, "jsd_entropy_lambda1", 1.0))
		self.jsd_entropy_lambda2 = float(getattr(self.config, "jsd_entropy_lambda2", 1.0))
		self.jsd_entropy_lambda3 = float(getattr(self.config, "jsd_entropy_lambda3", 1.0))

		self.jsd_entropy_grid_h = int(getattr(self.config, "jsd_entropy_grid_h", 24))
		self.jsd_entropy_grid_w = int(getattr(self.config, "jsd_entropy_grid_w", 24))

		self.jsd_entropy_topk_attention_mode = self._normalize_attention_mode(
			getattr(self.config, "jsd_entropy_topk_attention_mode", "prompt_image")
		)
		self.jsd_entropy_use_dynamic_boundaries = bool(
			getattr(self.config, "jsd_entropy_use_dynamic_boundaries", False)
		)
		self.jsd_entropy_abrupt_stage_ranges = str(
			getattr(self.config, "jsd_entropy_abrupt_stage_ranges", "2-4,5-15,16-20")
		)
		
		self.jsd_entropy_use_only_prompt2image_scoring = getattr(self.config, "jsd_entropy_use_only_prompt2image_scoring", True)
		self.jsd_entropy_use_adaptive_keep_ratio = getattr(self.config, "jsd_entropy_use_adaptive_keep_ratio", True)

		self.jsd_entropy_phase1_keep = None
		self.jsd_entropy_phase2_keep = None
		self.jsd_entropy_stage_layers = [
			self.jsd_entropy_phase1_prune_layer,
			self.jsd_entropy_phase2_prune_layer,
			self.jsd_entropy_phase3_prune_layer,
		]
		self.jsd_entropy_stage_keep_counts = []
		self.jsd_entropy_stage_scores = []
		self._jsd_entropy_stage_layers_locked = False

	def _safe_prob_from_vector(self, vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
		vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
		vec = vec.clamp_min(0.0)
		s = vec.sum()
		if s.item() <= 0:
			return torch.full_like(vec, 1.0 / max(vec.numel(), 1))
		return (vec / s).clamp_min(eps)

	def _normalized_jsd(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
		p = self._safe_prob_from_vector(p, eps=eps)
		q = self._safe_prob_from_vector(q, eps=eps)
		m = 0.5 * (p + q)
		kl_pm = torch.sum(p * torch.log((p / m).clamp_min(eps)))
		kl_qm = torch.sum(q * torch.log((q / m).clamp_min(eps)))
		jsd = 0.5 * (kl_pm + kl_qm)
		return float((jsd / math.log(2.0)).item())

	def _min_max_norm(self, values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
		if values.numel() == 0:
			return values
		values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
		v_min = values.min()
		v_max = values.max()
		denom = (v_max - v_min).abs()
		if denom.item() <= eps:
			return torch.zeros_like(values)
		return (values - v_min) / denom

	def _safe_zscore(self, value: float, mean: float, std: float) -> float:
		if abs(std) <= 1e-12:
			return 0.0
		return (float(value) - float(mean)) / float(std)

	def _safe_entropy_from_attention_vector(self, vec: torch.Tensor, eps: float = 1e-12) -> float:
		if vec.numel() <= 1:
			return 0.0
		p = self._safe_prob_from_vector(vec.float(), eps=eps)
		n = p.numel()
		if n <= 1:
			return 0.0
		ent = -(p * torch.log(p.clamp_min(eps))).sum() / math.log(float(n))
		return float(torch.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0).item())

	def _bidirectional_change_tensor(self, values: torch.Tensor) -> torch.Tensor:
		if values.numel() <= 1:
			return torch.zeros_like(values)
		left = torch.zeros_like(values)
		right = torch.zeros_like(values)
		diff = (values[1:] - values[:-1]).abs()
		left[1:] = diff
		right[:-1] = diff
		return 0.5 * (left + right)

	def _max_norm_tensor(self, values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
		if values.numel() == 0:
			return values
		m = torch.max(values)
		if float(m.item()) <= eps:
			return torch.zeros_like(values)
		return values / m

	def _parse_stage_ranges_from_config(self, total_layers: int) -> List[Tuple[int, int]]:
		raw = str(self.jsd_entropy_abrupt_stage_ranges or "").strip()
		ranges: List[Tuple[int, int]] = []
		if raw:
			for seg in raw.split(","):
				seg = seg.strip()
				if not seg:
					continue
				if "-" in seg:
					lo, hi = seg.split("-", 1)
				elif ":" in seg:
					lo, hi = seg.split(":", 1)
				else:
					lo, hi = seg, seg
				try:
					lo_i = int(lo)
					hi_i = int(hi)
				except ValueError:
					continue
				if lo_i > hi_i:
					lo_i, hi_i = hi_i, lo_i
				lo_i = max(0, min(lo_i, max(total_layers - 1, 0)))
				hi_i = max(0, min(hi_i, max(total_layers - 1, 0)))
				if lo_i <= hi_i:
					ranges.append((lo_i, hi_i))
		return ranges

	def _select_dynamic_stage_layers_from_vectors(
		self,
		layer_img_attn_vectors: List[torch.Tensor],
		total_layers: int,
	) -> Optional[Tuple[int, int, int]]:
		if len(layer_img_attn_vectors) <= 1 or total_layers <= 0:
			return None

		entropy_vals = []
		jsd_vals = [0.0]
		for i, vec in enumerate(layer_img_attn_vectors):
			entropy_vals.append(self._safe_entropy_from_attention_vector(vec))
			if i > 0:
				jsd_vals.append(self._normalized_jsd(vec, layer_img_attn_vectors[i - 1]))

		device = layer_img_attn_vectors[0].device
		entropy_t = torch.tensor(entropy_vals, dtype=torch.float32, device=device)
		jsd_t = torch.tensor(jsd_vals, dtype=torch.float32, device=device)

		entropy_change = self._bidirectional_change_tensor(torch.nan_to_num(entropy_t, nan=0.0, posinf=0.0, neginf=0.0))
		jsd_change = self._bidirectional_change_tensor(torch.nan_to_num(jsd_t, nan=0.0, posinf=0.0, neginf=0.0))
		combined_change = 0.5 * self._max_norm_tensor(entropy_change) + 0.5 * self._max_norm_tensor(jsd_change)

		ranges = self._parse_stage_ranges_from_config(total_layers=combined_change.numel())
		if len(ranges) == 0:
			return None

		abrupt_nodes: List[int] = []
		for lo, hi in ranges:
			seg = combined_change[lo:hi + 1]
			if seg.numel() <= 0:
				continue
			best_local = int(torch.argmax(seg).item())
			abrupt_nodes.append(int(lo + best_local))

		if len(abrupt_nodes) == 0:
			return None

		unique_nodes = sorted(set(abrupt_nodes))
		phase1 = unique_nodes[0]
		phase2 = unique_nodes[1] if len(unique_nodes) > 1 else max(phase1 + 1, self.jsd_entropy_phase2_prune_layer)
		phase3 = unique_nodes[2] if len(unique_nodes) > 2 else max(phase2 + 1, self.jsd_entropy_phase3_prune_layer)

		phase1 = max(0, min(phase1, total_layers - 1))
		phase2 = max(phase1 + 1, min(phase2, total_layers - 1))
		phase3 = max(phase2 + 1, min(phase3, total_layers - 1))
		return int(phase1), int(phase2), int(phase3)

	def _collect_prefill_image_attention_vectors(
		self,
		hidden_states: torch.Tensor,
		attention_mask: torch.Tensor,
		position_ids: torch.Tensor,
		sys_length: int,
		img_length: int,
	) -> List[torch.Tensor]:
		if img_length <= 0:
			return []

		probe_hidden = hidden_states
		vectors: List[torch.Tensor] = []
		for decoder_layer in self.layers:
			layer_outputs = decoder_layer(
				probe_hidden,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_value=None,
				output_attentions=True,
				use_cache=False,
			)
			probe_hidden = layer_outputs[0]
			layer_attn = layer_outputs[1] if len(layer_outputs) > 1 else None
			img_scores = self._extract_image_attention_scores(
				attn=layer_attn,
				sys_length=sys_length,
				img_length=img_length,
			).float()
			vectors.append(img_scores.detach())
		return vectors

	def _maybe_update_dynamic_stage_layers(
		self,
		hidden_states: torch.Tensor,
		attention_mask: torch.Tensor,
		position_ids: torch.Tensor,
		sys_length: int,
		img_length: int,
	):
		if not bool(self.jsd_entropy_use_dynamic_boundaries):
			return
		if bool(self._jsd_entropy_stage_layers_locked):
			return

		if not hasattr(self, "_jsd_dynamic_layer_accum"):
			self._jsd_dynamic_layer_accum = []

		layer_vectors = self._collect_prefill_image_attention_vectors(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			position_ids=position_ids,
			sys_length=sys_length,
			img_length=img_length,
		)
		dynamic_layers = self._select_dynamic_stage_layers_from_vectors(
			layer_img_attn_vectors=layer_vectors,
			total_layers=len(self.layers),
		)
		if dynamic_layers is None:
			return

		self._jsd_dynamic_layer_accum.append(dynamic_layers)

		avg_p1 = int(round(sum(x[0] for x in self._jsd_dynamic_layer_accum) / len(self._jsd_dynamic_layer_accum)))
		avg_p2 = int(round(sum(x[1] for x in self._jsd_dynamic_layer_accum) / len(self._jsd_dynamic_layer_accum)))
		avg_p3 = int(round(sum(x[2] for x in self._jsd_dynamic_layer_accum) / len(self._jsd_dynamic_layer_accum)))

		self.jsd_entropy_phase1_prune_layer = avg_p1
		self.jsd_entropy_phase2_prune_layer = avg_p2
		self.jsd_entropy_phase3_prune_layer = avg_p3
		self.jsd_entropy_stage_layers = [
			self.jsd_entropy_phase1_prune_layer,
			self.jsd_entropy_phase2_prune_layer,
			self.jsd_entropy_phase3_prune_layer,
		]

		if len(self._jsd_dynamic_layer_accum) >= 50:
			self._jsd_entropy_stage_layers_locked = True
			logger.info(f"Locked dynamic stage layers after 50 samples: {self.jsd_entropy_stage_layers}")

	def _log_sample_pruning_layers(self):
		phase1 = int(self.jsd_entropy_phase1_prune_layer)
		phase2 = int(self.jsd_entropy_phase2_prune_layer)
		phase3 = int(self.jsd_entropy_phase3_prune_layer)
		self._jsd_entropy_sample_counter += 1
		print(
			f"[JSD-Entropy] sample={self._jsd_entropy_sample_counter} "
			f"prune_layers_0based=(p1:{phase1}, p2:{phase2}, p3:{phase3}) "
			f"prune_layers_1based=(p1:{phase1 + 1}, p2:{phase2 + 1}, p3:{phase3 + 1})"
		)

	def _apply_phase1_alpha_beta_by_target(self):
		n_target = max(0, int(self.jsd_entropy_target_tokens))
		if n_target in {128, 192}:
			self.jsd_entropy_alpha = 24.0
			self.jsd_entropy_beta = 16.0
		elif n_target == 64:
			self.jsd_entropy_alpha = 9.0
			self.jsd_entropy_beta = 5.0

	def _normalize_attention_mode(self, mode) -> str:
		if not isinstance(mode, str):
			return "prompt_image"
		normalized = mode.strip().lower().replace("-", "_")
		if normalized in {"global", "all", "all_tokens", "global_image"}:
			return "global"
		return "prompt_image"

	def _extract_global_to_image_attention(self, attn, sys_length: int, img_length: int) -> torch.Tensor:
		if img_length <= 0:
			return torch.zeros((0,), dtype=torch.float32)

		if isinstance(attn, (tuple, list)):
			attn = attn[0] if len(attn) > 0 else None

		if attn is None:
			return torch.zeros((img_length,), dtype=torch.float32)

		attn_avg = torch.mean(attn, dim=1)[0]
		seq_len_k = attn_avg.shape[-1]
		img_span_end = min(sys_length + img_length, seq_len_k)
		local_img_len = max(img_span_end - sys_length, 0)
		image_scores = torch.zeros((img_length,), dtype=attn_avg.dtype, device=attn_avg.device)
		if local_img_len <= 0:
			return image_scores

		global_to_img = attn_avg[:, sys_length:sys_length + local_img_len]
		if global_to_img.numel() == 0:
			return image_scores

		global_scores = torch.mean(global_to_img, dim=0)
		image_scores[:local_img_len] = torch.nan_to_num(
			global_scores,
			nan=0.0,
			posinf=0.0,
			neginf=0.0,
		)
		return image_scores

	def _extract_prompt_to_image_attention(self, attn, sys_length: int, img_length: int) -> torch.Tensor:
		if img_length <= 0:
			return torch.zeros((0,), dtype=torch.float32)

		if isinstance(attn, (tuple, list)):
			attn = attn[0] if len(attn) > 0 else None

		if attn is None:
			return torch.zeros((img_length,), dtype=torch.float32)

		attn_avg = torch.mean(attn, dim=1)[0]
		seq_len_q = attn_avg.shape[-2]
		seq_len_k = attn_avg.shape[-1]

		img_span_end = min(sys_length + img_length, seq_len_k)
		local_img_len = max(img_span_end - sys_length, 0)
		image_scores = torch.zeros((img_length,), dtype=attn_avg.dtype, device=attn_avg.device)
		if local_img_len <= 0:
			return image_scores

		prompt_start = min(sys_length + img_length, seq_len_q)
		if prompt_start < seq_len_q:
			prompt_to_img = attn_avg[prompt_start:seq_len_q, sys_length:sys_length + local_img_len]
			if prompt_to_img.numel() > 0:
				prompt_scores = torch.mean(prompt_to_img, dim=0)
				image_scores[:local_img_len] = torch.nan_to_num(
					prompt_scores,
					nan=0.0,
					posinf=0.0,
					neginf=0.0,
				)
				return image_scores

		last_token_scores = attn_avg[-1, sys_length:sys_length + local_img_len]
		image_scores[:local_img_len] = torch.nan_to_num(
			last_token_scores,
			nan=0.0,
			posinf=0.0,
			neginf=0.0,
		)
		return image_scores

	def _extract_image_attention_scores(self, attn, sys_length: int, img_length: int) -> torch.Tensor:
		if self.jsd_entropy_topk_attention_mode == "global":
			return self._extract_global_to_image_attention(
				attn=attn,
				sys_length=sys_length,
				img_length=img_length,
			)
		return self._extract_prompt_to_image_attention(
			attn=attn,
			sys_length=sys_length,
			img_length=img_length,
		)

	def _extract_image_hidden_states(
		self,
		hidden_states: torch.Tensor,
		sys_length: int,
		num_text_tokens: int,
	) -> torch.Tensor:
		seq_len = hidden_states.shape[1]
		img_start = min(sys_length, seq_len)
		img_end = max(img_start, seq_len - num_text_tokens)
		if img_end <= img_start:
			return hidden_states.new_zeros((0, hidden_states.shape[-1]))
		return hidden_states[0, img_start:img_end, :]

	def _extract_visual_to_visual_entropy(self, attn, sys_length: int, img_length: int) -> torch.Tensor:
		if img_length <= 0:
			return torch.zeros((0,), dtype=torch.float32)

		if isinstance(attn, (tuple, list)):
			attn = attn[0] if len(attn) > 0 else None

		if attn is None:
			return torch.zeros((img_length,), dtype=torch.float32)

		attn_avg = torch.mean(attn, dim=1)[0]
		seq_len_q = attn_avg.shape[-2]
		seq_len_k = attn_avg.shape[-1]
		img_span_end_q = min(sys_length + img_length, seq_len_q)
		img_span_end_k = min(sys_length + img_length, seq_len_k)
		local_img_len = max(min(img_span_end_q - sys_length, img_span_end_k - sys_length), 0)
		if local_img_len <= 0:
			return torch.zeros((img_length,), dtype=attn_avg.dtype, device=attn_avg.device)

		vv = attn_avg[sys_length:sys_length + local_img_len, sys_length:sys_length + local_img_len]
		row_sums = vv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
		probs = (vv / row_sums).clamp_min(1e-12)

		if local_img_len > 1:
			entropy = -(probs * torch.log(probs)).sum(dim=-1) / math.log(local_img_len)
		else:
			entropy = torch.zeros((local_img_len,), dtype=vv.dtype, device=vv.device)

		out = torch.zeros((img_length,), dtype=vv.dtype, device=vv.device)
		out[:local_img_len] = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
		return out

	def _compute_h_vis(self, image_hidden: torch.Tensor) -> float:
		n = image_hidden.shape[0]
		if n <= 1:
			return 0.0
		normed = image_hidden / image_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-12)
		sim = torch.matmul(normed, normed.transpose(0, 1))
		mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
		if torch.count_nonzero(mask).item() == 0:
			mean_sim = 1.0
		else:
			mean_sim = float(sim[mask].mean().item())
		h_vis = 1.0 - mean_sim
		return float(max(0.0, min(1.0, h_vis)))

	def _compute_w_prompt(self, attn, sys_length: int, img_length: int) -> float:
		if img_length <= 0:
			return 0.0
		a = self._extract_prompt_to_image_attention(attn=attn, sys_length=sys_length, img_length=img_length).float()
		a_prob = self._safe_prob_from_vector(a)
		u = torch.full_like(a_prob, 1.0 / max(img_length, 1))
		jsd = self._normalized_jsd(a_prob, u)
		return float(max(0.0, min(1.0, 1.0 - jsd)))

	def _derive_budget_lengths(self, total_layers: int) -> Tuple[int, int, int]:
		phase1 = max(0, min(self.jsd_entropy_phase1_prune_layer, total_layers))
		phase2 = max(phase1 + 1, min(self.jsd_entropy_phase2_prune_layer, total_layers))
		phase3 = max(phase2 + 1, min(self.jsd_entropy_phase3_prune_layer, total_layers))
		l0 = max(1, phase1)
		l1 = max(1, phase2 - phase1)
		l2 = max(1, phase3 - phase2)
		return l0, l1, l2

	def _compute_phase1_keep_count(self, n0: int, h_vis: float, w_prompt: float) -> int:
		n_target = max(0, int(self.jsd_entropy_target_tokens))
		if n_target == 192:
			n1_base = float(self.jsd_entropy_n_base_192)
		elif n_target == 128:
			n1_base = float(self.jsd_entropy_n_base_128)
		elif n_target == 64:
			n1_base = float(self.jsd_entropy_n_base_64)
		else:
			n1_base = float(n0)

		if not self.jsd_entropy_use_adaptive_keep_ratio:
			return int(max(1, min(int(n1_base), n0)))

		z_h = self._safe_zscore(h_vis, self.jsd_entropy_mu_h, self.jsd_entropy_sigma_h)
		z_w = self._safe_zscore(w_prompt, self.jsd_entropy_mu_w, self.jsd_entropy_sigma_w)
		n1_raw = n1_base + self.jsd_entropy_alpha * z_h + self.jsd_entropy_beta * z_w
		n1 = int(max(1, min(int(n1_raw), n0)))
		return n1

	def _compute_phase2_keep_count(self, n0: int, n1: int) -> int:
		l = len(self.layers)
		l0, l1, l2 = self._derive_budget_lengths(l)
		n_target = max(0, int(self.jsd_entropy_target_tokens))

		n2_raw = float(n_target * l - n0 * l0 - n1 * l1) / max(l2, 1e-12)
		n2 = int(max(0, min(n2_raw, n1)))
		return n2

	def _compute_local_topology_similarity(self, image_hidden: torch.Tensor) -> torch.Tensor:
		n = image_hidden.shape[0]
		if n <= 0:
			return image_hidden.new_zeros((0,))

		grid_h = max(1, int(self.jsd_entropy_grid_h))
		grid_w = max(1, int(self.jsd_entropy_grid_w))
		if grid_h * grid_w != n:
			approx = int(round(math.sqrt(n)))
			if approx * approx == n:
				grid_h = approx
				grid_w = approx
			else:
				return image_hidden.new_zeros((n,))

		normed = image_hidden / image_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-12)
		normed = normed.view(grid_h, grid_w, -1)
		result = image_hidden.new_zeros((grid_h, grid_w))

		for r in range(grid_h):
			for c in range(grid_w):
				neighbors = []
				for dr in (-1, 0, 1):
					for dc in (-1, 0, 1):
						if dr == 0 and dc == 0:
							continue
						rr = r + dr
						cc = c + dc
						if 0 <= rr < grid_h and 0 <= cc < grid_w:
							neighbors.append((rr, cc))
				if len(neighbors) == 0:
					continue
				center = normed[r, c]
				sims = [torch.dot(center, normed[rr, cc]) for rr, cc in neighbors]
				result[r, c] = torch.stack(sims).mean()

		return result.view(-1)

	def _select_topk_indices(self, scores: torch.Tensor, keep_count: int) -> torch.Tensor:
		n = scores.numel()
		if keep_count <= 0 or n == 0:
			return torch.tensor([], dtype=torch.long, device=scores.device)
		keep_count = max(0, min(int(keep_count), n))
		if keep_count == 0:
			return torch.tensor([], dtype=torch.long, device=scores.device)
		return torch.topk(scores, k=keep_count, largest=True).indices

	def _apply_pruned_indices(
		self,
		hidden_states: torch.Tensor,
		position_ids: torch.Tensor,
		inputs_embeds: torch.Tensor,
		batch_size: int,
		sys_length: int,
		num_text_tokens: int,
		selected_img_indices: torch.Tensor,
	):
		current_seq_len = hidden_states.shape[1]
		current_img_len = max(current_seq_len - sys_length - num_text_tokens, 0)

		selected_img_indices = selected_img_indices.long()
		if selected_img_indices.numel() > 0:
			selected_img_indices = selected_img_indices.clamp_min(0).clamp_max(max(current_img_len - 1, 0)).unique()

		global_img_indices = selected_img_indices + sys_length
		text_start = sys_length + current_img_len
		text_indices = (
			torch.arange(text_start, current_seq_len, device=hidden_states.device)
			if text_start < current_seq_len
			else torch.tensor([], dtype=torch.long, device=hidden_states.device)
		)
		keep_indices = torch.cat(
			(
				torch.arange(min(sys_length, current_seq_len), device=hidden_states.device),
				global_img_indices,
				text_indices,
			)
		).sort().values

		hidden_states = hidden_states[:, keep_indices, :]
		base_pos = position_ids.squeeze(0)
		position_ids = base_pos[keep_indices].unsqueeze(0)
		new_seq_len = keep_indices.shape[0]
		new_attention_mask = self._prepare_decoder_attention_mask(None, (batch_size, new_seq_len), inputs_embeds, 0)
		return hidden_states, position_ids, new_attention_mask

	def _phase1_prune(
		self,
		hidden_states: torch.Tensor,
		position_ids: torch.Tensor,
		inputs_embeds: torch.Tensor,
		batch_size: int,
		sys_length: int,
		num_text_tokens: int,
		prev_layer_attn,
	):
		current_seq_len = hidden_states.shape[1]
		current_img_len = max(current_seq_len - sys_length - num_text_tokens, 0)
		if current_img_len <= 0:
			return hidden_states, position_ids, self._prepare_decoder_attention_mask(
				None, (batch_size, current_seq_len), inputs_embeds, 0
			)

		image_hidden = self._extract_image_hidden_states(hidden_states, sys_length, num_text_tokens)
		if self.jsd_entropy_use_adaptive_keep_ratio:
			h_vis = self._compute_h_vis(image_hidden)
			w_prompt = self._compute_w_prompt(prev_layer_attn, sys_length, current_img_len)
		else:
			h_vis = 0.0
			w_prompt = 0.0
		n1 = self._compute_phase1_keep_count(n0=current_img_len, h_vis=h_vis, w_prompt=w_prompt)
		n1 = max(1, min(n1, current_img_len))
		if getattr(self, "jsd_entropy_use_adaptive_keep_ratio", True):
			print(f"[Phase 1 Adaptive] keep count: {n1} / {current_img_len}")

		a_i = self._extract_prompt_to_image_attention(prev_layer_attn, sys_length, current_img_len).float()
		
		if self.jsd_entropy_use_only_prompt2image_scoring:
			score_1 = a_i
		else:
			e_i = self._extract_visual_to_visual_entropy(prev_layer_attn, sys_length, current_img_len).float()
			sim_local_i = self._compute_local_topology_similarity(image_hidden).float()

			norm_a = self._min_max_norm(a_i)
			norm_e = self._min_max_norm(e_i)
			norm_sim_local = self._min_max_norm(sim_local_i)

			score_1 = self.jsd_entropy_w1 * norm_a + self.jsd_entropy_w2 * norm_e - self.jsd_entropy_w3 * norm_sim_local
			
		selected_img_indices = self._select_topk_indices(score_1, n1)

		self.jsd_entropy_phase1_keep = int(n1)
		self.jsd_entropy_stage_keep_counts = [int(n1)]
		self.jsd_entropy_stage_scores = [float(h_vis), float(w_prompt)]

		return self._apply_pruned_indices(
			hidden_states=hidden_states,
			position_ids=position_ids,
			inputs_embeds=inputs_embeds,
			batch_size=batch_size,
			sys_length=sys_length,
			num_text_tokens=num_text_tokens,
			selected_img_indices=selected_img_indices,
		)

	def _phase2_prune(
		self,
		hidden_states: torch.Tensor,
		position_ids: torch.Tensor,
		inputs_embeds: torch.Tensor,
		batch_size: int,
		sys_length: int,
		num_text_tokens: int,
		prev_layer_attn,
	):
		current_seq_len = hidden_states.shape[1]
		current_img_len = max(current_seq_len - sys_length - num_text_tokens, 0)
		if current_img_len <= 0:
			return hidden_states, position_ids, self._prepare_decoder_attention_mask(
				None, (batch_size, current_seq_len), inputs_embeds, 0
			)

		n1_for_budget = int(self.jsd_entropy_phase1_keep or current_img_len)
		n2 = self._compute_phase2_keep_count(n0=self.jsd_entropy_n0, n1=n1_for_budget)
		n2 = max(0, min(n2, n1_for_budget, current_img_len))
		if getattr(self, "jsd_entropy_use_adaptive_keep_ratio", True):
			print(f"[Phase 2 Adaptive] keep count: {n2} / {n1_for_budget}")

		a_i = self._extract_prompt_to_image_attention(prev_layer_attn, sys_length, current_img_len).float()
		
		if self.jsd_entropy_use_only_prompt2image_scoring:
			score_2 = a_i
		else:
			anchor_k = max(1, min(10, current_img_len))
			anchor_indices = torch.topk(a_i, k=anchor_k, largest=True).indices if current_img_len > 0 else torch.tensor([], dtype=torch.long, device=hidden_states.device)

			image_hidden = self._extract_image_hidden_states(hidden_states, sys_length, num_text_tokens)
			if anchor_indices.numel() > 0 and image_hidden.shape[0] > 0:
				normed = image_hidden / image_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-12)
				anchor_features = normed[anchor_indices]
				sim_matrix = torch.matmul(normed, anchor_features.transpose(0, 1))
				s_anchor = sim_matrix.max(dim=-1).values
			else:
				s_anchor = torch.zeros((current_img_len,), dtype=image_hidden.dtype, device=image_hidden.device)

			h_i = self._extract_visual_to_visual_entropy(prev_layer_attn, sys_length, current_img_len).float()

			norm_a = self._min_max_norm(a_i)
			norm_s_anchor = self._min_max_norm(s_anchor)
			norm_h = self._min_max_norm(h_i)

			score_2 = (
				self.jsd_entropy_lambda1 * norm_a
				+ self.jsd_entropy_lambda2 * norm_s_anchor
				+ self.jsd_entropy_lambda3 * norm_h
			)
			
		selected_img_indices = self._select_topk_indices(score_2, n2)

		self.jsd_entropy_phase2_keep = int(n2)
		if len(self.jsd_entropy_stage_keep_counts) == 0:
			self.jsd_entropy_stage_keep_counts = [int(n1_for_budget)]
		self.jsd_entropy_stage_keep_counts.append(int(n2))

		return self._apply_pruned_indices(
			hidden_states=hidden_states,
			position_ids=position_ids,
			inputs_embeds=inputs_embeds,
			batch_size=batch_size,
			sys_length=sys_length,
			num_text_tokens=num_text_tokens,
			selected_img_indices=selected_img_indices,
		)

	def _phase3_drop_all_images(
		self,
		hidden_states: torch.Tensor,
		position_ids: torch.Tensor,
		inputs_embeds: torch.Tensor,
		batch_size: int,
		sys_length: int,
		num_text_tokens: int,
	):
		current_seq_len = hidden_states.shape[1]
		current_img_len = max(current_seq_len - sys_length - num_text_tokens, 0)
		if current_img_len <= 0:
			return hidden_states, position_ids, self._prepare_decoder_attention_mask(
				None, (batch_size, current_seq_len), inputs_embeds, 0
			)
		selected_img_indices = torch.tensor([], dtype=torch.long, device=hidden_states.device)
		if len(self.jsd_entropy_stage_keep_counts) <= 2:
			self.jsd_entropy_stage_keep_counts.append(0)
		return self._apply_pruned_indices(
			hidden_states=hidden_states,
			position_ids=position_ids,
			inputs_embeds=inputs_embeds,
			batch_size=batch_size,
			sys_length=sys_length,
			num_text_tokens=num_text_tokens,
			selected_img_indices=selected_img_indices,
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
		if not bool(self.use_jsd_entropy_pruning):
			return super().forward(
				input_ids=input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_values=past_key_values,
				inputs_embeds=inputs_embeds,
				use_cache=use_cache,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)

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
			seq_length_with_past = seq_length_with_past + past_key_values_length

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

		if self.gradient_checkpointing and self.training:
			if use_cache:
				logger.warning_once(
					"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
				)
				use_cache = False

		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		next_decoder_cache = () if use_cache else None

		sys_len = int(self.jsd_entropy_sys_length or 0)
		img_len = int(self.jsd_entropy_image_token_length or 0)
		num_text_tokens = max(seq_length_with_past - sys_len - img_len, 0)

		if past_key_values is None:
			self.jsd_entropy_phase1_keep = None
			self.jsd_entropy_phase2_keep = None

			# Only re-evaluate dynamic boundaries if not yet locked
			if not bool(self._jsd_entropy_stage_layers_locked):
				self.jsd_entropy_stage_layers = [
					self.jsd_entropy_phase1_prune_layer,
					self.jsd_entropy_phase2_prune_layer,
					self.jsd_entropy_phase3_prune_layer,
				]
				self.jsd_entropy_stage_keep_counts = []
				self.jsd_entropy_stage_scores = []
				self._maybe_update_dynamic_stage_layers(
					hidden_states=hidden_states,
					attention_mask=attention_mask,
					position_ids=position_ids,
					sys_length=sys_len,
					img_length=img_len,
				)
				self._log_sample_pruning_layers()
			else:
				# Even if locked, we may still want to log the locked sample layer usage occasionally
				self._log_sample_pruning_layers()

		phase1_layer = int(self.jsd_entropy_phase1_prune_layer)
		phase2_layer = int(self.jsd_entropy_phase2_prune_layer)
		phase3_layer = int(self.jsd_entropy_phase3_prune_layer)
		need_prune_attention_layers = {phase1_layer - 1, phase2_layer - 1}

		prev_layer_attn = None
		for idx, decoder_layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			past_key_value = past_key_values[idx] if past_key_values is not None else None

			if past_key_values is None and idx == phase1_layer:
				hidden_states, position_ids, new_attention_mask = self._phase1_prune(
					hidden_states=hidden_states,
					position_ids=position_ids,
					inputs_embeds=inputs_embeds,
					batch_size=batch_size,
					sys_length=sys_len,
					num_text_tokens=num_text_tokens,
					prev_layer_attn=prev_layer_attn,
				)
			elif past_key_values is None and idx == phase2_layer:
				hidden_states, position_ids, new_attention_mask = self._phase2_prune(
					hidden_states=hidden_states,
					position_ids=position_ids,
					inputs_embeds=inputs_embeds,
					batch_size=batch_size,
					sys_length=sys_len,
					num_text_tokens=num_text_tokens,
					prev_layer_attn=prev_layer_attn,
				)
			elif past_key_values is None and idx == phase3_layer:
				hidden_states, position_ids, new_attention_mask = self._phase3_drop_all_images(
					hidden_states=hidden_states,
					position_ids=position_ids,
					inputs_embeds=inputs_embeds,
					batch_size=batch_size,
					sys_length=sys_len,
					num_text_tokens=num_text_tokens,
				)
			else:
				current_seq_len = hidden_states.shape[1]
				if current_seq_len == seq_length_with_past:
					new_attention_mask = attention_mask
				else:
					new_attention_mask = self._prepare_decoder_attention_mask(
						None,
						(batch_size, current_seq_len),
						inputs_embeds,
						0,
					)

			need_attention_for_pruning = (idx in need_prune_attention_layers) and (past_key_values is None)
			layer_output_attentions = bool(output_attentions or need_attention_for_pruning)

			layer_outputs = decoder_layer(
				hidden_states,
				attention_mask=new_attention_mask,
				position_ids=position_ids,
				past_key_value=past_key_value,
				output_attentions=layer_output_attentions,
				use_cache=use_cache,
			)

			hidden_states = layer_outputs[0]
			prev_layer_attn = layer_outputs[1] if len(layer_outputs) > 1 else None

			if use_cache:
				cache_idx = 2 if layer_output_attentions else 1
				if len(layer_outputs) > cache_idx:
					next_decoder_cache += (layer_outputs[cache_idx],)

			if output_attentions and len(layer_outputs) > 1:
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
