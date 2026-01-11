"""Utilities for diversity-based token selection inside model inference layers.

Two selection methods are supported (set by ``method``):
- ``fps``: farthest point sampling over token embeddings.
- ``tome``: simplified Token Merging; pick high-degree seeds then keep one representative per local cluster.

Both methods work on per-batch hidden states and return the indices of the kept tokens
so callers can gather hidden states/attention masks before continuing the forward pass.
"""
from __future__ import annotations

from typing import Literal, Optional
import torch
import torch.nn.functional as F

SelectionMethod = Literal["fps", "tome"]
DistanceMetric = Literal["cosine", "l2"]


def _normalize_mask(attention_mask: Optional[torch.Tensor], hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    mask = attention_mask.to(dtype=torch.bool)
    if mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask shape must match hidden_states batch/sequence dimensions")
    return mask


def _pairwise_distance(tokens: torch.Tensor, metric: DistanceMetric) -> torch.Tensor:
    work = tokens.float()
    if metric == "cosine":
        work = F.normalize(work, dim=-1)
        # cosine distance = 1 - cosine similarity
        return 1.0 - torch.matmul(work, work.transpose(0, 1))
    if metric == "l2":
        return torch.cdist(work, work, p=2)
    raise ValueError(f"Unknown metric: {metric}")


def _pairwise_similarity(tokens: torch.Tensor) -> torch.Tensor:
    work = F.normalize(tokens.float(), dim=-1)
    return torch.matmul(work, work.transpose(0, 1))


class DiversityTokenSelector:
    """Diversity-based token selector for batching.

    Args:
        metric: distance metric used by fps ("cosine" or "l2").
    """

    def __init__(self, metric: DistanceMetric = "cosine"):
        self.metric = metric

    @torch.no_grad()
    def select(
        self,
        hidden_states: torch.Tensor,
        num_tokens: int,
        method: SelectionMethod = "fps",
        attention_mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.LongTensor:
        """Select token indices per batch.

        Args:
            hidden_states: [batch, seq_len, dim] token representations.
            num_tokens: number of tokens to keep (M).
            method: "fps" or "tome".
            attention_mask: optional mask where True/1 marks valid tokens.
            seed: optional seed for deterministic fps start token.

        Returns:
            LongTensor of shape [batch, num_tokens] with kept token indices; -1 when no token is available.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be [batch, seq_len, dim]")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")

        mask = _normalize_mask(attention_mask, hidden_states)
        batch, seq_len, _ = hidden_states.shape
        outputs = []
        for b in range(batch):
            outputs.append(
                self._select_single(
                    hidden_states[b],
                    num_tokens,
                    method=method,
                    mask=None if mask is None else mask[b],
                    seed=seed,
                )
            )
        return torch.stack(outputs, dim=0)

    def _select_single(
        self,
        tokens: torch.Tensor,
        keep: int,
        method: SelectionMethod,
        mask: Optional[torch.Tensor],
        seed: Optional[int],
    ) -> torch.LongTensor:
        if method == "fps":
            return self._fps(tokens, keep, mask, seed)
        if method == "tome":
            return self._tome(tokens, keep, mask)
        raise ValueError(f"Unknown method: {method}")

    def _valid_indices(self, seq_len: int, mask: Optional[torch.Tensor]) -> torch.LongTensor:
        if mask is None:
            return torch.arange(seq_len, device="cuda" if torch.cuda.is_available() else "cpu")
        valid = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        return valid

    def _fps(
        self,
        tokens: torch.Tensor,
        keep: int,
        mask: Optional[torch.Tensor],
        seed: Optional[int],
    ) -> torch.LongTensor:
        device = tokens.device
        valid_idx = self._valid_indices(tokens.shape[0], mask).to(device)
        if valid_idx.numel() == 0:
            return torch.full((keep,), -1, dtype=torch.long, device=device)

        keep = min(keep, valid_idx.numel())
        work_tokens = tokens[valid_idx]
        dist = _pairwise_distance(work_tokens, self.metric)

        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        start = torch.randint(0, work_tokens.shape[0], (1,), generator=rng, device=device).item()

        chosen = torch.zeros(work_tokens.shape[0], dtype=torch.bool, device=device)
        selected = torch.empty(keep, dtype=torch.long, device=device)
        selected[0] = start
        chosen[start] = True
        min_dist = dist[start]

        for i in range(1, keep):
            candidate_scores = min_dist.clone()
            candidate_scores[chosen] = -1.0
            next_idx = torch.argmax(candidate_scores).item()
            selected[i] = next_idx
            chosen[next_idx] = True
            min_dist = torch.minimum(min_dist, dist[next_idx])

        return valid_idx[selected]

    def _tome(
        self,
        tokens: torch.Tensor,
        keep: int,
        mask: Optional[torch.Tensor],
    ) -> torch.LongTensor:
        device = tokens.device
        valid_idx = self._valid_indices(tokens.shape[0], mask).to(device)
        if valid_idx.numel() == 0:
            return torch.full((keep,), -1, dtype=torch.long, device=device)

        keep = min(keep, valid_idx.numel())
        work_tokens = tokens[valid_idx]
        sim = _pairwise_similarity(work_tokens)
        degree = sim.sum(dim=1)

        seed_scores, seed_local_idx = torch.topk(degree, k=keep, largest=True)
        seed_tokens = work_tokens[seed_local_idx]

        # Assign tokens to nearest seed by similarity
        assign_scores = torch.matmul(F.normalize(work_tokens, dim=-1), F.normalize(seed_tokens, dim=-1).transpose(0, 1))
        assignments = torch.argmax(assign_scores, dim=1)

        selected_local = []
        for cluster_id in range(keep):
            members = torch.nonzero(assignments == cluster_id, as_tuple=False).squeeze(-1)
            if members.numel() == 0:
                selected_local.append(seed_local_idx[cluster_id])
                continue
            cluster_degrees = degree[members]
            best_member = members[torch.argmax(cluster_degrees)]
            selected_local.append(best_member)

        selected_local = torch.stack(selected_local)
        return valid_idx[selected_local]


__all__ = ["DiversityTokenSelector", "SelectionMethod", "DistanceMetric"]
