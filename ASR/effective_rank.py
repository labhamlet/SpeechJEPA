"""Effective-rank (RankMe) utilities for representation-collapse diagnostics.

References:
  - Roy & Vetterli, "The effective rank: a measure of effective
    dimensionality", EUSIPCO 2007.
  - Garrido et al., "RankMe: Assessing the Downstream Performance of
    Pretrained Self-Supervised Representations by Their Rank", ICML 2023.

All functions are model-agnostic: they take a (n_frames, dim) float tensor.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def effective_rank(
    feats: torch.Tensor,
    center: bool = True,
    eps: float = 1e-12,
) -> float:
    """RankMe: exp of the Shannon entropy of the normalized singular values.

    Args:
        feats: (n_frames, dim) feature matrix. Will be cast to float64.
        center: subtract the mean frame first. Keep True -- otherwise the
            mean direction contributes a large first singular value and a
            fully collapsed representation can still look rank > 1.
        eps: singular values below this are dropped.

    Returns:
        Effective rank in [1, min(n_frames, dim)].
    """
    if feats.ndim != 2:
        raise ValueError(f"expected (n, d), got shape {tuple(feats.shape)}")
    Z = feats.detach().to(torch.float64)
    if center:
        Z = Z - Z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Z)
    s = s[s > eps]
    if s.numel() == 0:
        return 1.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def dead_dimension_fraction(feats: torch.Tensor, tol: float = 1e-4) -> float:
    """Fraction of feature dimensions with (near-)zero std across frames.

    A cheap secondary collapse indicator to report alongside effective rank.
    """
    std = feats.detach().to(torch.float64).std(dim=0)
    return float((std < tol).float().mean())


@torch.no_grad()
def instance_norm_time(h: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Instance normalization over the time axis for a single utterance.

    h: (T, D). Normalizes each feature dimension across time -- the batch=1,
    no-padding case of the paper's padding-aware IN_t (Eq. 2), matching the
    eps used in masked_instance_normalize in the training code.
    """
    mean = h.mean(dim=0, keepdim=True)
    std = h.std(dim=0, unbiased=False, keepdim=True)
    return (h - mean) / (std + eps)


def subsample_rows(
    x: torch.Tensor,
    n: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly keep at most n rows of x (seeded via `generator`)."""
    if x.shape[0] <= n:
        return x
    idx = torch.randperm(x.shape[0], generator=generator)[:n]
    return x[idx]


class FrameCollector:
    """Accumulates a bounded, seeded random subsample of frames per feature.

    Keeps memory flat: from each utterance we keep at most `per_utt` frames
    per feature key, targeting `max_frames` total across `num_utts`.
    """

    def __init__(self, max_frames: int, num_utts: int, seed: int = 0):
        # small oversampling margin so short utterances don't leave us short
        self.per_utt = max(1, int(1.2 * max_frames / max(1, num_utts)))
        self.max_frames = max_frames
        self.gen = torch.Generator().manual_seed(seed)
        self.buffers: dict[str, list[torch.Tensor]] = {}

    def add(self, key: str, frames: torch.Tensor) -> None:
        """frames: (T, D) for one utterance; stored on CPU in float32."""
        kept = subsample_rows(frames.detach().float().cpu(), self.per_utt, self.gen)
        self.buffers.setdefault(key, []).append(kept)

    def matrix(self, key: str) -> torch.Tensor:
        full = torch.cat(self.buffers[key], dim=0)
        return subsample_rows(full, self.max_frames, self.gen)

    def keys(self):
        return list(self.buffers.keys())