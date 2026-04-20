"""Loss functions for crop classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss (Lin et al., 2017).

    ``loss = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Recovers cross-entropy at ``gamma=0``. Useful when the minority class is
    under-learned despite class weights.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"unknown reduction: {reduction!r}")
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # p_t = exp(-CE) when no smoothing; acceptable approximation with smoothing.
        pt = torch.exp(-ce).clamp(min=1e-8, max=1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_inverse_frequency_weights(
    labels: torch.Tensor | list[int],
    num_classes: int,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Return per-class weights ∝ 1 / (count + smooth), normalized to mean 1.0."""
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(list(labels), dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / (counts + smooth)
    weights = weights * (num_classes / weights.sum())
    return weights


__all__ = ["FocalLoss", "compute_inverse_frequency_weights"]
