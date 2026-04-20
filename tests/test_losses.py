"""Tests for gis_train.models.losses."""

from __future__ import annotations

import torch

from gis_train.models.losses import FocalLoss, compute_inverse_frequency_weights


def test_focal_recovers_ce_when_gamma_zero() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    focal = FocalLoss(gamma=0.0)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(focal(logits, targets), ce, atol=1e-5)


def test_focal_downweights_easy_examples() -> None:
    # Near-perfect predictions → focal loss should be much smaller than CE.
    logits = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    targets = torch.tensor([0, 0])
    ce = torch.nn.functional.cross_entropy(logits, targets)
    focal = FocalLoss(gamma=2.0)(logits, targets)
    assert focal < ce * 0.1


def test_inverse_frequency_weights_normalize_to_mean_one() -> None:
    labels = torch.tensor([0, 0, 0, 1, 2, 2])
    w = compute_inverse_frequency_weights(labels, num_classes=3)
    assert w.shape == (3,)
    assert abs(w.mean().item() - 1.0) < 1e-5
    # Rarest class (1, count=1) gets largest weight.
    assert w[1] > w[0] and w[1] > w[2]
