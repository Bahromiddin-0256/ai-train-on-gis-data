"""Tests for MixupCutmixCollate."""

from __future__ import annotations

import torch

from gis_train.data.mixup import MixupCutmixCollate


def _batch(n: int = 4, c: int = 3, s: int = 8, num_classes: int = 3):
    return [
        (torch.randn(c, s, s), torch.tensor(i % num_classes))
        for i in range(n)
    ]


def test_no_mix_returns_one_hot_labels() -> None:
    collate = MixupCutmixCollate(num_classes=3, mixup_alpha=0.0, cutmix_alpha=0.0, seed=0)
    images, labels = collate(_batch())
    assert images.shape == (4, 3, 8, 8)
    assert labels.shape == (4, 3)
    # Rows should sum to 1.
    assert torch.allclose(labels.sum(dim=1), torch.ones(4))
    # Labels should be one-hot (no mixing).
    assert ((labels == 0) | (labels == 1)).all()


def test_mixup_produces_soft_labels() -> None:
    collate = MixupCutmixCollate(
        num_classes=3, mixup_alpha=0.4, cutmix_alpha=0.0, p=1.0, seed=1
    )
    _, labels = collate(_batch(n=8))
    assert labels.shape == (8, 3)
    assert torch.allclose(labels.sum(dim=1), torch.ones(8), atol=1e-5)
    # At least one row should not be one-hot (mixed).
    one_hot_rows = ((labels == 0) | (labels == 1)).all(dim=1)
    assert not one_hot_rows.all()


def test_cutmix_preserves_image_shape() -> None:
    collate = MixupCutmixCollate(
        num_classes=3, mixup_alpha=0.0, cutmix_alpha=1.0, p=1.0, seed=2
    )
    images, labels = collate(_batch(n=6))
    assert images.shape == (6, 3, 8, 8)
    assert labels.shape == (6, 3)
