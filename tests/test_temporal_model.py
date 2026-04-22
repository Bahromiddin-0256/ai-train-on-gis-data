"""Tests for TemporalCropClassifier."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from gis_train.models.temporal import TemporalCropClassifier


def _build(**kw) -> TemporalCropClassifier:
    defaults = dict(
        n_windows=3,
        channels_per_window=10,
        num_classes=3,
        spatial_feature_dim=32,
        temporal_hidden=32,
        optimizer=OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 1.0e-3}),
    )
    defaults.update(kw)
    return TemporalCropClassifier(**defaults)


def test_forward_tempcnn() -> None:
    model = _build(temporal_encoder="tempcnn")
    x = torch.randn(2, 30, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 3)


def test_forward_gru() -> None:
    model = _build(temporal_encoder="gru")
    x = torch.randn(2, 30, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 3)


def test_forward_rejects_wrong_channel_count() -> None:
    model = _build()
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(2, 29, 32, 32))


def test_focal_loss_option() -> None:
    model = _build(loss="focal", focal_gamma=1.5)
    x = torch.randn(4, 30, 32, 32)
    y = torch.tensor([0, 1, 2, 0])
    logits = model(x)
    loss, hard = model._loss(logits, y)
    assert loss.ndim == 0
    assert torch.equal(hard, y)


def test_soft_label_path() -> None:
    model = _build()
    logits = torch.randn(2, 3)
    soft = torch.tensor([[0.7, 0.3, 0.0], [0.0, 0.5, 0.5]])
    loss, hard = model._loss(logits, soft)
    assert loss.ndim == 0
    assert hard.tolist() == [0, 1]
