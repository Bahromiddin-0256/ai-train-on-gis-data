"""Unit tests for ``gis_train.models.classifier.CropClassifier``."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from gis_train.models.classifier import CropClassifier


def _build_classifier(**overrides) -> CropClassifier:
    kwargs = dict(
        backbone="resnet18",
        in_channels=4,
        num_classes=2,
        pretrained=False,
        optimizer=OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 1.0e-3}),
        scheduler=None,
    )
    kwargs.update(overrides)
    return CropClassifier(**kwargs)


def test_forward_produces_logits() -> None:
    model = _build_classifier()
    x = torch.randn(2, 4, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 2)


def test_configure_optimizers_without_scheduler() -> None:
    model = _build_classifier()
    opt = model.configure_optimizers()
    assert isinstance(opt, torch.optim.Optimizer)


def test_configure_optimizers_with_scheduler() -> None:
    model = _build_classifier(
        scheduler=OmegaConf.create(
            {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 1, "gamma": 0.5}
        )
    )
    out = model.configure_optimizers()
    assert isinstance(out, dict)
    assert "optimizer" in out and "lr_scheduler" in out


def test_one_batch_overfit() -> None:
    """A single batch should push training accuracy above chance within a few steps."""
    torch.manual_seed(0)
    np.random.seed(0)

    model = _build_classifier()
    x = torch.randn(8, 4, 32, 32)
    # Linearly-separable labels: positive sum -> class 1, negative -> class 0.
    y = (x.sum(dim=(1, 2, 3)) > 0).long()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(20):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(x).argmax(dim=1)
        acc = (preds == y).float().mean().item()
    # Chance is 50%; require the model to have actually learned something.
    assert acc >= 0.75, f"expected train accuracy >= 0.75, got {acc:.3f}"
