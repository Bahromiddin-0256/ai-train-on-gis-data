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


def test_focal_loss_runs() -> None:
    model = _build_classifier(loss="focal", focal_gamma=2.0)
    x = torch.randn(2, 4, 32, 32)
    y = torch.tensor([0, 1])
    logits = model(x)
    loss = model.loss_fn(logits, y)
    assert loss.ndim == 0


def test_differential_lr_creates_two_param_groups() -> None:
    model = _build_classifier(backbone_lr_mult=0.1)
    opt = model.configure_optimizers()
    opt = opt["optimizer"] if isinstance(opt, dict) else opt
    assert len(opt.param_groups) == 2
    lrs = [g["lr"] for g in opt.param_groups]
    assert min(lrs) < max(lrs)


def test_warmup_scheduler_built() -> None:
    model = _build_classifier(
        warmup_epochs=2,
        scheduler=OmegaConf.create(
            {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 1, "gamma": 0.5}
        ),
    )
    out = model.configure_optimizers()
    assert isinstance(out, dict) and "lr_scheduler" in out


def test_tta_forward_shape_matches() -> None:
    model = _build_classifier(test_time_augmentation=True)
    model.eval()
    x = torch.randn(3, 4, 32, 32)
    with torch.no_grad():
        out = model._eval_forward(x)
    assert out.shape == (3, 2)


def test_soft_label_path() -> None:
    model = _build_classifier()
    x = torch.randn(2, 4, 32, 32)
    soft = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    loss, logits, hard = model._shared_step((x, soft))
    assert loss.ndim == 0
    assert hard.tolist() == [0, 1]


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
