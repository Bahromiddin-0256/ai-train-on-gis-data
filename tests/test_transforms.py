"""Unit tests for ``gis_train.data.transforms``."""

from __future__ import annotations

import pytest
import torch

from gis_train.data.transforms import (
    Compose,
    Normalize,
    RandomHFlip,
    RandomVFlip,
    ScaleReflectance,
    build_train_transforms,
    build_val_transforms,
)


def test_scale_reflectance_divides() -> None:
    tf = ScaleReflectance(scale=10_000.0)
    x = torch.full((1, 2, 2), 5000.0)
    out = tf(x)
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_normalize_matches_manual() -> None:
    mean = [0.1, 0.2]
    std = [0.5, 0.5]
    tf = Normalize(mean=mean, std=std)
    x = torch.tensor([[[0.6]], [[0.7]]])
    out = tf(x)
    assert torch.allclose(out, torch.tensor([[[1.0]], [[1.0]]]))


def test_normalize_rejects_wrong_channels() -> None:
    tf = Normalize(mean=[0.0, 0.0], std=[1.0, 1.0])
    with pytest.raises(ValueError, match="channels"):
        tf(torch.zeros(3, 2, 2))


def test_random_flips_preserve_shape() -> None:
    x = torch.randn(2, 4, 4)
    assert RandomHFlip(p=1.0)(x).shape == x.shape
    assert RandomVFlip(p=1.0)(x).shape == x.shape
    # p=0 is a no-op
    assert torch.equal(RandomHFlip(p=0.0)(x), x)


def test_random_hflip_actually_flips() -> None:
    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    flipped = RandomHFlip(p=1.0)(x)
    assert torch.equal(flipped, torch.flip(x, dims=[-1]))


def test_compose_chains_in_order() -> None:
    pipeline = Compose([ScaleReflectance(10_000.0), Normalize([0.5, 0.5], [0.5, 0.5])])
    x = torch.full((2, 1, 1), 10_000.0)
    out = pipeline(x)
    assert torch.allclose(out, torch.full_like(out, 1.0))


def test_builders_return_callable() -> None:
    mean, std = [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]
    train = build_train_transforms(mean=mean, std=std)
    val = build_val_transforms(mean=mean, std=std)
    x = torch.full((4, 4, 4), 5_000.0)
    assert train(x).shape == x.shape
    assert val(x).shape == x.shape
