"""Normalization + augmentation transforms for Sentinel-2 patches.

Kept framework-agnostic: each transform takes a ``torch.Tensor`` of shape
(C, H, W) and returns one of the same shape.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

Transform = Callable[[torch.Tensor], torch.Tensor]


def _as_tensor(values: Sequence[float], channels: int) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.numel() != channels:
        raise ValueError(f"expected {channels} values, got {t.numel()}")
    return t.view(channels, 1, 1)


class Normalize:
    """Channel-wise normalization: ``(x - mean) / std``."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        if len(mean) != len(std):
            raise ValueError("mean and std must have the same length")
        self._channels = len(mean)
        self._mean = _as_tensor(mean, self._channels)
        self._std = _as_tensor(std, self._channels)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self._channels:
            raise ValueError(f"input has {x.shape[0]} channels, transform expects {self._channels}")
        return (x - self._mean.to(x.device, x.dtype)) / self._std.to(x.device, x.dtype)


class ScaleReflectance:
    """Scale raw Sentinel-2 DN values by ``1 / scale`` (default 10_000 → [0, 1])."""

    def __init__(self, scale: float = 10_000.0) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        self._scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32) / self._scale


class RandomHFlip:
    """Random horizontal flip with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self._p:
            return torch.flip(x, dims=[-1])
        return x


class RandomVFlip:
    """Random vertical flip with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self._p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self._p:
            return torch.flip(x, dims=[-2])
        return x


class RandomRotation90:
    """Randomly rotate by a multiple of 90° (0, 90, 180, or 270 degrees).

    Appropriate for GIS imagery where objects have no preferred orientation.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(4, ()).item())
        return torch.rot90(x, k=k, dims=[-2, -1])


class SpectralJitter:
    """Per-band brightness and contrast jitter for spectral robustness.

    Simulates sensor calibration drift and seasonal reflectance variation
    common in multi-temporal Sentinel-2 data.
    """

    def __init__(self, brightness: float = 0.1, contrast: float = 0.1) -> None:
        self._brightness = brightness
        self._contrast = contrast

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[0]
        # Independent scale (contrast) and shift (brightness) per band.
        scale = 1.0 + (torch.rand(c, 1, 1) * 2 - 1) * self._contrast
        shift = (torch.rand(c, 1, 1) * 2 - 1) * self._brightness
        return x * scale.to(x.device, x.dtype) + shift.to(x.device, x.dtype)


class GaussianNoise:
    """Add channel-wise Gaussian noise to simulate sensor noise."""

    def __init__(self, std: float = 0.01) -> None:
        self._std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self._std


class Compose:
    """Sequentially apply a list of transforms."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        self._transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self._transforms:
            x = t(x)
        return x


def build_train_transforms(
    mean: Sequence[float],
    std: Sequence[float],
    scale: float = 10_000.0,
) -> Transform:
    """Return the default augmentation pipeline used during training."""
    return Compose(
        [
            ScaleReflectance(scale=scale),
            Normalize(mean=mean, std=std),
            RandomHFlip(p=0.5),
            RandomVFlip(p=0.5),
            RandomRotation90(),
            SpectralJitter(brightness=0.1, contrast=0.1),
            GaussianNoise(std=0.01),
        ]
    )


def build_val_transforms(
    mean: Sequence[float],
    std: Sequence[float],
    scale: float = 10_000.0,
) -> Transform:
    """Return the deterministic pipeline used during validation/test."""
    return Compose(
        [
            ScaleReflectance(scale=scale),
            Normalize(mean=mean, std=std),
        ]
    )
