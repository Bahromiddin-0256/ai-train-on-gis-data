"""PyTorch ``Dataset`` yielding Sentinel-2 patches with integer class labels.

Two construction paths are supported:

1. ``CropClassificationDataset.from_arrays(features, labels)`` — in-memory NumPy
   arrays. Used by the synthetic test fixtures and by the CropHarvest loader
   (which returns dense feature matrices per sample).

2. ``CropClassificationDataset.from_geotiffs(paths, labels)`` — list of GeoTIFF
   files, one per sample. Used when loading Sentinel-2 tiles from disk.

Both paths produce ``(image, label)`` tuples where ``image`` is a
``torch.FloatTensor`` of shape ``(C, H, W)`` and ``label`` is a ``torch.long``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

Transform = Callable[[torch.Tensor], torch.Tensor]


class CropClassificationDataset(Dataset):
    """Single-label image classification dataset for Sentinel-2 patches."""

    def __init__(
        self,
        images: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        geotiff_paths: Sequence[Path] | None = None,
        transform: Transform | None = None,
        num_classes: int | None = None,
    ) -> None:
        if (images is None) == (geotiff_paths is None):
            raise ValueError("provide exactly one of `images` or `geotiff_paths`")
        if labels is None:
            raise ValueError("`labels` is required")

        self._labels = np.asarray(labels, dtype=np.int64)
        self._transform = transform
        self._images = images
        self._geotiff_paths = [Path(p) for p in geotiff_paths] if geotiff_paths else None

        n = len(self._images) if self._images is not None else len(self._geotiff_paths or [])
        if n != len(self._labels):
            raise ValueError(f"got {n} images but {len(self._labels)} labels")

        if num_classes is not None and int(self._labels.max(initial=0)) >= num_classes:
            raise ValueError(
                f"label value {int(self._labels.max())} exceeds num_classes={num_classes}"
            )
        self._num_classes = num_classes

    @classmethod
    def from_arrays(
        cls,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Transform | None = None,
        num_classes: int | None = None,
    ) -> CropClassificationDataset:
        """Build a dataset from in-memory NumPy arrays.

        ``images`` must have shape ``(N, C, H, W)``; ``labels`` shape ``(N,)``.
        """
        images = np.asarray(images)
        if images.ndim != 4:
            raise ValueError(f"images must be 4-D (N, C, H, W); got shape {images.shape}")
        return cls(images=images, labels=labels, transform=transform, num_classes=num_classes)

    @classmethod
    def from_geotiffs(
        cls,
        paths: Sequence[Path | str],
        labels: np.ndarray,
        transform: Transform | None = None,
        num_classes: int | None = None,
    ) -> CropClassificationDataset:
        """Build a dataset that reads one GeoTIFF per sample, lazily, on ``__getitem__``."""
        return cls(
            geotiff_paths=[Path(p) for p in paths],
            labels=labels,
            transform=transform,
            num_classes=num_classes,
        )

    @property
    def num_classes(self) -> int:
        if self._num_classes is not None:
            return self._num_classes
        return int(self._labels.max()) + 1 if len(self._labels) else 0

    def __len__(self) -> int:
        return len(self._labels)

    def _load_image(self, index: int) -> torch.Tensor:
        if self._images is not None:
            arr = self._images[index]
        else:
            assert self._geotiff_paths is not None  # for type-checkers
            # Import rasterio lazily so the rest of the package (and tests for
            # the in-memory path) does not require the native GDAL stack.
            import rasterio

            with rasterio.open(self._geotiff_paths[index]) as src:
                arr = src.read()  # (C, H, W)
        tensor = torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32)
        if tensor.ndim != 3:
            raise ValueError(f"expected (C, H, W); got {tuple(tensor.shape)}")
        return tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._load_image(index)
        if self._transform is not None:
            image = self._transform(image)
        label = torch.as_tensor(self._labels[index], dtype=torch.long)
        return image, label
