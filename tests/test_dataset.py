"""Unit tests for ``gis_train.data.dataset``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gis_train.data.dataset import CropClassificationDataset


def test_from_arrays_yields_tensor_and_label(
    synthetic_batch: tuple[np.ndarray, np.ndarray],
) -> None:
    images, labels = synthetic_batch
    ds = CropClassificationDataset.from_arrays(images=images, labels=labels, num_classes=2)

    assert len(ds) == len(labels)
    img, lbl = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (4, 32, 32)
    assert img.dtype == torch.float32
    assert isinstance(lbl, torch.Tensor)
    assert lbl.dtype == torch.long


def test_from_arrays_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="4-D"):
        CropClassificationDataset.from_arrays(
            images=np.zeros((4, 4, 4), dtype=np.float32),
            labels=np.zeros(4, dtype=np.int64),
        )


def test_length_matches_labels_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="images but"):
        CropClassificationDataset.from_arrays(
            images=np.zeros((3, 4, 4, 4), dtype=np.float32),
            labels=np.zeros(4, dtype=np.int64),
        )


def test_transform_is_applied(synthetic_batch: tuple[np.ndarray, np.ndarray]) -> None:
    images, labels = synthetic_batch

    def double(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    ds = CropClassificationDataset.from_arrays(
        images=images, labels=labels, transform=double, num_classes=2
    )
    img, _ = ds[0]
    expected = torch.as_tensor(images[0], dtype=torch.float32) * 2
    assert torch.allclose(img, expected)


def test_num_classes_enforced(synthetic_batch: tuple[np.ndarray, np.ndarray]) -> None:
    images, labels = synthetic_batch
    labels = labels.copy()
    labels[0] = 5  # out of the configured range
    with pytest.raises(ValueError, match="exceeds num_classes"):
        CropClassificationDataset.from_arrays(images=images, labels=labels, num_classes=2)


def test_from_geotiffs_reads_all_bands(geotiff_dir, synthetic_batch) -> None:
    paths = sorted(geotiff_dir.glob("*.tif"))
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    ds = CropClassificationDataset.from_geotiffs(paths=paths, labels=labels, num_classes=2)
    assert len(ds) == 4
    img, lbl = ds[2]
    assert img.shape == (4, 16, 16)
    assert int(lbl) == 0
