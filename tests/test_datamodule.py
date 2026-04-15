"""Integration tests for ``gis_train.data.datamodule.CropDataModule``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gis_train.data.datamodule import CropDataModule


def _build_synthetic_dm(**overrides) -> CropDataModule:
    kwargs = dict(
        source="synthetic",
        image_size=16,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        synthetic_n=32,
        val_split=0.25,
        test_split=0.25,
    )
    kwargs.update(overrides)
    return CropDataModule(**kwargs)


def test_setup_produces_three_splits() -> None:
    dm = _build_synthetic_dm()
    dm.prepare_data()
    dm.setup()
    assert len(dm._train) + len(dm._val) + len(dm._test) == 32
    assert len(dm._val) == 8
    assert len(dm._test) == 8


def test_dataloaders_yield_expected_shapes() -> None:
    dm = _build_synthetic_dm()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    images, labels = batch
    assert images.shape == (4, 4, 16, 16)
    assert labels.shape == (4,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long


def test_band_mismatch_raises(tmp_path) -> None:
    # Save 2-channel arrays but configure 4 bands → should fail in setup().
    images = np.random.rand(8, 2, 16, 16).astype(np.float32)
    labels = np.random.randint(0, 2, size=8).astype(np.int64)
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "labels.npy", labels)

    dm = _build_synthetic_dm(
        source="local",
        data_dir=str(tmp_path),
        bands=("B02", "B03", "B04", "B08"),
    )
    dm.prepare_data()
    with pytest.raises(ValueError, match="channels"):
        dm.setup()


def test_local_source_requires_files(tmp_path) -> None:
    dm = _build_synthetic_dm(source="local", data_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        dm.setup()


def test_local_source_reads_npy(tmp_path) -> None:
    images = np.random.rand(8, 4, 16, 16).astype(np.float32)
    labels = np.random.randint(0, 2, size=8).astype(np.int64)
    np.save(tmp_path / "images.npy", images)
    np.save(tmp_path / "labels.npy", labels)

    dm = _build_synthetic_dm(
        source="local", data_dir=str(tmp_path), synthetic_n=8, val_split=0.25, test_split=0.25
    )
    dm.setup()
    total = len(dm._train) + len(dm._val) + len(dm._test)
    assert total == 8
