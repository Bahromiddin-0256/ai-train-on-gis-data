"""Shared pytest fixtures — synthetic tensors + on-disk GeoTIFF chips.

Everything here stays strictly offline: no network, no package cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

_RNG_SEED = 123


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(_RNG_SEED)


@pytest.fixture
def synthetic_batch(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)`` with per-class bias for trivial learnability."""
    n, c, h, w = 16, 4, 32, 32
    num_classes = 2
    images = rng.uniform(0.0, 3000.0, size=(n, c, h, w)).astype(np.float32)
    labels = rng.integers(0, num_classes, size=n).astype(np.int64)
    offsets = np.linspace(0.0, 6000.0, num=num_classes, dtype=np.float32)
    for i, lbl in enumerate(labels):
        images[i] += offsets[lbl]
    return images, labels


@pytest.fixture
def geotiff_dir(tmp_path: Path, rng: np.random.Generator) -> Path:
    """Write 4 tiny 4-band GeoTIFFs into a temp dir and return the dir."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    out = tmp_path / "tiles"
    out.mkdir()
    transform = from_origin(70.6, 41.4, 0.0001, 0.0001)
    for i in range(4):
        path = out / f"scene_{i}.tif"
        data = rng.integers(0, 10_000, size=(4, 16, 16)).astype(np.uint16)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=4,
            dtype="uint16",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)
    return out


@pytest.fixture
def tiny_config_overrides() -> list[str]:
    """Hydra CLI overrides that shrink a training run to a few seconds on CPU."""
    return [
        "data.source=synthetic",
        "data.synthetic_n=32",
        "data.data_dir=.",
        "data.image_size=16",
        "data.batch_size=4",
        "data.num_workers=0",
        "data.pin_memory=false",
        "data.val_split=0.25",
        "data.test_split=0.25",
        # Use 4 bands for synthetic data so the tiny test runs fast and
        # model.in_channels matches without needing 30-channel input.
        "data.bands=[B02,B03,B04,B08]",
        "data.mean=[0.1737,0.2178,0.2261,0.4629]",
        "data.std=[0.0451,0.0536,0.0838,0.0701]",
        "model.backbone=resnet18",
        "model.in_channels=4",
        "model.pretrained=false",
        "trainer.max_epochs=1",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "trainer.enable_progress_bar=false",
        "trainer.log_every_n_steps=1",
    ]


@pytest.fixture(autouse=True)
def _manual_seed() -> None:
    """Keep every test deterministic."""
    torch.manual_seed(_RNG_SEED)
    np.random.seed(_RNG_SEED)
