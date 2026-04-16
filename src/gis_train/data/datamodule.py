"""Lightning ``DataModule`` stitching labels + transforms into DataLoaders.

Three data sources are supported, selected via the ``source`` argument:

- ``"cropharvest"`` — load the CropHarvest Uzbekistan subset (requires the
  optional ``cropharvest`` extra).
- ``"synthetic"`` — generate a tiny random dataset in-memory. Used by the test
  suite so the full pipeline is exercised without any network / disk I/O.
- ``"local"`` — read ``images.npy`` + ``labels.npy`` from ``data_dir`` (the
  shape of ``prepare_labels.py``'s output).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from gis_train.data.dataset import CropClassificationDataset
from gis_train.data.transforms import build_train_transforms, build_val_transforms
from gis_train.utils.logging import get_logger

_log = get_logger(__name__)


def _make_splits(
    n: int,
    val_split: float,
    test_split: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 <= val_split < 1.0 and 0.0 <= test_split < 1.0):
        raise ValueError("val_split and test_split must be in [0, 1)")
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")
    indices = rng.permutation(n)
    n_test = round(n * test_split)
    n_val = round(n * val_split)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


def _make_synthetic(
    n: int,
    channels: int,
    image_size: int,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (features, labels) with a tiny class-dependent signal.

    Each class gets a different per-channel bias added to uniform noise so the
    smoke-train test can trivially fit the data and confirm the loop works.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n).astype(np.int64)
    images = rng.uniform(low=0.0, high=3000.0, size=(n, channels, image_size, image_size)).astype(
        np.float32
    )
    class_offsets = np.linspace(0.0, 6000.0, num=num_classes, dtype=np.float32)
    for i, label in enumerate(labels):
        images[i] += class_offsets[label]
    return images, labels


class CropDataModule(pl.LightningDataModule):
    """Lightning DataModule for crop classification on Sentinel-2 patches."""

    def __init__(
        self,
        source: str = "synthetic",
        data_dir: str | Path = "./data",
        external_test_dir: str | Path | None = None,
        country: str = "Uzbekistan",
        bbox: Sequence[float] = (70.60, 40.10, 72.90, 41.40),
        date_start: str = "2021-04-01",
        date_end: str = "2021-10-31",
        cloud_cover_max: float = 20.0,
        bands: Sequence[str] = ("B02", "B03", "B04", "B08"),
        mean: Sequence[float] = (0.1354, 0.1118, 0.1042, 0.3033),
        std: Sequence[float] = (0.0605, 0.0621, 0.0754, 0.1082),
        image_size: int = 64,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_split: float = 0.15,
        test_split: float = 0.15,
        classes: Sequence[str] = ("non_crop", "crop"),
        seed: int = 42,
        synthetic_n: int = 128,
    ) -> None:
        super().__init__()
        # Hydra-instantiated kwargs live on self via save_hyperparameters() so
        # they're serialized into Lightning checkpoints.
        self.save_hyperparameters()

        self.source = source
        self.data_dir = Path(data_dir)
        self.external_test_dir = Path(external_test_dir) if external_test_dir else None
        self.country = country
        self.bbox = tuple(bbox)
        self.date_start = date_start
        self.date_end = date_end
        self.cloud_cover_max = cloud_cover_max
        self.bands = tuple(bands)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.test_split = test_split
        self.classes = tuple(classes)
        self.seed = seed
        self.synthetic_n = synthetic_n

        self._train: CropClassificationDataset | Subset | None = None
        self._val: CropClassificationDataset | Subset | None = None
        self._test: CropClassificationDataset | Subset | None = None

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_bands(self) -> int:
        return len(self.bands)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Downloads / caches raw data. Called once on a single process."""
        if self.source in {"synthetic", "local"}:
            return
        if self.source == "cropharvest":
            # Touching the loader forces CropHarvest to cache its archives.
            from gis_train.data.labels import load_cropharvest_uzbekistan

            load_cropharvest_uzbekistan(root=str(self.data_dir / "cropharvest"))
            return
        raise ValueError(f"unsupported source: {self.source!r}")

    def setup(self, stage: str | None = None) -> None:
        images, labels = self._load_arrays()
        if images.shape[1] != self.num_bands:
            raise ValueError(
                f"features have {images.shape[1]} channels but {self.num_bands} bands are "
                "configured — check `data.bands`"
            )

        train_tf = build_train_transforms(mean=self.mean, std=self.std)
        eval_tf = build_val_transforms(mean=self.mean, std=self.std)

        if self.external_test_dir is not None:
            # External test set: use all of data_dir for train+val, external for test
            ext_images = np.load(self.external_test_dir / "images.npy")
            ext_labels = np.load(self.external_test_dir / "labels.npy")

            rng = np.random.default_rng(self.seed)
            n = images.shape[0]
            n_val = round(n * self.val_split)
            indices = rng.permutation(n)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            full_train = CropClassificationDataset.from_arrays(
                images=images, labels=labels, transform=train_tf, num_classes=self.num_classes
            )
            full_eval = CropClassificationDataset.from_arrays(
                images=images, labels=labels, transform=eval_tf, num_classes=self.num_classes
            )
            self._train = Subset(full_train, train_idx.tolist())
            self._val = Subset(full_eval, val_idx.tolist())
            self._test = CropClassificationDataset.from_arrays(
                images=ext_images, labels=ext_labels, transform=eval_tf,
                num_classes=self.num_classes,
            )
            _log.info(
                "prepared splits: train=%d val=%d test=%d (external, source=%s)",
                len(self._train), len(self._val), len(self._test), self.source,
            )
        else:
            rng = np.random.default_rng(self.seed)
            train_idx, val_idx, test_idx = _make_splits(
                n=images.shape[0],
                val_split=self.val_split,
                test_split=self.test_split,
                rng=rng,
            )

            full_train = CropClassificationDataset.from_arrays(
                images=images, labels=labels, transform=train_tf, num_classes=self.num_classes
            )
            full_eval = CropClassificationDataset.from_arrays(
                images=images, labels=labels, transform=eval_tf, num_classes=self.num_classes
            )

            self._train = Subset(full_train, train_idx.tolist())
            self._val = Subset(full_eval, val_idx.tolist())
            self._test = Subset(full_eval, test_idx.tolist())

            _log.info(
                "prepared splits: train=%d val=%d test=%d (source=%s)",
                len(self._train), len(self._val), len(self._test), self.source,
            )

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------

    def _loader(self, dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None, "call setup() first"
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None, "call setup() first"
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self._test is not None, "call setup() first"
        return self._loader(self._test, shuffle=False)

    # ------------------------------------------------------------------
    # Source-specific loading
    # ------------------------------------------------------------------

    def _load_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.source == "synthetic":
            _log.info("generating synthetic dataset (n=%d)", self.synthetic_n)
            return _make_synthetic(
                n=self.synthetic_n,
                channels=self.num_bands,
                image_size=self.image_size,
                num_classes=self.num_classes,
                seed=self.seed,
            )

        if self.source == "local":
            images_path = self.data_dir / "images.npy"
            labels_path = self.data_dir / "labels.npy"
            if not images_path.exists() or not labels_path.exists():
                raise FileNotFoundError(
                    f"expected {images_path} and {labels_path}; run "
                    "`python scripts/prepare_labels.py` first"
                )
            return np.load(images_path), np.load(labels_path)

        if self.source == "cropharvest":
            from gis_train.data.labels import load_cropharvest_uzbekistan

            samples = load_cropharvest_uzbekistan(
                root=str(self.data_dir / "cropharvest"),
                class_names=self.classes,
            )
            return samples.features, samples.labels

        raise ValueError(f"unsupported source: {self.source!r}")


__all__ = ["CropDataModule"]
