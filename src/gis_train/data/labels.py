"""Label loaders for supported public datasets.

The heavy CropHarvest / WorldCereal dependencies are imported lazily so that
``import gis_train`` stays cheap and works in CI environments that don't ship
their native geo stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gis_train.utils.geo import BBox
from gis_train.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class LabeledSamples:
    """Container for pre-extracted features + labels from a public dataset."""

    features: np.ndarray  # (N, C, H, W) float32
    labels: np.ndarray  # (N,) int64
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 4:
            raise ValueError(f"features must be 4-D, got shape {self.features.shape}")
        if self.labels.shape != (self.features.shape[0],):
            raise ValueError(
                f"labels shape {self.labels.shape} incompatible with features "
                f"N={self.features.shape[0]}"
            )


def load_cropharvest_uzbekistan(
    root: str | None = None,
    class_names: tuple[str, ...] = ("non_crop", "crop"),
) -> LabeledSamples:
    """Load the CropHarvest Uzbekistan subset as per-sample Sentinel-2 patches.

    This is a thin adapter around the ``cropharvest`` package. It's kept in its
    own function so callers can swap it out — see ``load_worldcereal`` for the
    parallel stub.

    Parameters
    ----------
    root:
        Directory where CropHarvest should cache its downloads. If ``None``,
        defaults to ``~/.cache/cropharvest``.
    class_names:
        Human-readable class names. The default binary setup mirrors CropHarvest's
        crop/non-crop task.

    Returns
    -------
    LabeledSamples
        In-memory features + labels ready to hand to
        :class:`gis_train.data.dataset.CropClassificationDataset`.
    """
    try:
        from cropharvest.datasets import CropHarvest  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "CropHarvest is not installed. Run `pip install -e '.[cropharvest]'` "
            "or pass `source=synthetic` in the data config for offline runs."
        ) from exc

    _log.info("Loading CropHarvest Uzbekistan subset (root=%s)", root)
    datasets = (
        CropHarvest.create_benchmark_datasets(root=root)
        if root
        else CropHarvest.create_benchmark_datasets()
    )  # type: ignore[attr-defined]
    uz = next((d for d in datasets if "Uzbekistan" in getattr(d, "id", "")), None)
    if uz is None:
        raise RuntimeError("Could not locate an Uzbekistan task in CropHarvest benchmarks")

    x, y = uz.as_array(flatten_x=False)
    # CropHarvest returns x of shape (N, T, C); collapse the temporal axis into
    # a spatial-ish one so the 2-D CNN can consume it. This matches the approach
    # used by the CropHarvest baseline notebooks.
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:
        n, t, c = x.shape
        side = int(np.ceil(np.sqrt(t)))
        pad = side * side - t
        if pad:
            x = np.concatenate([x, np.zeros((n, pad, c), dtype=x.dtype)], axis=1)
        x = x.reshape(n, side, side, c).transpose(0, 3, 1, 2)  # (N, C, H, W)
    elif x.ndim != 4:
        raise ValueError(f"unexpected CropHarvest feature shape: {x.shape}")

    y = np.asarray(y, dtype=np.int64).reshape(-1)
    return LabeledSamples(features=x, labels=y, class_names=tuple(class_names))


def load_worldcereal(
    bbox: BBox,
    year: int = 2021,
    **_: object,
) -> LabeledSamples:
    """Placeholder for the ESA WorldCereal loader.

    WorldCereal publishes global crop-type maps as COGs. Implementing this
    requires rasterizing the maps onto Sentinel-2 tiles and is tracked in
    the project README (TODO). Raising explicitly here keeps the interface
    stable while signalling that the loader is unimplemented.
    """
    raise NotImplementedError(
        "WorldCereal loader is not implemented yet. See README TODO and "
        "https://esa-worldcereal.org/ for access details."
    )
