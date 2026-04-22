"""Label loaders for supported public datasets.

The heavy CropHarvest / WorldCereal dependencies are imported lazily so that
``import gis_train`` stays cheap and works in CI environments that don't ship
their native geo stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


def filter_labels_with_worldcereal(
    labels_path: str | Path,
    cropland_raster: str | Path,
    wheat_raster: str | Path | None = None,
    *,
    label_field: str = "crop_type",
    id_field: str = "id",
    cropland_positive: tuple[int, ...] = (1,),
    wheat_positive: tuple[int, ...] = (1,),
    thresholds: object | None = None,
    keep_review: bool = False,
) -> list[dict]:
    """Return the subset of GeoJSON features that agree with WorldCereal.

    Pairs naturally with ``scripts/run_worldcereal.py`` (which produces the
    rasters) and ``scripts/validate_uzcosmos_worldcereal.py`` (which writes
    accepted/review/rejected splits to disk). Use this function when you
    want the filter inline in a Python pipeline — for example, before
    handing polygons to ``scripts/build_dataset.py`` for Sentinel-2 chip
    extraction.

    Parameters
    ----------
    labels_path:
        Input GeoJSON FeatureCollection with polygon geometries and
        ``properties[label_field]`` labels.
    cropland_raster:
        WorldCereal seasonal cropland GeoTIFF.
    wheat_raster:
        WorldCereal crop-type GeoTIFF (winter-cereals vs. others). Pass
        ``None`` to do a cropland-only filter.
    keep_review:
        If ``True``, ``review``-verdict features are returned alongside
        ``accepted`` ones. Defaults to ``False`` (strict).

    Returns
    -------
    list[dict]
        The surviving features, each with a ``worldcereal_verdict`` and
        ``worldcereal_*_fraction`` entry merged into ``properties``.
    """
    from gis_train.data.worldcereal import (
        Thresholds,
        Verdict,
        score_polygons,
    )

    t = thresholds or Thresholds()
    labels_path = Path(labels_path)
    data = json.loads(labels_path.read_text())
    features = data.get("features") if isinstance(data, dict) else None
    if not features:
        raise ValueError(f"{labels_path} does not contain a FeatureCollection")

    labelled = [f for f in features if (f.get("properties") or {}).get(label_field) is not None]
    scores = score_polygons(
        labelled,
        cropland_raster=Path(cropland_raster),
        wheat_raster=Path(wheat_raster) if wheat_raster is not None else None,
        label_field=label_field,
        id_field=id_field,
        cropland_positive=cropland_positive,
        wheat_positive=wheat_positive,
        thresholds=t,
    )

    allowed = {Verdict.ACCEPTED}
    if keep_review:
        allowed.add(Verdict.REVIEW)

    out: list[dict] = []
    for feat, score in zip(labelled, scores, strict=True):
        if score.verdict not in allowed:
            continue
        new_feat = dict(feat)
        props = dict(feat.get("properties") or {})
        props["worldcereal_verdict"] = score.verdict.value
        props["worldcereal_cropland_fraction"] = score.cropland_fraction
        props["worldcereal_wheat_fraction"] = score.wheat_fraction
        new_feat["properties"] = props
        out.append(new_feat)

    _log.info(
        "worldcereal filter: %d/%d features kept (keep_review=%s)",
        len(out),
        len(labelled),
        keep_review,
    )
    return out


def load_worldcereal(
    bbox: BBox,
    year: int = 2025,
    **_: object,
) -> LabeledSamples:
    """Bbox-style loader — not implemented by design.

    Unlike CropHarvest (which ships pre-extracted Sentinel-2 patches),
    WorldCereal only gives you raster masks. Turning a bbox + year into an
    in-memory ``LabeledSamples`` therefore requires a separate Sentinel-2
    chip-extraction step (see ``scripts/build_dataset.py``).

    For the common label-cleanup workflow, use
    :func:`filter_labels_with_worldcereal` instead — it returns the subset
    of your local polygon labels that WorldCereal agrees with, which is
    what ``build_dataset.py`` wants as input.
    """
    raise NotImplementedError(
        "load_worldcereal is intentionally not implemented for bbox+year — "
        "WorldCereal ships rasters, not pre-extracted chips. Use "
        "gis_train.data.labels.filter_labels_with_worldcereal() to filter "
        "local polygons, then run scripts/build_dataset.py for chip extraction."
    )
