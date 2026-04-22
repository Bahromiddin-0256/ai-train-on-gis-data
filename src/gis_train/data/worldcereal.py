"""Validate uzcosmos polygon labels against ESA WorldCereal rasters.

WorldCereal publishes per-season cropland + crop-type maps at 10 m. This
module compares local polygon labels (``bugdoy``, ``paxta``, ``other``)
against two single-band GeoTIFFs — a binary cropland mask and a crop-type
raster — and emits a per-polygon verdict:

* ``accepted``  — WorldCereal agrees with the label; keep for training.
* ``review``    — partial disagreement (e.g. mostly cropland but the
  crop-type doesn't match); flag for human inspection.
* ``rejected``  — strong disagreement (e.g. a field labelled ``bugdoy``
  that WorldCereal reports as non-cropland); drop from training.

The runner that produces the rasters lives in ``scripts/run_worldcereal.py``
(separate, since it requires CDSE openEO credentials). This module only
consumes the rasters and is therefore fully offline-testable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)


class Verdict(StrEnum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"


# uzcosmos compound labels → primary crop. Mirrors scripts/stats_noncrop.py
# and scripts/export_mongodb.py so a single source of truth is easy to spot.
_NORMALISE: dict[str, str] = {
    "bugdoy, other": "bugdoy",
    "bugdoy, paxta": "bugdoy",
    "other, bugdoy": "other",
}

# Canonical label groups — kept narrow on purpose.
WHEAT_LABELS = frozenset({"bugdoy"})
CROP_LABELS = frozenset({"bugdoy", "paxta"})
NONCROP_LABELS = frozenset({"other"})


def normalise_label(raw: str) -> str:
    """Lower-case and fold compound uzcosmos crop_type values to the primary."""
    key = raw.strip().lower()
    return _NORMALISE.get(key, key)


@dataclass(frozen=True)
class Thresholds:
    """Fractions of polygon area that drive the verdict.

    All values are in [0, 1]. ``crop_accept`` is the share of the polygon
    that must be WorldCereal-cropland before a crop label can be kept;
    ``wheat_accept`` is the additional share (of the polygon) that must be
    winter-cereal before a ``bugdoy`` polygon is accepted.
    """

    crop_accept: float = 0.70
    crop_reject: float = 0.30
    wheat_accept: float = 0.50
    wheat_reject: float = 0.10


_DEFAULT_THRESHOLDS = Thresholds()


@dataclass
class PolygonScore:
    """Per-polygon result of zonal stats + verdict."""

    polygon_id: str
    label: str
    label_normalised: str
    n_pixels: int
    cropland_fraction: float
    wheat_fraction: float
    verdict: Verdict
    reason: str


def _zonal_fraction(
    src: rasterio.io.DatasetReader,
    geom: BaseGeometry,
    positive_values: Iterable[int],
) -> tuple[int, float]:
    """Return (valid_pixel_count, fraction of pixels in positive_values).

    Rasterio's :func:`rasterio.mask.mask` with ``crop=True`` clips to the
    geometry; we then flatten and compare against the nodata value to
    exclude pixels outside the polygon.
    """
    positives = set(int(v) for v in positive_values)
    try:
        clipped, _ = rio_mask(src, [mapping(geom)], crop=True, filled=True, nodata=src.nodata)
    except ValueError:
        # Geometry does not overlap the raster.
        return 0, 0.0

    arr = clipped[0]
    valid = arr != src.nodata if src.nodata is not None else np.ones_like(arr, dtype=bool)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0, 0.0

    pos_mask = np.isin(arr, list(positives)) & valid
    return n_valid, float(pos_mask.sum()) / float(n_valid)


def score_polygon(
    geometry: BaseGeometry,
    label: str,
    cropland_src: rasterio.io.DatasetReader,
    wheat_src: rasterio.io.DatasetReader | None,
    polygon_id: str,
    cropland_positive: Iterable[int] = (1,),
    wheat_positive: Iterable[int] = (1,),
    thresholds: Thresholds = _DEFAULT_THRESHOLDS,
) -> PolygonScore:
    """Compute cropland/wheat agreement for a single polygon.

    ``wheat_src`` may be ``None`` — the validator then only does the
    cropland-vs-noncropland check, useful when only the cropland product
    has been generated so far.
    """
    n_pixels, crop_frac = _zonal_fraction(cropland_src, geometry, cropland_positive)
    wheat_frac = 0.0
    if wheat_src is not None and n_pixels > 0:
        _, wheat_frac = _zonal_fraction(wheat_src, geometry, wheat_positive)

    normalised = normalise_label(label)
    verdict, reason = _decide(
        normalised, crop_frac, wheat_frac, n_pixels, wheat_src is not None, thresholds
    )

    return PolygonScore(
        polygon_id=polygon_id,
        label=label,
        label_normalised=normalised,
        n_pixels=n_pixels,
        cropland_fraction=round(crop_frac, 4),
        wheat_fraction=round(wheat_frac, 4),
        verdict=verdict,
        reason=reason,
    )


def _decide(
    label: str,
    crop_frac: float,
    wheat_frac: float,
    n_pixels: int,
    have_wheat_layer: bool,
    t: Thresholds,
) -> tuple[Verdict, str]:
    # Polygons that never overlapped the raster are rejected — we can't
    # say anything useful about them.
    if n_pixels == 0:
        return Verdict.REJECTED, "no_overlap_with_worldcereal"

    if label in NONCROP_LABELS:
        # "other" = should NOT be cropland.
        if crop_frac >= t.crop_accept:
            return Verdict.REJECTED, "noncrop_label_but_worldcereal_is_cropland"
        if crop_frac >= t.crop_reject:
            return Verdict.REVIEW, "noncrop_label_but_partially_cropland"
        return Verdict.ACCEPTED, "noncrop_label_agrees"

    if label not in CROP_LABELS:
        # Unknown label: don't pretend to judge it.
        return Verdict.REVIEW, f"unknown_label:{label}"

    # Crop labels from here on.
    if crop_frac < t.crop_reject:
        return Verdict.REJECTED, "crop_label_but_worldcereal_is_noncropland"
    if crop_frac < t.crop_accept:
        return Verdict.REVIEW, "crop_label_but_cropland_evidence_weak"

    if not have_wheat_layer:
        return Verdict.ACCEPTED, "crop_label_agrees_cropland_only"

    if label in WHEAT_LABELS:
        if wheat_frac >= t.wheat_accept:
            return Verdict.ACCEPTED, "wheat_label_agrees"
        if wheat_frac < t.wheat_reject:
            return Verdict.REJECTED, "wheat_label_but_worldcereal_says_nonwheat"
        return Verdict.REVIEW, "wheat_label_but_wheat_evidence_weak"

    # Non-wheat crop (paxta): wheat fraction should be low.
    if wheat_frac >= t.wheat_accept:
        return Verdict.REJECTED, "nonwheat_crop_label_but_worldcereal_says_wheat"
    if wheat_frac >= t.wheat_reject:
        return Verdict.REVIEW, "nonwheat_crop_label_with_some_wheat_signal"
    return Verdict.ACCEPTED, "nonwheat_crop_label_agrees"


def score_polygons(
    features: Iterable[dict],
    cropland_raster: Path,
    wheat_raster: Path | None,
    label_field: str = "crop_type",
    id_field: str = "id",
    cropland_positive: Iterable[int] = (1,),
    wheat_positive: Iterable[int] = (1,),
    thresholds: Thresholds = _DEFAULT_THRESHOLDS,
) -> list[PolygonScore]:
    """Score a GeoJSON-style feature iterable against WorldCereal rasters.

    Each feature must have ``geometry`` and ``properties[label_field]``.
    The feature's id is pulled from ``properties[id_field]`` if present,
    otherwise from the top-level ``id`` field, otherwise a sequential index.
    """
    cropland_src = rasterio.open(cropland_raster)
    wheat_src = rasterio.open(wheat_raster) if wheat_raster is not None else None

    try:
        results: list[PolygonScore] = []
        for idx, feat in enumerate(features):
            geom = shape(feat["geometry"])
            props = feat.get("properties") or {}
            raw_label = props.get(label_field)
            if raw_label is None:
                _log.debug("skipping feature %s: no %s", idx, label_field)
                continue
            pid = str(props.get(id_field) or feat.get("id") or idx)
            results.append(
                score_polygon(
                    geometry=geom,
                    label=str(raw_label),
                    cropland_src=cropland_src,
                    wheat_src=wheat_src,
                    polygon_id=pid,
                    cropland_positive=cropland_positive,
                    wheat_positive=wheat_positive,
                    thresholds=thresholds,
                )
            )
    finally:
        cropland_src.close()
        if wheat_src is not None:
            wheat_src.close()

    return results
