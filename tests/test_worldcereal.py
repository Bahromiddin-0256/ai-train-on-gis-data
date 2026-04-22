"""Synthetic-fixture tests for the WorldCereal validator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from gis_train.data.worldcereal import (
    Thresholds,
    Verdict,
    _decide,
    normalise_label,
    score_polygons,
)

# Fixture raster layout:
#   10x10 pixels, pixel size 0.001 deg (~100 m at the equator — fine for tests).
#   Origin at (70.000, 41.010) so the raster covers lon [70.000, 70.010]
#   and lat [41.000, 41.010].
_PIX = 0.001
_ORIGIN_LON = 70.000
_ORIGIN_LAT = 41.010  # rasterio origin = upper-left
_SIZE = 10


def _write_raster(path: Path, data: np.ndarray) -> None:
    transform = from_origin(_ORIGIN_LON, _ORIGIN_LAT, _PIX, _PIX)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_SIZE,
        width=_SIZE,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        nodata=255,
    ) as dst:
        dst.write(data.astype("uint8"), 1)


@pytest.fixture
def rasters(tmp_path: Path) -> tuple[Path, Path]:
    """Top half of the AOI is cropland; the top-left quadrant is wheat."""
    cropland = np.zeros((_SIZE, _SIZE), dtype="uint8")
    cropland[:5, :] = 1  # upper half = cropland

    wheat = np.zeros((_SIZE, _SIZE), dtype="uint8")
    wheat[:5, :5] = 1  # upper-left quadrant = wheat

    crop_path = tmp_path / "cropland.tif"
    wheat_path = tmp_path / "wheat.tif"
    _write_raster(crop_path, cropland)
    _write_raster(wheat_path, wheat)
    return crop_path, wheat_path


def _polygon_at(pixel_row_start: int, pixel_col_start: int, size: int = 3) -> dict:
    """Build a GeoJSON polygon covering a pixel block in raster-array coords.

    Rows increase downward (raster convention), so the corresponding latitude
    range runs from ``_ORIGIN_LAT - row_end*_PIX`` up to
    ``_ORIGIN_LAT - row_start*_PIX``.
    """
    lon_min = _ORIGIN_LON + pixel_col_start * _PIX
    lon_max = lon_min + size * _PIX
    lat_max = _ORIGIN_LAT - pixel_row_start * _PIX
    lat_min = lat_max - size * _PIX
    return mapping(box(lon_min, lat_min, lon_max, lat_max))


def _feature(geom: dict, label: str, pid: str) -> dict:
    return {"geometry": geom, "properties": {"id": pid, "crop_type": label}}


def test_normalise_label_folds_compound_labels() -> None:
    assert normalise_label("bugdoy, paxta") == "bugdoy"
    assert normalise_label("BUGDOY, OTHER") == "bugdoy"
    assert normalise_label("other, bugdoy") == "other"
    assert normalise_label("paxta") == "paxta"


def test_wheat_label_in_wheat_quadrant_accepted(rasters: tuple[Path, Path]) -> None:
    cropland, wheat = rasters
    features = [_feature(_polygon_at(0, 0, 3), "bugdoy", "wheat-poly")]
    scored = score_polygons(features, cropland, wheat)
    assert len(scored) == 1
    s = scored[0]
    assert s.verdict == Verdict.ACCEPTED
    assert s.cropland_fraction == pytest.approx(1.0, abs=1e-3)
    assert s.wheat_fraction == pytest.approx(1.0, abs=1e-3)


def test_wheat_label_in_nonwheat_cropland_rejected(rasters: tuple[Path, Path]) -> None:
    # Upper-right quadrant is cropland but not wheat.
    cropland, wheat = rasters
    features = [_feature(_polygon_at(0, 6, 3), "bugdoy", "wheat-wrong")]
    scored = score_polygons(features, cropland, wheat)
    s = scored[0]
    assert s.verdict == Verdict.REJECTED
    assert s.cropland_fraction == pytest.approx(1.0, abs=1e-3)
    assert s.wheat_fraction == pytest.approx(0.0, abs=1e-3)
    assert "nonwheat" in s.reason


def test_cotton_label_in_nonwheat_cropland_accepted(rasters: tuple[Path, Path]) -> None:
    cropland, wheat = rasters
    features = [_feature(_polygon_at(0, 6, 3), "paxta", "cotton-ok")]
    scored = score_polygons(features, cropland, wheat)
    assert scored[0].verdict == Verdict.ACCEPTED


def test_cotton_label_in_wheat_quadrant_rejected(rasters: tuple[Path, Path]) -> None:
    cropland, wheat = rasters
    features = [_feature(_polygon_at(0, 0, 3), "paxta", "cotton-wrong")]
    assert score_polygons(features, cropland, wheat)[0].verdict == Verdict.REJECTED


def test_crop_label_on_noncropland_rejected(rasters: tuple[Path, Path]) -> None:
    # Lower half is non-cropland.
    cropland, wheat = rasters
    features = [_feature(_polygon_at(7, 0, 3), "bugdoy", "not-a-field")]
    s = score_polygons(features, cropland, wheat)[0]
    assert s.verdict == Verdict.REJECTED
    assert s.cropland_fraction == pytest.approx(0.0, abs=1e-3)


def test_noncrop_label_on_noncropland_accepted(rasters: tuple[Path, Path]) -> None:
    cropland, wheat = rasters
    features = [_feature(_polygon_at(7, 0, 3), "other", "bare-ground")]
    assert score_polygons(features, cropland, wheat)[0].verdict == Verdict.ACCEPTED


def test_noncrop_label_on_cropland_rejected(rasters: tuple[Path, Path]) -> None:
    cropland, wheat = rasters
    features = [_feature(_polygon_at(0, 0, 3), "other", "should-be-field")]
    assert score_polygons(features, cropland, wheat)[0].verdict == Verdict.REJECTED


def test_decide_review_zone_for_partial_cropland() -> None:
    v, _ = _decide(
        label="bugdoy",
        crop_frac=0.5,
        wheat_frac=0.0,
        n_pixels=100,
        have_wheat_layer=True,
        t=Thresholds(),
    )
    assert v == Verdict.REVIEW


def test_no_wheat_layer_accepts_crop_on_cropland(rasters: tuple[Path, Path]) -> None:
    cropland, _ = rasters
    features = [_feature(_polygon_at(0, 0, 3), "bugdoy", "wheat-only-cropland")]
    s = score_polygons(features, cropland, wheat_raster=None)[0]
    assert s.verdict == Verdict.ACCEPTED
    assert s.wheat_fraction == 0.0
    assert "cropland_only" in s.reason


def test_polygon_outside_raster_rejected(rasters: tuple[Path, Path]) -> None:
    cropland, wheat = rasters
    far_away = mapping(box(10.0, 10.0, 10.001, 10.001))
    features = [_feature(far_away, "bugdoy", "off-map")]
    s = score_polygons(features, cropland, wheat)[0]
    assert s.verdict == Verdict.REJECTED
    assert s.n_pixels == 0


def test_unknown_label_goes_to_review(rasters: tuple[Path, Path]) -> None:
    cropland, wheat = rasters
    features = [_feature(_polygon_at(0, 0, 3), "makka", "unknown-crop")]
    s = score_polygons(features, cropland, wheat)[0]
    assert s.verdict == Verdict.REVIEW
    assert s.reason.startswith("unknown_label")


def test_filter_labels_with_worldcereal_keeps_only_accepted(
    tmp_path: Path, rasters: tuple[Path, Path]
) -> None:
    import json as _json

    from gis_train.data.labels import filter_labels_with_worldcereal

    cropland, wheat = rasters
    features = [
        _feature(_polygon_at(0, 0, 3), "bugdoy", "keep"),  # wheat quadrant → accept
        _feature(_polygon_at(7, 0, 3), "bugdoy", "drop"),  # non-cropland → reject
        _feature(_polygon_at(0, 6, 3), "paxta", "keep2"),  # non-wheat cropland → accept
    ]
    labels_path = tmp_path / "labels.geojson"
    labels_path.write_text(_json.dumps({"type": "FeatureCollection", "features": features}))

    kept = filter_labels_with_worldcereal(
        labels_path=labels_path, cropland_raster=cropland, wheat_raster=wheat
    )
    kept_ids = {f["properties"]["id"] for f in kept}
    assert kept_ids == {"keep", "keep2"}
    for f in kept:
        assert f["properties"]["worldcereal_verdict"] == "accepted"
        assert "worldcereal_cropland_fraction" in f["properties"]
