"""Unit tests for ``gis_train.utils.geo``."""

from __future__ import annotations

import pytest

from gis_train.utils.geo import BBox, bbox_from_sequence


def test_bbox_as_tuple_is_stac_order() -> None:
    bbox = BBox(1.0, 2.0, 3.0, 4.0)
    assert bbox.as_tuple() == (1.0, 2.0, 3.0, 4.0)


def test_bbox_as_geojson_closes_the_ring() -> None:
    bbox = BBox(0.0, 0.0, 1.0, 1.0)
    gj = bbox.as_geojson()
    assert gj["type"] == "Polygon"
    ring = gj["coordinates"][0]
    assert ring[0] == ring[-1], "GeoJSON polygon rings must be closed"
    assert len(ring) == 5


def test_bbox_rejects_inverted_coords() -> None:
    with pytest.raises(ValueError, match="lon_min"):
        BBox(5.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="lat_min"):
        BBox(0.0, 5.0, 1.0, 1.0)


def test_bbox_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="latitude"):
        BBox(0.0, -91.0, 1.0, 1.0)


def test_bbox_from_sequence_wrong_length() -> None:
    with pytest.raises(ValueError, match="4 elements"):
        bbox_from_sequence([1.0, 2.0, 3.0])


def test_bbox_from_sequence_roundtrip() -> None:
    seq = [70.60, 40.10, 72.90, 41.40]
    bbox = bbox_from_sequence(seq)
    assert bbox.as_tuple() == tuple(seq)
