"""Small geospatial helpers — bbox handling and CRS utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Axis-aligned geographic bounding box in WGS84 (EPSG:4326).

    Coordinates are longitude (east+) and latitude (north+), all in degrees.
    """

    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

    def __post_init__(self) -> None:
        if self.lon_min >= self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) must be < lon_max ({self.lon_max})")
        if self.lat_min >= self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) must be < lat_max ({self.lat_max})")
        if not (-180.0 <= self.lon_min <= 180.0 and -180.0 <= self.lon_max <= 180.0):
            raise ValueError("longitude must be within [-180, 180]")
        if not (-90.0 <= self.lat_min <= 90.0 and -90.0 <= self.lat_max <= 90.0):
            raise ValueError("latitude must be within [-90, 90]")

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return (lon_min, lat_min, lon_max, lat_max) — STAC / rasterio order."""
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    def as_geojson(self) -> dict:
        """Return a GeoJSON Polygon covering this bbox."""
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [self.lon_min, self.lat_min],
                    [self.lon_max, self.lat_min],
                    [self.lon_max, self.lat_max],
                    [self.lon_min, self.lat_max],
                    [self.lon_min, self.lat_min],
                ]
            ],
        }


def bbox_from_sequence(seq: Sequence[float]) -> BBox:
    """Build a BBox from a 4-element sequence [lon_min, lat_min, lon_max, lat_max]."""
    if len(seq) != 4:
        raise ValueError(f"bbox sequence must have 4 elements, got {len(seq)}")
    return BBox(*map(float, seq))
