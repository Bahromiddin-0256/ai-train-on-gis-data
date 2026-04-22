"""Report how much uzcosmos_flats area lies on non-cropland per a crop mask.

Reads polygons from MongoDB ``gis-census.uzcosmos_flats`` and a binary crop
mask GeoTIFF (Dynamic World / WorldCereal style: pixel value == ``crop-value``
means cropland, everything else is non-cropland). For each polygon it
intersects the mask, counts crop vs non-crop pixels, and converts to hectares
using geodesic area.

Usage (from repo root)::

    .venv/bin/python scripts/stats_noncrop.py \\
        --tif /home/prog/Downloads/Uzbekistan_Active_Crops_2025.tif \\
        --crop-value 4

The default crop value ``4`` matches Dynamic World's ``crops`` class.

The script prints totals overall and broken down by normalised crop_type
(same normalisation as export_mongodb.py, e.g. ``bugdoy, paxta`` → ``bugdoy``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
import rasterio
from pyproj import Geod
from rasterio.mask import mask as rio_mask
from shapely import wkt
from shapely.geometry import mapping, shape
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gis_train.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)

_NORMALISE: dict[str, str] = {
    "bugdoy, other": "bugdoy",
    "bugdoy, paxta": "bugdoy",
    "other, bugdoy": "other",
}

_GEOD = Geod(ellps="WGS84")


def _primary(crop_type: str) -> str:
    return _NORMALISE.get(crop_type.strip().lower(), crop_type.strip().lower())


def _geodesic_area_ha(geom) -> float:
    """Return geodesic area in hectares for a lon/lat geometry."""
    area_m2, _ = _GEOD.geometry_area_perimeter(geom)
    return abs(area_m2) / 10_000.0


@dataclass
class ClassStats:
    n: int = 0
    total_ha: float = 0.0
    noncrop_ha: float = 0.0
    noncrop_fractions: list[float] = field(default_factory=list)

    def add(self, total_ha: float, noncrop_ha: float) -> None:
        self.n += 1
        self.total_ha += total_ha
        self.noncrop_ha += noncrop_ha
        if total_ha > 0:
            self.noncrop_fractions.append(noncrop_ha / total_ha)

    def summary(self) -> dict:
        pct = (self.noncrop_ha / self.total_ha * 100.0) if self.total_ha else 0.0
        arr = np.asarray(self.noncrop_fractions) if self.noncrop_fractions else None
        return {
            "n": self.n,
            "total_ha": round(self.total_ha, 2),
            "noncrop_ha": round(self.noncrop_ha, 2),
            "noncrop_pct": round(pct, 2),
            "median_noncrop_frac": round(float(np.median(arr)), 3) if arr is not None else None,
            "p90_noncrop_frac": round(float(np.percentile(arr, 90)), 3) if arr is not None else None,
        }


def _mostly_noncrop_count(fractions: list[float], threshold: float) -> int:
    return int(sum(1 for f in fractions if f >= threshold))


@click.command()
@click.option(
    "--tif",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("/home/prog/Downloads/Uzbekistan_Active_Crops_2025.tif"),
    show_default=True,
    help="Path to the binary crop mask GeoTIFF.",
)
@click.option("--crop-value", type=int, default=4, show_default=True,
              help="Pixel value that denotes cropland (Dynamic World = 4).")
@click.option("--uri", default="mongodb://localhost:27019", show_default=True)
@click.option("--db", default="gis-census", show_default=True)
@click.option("--collection", default="uzcosmos_flats", show_default=True)
@click.option("--geom-field", default="geom_2", show_default=True)
@click.option("--label-field", default="crop_type", show_default=True)
@click.option("--viloyat", default="", show_default=True)
@click.option("--tuman", default="", show_default=True)
@click.option("--tuman-code", type=int, default=0, show_default=True)
@click.option("--limit", type=int, default=0, show_default=True,
              help="0 = no limit.")
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Polygons whose non-crop fraction ≥ this are counted as 'mostly non-crop'.",
)
@click.option(
    "--out-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional per-polygon CSV (id, crop_type, total_ha, noncrop_ha, noncrop_frac).",
)
def main(
    tif: Path,
    crop_value: int,
    uri: str,
    db: str,
    collection: str,
    geom_field: str,
    label_field: str,
    viloyat: str,
    tuman: str,
    tuman_code: int,
    limit: int,
    threshold: float,
    out_csv: Path | None,
) -> None:
    try:
        from pymongo import MongoClient  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("pymongo is not installed. Run: pip install pymongo") from exc

    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    col = client[db][collection]

    query: dict = {}
    if viloyat:
        query["viloyat"] = {"$regex": viloyat, "$options": "i"}
    if tuman:
        query["tuman"] = {"$regex": tuman, "$options": "i"}
    if tuman_code:
        query["tuman_code"] = tuman_code

    projection = {geom_field: 1, label_field: 1, "_id": 1}
    total = col.count_documents(query)
    if limit:
        total = min(total, limit)
    cursor = col.find(query, projection)
    if limit:
        cursor = cursor.limit(limit)

    _log.info("scoring %d documents from %s.%s against %s", total, db, collection, tif)

    overall = ClassStats()
    per_class: dict[str, ClassStats] = {}
    skipped_parse = 0
    skipped_outside = 0

    csv_fh = None
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_fh = out_csv.open("w", encoding="utf-8")
        csv_fh.write("id,crop_type,total_ha,noncrop_ha,noncrop_frac\n")

    with rasterio.open(tif) as src:
        left, bottom, right, top = src.bounds
        nodata = src.nodata

        for doc in tqdm(cursor, total=total, desc="scoring", unit="poly"):
            raw_geom = doc.get(geom_field)
            raw_label = doc.get(label_field)
            if raw_geom is None or raw_label is None:
                skipped_parse += 1
                continue

            try:
                geom = wkt.loads(str(raw_geom))
            except Exception:
                skipped_parse += 1
                continue

            minx, miny, maxx, maxy = geom.bounds
            if maxx < left or minx > right or maxy < bottom or miny > top:
                skipped_outside += 1
                continue

            total_ha = _geodesic_area_ha(geom)
            if total_ha <= 0:
                continue

            try:
                arr, _ = rio_mask(src, [mapping(geom)], crop=True, filled=False)
            except ValueError:
                # Polygon does not intersect raster.
                skipped_outside += 1
                continue

            band = arr[0]
            valid = ~band.mask if hasattr(band, "mask") else np.ones_like(band, dtype=bool)
            if nodata is not None:
                valid &= (np.asarray(band) != nodata)
            vals = np.asarray(band)[valid]
            n_pixels = int(vals.size)
            if n_pixels == 0:
                # Polygon falls fully outside / on nodata.
                cls = _primary(str(raw_label))
                per_class.setdefault(cls, ClassStats()).add(total_ha, total_ha)
                overall.add(total_ha, total_ha)
                if csv_fh is not None:
                    csv_fh.write(f"{doc.get('_id')},{cls},{total_ha:.4f},{total_ha:.4f},1.0\n")
                continue

            noncrop_pixels = int(np.count_nonzero(vals != crop_value))
            noncrop_frac = noncrop_pixels / n_pixels
            noncrop_ha = total_ha * noncrop_frac

            cls = _primary(str(raw_label))
            per_class.setdefault(cls, ClassStats()).add(total_ha, noncrop_ha)
            overall.add(total_ha, noncrop_ha)

            if csv_fh is not None:
                csv_fh.write(
                    f"{doc.get('_id')},{cls},{total_ha:.4f},{noncrop_ha:.4f},{noncrop_frac:.4f}\n"
                )

    if csv_fh is not None:
        csv_fh.close()

    # ── Report ──────────────────────────────────────────────────────────
    print("\n=== uzcosmos_flats vs crop mask ===")
    print(f"mask: {tif}  (crop pixel value = {crop_value})")
    if viloyat or tuman or tuman_code:
        print(f"filter: viloyat={viloyat!r} tuman={tuman!r} tuman_code={tuman_code}")
    print(f"skipped (parse): {skipped_parse}   skipped (outside raster): {skipped_outside}")

    s = overall.summary()
    print("\n-- overall --")
    print(f"  polygons          : {s['n']:,}")
    print(f"  total area        : {s['total_ha']:,.2f} ha")
    print(f"  on non-cropland   : {s['noncrop_ha']:,.2f} ha  ({s['noncrop_pct']}%)")
    print(f"  median non-crop % : {s['median_noncrop_frac']}")
    print(f"  p90 non-crop %    : {s['p90_noncrop_frac']}")
    mostly = _mostly_noncrop_count(overall.noncrop_fractions, threshold)
    print(f"  polygons ≥{threshold:.0%} non-crop: {mostly:,} "
          f"({(mostly / max(overall.n, 1) * 100):.2f}%)")

    if per_class:
        print("\n-- by crop_type (normalised) --")
        print(f"  {'class':<12}{'n':>10}{'total_ha':>14}{'noncrop_ha':>14}"
              f"{'noncrop_%':>11}{'med_frac':>10}{'p90_frac':>10}")
        for cls in sorted(per_class):
            s = per_class[cls].summary()
            print(f"  {cls:<12}{s['n']:>10,}{s['total_ha']:>14,.2f}"
                  f"{s['noncrop_ha']:>14,.2f}{s['noncrop_pct']:>10}%"
                  f"{str(s['median_noncrop_frac']):>10}{str(s['p90_noncrop_frac']):>10}")


if __name__ == "__main__":
    main()