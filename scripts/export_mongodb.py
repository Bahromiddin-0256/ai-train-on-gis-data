"""Export crop-field polygons from MongoDB → GeoJSON for use with prepare_labels.py.

Usage
-----
    python scripts/export_mongodb.py \\
        --uri mongodb://localhost:27019 \\
        --db gis-census \\
        --collection uzcosmos_flats \\
        --out data/labels/uzcosmos.geojson

The script reads ``geom_2`` (WKT MULTIPOLYGON) and ``crop_type`` from each
document and writes a GeoJSON FeatureCollection.  Compound labels like
``"bugdoy, paxta"`` are normalised to the **primary** (first) crop so the
downstream classifier gets clean single-label targets:

    bugdoy, other   → bugdoy
    other, bugdoy   → other
    bugdoy, paxta   → bugdoy
    paxta           → paxta
    …

The resulting ``crop_type`` property is what you pass as ``--class-field``
to ``prepare_labels.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from shapely import wkt
from tqdm import tqdm

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

# Known compound labels → primary crop mapping.
# Add entries here if new compound values appear in the collection.
_NORMALISE: dict[str, str] = {
    "bugdoy, other": "bugdoy",
    "bugdoy, paxta": "bugdoy",
    "other, bugdoy": "other",
}


def _primary(crop_type: str) -> str:
    """Return the normalised primary crop label."""
    normalised = crop_type.strip().lower()
    return _NORMALISE.get(normalised, normalised)


def _wkt_to_geojson_geom(wkt_str: str) -> dict:
    """Convert a WKT geometry string to a GeoJSON geometry dict."""
    from shapely.geometry import mapping

    geom = wkt.loads(wkt_str)
    return mapping(geom)


@click.command()
@click.option(
    "--uri",
    default="mongodb://localhost:27019",
    show_default=True,
    help="MongoDB connection URI.",
)
@click.option("--db", default="gis-census", show_default=True, help="Database name.")
@click.option(
    "--collection",
    default="uzcosmos_flats",
    show_default=True,
    help="Collection name.",
)
@click.option(
    "--geom-field",
    default="geom_2",
    show_default=True,
    help="Document field containing the WKT geometry.",
)
@click.option(
    "--label-field",
    default="crop_type",
    show_default=True,
    help="Document field containing the crop-type label.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/labels/uzcosmos.geojson"),
    show_default=True,
    help="Output GeoJSON file path.",
)
@click.option(
    "--limit",
    default=0,
    show_default=True,
    help="Maximum documents to export (0 = all).",
)
@click.option(
    "--viloyat",
    default="",
    show_default=True,
    help="Filter to a single viloyat (empty = all).",
)
@click.option(
    "--tuman",
    default="",
    show_default=True,
    help="Filter to a single tuman by name (empty = all).",
)
@click.option(
    "--tuman-code",
    default=0,
    type=int,
    show_default=True,
    help="Filter by tuman_code integer (0 = no filter).",
)
@click.option(
    "--exclude-tuman-code",
    default="",
    show_default=True,
    help="Comma-separated tuman_code values to exclude (e.g. 1708215,1707210).",
)
@click.option(
    "--per-class",
    default=0,
    type=int,
    show_default=True,
    help="Randomly sample this many polygons per class (0 = no limit, use --limit instead).",
)
def main(
    uri: str,
    db: str,
    collection: str,
    geom_field: str,
    label_field: str,
    out: Path,
    limit: int,
    viloyat: str,
    tuman: str,
    tuman_code: int,
    exclude_tuman_code: str,
    per_class: int,
) -> None:
    """Export MongoDB crop polygons to a GeoJSON FeatureCollection."""
    try:
        from pymongo import MongoClient  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "pymongo is not installed. Run: pip install pymongo"
        ) from exc

    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    col = client[db][collection]

    query: dict = {}
    if viloyat:
        query["viloyat"] = {"$regex": viloyat, "$options": "i"}
    if tuman:
        query["tuman"] = {"$regex": tuman, "$options": "i"}
    if tuman_code:
        query["tuman_code"] = tuman_code
    if exclude_tuman_code:
        codes = [int(c.strip()) for c in exclude_tuman_code.split(",") if c.strip()]
        query["tuman_code"] = {"$nin": codes}

    projection = {geom_field: 1, label_field: 1, "_id": 0}
    cursor = col.find(query, projection)
    total = col.count_documents(query)
    if limit:
        cursor = cursor.limit(limit)
        total = min(total, limit)

    _log.info("exporting %d documents from %s.%s", total, db, collection)

    features: list[dict] = []
    skipped = 0

    for doc in tqdm(cursor, total=total, desc="exporting", unit="doc"):
        raw_geom = doc.get(geom_field)
        raw_label = doc.get(label_field)

        if raw_geom is None or raw_label is None:
            skipped += 1
            continue

        try:
            geom_dict = _wkt_to_geojson_geom(str(raw_geom))
        except Exception as exc:
            if skipped == 0:
                _log.warning("first geom parse error (will suppress further): %s", exc)
            skipped += 1
            continue

        label = _primary(str(raw_label))
        features.append(
            {
                "type": "Feature",
                "geometry": geom_dict,
                "properties": {label_field: label},
            }
        )

    _log.info("exported %d features, skipped %d", len(features), skipped)

    # --- per-class random sampling -----------------------------------------
    if per_class > 0:
        import random
        from collections import defaultdict

        by_class: dict[str, list[dict]] = defaultdict(list)
        for f in features:
            by_class[f["properties"][label_field]].append(f)

        sampled: list[dict] = []
        for cls, items in sorted(by_class.items()):
            chosen = random.sample(items, min(per_class, len(items)))
            sampled.extend(chosen)
            _log.info("class %r: sampled %d / %d", cls, len(chosen), len(items))
        features = sampled
        _log.info("total after per-class sampling: %d", len(features))

    out.parent.mkdir(parents=True, exist_ok=True)
    geojson = {"type": "FeatureCollection", "features": features}
    out.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    click.echo(f"wrote {len(features)} features → {out}")

    # Print class distribution
    from collections import Counter

    counts = Counter(f["properties"][label_field] for f in features)
    click.echo("\nClass distribution:")
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = n / len(features) * 100
        click.echo(f"  {cls:<20} {n:>8,}  ({pct:.1f}%)")

    # Compute and print the bounding box of all exported polygons.
    from shapely.geometry import shape

    all_geoms = [shape(f["geometry"]) for f in features]
    xs = [c for g in all_geoms for c in [g.bounds[0], g.bounds[2]]]
    ys = [c for g in all_geoms for c in [g.bounds[1], g.bounds[3]]]
    lon_min, lat_min, lon_max, lat_max = min(xs), min(ys), max(xs), max(ys)
    click.echo(
        f"\nBounding box of exported features (use with download_data.py):\n"
        f"  --bbox {lon_min:.4f},{lat_min:.4f},{lon_max:.4f},{lat_max:.4f}"
    )


if __name__ == "__main__":
    main()
