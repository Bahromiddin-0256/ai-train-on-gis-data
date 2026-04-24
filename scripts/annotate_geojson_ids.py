"""Annotate every data/labels/tuman_<code>.geojson with each feature's
MongoDB ``_id`` via geometry match — non-destructively.

For each feature this sets:
    feature["id"]              = str(doc["_id"])   # top-level GeoJSON id
    feature["properties"]["_id"] = str(doc["_id"])

Matching key is a rounded WKT (7 decimal places ≈ 11 mm at the equator)
computed on both sides of the join. That is stable to float-repr noise
because the GeoJSON geometries were originally produced by parsing the same
Mongo ``geom_2`` WKT strings — they round-trip identically.

Fallback when a rounded-WKT lookup misses (rare, e.g. geometry recomputed
by a downstream tool): snap to the nearest Mongo geometry whose centroid
is within ``--centroid-tol`` degrees AND whose area matches within
``--area-rtol`` relative tolerance.

Usage
-----
Dry-run one tuman::

    python scripts/annotate_geojson_ids.py --only 1735207 --dry-run

Full run with .bak backups::

    python scripts/annotate_geojson_ids.py
"""

from __future__ import annotations

import binascii
import json
import re
import shutil
import sys
from pathlib import Path

import click
from shapely import wkb, wkt, to_wkt
from shapely.geometry import mapping, shape
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

_TUMAN_RE = re.compile(r"^tuman_(\d+)\.geojson$")


# ---------------------------------------------------------------------------
# Geometry parsing (WKT / EWKT / WKB-hex / GeoJSON dict)
# ---------------------------------------------------------------------------

def _parse_mongo_geom(g):
    if g is None:
        return None
    if isinstance(g, dict):
        try:
            return shape(g)
        except Exception:
            return None

    s = str(g).strip()
    up = s.upper()
    if up.startswith(("MULTIPOLYGON", "POLYGON", "POINT", "LINESTRING")):
        try:
            return wkt.loads(s)
        except Exception:
            return None
    if ";" in s and up.startswith("SRID"):
        try:
            return wkt.loads(s.split(";", 1)[1])
        except Exception:
            return None
    clean = s.replace(" ", "")
    try:
        return wkb.loads(binascii.unhexlify(clean), hex=False)
    except Exception:
        pass
    try:
        return wkb.loads(clean, hex=True)
    except Exception:
        return None


def _fingerprint(geom) -> str:
    """Canonical rounded-WKT fingerprint; identical geoms → identical string."""
    return to_wkt(geom, rounding_precision=7, trim=True)


# ---------------------------------------------------------------------------
# Per-tuman worker
# ---------------------------------------------------------------------------

def _annotate_one(
    path: Path,
    code: int,
    col,
    geom_field: str,
    centroid_tol: float,
    area_rtol: float,
) -> dict:
    """Annotate one tuman geojson in place. Returns stats dict."""
    gj = json.loads(path.read_text())
    feats = gj.get("features") or []

    stats: dict = {
        "file": path.name,
        "tuman_code": code,
        "n_features": len(feats),
        "matched_exact": 0,
        "matched_centroid": 0,
        "missed": 0,
        "ambiguous": 0,
    }
    if not feats:
        return stats

    # --- load the Mongo candidate set for this tuman --------------------
    mongo_rows: list[tuple[str, object]] = []  # (_id_str, geometry)
    cursor = col.find(
        {"tuman_code": code, geom_field: {"$exists": True, "$ne": None}},
        {"_id": 1, geom_field: 1},
    )
    for doc in cursor:
        g = _parse_mongo_geom(doc.get(geom_field))
        if g is None or g.is_empty:
            continue
        mongo_rows.append((str(doc["_id"]), g))

    if not mongo_rows:
        stats["missed"] = len(feats)
        return stats

    # --- fingerprint and centroid index ---------------------------------
    by_fp: dict[str, list[str]] = {}
    centroid_bucket: dict[tuple[float, float], list[tuple[str, float]]] = {}
    tol = centroid_tol
    for oid, g in mongo_rows:
        fp = _fingerprint(g)
        by_fp.setdefault(fp, []).append(oid)
        cx, cy = g.centroid.x, g.centroid.y
        key = (round(cx / tol) * tol, round(cy / tol) * tol)
        centroid_bucket.setdefault(key, []).append((oid, g.area))

    # --- annotate ------------------------------------------------------
    for f in feats:
        geom_dict = f.get("geometry")
        if not geom_dict:
            stats["missed"] += 1
            continue
        try:
            g = shape(geom_dict)
        except Exception:
            stats["missed"] += 1
            continue

        fp = _fingerprint(g)
        hits = by_fp.get(fp, [])
        if len(hits) == 1:
            oid = hits[0]
            stats["matched_exact"] += 1
        elif len(hits) > 1:
            # Multiple identical footprints — fall through to centroid+area
            # disambiguation below.
            hits = []
            stats["ambiguous"] += 1

        if not hits:
            # Centroid-nearest fallback; accept only if area matches within
            # area_rtol, which almost always uniquely identifies the polygon.
            cx, cy = g.centroid.x, g.centroid.y
            key = (round(cx / tol) * tol, round(cy / tol) * tol)
            candidates: list[tuple[str, float]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidates.extend(
                        centroid_bucket.get((key[0] + dx * tol, key[1] + dy * tol), [])
                    )
            a = g.area
            best: tuple[str, float] | None = None
            best_score = float("inf")
            for oid, ca in candidates:
                if not ca and not a:
                    score = 0.0
                else:
                    score = abs(ca - a) / max(a, ca, 1e-30)
                if score < best_score:
                    best_score = score
                    best = (oid, ca)
            if best is not None and best_score <= area_rtol:
                oid = best[0]
                stats["matched_centroid"] += 1
            else:
                stats["missed"] += 1
                continue

        f["id"] = oid
        props = f.setdefault("properties", {})
        props["_id"] = oid

    return stats, gj  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--uri", default="mongodb://localhost:27019", show_default=True)
@click.option("--db", default="gis-census", show_default=True)
@click.option("--collection", default="uzcosmos_flats", show_default=True)
@click.option("--geom-field", default="geom_2", show_default=True)
@click.option(
    "--labels-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/labels"),
    show_default=True,
)
@click.option(
    "--only",
    default="",
    help="Comma-separated tuman_codes to process (empty = all tuman_<code>.geojson).",
)
@click.option("--centroid-tol", type=float, default=1e-6, show_default=True,
              help="Centroid bucket width (degrees); ~10 cm at the equator.")
@click.option("--area-rtol", type=float, default=0.02, show_default=True,
              help="Max relative area gap to accept a centroid-nearest fallback match.")
@click.option("--backup/--no-backup", default=True, show_default=True,
              help="Move each existing geojson to .geojson.bak before overwrite.")
@click.option("--dry-run", is_flag=True,
              help="Match and report but do not write anything.")
def main(
    uri: str,
    db: str,
    collection: str,
    geom_field: str,
    labels_dir: Path,
    only: str,
    centroid_tol: float,
    area_rtol: float,
    backup: bool,
    dry_run: bool,
) -> None:
    """Annotate every data/labels/tuman_<code>.geojson with Mongo _ids."""
    from pymongo import MongoClient

    restrict: set[int] = set()
    if only:
        restrict = {int(x.strip()) for x in only.split(",") if x.strip()}

    targets: list[tuple[Path, int]] = []
    for p in sorted(labels_dir.glob("tuman_*.geojson")):
        m = _TUMAN_RE.match(p.name)
        if not m:
            continue
        code = int(m.group(1))
        if restrict and code not in restrict:
            continue
        targets.append((p, code))

    if not targets:
        raise click.ClickException("no tuman_<code>.geojson matched")

    click.echo(f"annotating {len(targets)} tuman geojson(s) from {db}.{collection}")

    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    col = client[db][collection]

    totals = dict(n_features=0, matched_exact=0, matched_centroid=0,
                  missed=0, ambiguous=0, files_written=0)
    per_file: list[dict] = []

    for path, code in tqdm(targets, unit="tuman"):
        result = _annotate_one(
            path=path, code=code, col=col,
            geom_field=geom_field,
            centroid_tol=centroid_tol,
            area_rtol=area_rtol,
        )
        stats, new_gj = result  # type: ignore[misc]
        per_file.append(stats)
        for k in ("n_features", "matched_exact", "matched_centroid",
                  "missed", "ambiguous"):
            totals[k] += stats[k]

        if not dry_run and stats["missed"] == 0:
            if backup and path.exists():
                shutil.copy2(path, path.with_suffix(".geojson.bak"))
            path.write_text(json.dumps(new_gj, ensure_ascii=False))
            totals["files_written"] += 1
        elif stats["missed"] > 0:
            _log.warning(
                "%s: %d/%d features unmatched — not writing",
                path.name, stats["missed"], stats["n_features"],
            )

    client.close()

    # --- per-file table ------------------------------------------------
    click.echo(
        f"\n{'file':<32} {'feats':>6} {'exact':>6} {'centr':>6} "
        f"{'miss':>5} {'ambig':>5}"
    )
    click.echo("-" * 68)
    for s in per_file:
        click.echo(
            f"{s['file']:<32} {s['n_features']:>6} {s['matched_exact']:>6} "
            f"{s['matched_centroid']:>6} {s['missed']:>5} {s['ambiguous']:>5}"
        )

    click.echo("\nTotals")
    click.echo("-" * 40)
    click.echo(f"  tumans          : {len(per_file)}")
    click.echo(f"  features        : {totals['n_features']:,}")
    click.echo(f"  matched (exact) : {totals['matched_exact']:,}")
    click.echo(f"  matched (centr) : {totals['matched_centroid']:,}")
    click.echo(f"  unmatched       : {totals['missed']:,}")
    click.echo(f"  ambiguous fps   : {totals['ambiguous']:,}")
    if dry_run:
        click.echo(f"\n(dry-run) would have written {len(per_file) - sum(1 for s in per_file if s['missed'])} files")
    else:
        click.echo(f"  files written   : {totals['files_written']}")


if __name__ == "__main__":
    main()
