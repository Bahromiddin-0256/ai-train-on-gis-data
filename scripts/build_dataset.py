"""Build a regional dataset by selecting N tumans per viloyat from MongoDB.

Pipeline per tuman:
  1. export_mongodb.py  --tuman-code CODE --per-class LIMIT
  2. prepare_labels.py  --from-stac --vectors FILE --date-start ... --date-end ...

Then combine all per-tuman chips into one dataset.

Usage examples
--------------
# See which tumans would be selected (no I/O):
python scripts/build_dataset.py --dry-run

# Full run: 3 tumans per viloyat, max 500 polygons per class per tuman
python scripts/build_dataset.py \\
    --n-per-viloyat 3 --per-class 500 \\
    --out data/processed_regional

# Just combine already-extracted chips (skip MongoDB / STAC steps):
python scripts/build_dataset.py --combine-only --out data/processed_regional
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click
import numpy as np

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

_CLASS_NAMES = ["bugdoy", "other", "paxta"]


# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

def _query_tumans(uri: str, db: str, collection: str) -> list[dict]:
    """Return all (viloyat, tuman, tuman_code, polygon_count) rows from MongoDB."""
    try:
        from pymongo import MongoClient  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("pymongo not installed. Run: pip install pymongo") from exc

    client = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    col = client[db][collection]

    pipeline = [
        {"$match": {"tuman_code": {"$exists": True, "$ne": None}}},
        {
            "$group": {
                "_id": {
                    "viloyat": "$viloyat",
                    "tuman": "$tuman",
                    "tuman_code": "$tuman_code",
                },
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id.viloyat": 1, "count": -1}},
    ]
    return list(col.aggregate(pipeline))


def _select_tumans(rows: list[dict], n_per_viloyat: int) -> list[dict]:
    """For each viloyat, keep the top-N tumans by polygon count."""
    by_viloyat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        vil = (row["_id"].get("viloyat") or "unknown").strip()
        by_viloyat[vil].append(row)

    selected: list[dict] = []
    for vil in sorted(by_viloyat):
        selected.extend(by_viloyat[vil][:n_per_viloyat])
    return selected


# ---------------------------------------------------------------------------
# Combine helper
# ---------------------------------------------------------------------------

def _combine(processed_base: Path, out: Path) -> None:
    """Stack all processed_tuman_* subdirs into one images/labels .npy pair."""
    tuman_dirs = sorted(processed_base.glob("processed_tuman_*"))
    if not tuman_dirs:
        raise click.ClickException(f"No processed_tuman_* dirs found under {processed_base}")

    all_images: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for d in tuman_dirs:
        imgs_path = d / "images.npy"
        lbls_path = d / "labels.npy"
        if not imgs_path.exists() or not lbls_path.exists():
            click.echo(f"  [skip] {d.name} — missing images.npy / labels.npy")
            continue
        imgs = np.load(imgs_path)
        lbls = np.load(lbls_path)
        all_images.append(imgs)
        all_labels.append(lbls)
        click.echo(f"  + {d.name}: {len(imgs):,} chips")

    if not all_images:
        raise click.ClickException("Nothing to combine — all dirs were skipped.")

    combined_images = np.concatenate(all_images, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", combined_images)
    np.save(out / "labels.npy", combined_labels)

    click.echo(f"\nCombined dataset → {out}/")
    click.echo(f"  Total chips : {len(combined_images):,}")
    click.echo(f"  Shape       : {combined_images.shape}")

    counts = Counter(combined_labels.tolist())
    click.echo("\nClass distribution:")
    for idx, n in sorted(counts.items()):
        name = _CLASS_NAMES[idx] if idx < len(_CLASS_NAMES) else str(idx)
        pct = n / len(combined_labels) * 100
        click.echo(f"  {name:<20} {n:>8,}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--uri", default="mongodb://localhost:27019", show_default=True,
              help="MongoDB connection URI.")
@click.option("--db", default="gis-census", show_default=True,
              help="Database name.")
@click.option("--collection", default="uzcosmos_flats", show_default=True,
              help="Collection name.")
@click.option("--n-per-viloyat", default=3, type=int, show_default=True,
              help="Tumans to select per viloyat (sorted by polygon count desc).")
@click.option("--per-class", default=500, type=int, show_default=True,
              help="Max polygons per class per tuman (0 = no limit).")
@click.option("--bands", default="B02,B03,B04,B05,B06,B07,B08,B11,B12", show_default=True,
              help="Comma-separated Sentinel-2 band IDs.")
@click.option("--date-windows",
              default="2025-04-01:2025-05-31,2025-06-01:2025-07-31,2025-08-01:2025-09-30",
              show_default=True,
              help="Comma-separated 'start:end' date window pairs for multi-temporal extraction.")
              help="Comma-separated 'start:end' date window pairs for multi-temporal extraction.")
@click.option("--indices", type=str, default="ndvi", show_default=True,
              help="Comma-separated indices to compute (e.g. ndvi,ndre).")
@click.option("--labels-dir", type=Path, default=Path("data/labels"), show_default=True,
              help="Directory for per-tuman GeoJSON files.")
@click.option("--processed-base", type=Path, default=Path("data"), show_default=True,
              help="Parent directory for processed_tuman_* subdirectories.")
@click.option("--out", type=Path, default=Path("data/processed_regional"), show_default=True,
              help="Output directory for the combined dataset.")
@click.option("--dry-run", is_flag=True,
              help="Print the selection plan without any I/O.")
@click.option("--combine-only", is_flag=True,
              help="Skip MongoDB/STAC steps; just combine existing processed_tuman_* dirs.")
def main(
    uri: str,
    db: str,
    collection: str,
    n_per_viloyat: int,
    per_class: int,
    bands: str,
    date_windows: str,
    indices: str,
    labels_dir: Path,
    processed_base: Path,
    out: Path,
    dry_run: bool,
    combine_only: bool,
) -> None:
    """Build a regional Sentinel-2 dataset with N tumans per viloyat."""

    scripts_dir = Path(__file__).parent
    python = sys.executable

    # ------------------------------------------------------------------
    # Combine-only mode: skip MongoDB / STAC, just merge existing chips
    # ------------------------------------------------------------------
    if combine_only:
        click.echo("combine-only mode — merging existing processed_tuman_* dirs...")
        _combine(processed_base, out)
        return

    # ------------------------------------------------------------------
    # Step 1: discover tumans from MongoDB
    # ------------------------------------------------------------------
    click.echo(f"Querying MongoDB ({uri}) for viloyat/tuman coverage...")
    rows = _query_tumans(uri, db, collection)
    selected = _select_tumans(rows, n_per_viloyat)

    # Print selection summary
    by_vil: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        vil = (row["_id"].get("viloyat") or "unknown").strip()
        by_vil[vil].append(row)

    click.echo(
        f"\nSelected {len(selected)} tumans across {len(by_vil)} viloyats "
        f"({n_per_viloyat} per viloyat):\n"
    )
    for vil in sorted(by_vil):
        click.echo(f"  {vil}:")
        for t in by_vil[vil]:
            tcode = t["_id"].get("tuman_code")
            tname = t["_id"].get("tuman") or "?"
            count = t["count"]
            chips_dir = processed_base / f"processed_tuman_{tcode}_mt"
            status = " [already processed]" if (chips_dir / "images.npy").exists() else ""
            click.echo(f"    {tname:<30} code={tcode}  polygons={count:>6,}{status}")

    if dry_run:
        click.echo("\n[dry-run] Nothing written.")
        return

    # ------------------------------------------------------------------
    # Step 2: export + extract chips per tuman
    # ------------------------------------------------------------------
    labels_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for row in selected:
        tcode = row["_id"].get("tuman_code")
        tname = (row["_id"].get("tuman") or str(tcode)).strip()
        vilname = (row["_id"].get("viloyat") or "unknown").strip()

        geojson_path = labels_dir / f"tuman_{tcode}.geojson"
        chips_dir = processed_base / f"processed_tuman_{tcode}_mt"

        if (chips_dir / "images.npy").exists():
            click.echo(f"\n[skip] {tname} ({vilname})  — chips already at {chips_dir}")
            continue

        click.echo(f"\n{'='*60}")
        click.echo(f"Processing: {tname}  ({vilname}, code={tcode})")

        # 2a. Export labels from MongoDB (skip if GeoJSON already present)
        if not geojson_path.exists():
            click.echo(f"  → exporting labels → {geojson_path}")
            cmd = [
                python, str(scripts_dir / "export_mongodb.py"),
                "--uri", uri,
                "--db", db,
                "--collection", collection,
                "--tuman-code", str(tcode),
                "--out", str(geojson_path),
            ]
            if per_class > 0:
                cmd += ["--per-class", str(per_class)]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                msg = f"export_mongodb.py failed for tuman_code={tcode}"
                click.echo(f"  [ERROR] {msg}", err=True)
                errors.append(msg)
                continue
        else:
            click.echo(f"  → using existing labels: {geojson_path}")

        # 2b. Extract chips via Planetary Computer STAC (multi-temporal)
        click.echo(f"  → extracting chips (STAC multi-temporal) → {chips_dir}")
        cmd = [
            python, str(scripts_dir / "prepare_labels.py"),
            "--from-stac",
            "--vectors", str(geojson_path),
            "--date-windows", date_windows,
            "--bands", bands,
            "--chip-size", "64",
            "--out", str(chips_dir),
        ]
        if indices:
            cmd.extend(["--indices", indices])
        result = subprocess.run(cmd)
        if result.returncode != 0:
            msg = f"prepare_labels.py failed for tuman_code={tcode}"
            click.echo(f"  [ERROR] {msg}", err=True)
            errors.append(msg)

    # ------------------------------------------------------------------
    # Step 3: combine all processed_tuman_* into one dataset
    # ------------------------------------------------------------------
    click.echo(f"\n{'='*60}")
    click.echo("Combining all processed_tuman_* directories...")
    _combine(processed_base, out)

    if errors:
        click.echo(f"\n[WARNING] {len(errors)} tuman(s) failed:")
        for e in errors:
            click.echo(f"  • {e}")


if __name__ == "__main__":
    main()
