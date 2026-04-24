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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np

from gis_train.data.phenology import format_windows_cli, get_stack_windows
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
    """Stack all processed_tuman_* subdirs into one images/labels .npy pair.

    Streams via np.lib.format.open_memmap so peak RAM stays bounded to a
    single tuman's chips (not the full combined array, which can exceed
    100 GB for the 6-window 90-channel phenology stack).
    """
    tuman_dirs = sorted(processed_base.glob("processed_tuman_*"))
    if not tuman_dirs:
        raise click.ClickException(f"No processed_tuman_* dirs found under {processed_base}")

    # Pass 1: inspect each tuman to determine total size and chip shape.
    plan: list[tuple[Path, Path, Path | None, int]] = []
    total_n = 0
    chip_shape: tuple[int, ...] | None = None
    chip_dtype = None
    label_dtype = None
    all_have_ids = True

    for d in tuman_dirs:
        imgs_path = d / "images.npy"
        lbls_path = d / "labels.npy"
        ids_path = d / "ids.npy"
        if not imgs_path.exists() or not lbls_path.exists():
            click.echo(f"  [skip] {d.name} — missing images.npy / labels.npy")
            continue
        imgs_header = np.load(imgs_path, mmap_mode="r")
        lbls_header = np.load(lbls_path, mmap_mode="r")
        n = int(imgs_header.shape[0])
        if n != int(lbls_header.shape[0]):
            click.echo(f"  [skip] {d.name} — images/labels length mismatch")
            continue
        if chip_shape is None:
            chip_shape = tuple(int(x) for x in imgs_header.shape[1:])
            chip_dtype = imgs_header.dtype
            label_dtype = lbls_header.dtype
        elif tuple(int(x) for x in imgs_header.shape[1:]) != chip_shape:
            raise click.ClickException(
                f"shape mismatch at {d.name}: expected (*,{chip_shape}), got {imgs_header.shape}"
            )
        has_ids = ids_path.exists()
        if has_ids and int(np.load(ids_path, allow_pickle=True).shape[0]) != n:
            click.echo(f"  [skip-ids] {d.name} — ids.npy length mismatch; omitting ids")
            has_ids = False
        all_have_ids = all_have_ids and has_ids
        plan.append((imgs_path, lbls_path, ids_path if has_ids else None, n))
        total_n += n
        click.echo(f"  + {d.name}: {n:,} chips{' (+ids)' if has_ids else ''}")

    if not plan or chip_shape is None:
        raise click.ClickException("Nothing to combine — all dirs were skipped.")

    out.mkdir(parents=True, exist_ok=True)
    out_imgs_path = out / "images.npy"
    out_lbls_path = out / "labels.npy"
    out_ids_path = out / "ids.npy"
    click.echo(
        f"\nAllocating output: images=({total_n},{chip_shape}) dtype={chip_dtype}, "
        f"labels=({total_n},) dtype={label_dtype}"
    )
    out_imgs = np.lib.format.open_memmap(
        out_imgs_path, mode="w+", dtype=chip_dtype, shape=(total_n, *chip_shape),
    )
    out_lbls = np.lib.format.open_memmap(
        out_lbls_path, mode="w+", dtype=label_dtype, shape=(total_n,),
    )
    # ids.npy holds Python strings (object dtype) — too variable-length for memmap;
    # build it in-memory (~100 bytes/row worst case, trivial for 74k rows).
    combined_ids: np.ndarray | None = (
        np.empty(total_n, dtype=object) if all_have_ids else None
    )

    # Pass 2: stream each tuman into the preallocated memmap.
    offset = 0
    for imgs_path, lbls_path, ids_path, n in plan:
        imgs = np.load(imgs_path, mmap_mode="r")
        lbls = np.load(lbls_path, mmap_mode="r")
        out_imgs[offset:offset + n] = imgs
        out_lbls[offset:offset + n] = lbls
        if combined_ids is not None and ids_path is not None:
            combined_ids[offset:offset + n] = np.load(ids_path, allow_pickle=True)
        offset += n
        del imgs, lbls  # release mmap handles
    assert offset == total_n, (offset, total_n)
    out_imgs.flush()
    out_lbls.flush()
    del out_imgs, out_lbls
    if combined_ids is not None:
        np.save(out_ids_path, combined_ids)
        click.echo(f"  wrote combined ids.npy → {out_ids_path}")
    elif not all_have_ids:
        click.echo(
            "  (some tumans missing ids.npy — combined ids.npy not written; "
            "re-export those tumans with the updated export_mongodb.py to populate)"
        )

    click.echo(f"\nCombined dataset → {out}/")
    click.echo(f"  Total chips : {total_n:,}")
    click.echo(f"  Shape       : ({total_n},{chip_shape})")

    # Reload labels via mmap for class distribution (cheap — a few MB).
    combined_labels = np.load(out_lbls_path, mmap_mode="r")
    counts = Counter(combined_labels.tolist())
    click.echo("\nClass distribution:")
    for idx, n in sorted(counts.items()):
        name = _CLASS_NAMES[idx] if idx < len(_CLASS_NAMES) else str(idx)
        pct = n / total_n * 100
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
@click.option(
    "--date-windows",
    default=format_windows_cli(get_stack_windows(2025)),
    show_default=True,
    help="Comma-separated 'start:end' date window pairs for multi-temporal extraction (phenologically optimised 6-window stack by default).",
)
@click.option("--indices", type=str, default="ndvi,evi,ndwi,ndre,msi,nbr", show_default=True,
              help="Comma-separated indices to compute (e.g. ndvi,ndre).")
@click.option("--labels-dir", type=Path, default=Path("data/labels"), show_default=True,
              help="Directory for per-tuman GeoJSON files.")
@click.option("--processed-base", type=Path, default=Path("data"), show_default=True,
              help="Parent directory for processed_tuman_* subdirectories.")
@click.option("--out", type=Path, default=Path("data/processed_regional"), show_default=True,
              help="Output directory for the combined dataset.")
@click.option("--num-proc", default=6, type=int, show_default=True,
              help="Processes for parallel window fetches inside prepare_labels.py (default matches 6-window stack).")
@click.option("--num-threads", default=4, type=int, show_default=True,
              help="Threads per window for parallel scene reads inside prepare_labels.py.")
@click.option("--parallel-tumans", default=1, type=int, show_default=True,
              help="Number of tumans to process concurrently. Each tuman spawns prepare_labels with num_proc*num_threads workers, so total concurrency = parallel_tumans × num_proc × num_threads.")
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
    num_proc: int,
    num_threads: int,
    parallel_tumans: int,
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
    tuman_logs_dir = processed_base / "logs"
    tuman_logs_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    def _process_tuman(row: dict) -> tuple[bool, str]:
        tcode = row["_id"].get("tuman_code")
        tname = (row["_id"].get("tuman") or str(tcode)).strip()
        vilname = (row["_id"].get("viloyat") or "unknown").strip()

        geojson_path = labels_dir / f"tuman_{tcode}.geojson"
        chips_dir = processed_base / f"processed_tuman_{tcode}_mt"
        tuman_log = tuman_logs_dir / f"tuman_{tcode}.log"

        if (chips_dir / "images.npy").exists():
            click.echo(f"[skip] {tname} ({vilname})  — chips already at {chips_dir}")
            return True, ""

        click.echo(f"[start] {tname}  ({vilname}, code={tcode}) → log: {tuman_log}")

        with tuman_log.open("w") as logf:
            # 2a. Export labels from MongoDB (skip if GeoJSON already present)
            if not geojson_path.exists():
                logf.write(f"→ exporting labels → {geojson_path}\n"); logf.flush()
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
                result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    msg = f"export_mongodb.py failed for tuman_code={tcode}"
                    click.echo(f"[ERROR] {msg}", err=True)
                    return False, msg
            else:
                logf.write(f"→ using existing labels: {geojson_path}\n"); logf.flush()

            # 2b. Extract chips via Planetary Computer STAC (multi-temporal)
            logf.write(f"→ extracting chips (STAC multi-temporal) → {chips_dir}\n"); logf.flush()
            cmd = [
                python, str(scripts_dir / "prepare_labels.py"),
                "--from-stac",
                "--vectors", str(geojson_path),
                "--date-windows", date_windows,
                "--bands", bands,
                "--chip-size", "64",
                "--out", str(chips_dir),
                "--num-proc", str(num_proc),
                "--num-threads", str(num_threads),
            ]
            if indices:
                cmd.extend(["--indices", indices])
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                msg = f"prepare_labels.py failed for tuman_code={tcode}"
                click.echo(f"[ERROR] {msg}", err=True)
                return False, msg

        click.echo(f"[done]  {tname}  ({vilname}, code={tcode})")
        return True, ""

    if parallel_tumans <= 1:
        for row in selected:
            ok, msg = _process_tuman(row)
            if not ok:
                errors.append(msg)
    else:
        click.echo(f"\nProcessing {len(selected)} tumans with {parallel_tumans} parallel workers...")
        with ThreadPoolExecutor(max_workers=parallel_tumans) as pool:
            futures = {pool.submit(_process_tuman, row): row for row in selected}
            for fut in as_completed(futures):
                try:
                    ok, msg = fut.result()
                    if not ok:
                        errors.append(msg)
                except Exception as exc:  # pragma: no cover - defensive
                    row = futures[fut]
                    tcode = row["_id"].get("tuman_code")
                    errors.append(f"tuman_code={tcode} crashed: {exc!r}")

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
