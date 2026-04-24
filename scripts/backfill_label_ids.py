"""Re-export every per-tuman GeoJSON in data/labels/ so each feature carries
its MongoDB ObjectId (top-level ``id`` + ``properties._id``).

The expected filename pattern is ``tuman_<tuman_code>.geojson``; the tuman_code
integer is parsed from the filename and fed to
``scripts/export_mongodb.py --tuman-code <code>``.

IMPORTANT — chip re-alignment
-----------------------------
Per-class random sampling in export_mongodb.py is non-deterministic unless a
``--seed`` is passed. This backfill always passes ``--seed`` (default 0, override
with ``--seed``) so repeated runs produce identical output, but the new GeoJSONs
will NOT match the unseeded originals feature-for-feature. Any existing chip
directories (``data/v6win/processed_tuman_*_mt/``) were extracted against the
original sample and therefore cannot be re-aligned to these new GeoJSONs — plan
to re-run chip extraction after the backfill so ``ids.npy`` is produced.

Files that do NOT match the ``tuman_<code>.geojson`` pattern
(``all_uzbekistan.geojson``, ``train_13tumans.geojson``, ``val_random.geojson``,
``dostlik.geojson``, ``uzcosmos.geojson``, ``zarbdor.geojson``) are skipped —
their source filter args are not recoverable from the filename alone.

Usage
-----
    # Dry run: show what would be re-exported
    python scripts/backfill_label_ids.py --dry-run

    # Real run with seed=0 and .bak backups
    python scripts/backfill_label_ids.py --per-class 500 --seed 0 --backup

    # Targeted
    python scripts/backfill_label_ids.py --only 1735207,1703206
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import click

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

_TUMAN_RE = re.compile(r"^tuman_(\d+)\.geojson$")


def _parse_tuman_code(path: Path) -> int | None:
    m = _TUMAN_RE.match(path.name)
    return int(m.group(1)) if m else None


@click.command()
@click.option(
    "--labels-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/labels"),
    show_default=True,
)
@click.option("--uri", default="mongodb://localhost:27019", show_default=True)
@click.option("--db", default="gis-census", show_default=True)
@click.option("--collection", default="uzcosmos_flats", show_default=True)
@click.option("--per-class", type=int, default=500, show_default=True,
              help="Max polygons per class per tuman (0 = no limit).")
@click.option("--seed", type=int, default=0, show_default=True,
              help="Random seed for per-class sampling (must be fixed to be reproducible).")
@click.option("--only", default="", help="Comma-separated tuman_codes to restrict to.")
@click.option("--backup/--no-backup", default=True, show_default=True,
              help="Move the existing .geojson to .geojson.bak before overwrite.")
@click.option("--dry-run", is_flag=True, help="Print commands without executing.")
def main(
    labels_dir: Path,
    uri: str,
    db: str,
    collection: str,
    per_class: int,
    seed: int,
    only: str,
    backup: bool,
    dry_run: bool,
) -> None:
    """Re-run export_mongodb.py for every data/labels/tuman_*.geojson."""
    restrict: set[int] = set()
    if only:
        restrict = {int(x.strip()) for x in only.split(",") if x.strip()}

    geojsons = sorted(labels_dir.glob("*.geojson"))
    tuman_files: list[tuple[Path, int]] = []
    skipped: list[str] = []
    for p in geojsons:
        code = _parse_tuman_code(p)
        if code is None:
            skipped.append(p.name)
            continue
        if restrict and code not in restrict:
            continue
        tuman_files.append((p, code))

    if not tuman_files:
        raise click.ClickException("no tuman_<code>.geojson files matched")

    click.echo(f"backfilling {len(tuman_files)} tuman file(s)  (seed={seed}, per_class={per_class})")
    if skipped:
        click.echo(f"skipping non-tuman files: {skipped}")

    export_script = Path(__file__).resolve().parent / "export_mongodb.py"
    python = sys.executable

    ok = 0
    failed: list[tuple[int, str]] = []
    for path, code in tuman_files:
        cmd = [
            python, str(export_script),
            "--uri", uri,
            "--db", db,
            "--collection", collection,
            "--tuman-code", str(code),
            "--per-class", str(per_class),
            "--seed", str(seed),
            "--out", str(path),
        ]
        click.echo(f"\n[{code}] {' '.join(cmd)}")
        if dry_run:
            continue

        if backup and path.exists():
            bak = path.with_suffix(".geojson.bak")
            shutil.copy2(path, bak)

        try:
            subprocess.run(cmd, check=True)
            ok += 1
        except subprocess.CalledProcessError as exc:
            failed.append((code, str(exc)))
            click.echo(f"  ✗ failed for tuman_code={code}: {exc}")

    click.echo(f"\nbackfill done: ok={ok}  failed={len(failed)}  total={len(tuman_files)}")
    if failed:
        for code, err in failed:
            click.echo(f"  failed: {code} — {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
