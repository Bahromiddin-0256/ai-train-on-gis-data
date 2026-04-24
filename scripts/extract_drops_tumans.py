"""Re-extract chips for every processed_tuman_<code>_mt/ dir that does not
yet have an ids.npy — the 30 DROPS tumans where STAC-replay failed to exactly
reproduce the original drop mask.

Because the input geojsons already carry ``_id`` per feature (annotated by
``scripts/annotate_geojson_ids.py``) and ``prepare_labels.py`` has been
patched to emit ``ids.npy`` whenever the gdf has an ``_id`` column, a vanilla
``prepare_labels.py --from-stac`` run produces fresh chips + ids.npy together.

The new chips will be close to but not byte-identical to the old ones due to
STAC catalog drift. The old images.npy/labels.npy get overwritten. Old
``predictions_resnet50plus.csv/.npy`` in the same dir will be stale until
you re-run ``score_uzcosmos_resnet50plus.py --chips-root data/v6win``.

Usage
-----
Dry-run (prints commands, nothing executed)::

    python scripts/extract_drops_tumans.py --dry-run

Default run (sequential, matches build_dataset.py defaults)::

    python scripts/extract_drops_tumans.py

Parallel across tumans::

    python scripts/extract_drops_tumans.py --parallel-tumans 3

Subset by tuman_code::

    python scripts/extract_drops_tumans.py --only 1727259,1735207
"""

from __future__ import annotations

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gis_train.data.phenology import format_windows_cli, get_stack_windows
from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

_CHIPDIR_RE = re.compile(r"^processed_tuman_(\d+)(?:_mt)?$")

# Same defaults as build_dataset.py.
DEFAULT_BANDS = "B02,B03,B04,B05,B06,B07,B08,B11,B12"
DEFAULT_INDICES = "ndvi,evi,ndwi,ndre,msi,nbr"
DEFAULT_CHIP_SIZE = 64
DEFAULT_NUM_PROC = 6
DEFAULT_NUM_THREADS = 4


def _find_targets(
    chips_root: Path,
    labels_dir: Path,
    restrict: set[int],
    include_existing_ids: bool,
) -> list[tuple[int, Path, Path]]:
    """Return (tuman_code, chip_dir, geojson_path) for each DROPS dir."""
    out: list[tuple[int, Path, Path]] = []
    for d in sorted(chips_root.iterdir()):
        m = _CHIPDIR_RE.match(d.name) if d.is_dir() else None
        if not m:
            continue
        code = int(m.group(1))
        if restrict and code not in restrict:
            continue
        ids_path = d / "ids.npy"
        if ids_path.exists() and not include_existing_ids:
            continue
        geojson = labels_dir / f"tuman_{code}.geojson"
        if not geojson.exists():
            _log.warning("skip %s: no geojson at %s", d.name, geojson)
            continue
        out.append((code, d, geojson))
    return out


def _run_one(
    code: int,
    chip_dir: Path,
    geojson: Path,
    logs_dir: Path,
    year: int,
    bands: str,
    indices: str,
    num_proc: int,
    num_threads: int,
    dry_run: bool,
) -> tuple[int, int, str]:
    """Run prepare_labels.py for one tuman. Returns (code, returncode, logpath)."""
    python = sys.executable
    scripts_dir = Path(__file__).resolve().parent
    date_windows = format_windows_cli(get_stack_windows(year))

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"tuman_{code}_refresh.log"

    cmd = [
        python, str(scripts_dir / "prepare_labels.py"),
        "--from-stac",
        "--vectors", str(geojson),
        "--date-windows", date_windows,
        "--bands", bands,
        "--indices", indices,
        "--chip-size", str(DEFAULT_CHIP_SIZE),
        "--out", str(chip_dir),
        "--num-proc", str(num_proc),
        "--num-threads", str(num_threads),
    ]

    if dry_run:
        return code, 0, " ".join(cmd)

    with log_path.open("w") as logf:
        logf.write(f"cmd: {' '.join(cmd)}\n")
        logf.flush()
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT).returncode
    return code, rc, str(log_path)


@click.command()
@click.option(
    "--chips-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/v6win"),
    show_default=True,
)
@click.option(
    "--labels-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/labels"),
    show_default=True,
)
@click.option("--year", type=int, default=2025, show_default=True,
              help="Growing-season year for the 6-window phenology stack.")
@click.option("--bands", default=DEFAULT_BANDS, show_default=True)
@click.option("--indices", default=DEFAULT_INDICES, show_default=True)
@click.option("--num-proc", type=int, default=DEFAULT_NUM_PROC, show_default=True,
              help="Processes for parallel window fetches inside prepare_labels.py.")
@click.option("--num-threads", type=int, default=DEFAULT_NUM_THREADS, show_default=True,
              help="Threads per window for parallel scene reads inside prepare_labels.py.")
@click.option("--parallel-tumans", type=int, default=1, show_default=True,
              help="Run this many tumans concurrently (total STAC load = "
                   "parallel_tumans × num_proc × num_threads).")
@click.option("--only", default="",
              help="Comma-separated tuman_codes (empty = all dirs lacking ids.npy).")
@click.option("--include-existing-ids", is_flag=True,
              help="Also re-extract dirs that already have an ids.npy.")
@click.option("--dry-run", is_flag=True,
              help="Print the commands that would run; no STAC traffic.")
def main(
    chips_root: Path,
    labels_dir: Path,
    year: int,
    bands: str,
    indices: str,
    num_proc: int,
    num_threads: int,
    parallel_tumans: int,
    only: str,
    include_existing_ids: bool,
    dry_run: bool,
) -> None:
    """Re-extract the DROPS tumans with id-carrying geojsons."""
    restrict: set[int] = set()
    if only:
        restrict = {int(x.strip()) for x in only.split(",") if x.strip()}

    targets = _find_targets(chips_root, labels_dir, restrict, include_existing_ids)
    if not targets:
        click.echo("nothing to do — every chip dir already has ids.npy")
        return

    logs_dir = chips_root / "logs"
    click.echo(f"re-extracting {len(targets)} tuman(s)"
               f"  (num_proc={num_proc}, num_threads={num_threads}, "
               f"parallel_tumans={parallel_tumans})")
    for code, d, g in targets:
        click.echo(f"  - {code:>8}  {d}  (geojson: {g.name})")

    if dry_run:
        click.echo("\n[dry-run] commands:")
        for code, d, g in targets:
            _, _, cmd_str = _run_one(
                code, d, g, logs_dir, year, bands, indices,
                num_proc, num_threads, dry_run=True,
            )
            click.echo(f"  [{code}] {cmd_str}")
        return

    ok: list[int] = []
    failed: list[tuple[int, str]] = []

    if parallel_tumans <= 1:
        for code, d, g in targets:
            click.echo(f"[start] tuman {code} → {d}")
            _, rc, log_path = _run_one(
                code, d, g, logs_dir, year, bands, indices,
                num_proc, num_threads, dry_run=False,
            )
            if rc == 0:
                ok.append(code)
                click.echo(f"[done]  tuman {code}   log: {log_path}")
            else:
                failed.append((code, log_path))
                click.echo(f"[fail]  tuman {code}   log: {log_path}", err=True)
    else:
        with ThreadPoolExecutor(max_workers=parallel_tumans) as pool:
            futures = {
                pool.submit(
                    _run_one, code, d, g, logs_dir, year, bands, indices,
                    num_proc, num_threads, False,
                ): code
                for code, d, g in targets
            }
            for fut in as_completed(futures):
                code, rc, log_path = fut.result()
                if rc == 0:
                    ok.append(code)
                    click.echo(f"[done]  tuman {code}   log: {log_path}")
                else:
                    failed.append((code, log_path))
                    click.echo(f"[fail]  tuman {code}   log: {log_path}", err=True)

    click.echo(f"\nsummary: ok={len(ok)}  failed={len(failed)}  total={len(targets)}")
    if failed:
        click.echo("failed tumans (check logs):")
        for code, log in failed:
            click.echo(f"  {code}  →  {log}")
        sys.exit(1)


if __name__ == "__main__":
    main()
