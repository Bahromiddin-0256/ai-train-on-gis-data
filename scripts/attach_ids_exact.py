"""Attach row-aligned ids.npy to the 'EXACT' chip dirs identified by
scripts/diagnose_chip_id_recovery.py — those where
``len(labels.npy) == count(features with mappable crop_type)``.

No STAC traffic, no pixel reads. The script:

1. Reads data/labels/tuman_<code>.geojson (which must carry _id per feature —
   run scripts/annotate_geojson_ids.py first).
2. Filters features to those whose crop_type maps to a known class, preserving
   the geojson's feature order. This mirrors
   ``gdf[gdf["class_idx"].notna()]`` in prepare_labels.py.
3. Requires ``len(filtered) == len(labels.npy)``; otherwise skips the dir as
   a DROPS case (needs the STAC-replay step).
4. Validates row-by-row: ``class_to_idx[crop_type] == labels.npy[k]`` for every
   row. Any mismatch aborts that tuman — indicates the geojson was resampled
   after the chips were extracted.
5. Writes ``<chip_dir>/ids.npy`` (object dtype, strings).

Usage
-----
    python scripts/attach_ids_exact.py --dry-run
    python scripts/attach_ids_exact.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click
import numpy as np

CLASSES = ("bugdoy", "other", "paxta")
_NORMALISE = {
    "bugdoy, other": "bugdoy",
    "bugdoy, paxta": "bugdoy",
    "other, bugdoy": "other",
}
_CHIPDIR_RE = re.compile(r"^processed_tuman_(\d+)(?:_mt)?$")


def _primary(s: str) -> str:
    s = s.strip().lower()
    return _NORMALISE.get(s, s)


def _attach_one(chip_dir: Path, geojson: Path) -> dict:
    res: dict = {"tuman": chip_dir.name, "status": "?"}
    labels_path = chip_dir / "labels.npy"
    if not labels_path.exists():
        res["status"] = "NO_LABELS"
        return res
    if not geojson.exists():
        res["status"] = "NO_GEO"
        return res

    labels = np.load(labels_path)
    res["n_chips"] = int(labels.shape[0])

    gj = json.loads(geojson.read_text())
    feats = gj.get("features") or []

    kept: list[tuple[int, str]] = []   # (class_idx, _id_str)
    missing_id = 0
    for f in feats:
        props = f.get("properties") or {}
        cls = _primary(str(props.get("crop_type", "")))
        if cls not in CLASSES:
            continue
        oid = props.get("_id") or f.get("id")
        if not oid:
            missing_id += 1
            continue
        kept.append((CLASSES.index(cls), str(oid)))

    res["n_geo"] = len(kept)
    res["missing_id_in_geo"] = missing_id

    if missing_id:
        res["status"] = "MISSING_IDS_IN_GEO"
        return res

    if len(kept) != len(labels):
        res["status"] = "DROPS"   # not exact; defer to STAC-replay step
        return res

    # Row-by-row class validation.
    mismatched: list[int] = []
    for k, (cls_idx, _) in enumerate(kept):
        if int(labels[k]) != cls_idx:
            mismatched.append(k)
            if len(mismatched) > 5:
                break

    if mismatched:
        res["status"] = "CLASS_MISMATCH"
        res["mismatched_rows"] = mismatched
        return res

    res["status"] = "OK"
    res["_ids"] = [oid for _, oid in kept]
    return res


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
@click.option("--only", default="",
              help="Comma-separated tuman_codes to process (empty = all).")
@click.option("--dry-run", is_flag=True,
              help="Validate and report but do not write ids.npy.")
@click.option("--overwrite", is_flag=True,
              help="Rewrite ids.npy even if it already exists.")
def main(
    chips_root: Path,
    labels_dir: Path,
    only: str,
    dry_run: bool,
    overwrite: bool,
) -> None:
    """Attach ids.npy to EXACT chip dirs using the annotated geojsons."""
    restrict: set[int] = set()
    if only:
        restrict = {int(x.strip()) for x in only.split(",") if x.strip()}

    chip_dirs: list[tuple[int, Path]] = []
    for d in sorted(chips_root.iterdir()):
        m = _CHIPDIR_RE.match(d.name)
        if d.is_dir() and m:
            code = int(m.group(1))
            if restrict and code not in restrict:
                continue
            chip_dirs.append((code, d))

    if not chip_dirs:
        raise click.ClickException(f"no processed_tuman_* dirs under {chips_root}")

    click.echo(f"{'tuman':<32} {'n_chips':>8} {'n_geo':>7} {'status':<18} {'note'}")
    click.echo("-" * 80)

    counts = {"OK": 0, "DROPS": 0, "CLASS_MISMATCH": 0,
              "MISSING_IDS_IN_GEO": 0, "NO_GEO": 0, "NO_LABELS": 0}
    written = 0
    ok_chips = 0

    for code, d in chip_dirs:
        ids_path = d / "ids.npy"
        if ids_path.exists() and not overwrite:
            click.echo(f"{d.name:<32} {'-':>8} {'-':>7} {'SKIP_EXISTS':<18} ids.npy present (use --overwrite)")
            continue

        res = _attach_one(d, labels_dir / f"tuman_{code}.geojson")
        status = res["status"]
        counts[status] = counts.get(status, 0) + 1
        n_chips = res.get("n_chips", "-")
        n_geo = res.get("n_geo", "-")
        note = ""
        if status == "DROPS":
            if isinstance(n_chips, int) and isinstance(n_geo, int):
                diff = n_geo - n_chips
                note = f"drops={diff}" if diff > 0 else f"extra={-diff}"
        elif status == "CLASS_MISMATCH":
            note = f"first rows: {res.get('mismatched_rows', [])}"
        elif status == "MISSING_IDS_IN_GEO":
            note = f"features missing _id: {res['missing_id_in_geo']}"

        click.echo(f"{d.name:<32} {n_chips:>8} {n_geo:>7} {status:<18} {note}")

        if status == "OK":
            ok_chips += res["n_chips"]
            if not dry_run:
                ids_arr = np.asarray(res["_ids"], dtype=object)
                np.save(ids_path, ids_arr)
                written += 1

    click.echo("\nSummary")
    click.echo("-" * 40)
    for k in ("OK", "DROPS", "CLASS_MISMATCH", "MISSING_IDS_IN_GEO", "NO_GEO", "NO_LABELS"):
        click.echo(f"  {k:<22}: {counts.get(k, 0)}")
    click.echo(f"  OK chips covered       : {ok_chips:,}")
    if dry_run:
        click.echo(f"  (dry-run) would write  : {counts.get('OK', 0)} ids.npy files")
    else:
        click.echo(f"  ids.npy files written  : {written}")


if __name__ == "__main__":
    main()
