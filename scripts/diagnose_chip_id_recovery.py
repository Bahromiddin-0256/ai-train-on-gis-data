"""Diagnostic: for each processed_tuman_<code>_mt/ chip dir, decide whether
its ids.npy can be rebuilt trivially (no drops) or needs a STAC-replay attach.

Compares:
    n_chips = len(processed_tuman_<code>_mt/labels.npy)
    n_geo   = features in data/labels/tuman_<code>.geojson whose crop_type
              maps to a known class (bugdoy/other/paxta)

Outcomes
--------
    EXACT        n_chips == n_geo     → ids.npy = gdf["_id"] directly.
    DROPS        n_chips <  n_geo     → need STAC-replay to reproduce drop mask.
    EXTRA        n_chips >  n_geo     → impossible under prepare_labels semantics;
                                        likely the wrong geojson was paired.
    NO_GEO       matching geojson is missing
    NO_IDS_GEO   geojson has no _id property yet (run backfill / geometry match)

Usage::

    python scripts/diagnose_chip_id_recovery.py \\
        --chips-root data/v6win \\
        --labels-dir data/labels
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import click
import numpy as np

CLASSES = ("bugdoy", "other", "paxta")
_CHIPDIR_RE = re.compile(r"^processed_tuman_(\d+)(?:_mt)?$")

# Known compound-label normalisations — mirrors scripts/export_mongodb.py.
_NORMALISE = {
    "bugdoy, other": "bugdoy",
    "bugdoy, paxta": "bugdoy",
    "other, bugdoy": "other",
}


def _primary(crop_type: str) -> str:
    s = crop_type.strip().lower()
    return _NORMALISE.get(s, s)


def _diagnose_one(chip_dir: Path, geojson: Path | None) -> dict:
    row: dict = {"tuman": chip_dir.name, "status": "?"}

    labels_path = chip_dir / "labels.npy"
    if not labels_path.exists():
        row["status"] = "NO_LABELS"
        return row
    lbls = np.load(labels_path)
    row["n_chips"] = int(lbls.shape[0])
    row["chip_dist"] = {CLASSES[i]: int((lbls == i).sum()) for i in range(len(CLASSES))}
    row["has_ids_npy"] = (chip_dir / "ids.npy").exists()

    if geojson is None or not geojson.exists():
        row["status"] = "NO_GEO"
        return row

    try:
        gj = json.loads(geojson.read_text())
    except Exception as exc:
        row["status"] = f"GEO_PARSE_FAIL: {exc}"
        return row

    feats = gj.get("features") or []
    row["n_geo_total"] = len(feats)

    # Count features with a class we actually map, matching prepare_labels
    # semantics (`gdf[gdf["class_idx"].notna()]`).
    dist = Counter()
    has_id = 0
    for f in feats:
        props = f.get("properties") or {}
        label = _primary(str(props.get("crop_type", "")))
        if label in CLASSES:
            dist[label] += 1
        if "_id" in props or f.get("id") is not None:
            has_id += 1
    row["geo_dist"] = {c: dist.get(c, 0) for c in CLASSES}
    row["n_geo_mapped"] = sum(dist.values())
    row["geo_has_ids"] = has_id == len(feats) and has_id > 0

    n_chips = row["n_chips"]
    n_geo = row["n_geo_mapped"]
    if n_chips == n_geo:
        row["status"] = "EXACT"
    elif n_chips < n_geo:
        row["status"] = "DROPS"
        row["drops"] = n_geo - n_chips
    else:
        row["status"] = "EXTRA"
        row["extra"] = n_chips - n_geo

    if row["status"] != "NO_GEO" and not row["geo_has_ids"]:
        row["status"] += "+NO_IDS_GEO"
    return row


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
def main(chips_root: Path, labels_dir: Path) -> None:
    """Classify each processed_tuman_*_mt dir by id-recovery difficulty."""
    chip_dirs: list[tuple[int, Path]] = []
    for d in sorted(chips_root.iterdir()):
        m = _CHIPDIR_RE.match(d.name)
        if d.is_dir() and m:
            chip_dirs.append((int(m.group(1)), d))

    if not chip_dirs:
        raise click.ClickException(f"no processed_tuman_* dirs under {chips_root}")

    rows: list[dict] = []
    for code, d in chip_dirs:
        gj = labels_dir / f"tuman_{code}.geojson"
        rows.append(_diagnose_one(d, gj if gj.exists() else None))

    # --- per-row table ---------------------------------------------------
    click.echo(
        f"{'tuman':<32} {'n_chips':>8} {'n_geo':>7} {'status':<22} {'ids?':<5} {'notes'}"
    )
    click.echo("-" * 96)
    for r in rows:
        tuman = r["tuman"]
        n_chips = r.get("n_chips", "-")
        n_geo = r.get("n_geo_mapped", "-")
        status = r["status"]
        ids_flag = "yes" if r.get("has_ids_npy") else "no"
        note = ""
        if r["status"].startswith("DROPS"):
            note = f"drops={r.get('drops', '?')}"
        elif r["status"].startswith("EXTRA"):
            note = f"extra={r.get('extra', '?')}  (unexpected — wrong geojson?)"
        click.echo(
            f"{tuman:<32} {n_chips:>8} {n_geo:>7} {status:<22} {ids_flag:<5} {note}"
        )

    # --- summary ---------------------------------------------------------
    n_exact = sum(1 for r in rows if r["status"].startswith("EXACT"))
    n_drops = sum(1 for r in rows if r["status"].startswith("DROPS"))
    n_extra = sum(1 for r in rows if r["status"].startswith("EXTRA"))
    n_nogeo = sum(1 for r in rows if r["status"] == "NO_GEO")
    n_have_ids_npy = sum(1 for r in rows if r.get("has_ids_npy"))
    n_need_geo_ids = sum(1 for r in rows if "NO_IDS_GEO" in r["status"])

    total_chips = sum(r.get("n_chips", 0) for r in rows)
    exact_chips = sum(r.get("n_chips", 0) for r in rows if r["status"].startswith("EXACT"))
    drops_chips = sum(r.get("n_chips", 0) for r in rows if r["status"].startswith("DROPS"))

    click.echo("\nSummary")
    click.echo("-" * 40)
    click.echo(f"  total tumans        : {len(rows)}")
    click.echo(f"  EXACT  (trivial)    : {n_exact}   chips={exact_chips:,}")
    click.echo(f"  DROPS  (need replay): {n_drops}   chips={drops_chips:,}")
    click.echo(f"  EXTRA  (mismatch)   : {n_extra}")
    click.echo(f"  NO_GEO              : {n_nogeo}")
    click.echo(f"  already have ids.npy: {n_have_ids_npy}")
    click.echo(f"  need geojson-id bkfl: {n_need_geo_ids}")
    click.echo(f"  total chips covered : {total_chips:,}")
    if total_chips:
        click.echo(
            f"  coverage trivially  : {100 * exact_chips / total_chips:.1f}% of chips"
        )


if __name__ == "__main__":
    main()
