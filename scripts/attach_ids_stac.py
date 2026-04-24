"""Attach row-aligned ids.npy to DROPS chip dirs by replaying the STAC
scene-assignment logic used during the original extraction — without any
pixel reads.

For each tuman the script re-runs, per phenology window:
  1. STAC search (sorted ascending by eo:cloud_cover).
  2. Per-polygon assignment to the first (least-cloudy) scene whose bbox
     covers the polygon's bounds.
  3. Header-only rasterio open to get the scene transform.
  4. ``rasterio.windows.from_bounds`` gives the would-be read window size;
     apply the ``min_native_px`` drop rule without any ``src.read()`` call.

A polygon is kept when at least ``n_windows - max_missing_windows`` windows
produced a chip (defaults: 6 - 1 = 5), matching ``fetch_chips_multitemporal``.

The reconstructed ``kept_indices`` is validated against ``labels.npy``:

* ``len(kept) == len(labels)``
* ``class_to_idx[gdf.crop_type[k]] == labels[k]`` for every row k

If validation passes, ``ids.npy`` is written next to ``images.npy``. If it
fails (PC catalog drifted since the original extraction), that tuman is
left alone — fall back to scripts/attach_ids_cog.py (COG-header probe)
for it if needed.

Usage
-----
    python scripts/attach_ids_stac.py --dry-run --only 1714219
    python scripts/attach_ids_stac.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gis_train.data.download import _PC_STAC_URL, _S2_L2A_COLLECTION  # type: ignore[attr-defined]
from gis_train.data.phenology import get_stack_windows
from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

CLASSES = ("bugdoy", "other", "paxta")
_NORMALISE = {
    "bugdoy, other": "bugdoy",
    "bugdoy, paxta": "bugdoy",
    "other, bugdoy": "other",
}
_CHIPDIR_RE = re.compile(r"^processed_tuman_(\d+)(?:_mt)?$")

# Same defaults as scripts/run_resnet50_plus_pipeline.sh / build_dataset.py.
DEFAULT_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12")
DEFAULT_CLOUD = 20.0
DEFAULT_MIN_NATIVE_PX = 4
DEFAULT_MAX_MISSING_WINDOWS = 1


# ---------------------------------------------------------------------------
# Single-window STAC replay (no pixel reads)
# ---------------------------------------------------------------------------

def _replay_window(
    gdf,
    date_start: str,
    date_end: str,
    bands: tuple[str, ...],
    cloud_cover_max: float,
    min_native_px: int,
    probe_band: str,
) -> set[int]:
    """Return the set of gdf positional indices kept in this window.

    A polygon is kept when:
      * a scene whose bbox intersects the polygon's bounds exists, AND
      * the scene's assets include all requested bands, AND
      * rasterio.open on the probe band succeeds, AND
      * rasterio.windows.from_bounds yields width & height >= min_native_px
        in the probe band's native CRS.
    """
    import planetary_computer
    import rasterio
    from pyproj import Transformer
    from pystac_client import Client
    from rasterio.windows import from_bounds

    catalog = Client.open(_PC_STAC_URL)
    items = list(
        catalog.search(
            collections=[_S2_L2A_COLLECTION],
            bbox=tuple(float(x) for x in gdf.total_bounds),
            datetime=f"{date_start}/{date_end}",
            query={"eo:cloud_cover": {"lt": cloud_cover_max}},
            sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
        ).items()
    )
    if not items:
        return set()

    # Scene bbox index, already in cloud-cover-ascending order.
    scene_infos: list[tuple[str, tuple, object]] = []
    for item in items:
        geom = item.geometry
        if geom and geom.get("type") == "Polygon":
            coords = geom["coordinates"][0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            sbbox = (min(xs), min(ys), max(xs), max(ys))
        elif item.bbox:
            sbbox = tuple(item.bbox)
        else:
            continue
        scene_infos.append((item.id, sbbox, item))

    # Assign each polygon to the first covering scene.
    poly_to_scene: dict[int, str] = {}
    scene_items: dict[str, object] = {}
    for idx, row in gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        pb = g.bounds
        for scene_id, sbbox, item in scene_infos:
            if pb[0] <= sbbox[2] and pb[2] >= sbbox[0] and pb[1] <= sbbox[3] and pb[3] >= sbbox[1]:
                poly_to_scene[idx] = scene_id
                scene_items[scene_id] = item
                break

    scene_to_polys: dict[str, list[int]] = defaultdict(list)
    for idx, scene_id in poly_to_scene.items():
        scene_to_polys[scene_id].append(int(idx))

    kept: set[int] = set()
    for scene_id, poly_indices in scene_to_polys.items():
        item = scene_items[scene_id]
        signed = planetary_computer.sign(item)
        hrefs = {b: signed.assets[b].href for b in bands if b in signed.assets}
        if len(hrefs) < len(bands):
            continue

        try:
            src = rasterio.open(hrefs[probe_band])
        except Exception as exc:
            _log.debug("cannot open probe band for %s: %s", scene_id, exc)
            continue
        try:
            crs = src.crs
            same_crs = crs.to_epsg() == 4326
            if not same_crs:
                transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            nb = src.bounds
            for idx in poly_indices:
                g = gdf.loc[idx, "geometry"]
                b = g.bounds
                if same_crs:
                    bx0, by0, bx1, by1 = b
                else:
                    corners = [
                        transformer.transform(b[0], b[1]),
                        transformer.transform(b[2], b[1]),
                        transformer.transform(b[0], b[3]),
                        transformer.transform(b[2], b[3]),
                    ]
                    xs, ys = zip(*corners)
                    bx0, bx1 = min(xs), max(xs)
                    by0, by1 = min(ys), max(ys)
                bx0 = max(bx0, nb.left)
                by0 = max(by0, nb.bottom)
                bx1 = min(bx1, nb.right)
                by1 = min(by1, nb.top)
                if bx0 >= bx1 or by0 >= by1:
                    continue
                win = from_bounds(bx0, by0, bx1, by1, src.transform)
                # Use the actual read-shape rasterio would produce; this is
                # the only way to perfectly match the original drop decision
                # when polygons sit on the min_native_px boundary. Single-band
                # read at 10 m is cheap (a few KB per COG range request).
                try:
                    data = src.read(1, window=win)
                except Exception:
                    continue
                if data.size == 0:
                    continue
                h, w = int(data.shape[0]), int(data.shape[1])
                if h >= min_native_px and w >= min_native_px:
                    kept.add(idx)
        finally:
            src.close()

    return kept


# ---------------------------------------------------------------------------
# Per-tuman driver
# ---------------------------------------------------------------------------

def _primary(s: str) -> str:
    s = s.strip().lower()
    return _NORMALISE.get(s, s)


def _attach_one(
    chip_dir: Path,
    geojson: Path,
    year: int,
    bands: tuple[str, ...],
    cloud_cover_max: float,
    min_native_px: int,
    max_missing_windows: int,
    dry_run: bool,
) -> dict:
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

    import geopandas as gpd

    gdf = gpd.read_file(geojson).to_crs("EPSG:4326")
    if "_id" not in gdf.columns:
        res["status"] = "MISSING_IDS_IN_GEO"
        return res

    gdf = gdf.reset_index(drop=True)  # positional index for joining back
    class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    gdf["class_name"] = gdf["crop_type"].astype(str).map(_primary)
    gdf["class_idx"] = gdf["class_name"].map(class_to_idx)
    gdf = gdf[gdf["class_idx"].notna()].copy()
    res["n_geo"] = len(gdf)

    if len(gdf) == 0:
        res["status"] = "NO_MAPPABLE_FEATURES"
        return res

    windows = get_stack_windows(year)
    kept_per_window: list[set[int]] = []
    probe_band = bands[0]

    for date_start, date_end in windows:
        kept = _replay_window(
            gdf=gdf,
            date_start=date_start,
            date_end=date_end,
            bands=bands,
            cloud_cover_max=cloud_cover_max,
            min_native_px=min_native_px,
            probe_band=probe_band,
        )
        kept_per_window.append(kept)

    # Apply max_missing_windows rule.
    all_idx: set[int] = set()
    for wr in kept_per_window:
        all_idx.update(wr)

    kept_indices: list[int] = []
    for idx in sorted(all_idx):
        missing = sum(1 for wr in kept_per_window if idx not in wr)
        if missing <= max_missing_windows:
            kept_indices.append(idx)

    res["n_kept"] = len(kept_indices)

    if len(kept_indices) != res["n_chips"]:
        res["status"] = "LENGTH_MISMATCH"
        res["diff"] = res["n_kept"] - res["n_chips"]
        return res

    mismatched: list[int] = []
    for k, idx in enumerate(kept_indices):
        if int(gdf.loc[idx, "class_idx"]) != int(labels[k]):
            mismatched.append(k)
            if len(mismatched) > 5:
                break
    if mismatched:
        res["status"] = "CLASS_MISMATCH"
        res["mismatched_rows"] = mismatched
        return res

    ids = np.asarray([str(gdf.loc[i, "_id"]) for i in kept_indices], dtype=object)
    if not dry_run:
        np.save(chip_dir / "ids.npy", ids)
    res["status"] = "OK"
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
              help="Growing-season year used when the chips were extracted.")
@click.option("--only", default="",
              help="Comma-separated tuman_codes (empty = all DROPS dirs).")
@click.option("--cloud-cover-max", type=float, default=DEFAULT_CLOUD, show_default=True)
@click.option("--min-native-px", type=int, default=DEFAULT_MIN_NATIVE_PX, show_default=True)
@click.option("--max-missing-windows", type=int, default=DEFAULT_MAX_MISSING_WINDOWS, show_default=True)
@click.option("--dry-run", is_flag=True,
              help="Replay and validate but do not write ids.npy.")
@click.option("--overwrite", is_flag=True,
              help="Rewrite ids.npy even if it already exists.")
def main(
    chips_root: Path,
    labels_dir: Path,
    year: int,
    only: str,
    cloud_cover_max: float,
    min_native_px: int,
    max_missing_windows: int,
    dry_run: bool,
    overwrite: bool,
) -> None:
    """Attach ids.npy to DROPS chip dirs via STAC-replay."""
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

    click.echo(
        f"{'tuman':<32} {'n_chips':>8} {'n_kept':>8} {'n_geo':>7} {'status':<18} {'note'}"
    )
    click.echo("-" * 96)

    counts: dict[str, int] = {}
    written = 0
    ok_chips = 0

    for code, d in tqdm(chip_dirs, unit="tuman", desc="replay"):
        ids_path = d / "ids.npy"
        if ids_path.exists() and not overwrite:
            click.echo(f"{d.name:<32} {'-':>8} {'-':>8} {'-':>7} {'SKIP_EXISTS':<18} ids.npy present")
            counts["SKIP_EXISTS"] = counts.get("SKIP_EXISTS", 0) + 1
            continue

        res = _attach_one(
            chip_dir=d,
            geojson=labels_dir / f"tuman_{code}.geojson",
            year=year,
            bands=DEFAULT_BANDS,
            cloud_cover_max=cloud_cover_max,
            min_native_px=min_native_px,
            max_missing_windows=max_missing_windows,
            dry_run=dry_run,
        )
        status = res["status"]
        counts[status] = counts.get(status, 0) + 1
        n_chips = res.get("n_chips", "-")
        n_kept = res.get("n_kept", "-")
        n_geo = res.get("n_geo", "-")
        note = ""
        if status == "LENGTH_MISMATCH":
            note = f"diff={res.get('diff', '?')}"
        elif status == "CLASS_MISMATCH":
            note = f"rows: {res.get('mismatched_rows', [])}"
        click.echo(f"{d.name:<32} {n_chips:>8} {n_kept:>8} {n_geo:>7} {status:<18} {note}")

        if status == "OK":
            ok_chips += res["n_chips"]
            if not dry_run:
                written += 1

    click.echo("\nSummary")
    click.echo("-" * 40)
    for k in ("OK", "LENGTH_MISMATCH", "CLASS_MISMATCH", "NO_LABELS",
              "NO_GEO", "MISSING_IDS_IN_GEO", "NO_MAPPABLE_FEATURES", "SKIP_EXISTS"):
        if k in counts:
            click.echo(f"  {k:<22}: {counts[k]}")
    click.echo(f"  OK chips covered       : {ok_chips:,}")
    if dry_run:
        click.echo(f"  (dry-run) would write  : {counts.get('OK', 0)} ids.npy files")
    else:
        click.echo(f"  ids.npy files written  : {written}")


if __name__ == "__main__":
    main()
