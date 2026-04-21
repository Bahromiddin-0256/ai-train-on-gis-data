"""Rasterize labeled polygons (GeoJSON) onto Sentinel-2 tiles — polygon-level chips.

For every polygon in the vector file:
  1. Find the Sentinel-2 scene whose extent covers the polygon.
  2. Do a windowed read (polygon bbox) across all band files for that scene.
  3. Resize the window to ``chip_size × chip_size`` (bilinear, via PyTorch).
  4. Assign the polygon's exact label — no majority-vote approximation.

This produces much cleaner labels than the old tile-slicing approach, because
each chip corresponds to exactly one labelled field.

Output: ``images.npy`` (N, C, chip_size, chip_size) + ``labels.npy`` (N,).

Multi-temporal mode (``--date-windows``):
  Pass multiple date windows as comma-separated ``start:end`` pairs.
  E.g. ``--date-windows 2025-04-01:2025-05-31,2025-06-01:2025-07-31``.
  Each polygon gets chips from all windows concatenated along the channel axis.
  Windows with no STAC coverage are zero-padded.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)


def _extract_record_local(
    record: tuple[int, bytes | None, int | None],
    scene_index: list[dict],
    chip_size: int,
    min_pixels: int,
    retries: int,
    retry_backoff: float,
) -> tuple[int, str, np.ndarray | None, int | None]:
    """Extract one polygon chip and return (idx, status, chip, label)."""
    from shapely import wkb

    idx, geom_wkb, label_idx = record
    if geom_wkb is None or label_idx is None:
        return idx, "no_label", None, None

    geom = wkb.loads(geom_wkb)
    gb = geom.bounds
    candidates = []
    for scene in scene_index:
        sb = scene["bounds_wgs84"]
        if gb[2] < sb[0] or gb[0] > sb[2] or gb[3] < sb[1] or gb[1] > sb[3]:
            continue
        candidates.append(scene)

    if not candidates:
        return idx, "no_scene", None, None

    for scene in candidates:
        delay = retry_backoff
        for attempt in range(retries + 1):
            try:
                chip = _extract_chip(scene, geom, chip_size=chip_size, min_native_px=min_pixels)
                if chip is not None:
                    return idx, "ok", chip, label_idx
                break
            except Exception:
                if attempt >= retries:
                    break
                time.sleep(delay)
                delay *= 2

    return idx, "too_small", None, None


def _process_local_chunk(
    records: list[tuple[int, bytes | None, int | None]],
    scene_index: list[dict],
    chip_size: int,
    min_pixels: int,
    num_threads: int,
    retries: int,
    retry_backoff: float,
) -> list[tuple[int, str, np.ndarray | None, int | None]]:
    """Process a list of polygon records, optionally using threads per process."""
    if num_threads <= 1:
        return [
            _extract_record_local(
                rec,
                scene_index=scene_index,
                chip_size=chip_size,
                min_pixels=min_pixels,
                retries=retries,
                retry_backoff=retry_backoff,
            )
            for rec in records
        ]

    out: list[tuple[int, str, np.ndarray | None, int | None]] = []
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {
            pool.submit(
                _extract_record_local,
                rec,
                scene_index,
                chip_size,
                min_pixels,
                retries,
                retry_backoff,
            ): rec[0]
            for rec in records
        }
        for future in as_completed(futures):
            out.append(future.result())
    return out


# ---------------------------------------------------------------------------
# Scene grouping (unchanged from previous version)
# ---------------------------------------------------------------------------

def _group_tiles_by_scene(tiles: list[Path]) -> list[list[Path]]:
    """Group per-band GeoTIFFs by scene ID.

    Files named ``<scene_id>_B02.tif``, ``<scene_id>_B03.tif`` … are grouped
    so each scene becomes one multi-band stack.
    """
    import re
    from collections import defaultdict

    band_suffix = re.compile(r"_B\d+[a-zA-Z]?$")
    groups: dict[str, list[Path]] = defaultdict(list)
    for tile in tiles:
        stem = band_suffix.sub("", tile.stem)
        groups[stem].append(tile)

    return [sorted(v) for v in groups.values()]


# ---------------------------------------------------------------------------
# Spatial index over scenes
# ---------------------------------------------------------------------------

def _build_scene_index(scene_groups: list[list[Path]]) -> list[dict]:
    """Return a list of scene descriptors with WGS-84 bounds for fast lookup.

    Each entry: ``{bounds_wgs84, crs, band_files, native_bounds}``.
    """
    import rasterio
    from pyproj import Transformer

    index = []
    for band_files in scene_groups:
        with rasterio.open(band_files[0]) as src:
            crs = src.crs
            b = src.bounds
            if crs.to_epsg() == 4326:
                bounds_wgs84 = (b.left, b.bottom, b.right, b.top)
            else:
                t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                corners = [
                    t.transform(b.left, b.bottom),
                    t.transform(b.right, b.bottom),
                    t.transform(b.left, b.top),
                    t.transform(b.right, b.top),
                ]
                xs, ys = zip(*corners)
                bounds_wgs84 = (min(xs), min(ys), max(xs), max(ys))

            index.append(
                dict(
                    bounds_wgs84=bounds_wgs84,
                    crs=crs,
                    band_files=band_files,
                    native_bounds=b,
                )
            )
    return index


# ---------------------------------------------------------------------------
# Per-polygon chip extraction
# ---------------------------------------------------------------------------

def _extract_chip(
    scene: dict,
    geom_wgs84,
    chip_size: int,
    min_native_px: int,
) -> np.ndarray | None:
    """Read the polygon's bounding window from ``scene`` and resize to chip_size².

    Returns ``None`` if the polygon does not overlap the scene or the extracted
    window is smaller than ``min_native_px`` in either dimension.
    """
    import rasterio
    import torch
    import torch.nn.functional as F
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    crs = scene["crs"]
    band_files = scene["band_files"]
    nb = scene["native_bounds"]

    # --- reproject polygon bbox from WGS-84 to tile CRS --------------------
    if crs.to_epsg() == 4326:
        bx0, by0, bx1, by1 = geom_wgs84.bounds
    else:
        t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        b = geom_wgs84.bounds  # (minx, miny, maxx, maxy) in WGS-84
        corners = [
            t.transform(b[0], b[1]),
            t.transform(b[2], b[1]),
            t.transform(b[0], b[3]),
            t.transform(b[2], b[3]),
        ]
        xs, ys = zip(*corners)
        bx0, bx1 = min(xs), max(xs)
        by0, by1 = min(ys), max(ys)

    # --- clip to tile extent ------------------------------------------------
    bx0 = max(bx0, nb.left)
    by0 = max(by0, nb.bottom)
    bx1 = min(bx1, nb.right)
    by1 = min(by1, nb.top)

    if bx0 >= bx1 or by0 >= by1:
        return None

    # --- windowed read across all band files --------------------------------
    bands: list[np.ndarray] = []
    for bf in band_files:
        with rasterio.open(bf) as src:
            win = from_bounds(bx0, by0, bx1, by1, src.transform)
            data = src.read(window=win)  # (1, h, w) for single-band files
            if data.size == 0 or data.shape[1] < 1 or data.shape[2] < 1:
                return None
            bands.append(data)

    arr = np.concatenate(bands, axis=0).astype(np.float32)  # (C, h, w)
    _, h, w = arr.shape

    if h < min_native_px or w < min_native_px:
        return None  # polygon too small — skip

    # --- resize to chip_size x chip_size ------------------------------------
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, C, h, w)
    resized = F.interpolate(
        tensor, size=(chip_size, chip_size), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0).numpy()  # (C, chip_size, chip_size)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--tiles-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory with downloaded Sentinel-2 GeoTIFF tiles. "
         "Required unless --from-stac is set.",
)
@click.option(
    "--vectors",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="GeoJSON file with labelled polygons (output of `export_mongodb.py`).",
)
@click.option(
    "--class-field",
    type=str,
    default="crop_type",
    show_default=True,
    help="Attribute column holding class names.",
)
@click.option(
    "--chip-size",
    type=int,
    default=64,
    show_default=True,
    help="Edge size (pixels) of extracted image chips.",
)
@click.option(
    "--min-pixels",
    type=int,
    default=4,
    show_default=True,
    help="Skip polygons whose native window is smaller than this in either dimension.",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/processed"),
    show_default=True,
    help="Destination for images.npy + labels.npy.",
)
@click.option(
    "--from-stac",
    is_flag=True,
    default=False,
    help="Read chips directly from Planetary Computer COGs — no tile download needed.",
)
@click.option(
    "--date-start",
    type=str,
    default="2025-06-01",
    show_default=True,
    help="Start date for STAC search (used only with --from-stac).",
)
@click.option(
    "--date-end",
    type=str,
    default="2025-08-31",
    show_default=True,
    help="End date for STAC search (used only with --from-stac).",
)
@click.option(
    "--bands",
    type=str,
    default="B02,B03,B04,B08",
    show_default=True,
    help="Comma-separated band IDs (used only with --from-stac).",
)
@click.option(
    "--date-windows",
    type=str,
    default=None,
    help=(
        "Comma-separated 'start:end' pairs for multi-temporal windows. "
        "E.g. '2025-04-01:2025-05-31,2025-06-01:2025-07-31,2025-08-01:2025-09-30'. "
        "When set, overrides --date-start/--date-end."
    ),
)
@click.option(
    "--indices",
    type=str,
    default="ndvi,evi,ndwi,ndre,msi,nbr",
    show_default=True,
    help="Comma-separated indices to append as extra channels (e.g., 'ndvi,ndre').",
)
@click.option(
    "--num-proc",
    type=int,
    default=1,
    show_default=True,
    help="Number of extraction processes for local-tile mode.",
)
@click.option(
    "--num-threads",
    type=int,
    default=1,
    show_default=True,
    help="Threads per process for local-tile mode.",
)
@click.option(
    "--retries",
    type=int,
    default=2,
    show_default=True,
    help="Retries for transient per-polygon extraction errors.",
)
@click.option(
    "--retry-backoff",
    type=float,
    default=0.5,
    show_default=True,
    help="Initial retry backoff in seconds (doubles per retry).",
)
def main(
    tiles_dir: Path | None,
    vectors: Path,
    class_field: str,
    chip_size: int,
    min_pixels: int,
    out: Path,
    from_stac: bool,
    date_start: str,
    date_end: str,
    bands: str,
    date_windows: str | None,
    indices: str | None,
    num_proc: int,
    num_threads: int,
    retries: int,
    retry_backoff: float,
) -> None:
    """Produce images.npy + labels.npy using polygon-level chips (one chip per field).

    Two modes:

    \b
    1. Local tiles (default):
         --tiles-dir data/raw/s2_zarbdor --vectors data/labels/zarbdor.geojson

    \b
    2. Direct STAC read (no download):
         --from-stac --vectors data/labels/val_random.geojson
    """
    import geopandas as gpd

    # --- load vector labels -------------------------------------------------
    gdf = gpd.read_file(vectors).to_crs("EPSG:4326")
    if class_field not in gdf.columns:
        raise click.UsageError(f"class field {class_field!r} not in {vectors}")

    classes = sorted(gdf[class_field].dropna().unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(classes)}
    _log.info("classes: %s", class_to_idx)

    # -----------------------------------------------------------------------
    # Mode A: read directly from Planetary Computer STAC
    # -----------------------------------------------------------------------
    if from_stac:
        band_list = [b.strip() for b in bands.split(",") if b.strip()]
        index_list = [i.strip() for i in indices.split(",")] if indices else None

        if index_list:
            from gis_train.data.indices import required_bands
            req = required_bands(index_list)
            missing = req - set(band_list)
            if missing:
                raise click.UsageError(
                    f"Requested indices {index_list} require bands {req}, "
                    f"but missing: {missing}"
                )

        gdf["class_idx"] = gdf[class_field].map(class_to_idx)
        gdf = gdf[gdf["class_idx"].notna()].copy()

        if date_windows is not None:
            from gis_train.data.download import fetch_chips_multitemporal

            # Parse "start1:end1,start2:end2,..." into list of (start, end) tuples
            windows: list[tuple[str, str]] = []
            for pair in date_windows.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                parts = pair.split(":")
                if len(parts) != 2:
                    raise click.UsageError(
                        f"Invalid date window {pair!r}; expected 'YYYY-MM-DD:YYYY-MM-DD'"
                    )
                windows.append((parts[0].strip(), parts[1].strip()))

            chips, label_list = fetch_chips_multitemporal(
                gdf=gdf,
                bands=band_list,
                date_windows=windows,
                chip_size=chip_size,
                min_native_px=min_pixels,
                indices=index_list,
            )
        else:
            from gis_train.data.download import fetch_chips_from_stac

            chips, label_list = fetch_chips_from_stac(
                gdf=gdf,
                bands=band_list,
                date_start=date_start,
                date_end=date_end,
                chip_size=chip_size,
                min_native_px=min_pixels,
            )

        if not chips:
            raise RuntimeError("no chips produced — check vectors and date range")

        images_arr = np.stack(chips).astype(np.float32)
        labels_arr = np.asarray(label_list, dtype=np.int64)

    # -----------------------------------------------------------------------
    # Mode B: read from pre-downloaded local tiles
    # -----------------------------------------------------------------------
    else:
        from tqdm import tqdm

        if tiles_dir is None:
            raise click.UsageError("--tiles-dir is required unless --from-stac is set")
        if not tiles_dir.exists():
            raise click.UsageError(f"tiles-dir does not exist: {tiles_dir}")

        tiles = sorted(tiles_dir.glob("*.tif"))
        if not tiles:
            raise click.UsageError(f"no .tif files found in {tiles_dir}")
        _log.info("found %d tile files", len(tiles))

        scene_groups = _group_tiles_by_scene(tiles)
        _log.info("grouped into %d scenes", len(scene_groups))
        scene_index = _build_scene_index(scene_groups)

        if num_proc < 1:
            raise click.UsageError("--num-proc must be >= 1")
        if num_threads < 1:
            raise click.UsageError("--num-threads must be >= 1")
        if retries < 0:
            raise click.UsageError("--retries must be >= 0")
        if retry_backoff < 0:
            raise click.UsageError("--retry-backoff must be >= 0")

        records: list[tuple[int, bytes | None, int | None]] = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            label_str = row.get(class_field)
            label_idx = class_to_idx.get(label_str)
            geom_wkb = None
            if geom is not None and not geom.is_empty:
                geom_wkb = bytes(geom.wkb)
            records.append((int(idx), geom_wkb, label_idx))

        results: list[tuple[int, str, np.ndarray | None, int | None]] = []
        if num_proc == 1:
            results = _process_local_chunk(
                records=records,
                scene_index=scene_index,
                chip_size=chip_size,
                min_pixels=min_pixels,
                num_threads=num_threads,
                retries=retries,
                retry_backoff=retry_backoff,
            )
        else:
            chunk_count = min(num_proc, max(1, len(records)))
            chunks: list[list[tuple[int, bytes | None, int | None]]] = [
                records[i::chunk_count] for i in range(chunk_count)
            ]
            with ProcessPoolExecutor(max_workers=chunk_count) as pool:
                futures = [
                    pool.submit(
                        _process_local_chunk,
                        chunk,
                        scene_index,
                        chip_size,
                        min_pixels,
                        num_threads,
                        retries,
                        retry_backoff,
                    )
                    for chunk in chunks
                    if chunk
                ]
                for future in tqdm(as_completed(futures), total=len(futures), desc="chunks", unit="chunk"):
                    results.extend(future.result())

        results.sort(key=lambda x: x[0])

        images = [chip for _, status, chip, _ in results if status == "ok" and chip is not None]
        labels = [label for _, status, _, label in results if status == "ok" and label is not None]
        skipped_no_scene = sum(1 for _, status, _, _ in results if status == "no_scene")
        skipped_small = sum(1 for _, status, _, _ in results if status == "too_small")
        skipped_label = sum(1 for _, status, _, _ in results if status == "no_label")

        if not images:
            raise RuntimeError("no chips produced — check tiles-dir and vectors overlap")

        images_arr = np.stack(images).astype(np.float32)
        labels_arr = np.asarray(labels, dtype=np.int64)

        click.echo(f"  skipped : {skipped_no_scene} (no scene), "
                   f"{skipped_small} (too small), {skipped_label} (no label)")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", images_arr)
    np.save(out / "labels.npy", labels_arr)

    click.echo(f"\nwrote {len(images_arr)} chips → {out}/")
    click.echo(f"  shape   : {images_arr.shape}")
    click.echo("\nClass distribution:")
    from collections import Counter
    counts = Counter(labels_arr.tolist())
    for idx, n in sorted(counts.items()):
        name = classes[idx]
        pct = n / len(labels_arr) * 100
        click.echo(f"  {name:<20} {n:>6}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
