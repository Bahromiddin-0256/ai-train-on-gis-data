"""Rasterize labeled polygons (GeoJSON) onto Sentinel-2 tiles — polygon-level chips.

For every polygon in the vector file:
  1. Find the Sentinel-2 scene whose extent covers the polygon.
  2. Do a windowed read (polygon bbox) across all band files for that scene.
  3. Resize the window to ``chip_size × chip_size`` (bilinear, via PyTorch).
  4. Assign the polygon's exact label — no majority-vote approximation.

This produces much cleaner labels than the old tile-slicing approach, because
each chip corresponds to exactly one labelled field.

Output: ``images.npy`` (N, C, chip_size, chip_size) + ``labels.npy`` (N,).
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)


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
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory with Sentinel-2 GeoTIFF tiles (output of `download_data.py`).",
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
def main(
    tiles_dir: Path,
    vectors: Path,
    class_field: str,
    chip_size: int,
    min_pixels: int,
    out: Path,
) -> None:
    """Produce images.npy + labels.npy using polygon-level chips (one chip per field)."""
    import geopandas as gpd
    from tqdm import tqdm

    # --- load tiles and build scene index -----------------------------------
    tiles = sorted(tiles_dir.glob("*.tif"))
    if not tiles:
        raise click.UsageError(f"no .tif files found in {tiles_dir}")
    _log.info("found %d tile files", len(tiles))

    scene_groups = _group_tiles_by_scene(tiles)
    _log.info("grouped into %d scenes", len(scene_groups))
    scene_index = _build_scene_index(scene_groups)

    # --- load vector labels -------------------------------------------------
    gdf = gpd.read_file(vectors).to_crs("EPSG:4326")
    if class_field not in gdf.columns:
        raise click.UsageError(f"class field {class_field!r} not in {vectors}")

    classes = sorted(gdf[class_field].dropna().unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(classes)}
    _log.info("classes: %s", class_to_idx)

    # --- extract one chip per polygon ---------------------------------------
    images: list[np.ndarray] = []
    labels: list[int] = []
    skipped_no_scene = 0
    skipped_small = 0
    skipped_label = 0

    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="extracting chips", unit="poly"):
        geom = row.geometry
        label_str = row.get(class_field)

        if geom is None or geom.is_empty:
            skipped_label += 1
            continue
        if label_str not in class_to_idx:
            skipped_label += 1
            continue

        gb = geom.bounds  # WGS-84

        chip = None
        for scene in scene_index:
            sb = scene["bounds_wgs84"]
            # Quick AABB overlap check
            if gb[2] < sb[0] or gb[0] > sb[2] or gb[3] < sb[1] or gb[1] > sb[3]:
                continue
            chip = _extract_chip(scene, geom, chip_size=chip_size, min_native_px=min_pixels)
            if chip is not None:
                break

        if chip is None:
            # Distinguish "no scene" from "too small"
            covered = any(
                not (gb[2] < s["bounds_wgs84"][0] or gb[0] > s["bounds_wgs84"][2]
                     or gb[3] < s["bounds_wgs84"][1] or gb[1] > s["bounds_wgs84"][3])
                for s in scene_index
            )
            if covered:
                skipped_small += 1
            else:
                skipped_no_scene += 1
            continue

        images.append(chip)
        labels.append(class_to_idx[label_str])

    if not images:
        raise RuntimeError("no chips produced — check tiles-dir and vectors overlap")

    images_arr = np.stack(images).astype(np.float32)   # (N, C, chip_size, chip_size)
    labels_arr = np.asarray(labels, dtype=np.int64)     # (N,)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", images_arr)
    np.save(out / "labels.npy", labels_arr)

    click.echo(f"\nwrote {len(images_arr)} chips → {out}/")
    click.echo(f"  shape   : {images_arr.shape}")
    click.echo(f"  skipped : {skipped_no_scene} (no scene), "
               f"{skipped_small} (too small), {skipped_label} (no label)")
    click.echo(f"\nClass distribution:")
    from collections import Counter
    counts = Counter(labels_arr.tolist())
    for idx, n in sorted(counts.items()):
        name = classes[idx]
        pct = n / len(labels_arr) * 100
        click.echo(f"  {name:<20} {n:>6}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
