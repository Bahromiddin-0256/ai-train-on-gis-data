"""Rasterize a labeled vector file (shapefile / GeoJSON) onto Sentinel-2 tiles.

This is the bridge between "user-supplied vector parcels" and the
``source=local`` path in ``CropDataModule``: it writes ``images.npy`` +
``labels.npy`` into the data directory, ready to be loaded without any
further geospatial dependencies at training time.

The script intentionally stays minimal — callers with more complex label
schemas should fork it rather than try to configure every edge case.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)


def _chip_rasters(tiles: list[Path], chip_size: int) -> np.ndarray:
    """Slice each GeoTIFF into non-overlapping chip_size x chip_size patches."""
    import rasterio

    chips: list[np.ndarray] = []
    for tile in tiles:
        with rasterio.open(tile) as src:
            arr = src.read()  # (C, H, W)
        _, h, w = arr.shape
        for r in range(0, h - chip_size + 1, chip_size):
            for c in range(0, w - chip_size + 1, chip_size):
                chips.append(arr[:, r : r + chip_size, c : c + chip_size])
    if not chips:
        raise RuntimeError("no chips produced — check input tiles and chip_size")
    return np.stack(chips).astype(np.float32)


def _rasterize_labels(vectors_path: Path, tiles: list[Path], class_field: str) -> np.ndarray:
    """Return one integer class label per chip produced by ``_chip_rasters``."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import box

    gdf = gpd.read_file(vectors_path)
    if class_field not in gdf.columns:
        raise ValueError(f"class field {class_field!r} missing from {vectors_path}")
    classes = sorted(gdf[class_field].dropna().unique().tolist())
    class_to_idx = {name: i for i, name in enumerate(classes)}
    _log.info("found %d classes: %s", len(classes), class_to_idx)

    labels: list[int] = []
    chip_size = None
    for tile in tiles:
        with rasterio.open(tile) as src:
            transform = src.transform
            height, width = src.height, src.width
            bounds = src.bounds
            proj_gdf = gdf.to_crs(src.crs)
        # Rasterize once per tile at native resolution, then aggregate per chip.
        shapes = (
            (geom, class_to_idx[cls])
            for geom, cls in zip(proj_gdf.geometry, proj_gdf[class_field], strict=False)
            if geom is not None and cls in class_to_idx and geom.intersects(box(*bounds))
        )
        label_raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=-1,
            dtype=np.int32,
        )
        if chip_size is None:
            chip_size = min(height, width, 64)
        for r in range(0, height - chip_size + 1, chip_size):
            for c in range(0, width - chip_size + 1, chip_size):
                patch = label_raster[r : r + chip_size, c : c + chip_size]
                valid = patch[patch >= 0]
                if valid.size == 0:
                    labels.append(0)  # default to first class for all-unlabeled chips
                else:
                    # Majority vote.
                    values, counts = np.unique(valid, return_counts=True)
                    labels.append(int(values[counts.argmax()]))
    return np.asarray(labels, dtype=np.int64)


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
    help="Vector file with labeled polygons (shapefile or GeoJSON).",
)
@click.option(
    "--class-field",
    type=str,
    default="class",
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
    out: Path,
) -> None:
    """Produce images.npy + labels.npy for the ``source=local`` pipeline."""
    tiles = sorted(tiles_dir.glob("*.tif"))
    if not tiles:
        raise click.UsageError(f"no .tif files found in {tiles_dir}")
    _log.info("found %d tiles", len(tiles))

    images = _chip_rasters(tiles, chip_size=chip_size)
    labels = _rasterize_labels(vectors, tiles, class_field=class_field)
    if len(images) != len(labels):
        raise RuntimeError(f"image/label count mismatch: {len(images)} vs {len(labels)}")

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", images)
    np.save(out / "labels.npy", labels)
    click.echo(f"wrote {len(images)} chips to {out}/images.npy and {out}/labels.npy")


if __name__ == "__main__":
    main()
