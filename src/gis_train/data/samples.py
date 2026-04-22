"""Convert GeoTIFF rasters + vector labels to ML-ready tabular format.

Samples pixel values from each band GeoTIFF at the centroid of every label
polygon and exports a CSV with one row per polygon.  Designed to work on the
output of ``download_sentinel2_l2a`` (individual scenes) or
``composite_scenes`` (temporal composites).
"""

from __future__ import annotations

from pathlib import Path

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

# Prefer composite files if present; fall back to individual scene files.
_COMPOSITE_GLOB = "composite_*.tif"
_MONTHLY_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]_*.tif"
_SCENE_GLOB = "*.tif"


def _find_band_tifs(raster_dir: Path) -> list[Path]:
    """Return the best available set of band TIFs for tabular export."""
    composites = sorted(raster_dir.glob(_COMPOSITE_GLOB))
    if composites:
        _log.info("Using %d composite TIFs", len(composites))
        return composites
    monthly = sorted(raster_dir.glob(_MONTHLY_GLOB))
    if monthly:
        _log.info("Using %d monthly composite TIFs", len(monthly))
        return monthly
    # Fall back to all scene TIFs
    all_tifs = sorted(raster_dir.glob(_SCENE_GLOB))
    _log.info("Using %d scene TIFs (no composites found)", len(all_tifs))
    return all_tifs


def raster_to_tabular(
    raster_dir: Path | str,
    labels_path: str | Path,
    out_csv: Path | str | None = None,
) -> Path:
    """Sample raster values at label polygon centroids and export as CSV.

    Parameters
    ----------
    raster_dir:
        Directory containing GeoTIFF files (output of download or composite).
    labels_path:
        GeoJSON (or any geopandas-readable) file with label polygons.  Must
        have a ``class`` or ``class_idx`` column.
    out_csv:
        Output path.  Defaults to ``{raster_dir}/samples.csv``.

    Returns
    -------
    Path
        Path to the written CSV.
    """
    try:
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "geopandas, pandas, and rasterio are required for raster_to_tabular"
        ) from exc

    raster_dir = Path(raster_dir)
    labels_path = Path(labels_path)
    out_csv = Path(out_csv) if out_csv else raster_dir / "samples.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")

    gdf = gpd.read_file(labels_path)
    if "class_idx" not in gdf.columns and "class" not in gdf.columns:
        raise ValueError("labels file must contain a 'class' or 'class_idx' column")

    label_col = "class_idx" if "class_idx" in gdf.columns else "class"

    # Reproject to EPSG:4326 if needed
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    centroids_lon = gdf.geometry.centroid.x.values
    centroids_lat = gdf.geometry.centroid.y.values

    tif_files = _find_band_tifs(raster_dir)
    if not tif_files:
        raise RuntimeError(f"No GeoTIFF files found in {raster_dir}")

    rows: dict[str, list] = {label_col: list(gdf[label_col])}

    for tif_path in tif_files:
        col_name = tif_path.stem
        values = []
        with rasterio.open(tif_path) as src:
            # Reproject centroids to raster CRS for sampling
            if src.crs.to_epsg() == 4326:
                xs, ys = centroids_lon, centroids_lat
            else:
                from pyproj import Transformer
                t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xs_arr, ys_arr = t.transform(centroids_lon, centroids_lat)
                xs, ys = xs_arr, ys_arr

            data = src.read(1)
            for x, y in zip(xs, ys):
                try:
                    row_idx, col_idx = src.index(x, y)
                    row_idx = int(np.clip(row_idx, 0, data.shape[0] - 1))
                    col_idx = int(np.clip(col_idx, 0, data.shape[1] - 1))
                    values.append(float(data[row_idx, col_idx]))
                except Exception:
                    values.append(float("nan"))
        rows[col_name] = values
        _log.debug("sampled %s → %d values", tif_path.name, len(values))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    _log.info("Exported %d samples × %d features → %s", len(df), len(df.columns) - 1, out_csv)
    return out_csv


__all__ = ["raster_to_tabular"]
