"""CLI wrapper around ``gis_train.data.download``.

Example::

    python scripts/download_data.py \\
        --aoi fergana \\
        --date-start 2023-04-01 \\
        --date-end 2023-10-01 \\
        --bands B02,B03,B04,B05,B06,B07,B08,B11,B12 \\
        --cloud-cover-max 15 \\
        --compute-indices \\
        --composite monthly \\
        --sar \\
        --export-csv
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from gis_train.data.download import download_sentinel2_l2a
from gis_train.utils.geo import BBox, bbox_from_sequence
from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

# Preset areas of interest — extend as new regions are added.
_AOI_PRESETS: dict[str, BBox] = {
    "fergana": BBox(70.60, 40.10, 72.90, 41.40),
    "tashkent": BBox(68.80, 40.90, 69.80, 41.60),
    "samarkand": BBox(66.50, 39.40, 67.40, 39.90),
}


def _parse_bbox(value: str) -> BBox:
    parts = [float(x) for x in value.split(",")]
    return bbox_from_sequence(parts)


@click.command()
@click.option(
    "--aoi",
    type=click.Choice(sorted(_AOI_PRESETS.keys())),
    default="fergana",
    show_default=True,
    help="Named preset AOI.",
)
@click.option(
    "--bbox",
    type=str,
    default=None,
    help="Override AOI with a custom WGS84 bbox: lon_min,lat_min,lon_max,lat_max",
)
@click.option("--date-start", type=str, required=True, help="Start date (YYYY-MM-DD).")
@click.option("--date-end", type=str, required=True, help="End date (YYYY-MM-DD).")
@click.option(
    "--bands",
    type=str,
    default="B02,B03,B04,B05,B06,B07,B08,B11,B12",
    show_default=True,
    help="Comma-separated Sentinel-2 band IDs.",
)
@click.option(
    "--cloud-cover-max",
    type=float,
    default=20.0,
    show_default=True,
    help="Maximum scene cloud cover percentage.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of scenes to download (for testing).",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/raw/s2"),
    show_default=True,
    help="Output directory for downloaded GeoTIFFs.",
)
@click.option(
    "--no-clip",
    is_flag=True,
    default=False,
    help="Disable bbox clipping (download full 110×110 km tiles).",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Parallel download threads (default: min(8, CPU count)).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Only count scenes and files — do not download anything.",
)
@click.option(
    "--compute-indices",
    is_flag=True,
    default=False,
    help="Compute NDVI, EVI, NDWI, NDMI after download.",
)
@click.option(
    "--composite",
    type=click.Choice(["none", "monthly", "median"]),
    default="none",
    show_default=True,
    help="Temporal compositing strategy applied after download.",
)
@click.option(
    "--sar",
    is_flag=True,
    default=False,
    help="Download Sentinel-1 GRD VV/VH alongside Sentinel-2.",
)
@click.option(
    "--export-csv",
    is_flag=True,
    default=False,
    help="Export pixel samples as ML-ready CSV (requires data/labels.geojson).",
)
def main(
    aoi: str,
    bbox: str | None,
    date_start: str,
    date_end: str,
    bands: str,
    cloud_cover_max: float,
    limit: int | None,
    out: Path,
    no_clip: bool,
    workers: int | None,
    dry_run: bool,
    compute_indices: bool,
    composite: str,
    sar: bool,
    export_csv: bool,
) -> None:
    """Download Sentinel-2 L2A imagery for the given AOI + time window."""
    band_list = [b.strip() for b in bands.split(",") if b.strip()]
    if not band_list:
        raise click.BadParameter("At least one band must be specified.", param_hint="--bands")

    bbox_obj = _parse_bbox(bbox) if bbox else _AOI_PRESETS[aoi]
    effective_workers = workers if workers is not None else min(8, os.cpu_count() or 4)

    _log.info("Date range: %s → %s", date_start, date_end)
    _log.info("Cloud cover max: %s%%", cloud_cover_max)
    _log.info(
        "AOI=%s bbox=%s bands=%s clip=%s workers=%d",
        aoi if not bbox else "custom",
        bbox_obj.as_tuple(),
        band_list,
        not no_clip,
        effective_workers,
    )

    if dry_run:
        from gis_train.data.download import search_sentinel2_l2a
        items = list(search_sentinel2_l2a(
            bbox=bbox_obj,
            date_start=date_start,
            date_end=date_end,
            cloud_cover_max=cloud_cover_max,
            limit=limit,
        ))
        n_files = len(items) * len(band_list)
        click.echo(f"\nDry run — nothing downloaded.")
        click.echo(f"  Scenes : {len(items)}")
        click.echo(f"  Bands  : {len(band_list)}  {band_list}")
        click.echo(f"  Files  : {n_files}")
        click.echo(f"  Dates  : {items[-1].datetime.date() if items else '—'}"
                   f" → {items[0].datetime.date() if items else '—'}")
        return

    result = download_sentinel2_l2a(
        bbox=bbox_obj,
        date_start=date_start,
        date_end=date_end,
        out_dir=out,
        bands=band_list,
        cloud_cover_max=cloud_cover_max,
        limit=limit,
        clip=not no_clip,
        max_workers=effective_workers,
    )
    click.echo(
        f"Downloaded {result.assets} assets across {result.scenes} scenes -> {result.out_dir}"
    )

    if sar:
        from gis_train.data.download import download_sentinel1
        sar_out = out.parent / "s1"
        sar_result = download_sentinel1(
            bbox=bbox_obj,
            date_start=date_start,
            date_end=date_end,
            out_dir=sar_out,
            max_workers=effective_workers,
        )
        click.echo(
            f"SAR: downloaded {sar_result.assets} assets across"
            f" {sar_result.scenes} scenes -> {sar_result.out_dir}"
        )

    if composite != "none":
        from gis_train.data.composite import composite_scenes
        composite_scenes(out, strategy=composite, bands=band_list)
        click.echo(f"Composite ({composite}) written to {out}")

    if compute_indices:
        from gis_train.data.indices import compute_indices_folder
        compute_indices_folder(out, bands=band_list)
        click.echo(f"Spectral indices written to {out}")

    if export_csv:
        from gis_train.data.samples import raster_to_tabular
        csv_path = raster_to_tabular(out, labels_path="data/labels.geojson")
        click.echo(f"ML-ready CSV written to {csv_path}")


if __name__ == "__main__":
    main()
