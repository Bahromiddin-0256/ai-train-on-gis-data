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

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
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


def _split_date_range(date_start: str, date_end: str, parts: int) -> list[tuple[str, str]]:
    start = date.fromisoformat(date_start)
    end = date.fromisoformat(date_end)
    if end < start:
        raise click.BadParameter("--date-end must be >= --date-start")

    total_days = (end - start).days + 1
    parts = max(1, min(parts, total_days))

    base = total_days // parts
    rem = total_days % parts
    ranges: list[tuple[str, str]] = []

    cursor = start
    for i in range(parts):
        length = base + (1 if i < rem else 0)
        chunk_end = cursor + timedelta(days=length - 1)
        ranges.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)

    return ranges


def _write_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _download_chunk_worker(
    *,
    bbox_tuple: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    out_dir: str,
    bands: list[str],
    cloud_cover_max: float,
    limit: int | None,
    clip: bool,
    threads_per_process: int,
) -> tuple[int, int]:
    bbox_obj = BBox(*bbox_tuple)
    result = download_sentinel2_l2a(
        bbox=bbox_obj,
        date_start=date_start,
        date_end=date_end,
        out_dir=Path(out_dir),
        bands=bands,
        cloud_cover_max=cloud_cover_max,
        limit=limit,
        clip=clip,
        max_workers=threads_per_process,
    )
    return result.scenes, result.assets


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
    "--processes",
    type=int,
    default=1,
    show_default=True,
    help="Download processes for date-range chunking.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Resume chunked downloads from checkpoint on retry.",
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
    processes: int,
    resume: bool,
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
    threads_per_process = workers if workers is not None else min(8, os.cpu_count() or 4)
    if processes < 1:
        raise click.BadParameter("--processes must be >= 1", param_hint="--processes")

    _log.info("Date range: %s → %s", date_start, date_end)
    _log.info("Cloud cover max: %s%%", cloud_cover_max)
    _log.info(
        "AOI=%s bbox=%s bands=%s clip=%s processes=%d threads/process=%d",
        aoi if not bbox else "custom",
        bbox_obj.as_tuple(),
        band_list,
        not no_clip,
        processes,
        threads_per_process,
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

    out.mkdir(parents=True, exist_ok=True)

    if processes == 1:
        result = download_sentinel2_l2a(
            bbox=bbox_obj,
            date_start=date_start,
            date_end=date_end,
            out_dir=out,
            bands=band_list,
            cloud_cover_max=cloud_cover_max,
            limit=limit,
            clip=not no_clip,
            max_workers=threads_per_process,
        )
        click.echo(
            f"Downloaded {result.assets} assets across {result.scenes} scenes -> {result.out_dir}"
        )
    else:
        chunks = _split_date_range(date_start, date_end, processes)
        checkpoint_path = out / ".download_chunks_checkpoint.json"
        checkpoint_sig = {
            "bbox": bbox_obj.as_tuple(),
            "date_start": date_start,
            "date_end": date_end,
            "bands": band_list,
            "cloud_cover_max": cloud_cover_max,
            "limit": limit,
            "clip": not no_clip,
            "processes": len(chunks),
            "threads_per_process": threads_per_process,
        }

        completed: set[int] = set()
        if resume:
            prior = _load_checkpoint(checkpoint_path)
            if prior and prior.get("signature") == checkpoint_sig:
                completed = {int(i) for i in prior.get("completed", [])}
                _log.info("Resuming: %d/%d chunks already completed", len(completed), len(chunks))

        pending = [
            (idx, chunk_start, chunk_end)
            for idx, (chunk_start, chunk_end) in enumerate(chunks)
            if idx not in completed
        ]

        total_scenes = 0
        total_assets = 0
        if not pending:
            _log.info("All chunks already completed per checkpoint")
        else:
            with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
                futures = {
                    pool.submit(
                        _download_chunk_worker,
                        bbox_tuple=bbox_obj.as_tuple(),
                        date_start=chunk_start,
                        date_end=chunk_end,
                        out_dir=str(out),
                        bands=band_list,
                        cloud_cover_max=cloud_cover_max,
                        limit=limit,
                        clip=not no_clip,
                        threads_per_process=threads_per_process,
                    ): idx
                    for idx, chunk_start, chunk_end in pending
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    scenes, assets = future.result()
                    total_scenes += scenes
                    total_assets += assets
                    completed.add(idx)
                    if resume:
                        _write_checkpoint(
                            checkpoint_path,
                            {
                                "signature": checkpoint_sig,
                                "completed": sorted(completed),
                                "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                            },
                        )

        if resume and len(completed) == len(chunks):
            checkpoint_path.unlink(missing_ok=True)

        click.echo(
            f"Downloaded {total_assets} assets across {total_scenes} chunk-scenes -> {out}"
        )

    if sar:
        from gis_train.data.download import download_sentinel1
        sar_out = out.parent / "s1"
        sar_result = download_sentinel1(
            bbox=bbox_obj,
            date_start=date_start,
            date_end=date_end,
            out_dir=sar_out,
            max_workers=threads_per_process,
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
