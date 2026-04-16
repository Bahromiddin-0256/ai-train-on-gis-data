"""Fetch Sentinel-2 L2A scenes from Microsoft Planetary Computer.

The heavy STAC clients are imported lazily so that importing ``gis_train`` in a
lean environment (e.g. CI) never needs network or GDAL-linked wheels.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from gis_train.utils.geo import BBox
from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

_PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_S2_L2A_COLLECTION = "sentinel-2-l2a"


@dataclass
class DownloadResult:
    """Summary of a download call — kept small on purpose."""

    out_dir: Path
    scenes: int
    assets: int


def search_sentinel2_l2a(
    bbox: BBox,
    date_start: str,
    date_end: str,
    cloud_cover_max: float = 20.0,
    limit: int | None = None,
):
    """Return an iterable of STAC items matching the search window.

    Thin wrapper around ``pystac_client.Client.search`` — kept separate so
    callers can filter / inspect items before downloading.
    """
    try:
        from pystac_client import Client  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pystac-client is required for Sentinel-2 downloads") from exc

    catalog = Client.open(_PC_STAC_URL)
    _log.info(
        "Searching %s over %s for %s..%s (clouds <= %s%%)",
        _S2_L2A_COLLECTION,
        bbox.as_tuple(),
        date_start,
        date_end,
        cloud_cover_max,
    )
    search = catalog.search(
        collections=[_S2_L2A_COLLECTION],
        bbox=bbox.as_tuple(),
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        limit=limit,
    )
    return search.items()


def _download_with_retry(
    url: str,
    dest: Path,
    band: str,
    retries: int = 5,
    backoff: float = 3.0,
) -> None:
    """Stream *url* to *dest*, retrying on connection errors with exponential backoff."""
    import time

    import requests
    from tqdm import tqdm

    delay = backoff
    for attempt in range(1, retries + 1):
        try:
            dest.unlink(missing_ok=True)
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                total_bytes = int(resp.headers.get("Content-Length", 0)) or None
                with (
                    dest.open("wb") as fh,
                    tqdm(
                        total=total_bytes,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"  {band}",
                        leave=False,
                        dynamic_ncols=True,
                    ) as file_bar,
                ):
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        file_bar.update(len(chunk))
            return  # success
        except (requests.ConnectionError, requests.Timeout) as exc:
            dest.unlink(missing_ok=True)
            if attempt == retries:
                raise
            _log.warning(
                "attempt %d/%d failed (%s); retrying in %.0fs", attempt, retries, exc, delay
            )
            time.sleep(delay)
            delay *= 2


def _clip_tif_to_bbox(src_path: Path, dst_path: Path, bbox: BBox) -> None:
    """Crop *src_path* GeoTIFF to *bbox* and write to *dst_path*.

    Uses rasterio windowed reading so only the needed pixels are kept in memory.
    The output is written as a compressed Cloud-Optimised GeoTIFF.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    with rasterio.open(src_path) as src:
        # Re-project the WGS84 bbox into the tile's native CRS.
        dst_crs = CRS.from_epsg(4326)
        left, bottom, right, top = transform_bounds(
            dst_crs, src.crs, *bbox.as_tuple()
        )
        window = window_from_bounds(left, bottom, right, top, src.transform)
        window = window.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )
        if window.width <= 0 or window.height <= 0:
            raise ValueError(f"bbox does not intersect tile {src_path.name}")

        data = src.read(window=window)
        win_transform = src.window_transform(window)
        profile = src.profile.copy()

    profile.update(
        width=data.shape[2],
        height=data.shape[1],
        transform=win_transform,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)

    original_mb = src_path.stat().st_size / 1024 / 1024
    clipped_mb = dst_path.stat().st_size / 1024 / 1024
    _log.debug(
        "clipped %s: %.1f MB → %.1f MB (%.0f%% reduction)",
        src_path.name,
        original_mb,
        clipped_mb,
        (1 - clipped_mb / original_mb) * 100,
    )


def download_sentinel2_l2a(
    bbox: BBox,
    date_start: str,
    date_end: str,
    out_dir: Path | str,
    bands: Sequence[str] = ("B02", "B03", "B04", "B08"),
    cloud_cover_max: float = 20.0,
    limit: int | None = None,
    clip: bool = True,
) -> DownloadResult:
    """Download the requested Sentinel-2 L2A band assets into ``out_dir``.

    When *clip* is ``True`` (default) each tile is cropped to *bbox* after
    download, which can reduce file size by 10–50× for small AOIs.  The full
    tile is downloaded to a ``.tmp`` file first, then clipped; the ``.tmp`` is
    removed afterwards regardless of success.

    Returns a :class:`DownloadResult` with the number of scenes / assets
    written. Each asset is saved as ``{scene_id}_{band}.tif``.
    """
    try:
        import planetary_computer  # type: ignore[import-not-found]
        import requests  # noqa: F401 — checked here so error is early
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "planetary-computer and requests are required for Sentinel-2 downloads"
        ) from exc

    from tqdm import tqdm

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _log.info("Fetching scene list (metadata only, no download)...")
    all_items = list(
        search_sentinel2_l2a(
            bbox=bbox,
            date_start=date_start,
            date_end=date_end,
            cloud_cover_max=cloud_cover_max,
            limit=limit,
        )
    )
    n_scenes = len(all_items)
    n_files = n_scenes * len(bands)
    already = sum(1 for item in all_items for band in bands
                  if (out / f"{item.id}_{band}.tif").exists())
    _log.info(
        "Found %d scenes × %d bands = %d files  (already downloaded: %d, remaining: %d)",
        n_scenes, len(bands), n_files, already, n_files - already,
    )

    scenes = 0
    assets = 0
    scene_bar = tqdm(all_items, total=n_scenes, desc="scenes", unit="scene", dynamic_ncols=True)
    for item in scene_bar:
        scenes += 1
        scene_bar.set_postfix(scene=item.id[:30])
        signed = planetary_computer.sign(item)
        for band in bands:
            asset = signed.assets.get(band)
            if asset is None:
                _log.warning("scene %s missing band %s; skipping", item.id, band)
                continue
            dest = out / f"{item.id}_{band}.tif"
            tmp = dest.with_suffix(".tmp")
            if dest.exists():
                _log.debug("already have %s; skipping", dest.name)
                assets += 1
                continue
            # Remove any leftover partial download before starting fresh.
            tmp.unlink(missing_ok=True)
            try:
                _download_with_retry(asset.href, tmp, band=band, retries=5, backoff=3.0)
                if clip:
                    _clip_tif_to_bbox(tmp, dest, bbox)
                else:
                    tmp.rename(dest)
            finally:
                tmp.unlink(missing_ok=True)
            assets += 1

    _log.info("downloaded %d assets across %d scenes into %s", assets, scenes, out)
    return DownloadResult(out_dir=out, scenes=scenes, assets=assets)
