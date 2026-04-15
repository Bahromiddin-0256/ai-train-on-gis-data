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


def download_sentinel2_l2a(
    bbox: BBox,
    date_start: str,
    date_end: str,
    out_dir: Path | str,
    bands: Sequence[str] = ("B02", "B03", "B04", "B08"),
    cloud_cover_max: float = 20.0,
    limit: int | None = None,
) -> DownloadResult:
    """Download the requested Sentinel-2 L2A band assets into ``out_dir``.

    Returns a :class:`DownloadResult` with the number of scenes / assets
    written. Each asset is saved as ``{scene_id}_{band}.tif``.
    """
    try:
        import planetary_computer  # type: ignore[import-not-found]
        import requests
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "planetary-computer and requests are required for Sentinel-2 downloads"
        ) from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    items: Iterable = search_sentinel2_l2a(
        bbox=bbox,
        date_start=date_start,
        date_end=date_end,
        cloud_cover_max=cloud_cover_max,
        limit=limit,
    )

    scenes = 0
    assets = 0
    for item in items:
        scenes += 1
        signed = planetary_computer.sign(item)
        for band in bands:
            asset = signed.assets.get(band)
            if asset is None:
                _log.warning("scene %s missing band %s; skipping", item.id, band)
                continue
            dest = out / f"{item.id}_{band}.tif"
            if dest.exists():
                _log.debug("already have %s; skipping", dest.name)
                assets += 1
                continue
            _log.info("downloading %s -> %s", asset.href.split("?")[0], dest.name)
            with requests.get(asset.href, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            assets += 1

    _log.info("downloaded %d assets across %d scenes into %s", assets, scenes, out)
    return DownloadResult(out_dir=out, scenes=scenes, assets=assets)
