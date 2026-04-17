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


def fetch_chips_from_stac(
    gdf,
    bands: Sequence[str],
    date_start: str,
    date_end: str,
    chip_size: int = 64,
    cloud_cover_max: float = 20.0,
    min_native_px: int = 4,
) -> tuple[list, list[int]]:
    """Read per-polygon chips directly from Planetary Computer COGs (no tile download).

    Scene-first approach — O(scenes) STAC queries instead of O(polygons):
      1. One STAC query covering the full bbox of all polygons.
      2. Each polygon is assigned to the least-cloudy scene that covers it.
      3. Polygons are grouped by scene; each COG band file is opened once per scene.
      4. All polygon windows are extracted in a single pass per scene.

    Returns ``(chips, labels)`` where chips is a list of ``np.ndarray`` and
    labels is a list of int class indices.

    Parameters
    ----------
    gdf:
        GeoDataFrame in EPSG:4326.  Must have a ``class_idx`` integer column.
    """
    from collections import defaultdict

    import numpy as np
    import planetary_computer  # type: ignore[import-not-found]
    import torch
    import torch.nn.functional as F
    from pyproj import Transformer
    from pystac_client import Client  # type: ignore[import-not-found]
    from rasterio.windows import from_bounds
    from tqdm import tqdm

    import rasterio

    catalog = Client.open(_PC_STAC_URL)

    # ------------------------------------------------------------------
    # Step 1: single STAC query covering all polygons
    # ------------------------------------------------------------------
    total_bounds = tuple(float(x) for x in gdf.total_bounds)  # (minx, miny, maxx, maxy)
    _log.info(
        "Querying STAC: bbox=(%.4f,%.4f,%.4f,%.4f) %s..%s",
        *total_bounds, date_start, date_end,
    )
    all_items = list(
        catalog.search(
            collections=[_S2_L2A_COLLECTION],
            bbox=total_bounds,
            datetime=f"{date_start}/{date_end}",
            query={"eo:cloud_cover": {"lt": cloud_cover_max}},
            sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
        ).items()
    )
    _log.info("Found %d candidate scenes", len(all_items))

    if not all_items:
        _log.warning("No scenes found — check date range and cloud_cover_max")
        return [], []

    # ------------------------------------------------------------------
    # Step 2: build scene bbox index (WGS-84)
    # sorted ascending by cloud cover so first match = least cloudy
    # ------------------------------------------------------------------
    scene_infos: list[tuple[str, tuple, object]] = []  # (id, bbox_wgs84, item)
    for item in all_items:
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

    # ------------------------------------------------------------------
    # Step 3: assign each polygon → best (least cloudy) covering scene
    # ------------------------------------------------------------------
    poly_to_scene: dict[int, str] = {}
    scene_items: dict[str, object] = {}

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        pb = geom.bounds
        for scene_id, sbbox, item in scene_infos:
            if pb[0] <= sbbox[2] and pb[2] >= sbbox[0] and pb[1] <= sbbox[3] and pb[3] >= sbbox[1]:
                poly_to_scene[idx] = scene_id
                scene_items[scene_id] = item
                break

    skipped_no_scene = len(gdf) - len(poly_to_scene)
    _log.info(
        "Assigned %d/%d polygons across %d scenes (%d unassigned)",
        len(poly_to_scene), len(gdf), len(scene_items), skipped_no_scene,
    )

    scene_to_polys: dict[str, list[int]] = defaultdict(list)
    for idx, scene_id in poly_to_scene.items():
        scene_to_polys[scene_id].append(idx)

    # ------------------------------------------------------------------
    # Step 4: per-scene windowed reads — open each COG once
    # ------------------------------------------------------------------
    def _read_window_from_src(src, geom) -> np.ndarray | None:
        nb = src.bounds
        if src.crs.to_epsg() == 4326:
            bx0, by0, bx1, by1 = geom.bounds
        else:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            b = geom.bounds
            corners = [
                t.transform(b[0], b[1]), t.transform(b[2], b[1]),
                t.transform(b[0], b[3]), t.transform(b[2], b[3]),
            ]
            xs, ys = zip(*corners)
            bx0, bx1 = min(xs), max(xs)
            by0, by1 = min(ys), max(ys)
        bx0 = max(bx0, nb.left);  by0 = max(by0, nb.bottom)
        bx1 = min(bx1, nb.right); by1 = min(by1, nb.top)
        if bx0 >= bx1 or by0 >= by1:
            return None
        win = from_bounds(bx0, by0, bx1, by1, src.transform)
        return src.read(window=win)

    chips: list[np.ndarray] = []
    labels: list[int] = []
    skipped_small = 0

    for scene_id, poly_indices in tqdm(
        scene_to_polys.items(), desc="scenes", unit="scene"
    ):
        item = scene_items[scene_id]
        signed = planetary_computer.sign(item)
        hrefs = {b: signed.assets[b].href for b in bands if b in signed.assets}
        if len(hrefs) < len(bands):
            continue

        # Open all band COGs for this scene at once
        try:
            band_srcs = [rasterio.open(hrefs[b]) for b in bands]
        except Exception as exc:
            _log.warning("Cannot open scene %s: %s", scene_id, exc)
            continue

        try:
            for idx in tqdm(poly_indices, desc=f"  {scene_id[:28]}", unit="poly", leave=False):
                row = gdf.loc[idx]
                geom = row.geometry

                band_arrays: list[np.ndarray] = []
                ok = True
                for src in band_srcs:
                    arr = _read_window_from_src(src, geom)
                    if arr is None or arr.shape[1] < min_native_px or arr.shape[2] < min_native_px:
                        ok = False
                        skipped_small += 1
                        break
                    band_arrays.append(arr)

                if not ok:
                    continue

                stacked = np.concatenate(band_arrays, axis=0).astype(np.float32)
                t = torch.from_numpy(stacked).unsqueeze(0)
                resized = F.interpolate(
                    t, size=(chip_size, chip_size), mode="bilinear", align_corners=False
                )
                chips.append(resized.squeeze(0).numpy())
                labels.append(int(row["class_idx"]))
        finally:
            for src in band_srcs:
                src.close()

    _log.info(
        "Done: %d chips (skipped %d no-scene, %d too-small)",
        len(chips), skipped_no_scene, skipped_small,
    )
    return chips, labels


def _fetch_single_window_chips(
    gdf,
    bands: Sequence[str],
    date_start: str,
    date_end: str,
    chip_size: int,
    cloud_cover_max: float,
    min_native_px: int,
    add_ndvi: bool,
    catalog,
) -> dict:
    """Extract chips for one time window. Returns {poly_idx: np.ndarray}.

    Same pipeline as fetch_chips_from_stac but:
    - Returns a dict keyed by polygon index, not a list.
    - Optionally appends NDVI as an extra channel after resizing.
    - Does NOT require class_idx column (handled by caller).
    - Skips min_native_px check for 20m-resolution bands (identified by
      src.res[0] > 0.00015 degrees, since 10m ~ 0.0001 deg, 20m ~ 0.0002 deg).
    """
    from collections import defaultdict

    import numpy as np
    import planetary_computer  # type: ignore[import-not-found]
    import torch
    import torch.nn.functional as F
    from pyproj import Transformer
    from rasterio.windows import from_bounds
    from tqdm import tqdm

    import rasterio

    # ------------------------------------------------------------------
    # Step 1: single STAC query covering all polygons
    # ------------------------------------------------------------------
    total_bounds = tuple(float(x) for x in gdf.total_bounds)
    _log.info(
        "Querying STAC: bbox=(%.4f,%.4f,%.4f,%.4f) %s..%s",
        *total_bounds, date_start, date_end,
    )
    all_items = list(
        catalog.search(
            collections=[_S2_L2A_COLLECTION],
            bbox=total_bounds,
            datetime=f"{date_start}/{date_end}",
            query={"eo:cloud_cover": {"lt": cloud_cover_max}},
            sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
        ).items()
    )
    _log.info("Found %d candidate scenes", len(all_items))

    if not all_items:
        _log.warning("No scenes found for window %s..%s", date_start, date_end)
        return {}

    # ------------------------------------------------------------------
    # Step 2: build scene bbox index (WGS-84), sorted ascending by cloud cover
    # ------------------------------------------------------------------
    scene_infos: list[tuple[str, tuple, object]] = []
    for item in all_items:
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

    # ------------------------------------------------------------------
    # Step 3: assign each polygon → best (least cloudy) covering scene
    # ------------------------------------------------------------------
    poly_to_scene: dict[int, str] = {}
    scene_items: dict[str, object] = {}

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        pb = geom.bounds
        for scene_id, sbbox, item in scene_infos:
            if pb[0] <= sbbox[2] and pb[2] >= sbbox[0] and pb[1] <= sbbox[3] and pb[3] >= sbbox[1]:
                poly_to_scene[idx] = scene_id
                scene_items[scene_id] = item
                break

    skipped_no_scene = len(gdf) - len(poly_to_scene)
    _log.info(
        "Assigned %d/%d polygons across %d scenes (%d unassigned)",
        len(poly_to_scene), len(gdf), len(scene_items), skipped_no_scene,
    )

    scene_to_polys: dict[str, list[int]] = defaultdict(list)
    for idx, scene_id in poly_to_scene.items():
        scene_to_polys[scene_id].append(idx)

    # ------------------------------------------------------------------
    # Step 4: per-scene windowed reads — open each COG once
    # ------------------------------------------------------------------
    def _read_window_from_src(src, geom) -> np.ndarray | None:
        nb = src.bounds
        if src.crs.to_epsg() == 4326:
            bx0, by0, bx1, by1 = geom.bounds
        else:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            b = geom.bounds
            corners = [
                t.transform(b[0], b[1]), t.transform(b[2], b[1]),
                t.transform(b[0], b[3]), t.transform(b[2], b[3]),
            ]
            xs, ys = zip(*corners)
            bx0, bx1 = min(xs), max(xs)
            by0, by1 = min(ys), max(ys)
        bx0 = max(bx0, nb.left);  by0 = max(by0, nb.bottom)
        bx1 = min(bx1, nb.right); by1 = min(by1, nb.top)
        if bx0 >= bx1 or by0 >= by1:
            return None
        win = from_bounds(bx0, by0, bx1, by1, src.transform)
        return src.read(window=win)

    result: dict[int, np.ndarray] = {}
    skipped_small = 0

    for scene_id, poly_indices in tqdm(
        scene_to_polys.items(), desc="scenes", unit="scene"
    ):
        item = scene_items[scene_id]
        signed = planetary_computer.sign(item)
        hrefs = {b: signed.assets[b].href for b in bands if b in signed.assets}
        if len(hrefs) < len(bands):
            continue

        try:
            band_srcs = [rasterio.open(hrefs[b]) for b in bands]
        except Exception as exc:
            _log.warning("Cannot open scene %s: %s", scene_id, exc)
            continue

        try:
            for idx in tqdm(poly_indices, desc=f"  {scene_id[:28]}", unit="poly", leave=False):
                row = gdf.loc[idx]
                geom = row.geometry

                band_arrays: list[np.ndarray] = []
                ok = True
                for b_name, src in zip(bands, band_srcs):
                    arr = _read_window_from_src(src, geom)
                    if arr is None:
                        ok = False
                        skipped_small += 1
                        break
                    # For 20m bands (res[0] > 0.00015 deg), skip min_native_px check
                    # since F.interpolate will resize to chip_size anyway.
                    is_20m = src.res[0] > 0.00015
                    if not is_20m and (arr.shape[1] < min_native_px or arr.shape[2] < min_native_px):
                        ok = False
                        skipped_small += 1
                        break
                    band_arrays.append(arr)

                if not ok:
                    continue

                # Resize each band individually before concatenating —
                # 10m and 20m bands have different native pixel counts for the same window.
                resized_bands: list[np.ndarray] = []
                for arr in band_arrays:
                    t_band = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
                    r = F.interpolate(t_band, size=(chip_size, chip_size), mode="bilinear", align_corners=False)
                    resized_bands.append(r.squeeze(0).numpy())
                chip = np.concatenate(resized_bands, axis=0)  # (C, chip_size, chip_size)

                # Optionally append NDVI as an extra channel (computed from raw DN values)
                if add_ndvi and "B08" in bands and "B04" in bands:
                    b08 = chip[list(bands).index("B08")]
                    b04 = chip[list(bands).index("B04")]
                    ndvi_raw = (b08 - b04) / (b08 + b04 + 1e-3)  # [-1, 1]
                    ndvi_dn = (ndvi_raw + 1.0) * 5000.0  # scale to [0, 10000]
                    chip = np.concatenate([chip, ndvi_dn[np.newaxis]], axis=0)

                result[idx] = chip
        finally:
            for src in band_srcs:
                src.close()

    _log.info(
        "Window %s..%s: %d chips extracted (%d no-scene, %d too-small)",
        date_start, date_end, len(result), skipped_no_scene, skipped_small,
    )
    return result


def fetch_chips_multitemporal(
    gdf,
    bands: Sequence[str],
    date_windows: Sequence[tuple[str, str]],
    chip_size: int = 64,
    cloud_cover_max: float = 20.0,
    min_native_px: int = 4,
    add_ndvi: bool = True,
    max_missing_windows: int = 1,
) -> tuple[list, list[int]]:
    """Fetch per-polygon chips across multiple time windows.

    For each polygon: collects chips from each window, concatenates along
    the channel axis. Windows with no STAC coverage are zero-padded. Polygons
    missing more than ``max_missing_windows`` windows are dropped.

    Returns ``(chips, labels)`` where each chip has shape
    ``(len(windows) * channels_per_window, chip_size, chip_size)``.

    Parameters
    ----------
    gdf:
        GeoDataFrame in EPSG:4326. Must have a ``class_idx`` integer column.
    bands:
        Sentinel-2 band IDs to fetch per window (e.g. ``["B02", "B03", "B04", "B08"]``).
    date_windows:
        Sequence of ``(date_start, date_end)`` pairs defining the time windows.
    add_ndvi:
        Append NDVI as an extra channel after the band channels for each window.
    max_missing_windows:
        Polygons with more than this many zero-padded windows are dropped.
    """
    import numpy as np
    from pystac_client import Client  # type: ignore[import-not-found]

    catalog = Client.open(_PC_STAC_URL)

    n_channels_per_window = len(bands) + (1 if add_ndvi else 0)
    zero_window = np.zeros((n_channels_per_window, chip_size, chip_size), dtype=np.float32)

    # Collect chips per window: list of {poly_idx: chip}
    window_results: list[dict[int, np.ndarray]] = []
    for date_start, date_end in date_windows:
        _log.info("Fetching window %s .. %s", date_start, date_end)
        window_chips = _fetch_single_window_chips(
            gdf=gdf,
            bands=bands,
            date_start=date_start,
            date_end=date_end,
            chip_size=chip_size,
            cloud_cover_max=cloud_cover_max,
            min_native_px=min_native_px,
            add_ndvi=add_ndvi,
            catalog=catalog,
        )
        window_results.append(window_chips)

    # Collect all polygon indices that appear in ANY window
    all_poly_indices: set[int] = set()
    for wr in window_results:
        all_poly_indices.update(wr.keys())

    chips: list[np.ndarray] = []
    labels: list[int] = []

    for poly_idx in sorted(all_poly_indices):
        missing = sum(1 for wr in window_results if poly_idx not in wr)
        if missing > max_missing_windows:
            continue

        window_chips_list = [
            wr.get(poly_idx, zero_window) for wr in window_results
        ]
        combined = np.concatenate(window_chips_list, axis=0)  # (n_windows * C, H, W)
        chips.append(combined)
        labels.append(int(gdf.loc[poly_idx, "class_idx"]))

    _log.info(
        "Multi-temporal done: %d chips from %d windows × %d ch/window",
        len(chips), len(date_windows), n_channels_per_window,
    )
    return chips, labels


def download_sentinel2_l2a(
    bbox: BBox,
    date_start: str,
    date_end: str,
    out_dir: Path | str,
    bands: Sequence[str] = ("B02", "B03", "B04", "B08"),
    cloud_cover_max: float = 20.0,
    limit: int | None = None,
    clip: bool = True,
    max_workers: int = 4,
) -> DownloadResult:
    """Download the requested Sentinel-2 L2A band assets into ``out_dir``.

    Downloads up to *max_workers* files in parallel using threads.
    When *clip* is ``True`` (default) each tile is cropped to *bbox* after
    download, reducing file size by 10–50× for small AOIs.

    Returns a :class:`DownloadResult` with the number of scenes / assets
    written. Each asset is saved as ``{scene_id}_{band}.tif``.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

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
    already = sum(
        1 for item in all_items for band in bands
        if (out / f"{item.id}_{band}.tif").exists()
    )
    _log.info(
        "Found %d scenes × %d bands = %d files  (already downloaded: %d, remaining: %d)",
        n_scenes, len(bands), n_files, already, n_files - already,
    )

    # Build a flat list of (dest, url) for every file that needs downloading.
    tasks: list[tuple[Path, str, str]] = []  # (dest, url, band)
    for item in all_items:
        signed = planetary_computer.sign(item)
        for band in bands:
            dest = out / f"{item.id}_{band}.tif"
            if dest.exists():
                continue
            asset = signed.assets.get(band)
            if asset is None:
                _log.warning("scene %s missing band %s; skipping", item.id, band)
                continue
            tasks.append((dest, asset.href, band))

    assets_lock = threading.Lock()
    completed_assets = already

    def _fetch(dest: Path, url: str, band: str) -> bool:
        tmp = dest.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        try:
            _download_with_retry(url, tmp, band=band, retries=5, backoff=3.0)
            if clip:
                _clip_tif_to_bbox(tmp, dest, bbox)
            else:
                tmp.rename(dest)
            return True
        finally:
            tmp.unlink(missing_ok=True)

    progress = tqdm(
        total=n_files,
        initial=already,
        desc="files",
        unit="file",
        dynamic_ncols=True,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, dest, url, band): dest for dest, url, band in tasks}
        for future in as_completed(futures):
            dest = futures[future]
            try:
                future.result()
            except Exception as exc:
                _log.error("failed to download %s: %s", dest.name, exc)
            with assets_lock:
                completed_assets += 1
            progress.update(1)
            progress.set_postfix(file=dest.name[:40])

    progress.close()
    _log.info("downloaded %d assets across %d scenes into %s", completed_assets, n_scenes, out)
    return DownloadResult(out_dir=out, scenes=n_scenes, assets=completed_assets)
