"""Temporal compositing utilities for downloaded Sentinel-2 GeoTIFFs.

Operates on directories produced by ``download_sentinel2_l2a``, where files are
named ``{scene_id}_{band}.tif``.  Composite outputs are written into the same
directory as ``composite_{band}.tif`` (median) or ``{YYYY-MM}_{band}.tif`` (monthly).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

# Sentinel-2 PC scene IDs embed the acquisition date at the 3rd "_"-delimited token.
# e.g. S2A_MSIL2A_20230415T063631_R006_T39SUV_20230415T135453
_DATE_PATTERN = re.compile(r"_(\d{8})T\d{6}_")


def _scene_date(scene_id: str) -> str | None:
    """Return 'YYYYMMDD' from a Sentinel-2 STAC scene ID, or None if not parseable."""
    m = _DATE_PATTERN.search(scene_id)
    return m.group(1) if m else None


def _median_stack(paths: list[Path]) -> "np.ndarray":
    import numpy as np
    import rasterio

    arrays = []
    for p in paths:
        with rasterio.open(p) as src:
            arrays.append(src.read(1).astype(np.float32))
    return np.nanmedian(np.stack(arrays, axis=0), axis=0)


def _write_composite(arr: "np.ndarray", reference: Path, dest: Path) -> None:
    import rasterio

    with rasterio.open(reference) as ref:
        profile = ref.profile.copy()
    profile.update(dtype="float32", count=1, compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(arr, 1)
    _log.debug("wrote composite %s", dest.name)


def composite_scenes(
    out_dir: Path | str,
    strategy: str,
    bands: list[str],
) -> None:
    """Compute temporal composites from per-scene GeoTIFFs.

    Parameters
    ----------
    out_dir:
        Directory containing ``{scene_id}_{band}.tif`` files.
    strategy:
        ``"median"`` — single nanmedian across all scenes per band.
        ``"monthly"`` — one nanmedian composite per calendar month per band.
    bands:
        Band IDs to process (e.g. ``["B02", "B08"]``).
    """
    if strategy not in ("median", "monthly"):
        raise ValueError(f"strategy must be 'median' or 'monthly', got {strategy!r}")

    out = Path(out_dir)

    # Collect {band: {scene_id: Path}}
    band_scenes: dict[str, dict[str, Path]] = defaultdict(dict)
    for tif in sorted(out.glob("*.tif")):
        parts = tif.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        scene_id, band = parts
        if band in bands:
            band_scenes[band][scene_id] = tif

    for band in bands:
        scene_map = band_scenes.get(band, {})
        if not scene_map:
            _log.warning("no scenes found for band %s in %s", band, out)
            continue

        if strategy == "median":
            dest = out / f"composite_{band}.tif"
            if dest.exists():
                _log.debug("skipping %s (exists)", dest.name)
                continue
            paths = list(scene_map.values())
            arr = _median_stack(paths)
            _write_composite(arr, paths[0], dest)
            _log.info("median composite %s: %d scenes → %s", band, len(paths), dest.name)

        elif strategy == "monthly":
            monthly: dict[str, list[Path]] = defaultdict(list)
            for scene_id, path in scene_map.items():
                date = _scene_date(scene_id)
                if date is None:
                    _log.warning("cannot parse date from scene ID %r; skipping", scene_id)
                    continue
                month_key = f"{date[:4]}-{date[4:6]}"
                monthly[month_key].append(path)

            for month_key, paths in sorted(monthly.items()):
                dest = out / f"{month_key}_{band}.tif"
                if dest.exists():
                    _log.debug("skipping %s (exists)", dest.name)
                    continue
                arr = _median_stack(paths)
                _write_composite(arr, paths[0], dest)
                _log.info(
                    "monthly composite %s %s: %d scenes → %s",
                    month_key, band, len(paths), dest.name,
                )


__all__ = ["composite_scenes"]
