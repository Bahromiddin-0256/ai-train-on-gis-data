"""Sentinel-2 vegetation / water indices.

Inputs are per-band reflectance arrays in raw DN units (0–10000, as downloaded
from Planetary Computer). Outputs are rescaled to the same DN range so they
slot next to raw bands with minimal normalization drift.
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

DN_SCALE = 10_000.0
_EPS = 1e-3


def _normalized_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    raw = (a - b) / (a + b + _EPS)  # [-1, 1]
    return (raw + 1.0) * (DN_SCALE / 2.0)  # [0, DN_SCALE]


def ndvi(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    return _normalized_diff(bands["B08"], bands["B04"])


def ndre(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Red-edge NDVI; better cereal/legume separation than NDVI."""
    return _normalized_diff(bands["B08"], bands["B05"])


def ndwi(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Gao water index (canopy water content)."""
    return _normalized_diff(bands["B08"], bands["B03"])


def ndmi(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Moisture index using SWIR."""
    return _normalized_diff(bands["B08"], bands["B11"])


def evi(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Enhanced Vegetation Index (no saturation in dense canopy)."""
    b8 = bands["B08"] / DN_SCALE
    b4 = bands["B04"] / DN_SCALE
    b2 = bands["B02"] / DN_SCALE
    raw = 2.5 * (b8 - b4) / (b8 + 6.0 * b4 - 7.5 * b2 + 1.0 + _EPS)  # roughly [-1, 1]
    return np.clip((raw + 1.0) * (DN_SCALE / 2.0), 0.0, DN_SCALE)


def savi(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Soil-Adjusted VI (L=0.5); robust for sparse canopies."""
    b8 = bands["B08"] / DN_SCALE
    b4 = bands["B04"] / DN_SCALE
    L = 0.5
    raw = ((b8 - b4) / (b8 + b4 + L + _EPS)) * (1.0 + L)  # [-1, 1]
    return (raw + 1.0) * (DN_SCALE / 2.0)


def msi(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Moisture Stress Index: SWIR1/NIR.  Higher values = more water stress."""
    b8 = bands["B08"].astype(np.float32)
    b11 = bands["B11"].astype(np.float32)
    # Ratio typically in [0, 3]; clip and rescale to DN range.
    raw = b11 / (b8 + _EPS)
    return np.clip(raw, 0.0, 3.0) * (DN_SCALE / 3.0)


def nbr(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Normalized Burn Ratio (NIR-SWIR2)/(NIR+SWIR2).  Proxy for dryness."""
    return _normalized_diff(bands["B08"], bands["B12"])


def vv_vh_ratio(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """SAR cross-polarization ratio VV/VH.  Sensitive to crop structure."""
    vv = bands["VV"].astype(np.float32)
    vh = bands["VH"].astype(np.float32)
    # Ratio typically in [0, 10]; clip and rescale.
    raw = vv / (vh + _EPS)
    return np.clip(raw, 0.0, 10.0) * (DN_SCALE / 10.0)


IndexFn = Callable[[Mapping[str, np.ndarray]], np.ndarray]
INDEX_REGISTRY: dict[str, IndexFn] = {
    "ndvi": ndvi,
    "ndre": ndre,
    "ndwi": ndwi,
    "ndmi": ndmi,
    "evi": evi,
    "savi": savi,
    "msi": msi,
    "nbr": nbr,
    "vv_vh_ratio": vv_vh_ratio,
}


def required_bands(names: list[str]) -> set[str]:
    """Bands needed to compute the requested indices."""
    req = {
        "ndvi": {"B04", "B08"},
        "ndre": {"B05", "B08"},
        "ndwi": {"B03", "B08"},
        "ndmi": {"B08", "B11"},
        "evi": {"B02", "B04", "B08"},
        "savi": {"B04", "B08"},
        "msi": {"B08", "B11"},
        "nbr": {"B08", "B12"},
        "vv_vh_ratio": {"VV", "VH"},
    }
    out: set[str] = set()
    for n in names:
        out |= req[n]
    return out


def compute_indices(
    chip: np.ndarray,
    bands: list[str],
    indices: list[str],
) -> np.ndarray:
    """Return a ``(len(indices), H, W)`` stack of requested indices.

    ``chip`` is ``(C, H, W)`` with channel order matching ``bands``.
    """
    band_map: dict[str, np.ndarray] = {}
    for i, b in enumerate(bands):
        band_map[b] = chip[i].astype(np.float32)

    out = []
    for name in indices:
        name = name.lower()
        if name not in INDEX_REGISTRY:
            raise KeyError(f"unknown index: {name!r}; known: {sorted(INDEX_REGISTRY)}")
        out.append(INDEX_REGISTRY[name](band_map).astype(np.float32))
    return np.stack(out, axis=0)


def compute_indices_folder(
    out_dir,
    bands: list[str],
    index_names: list[str] | None = None,
) -> None:
    """Compute spectral indices from per-band GeoTIFFs in ``out_dir``.

    Expects files named ``{scene_id}_{band}.tif``. Writes new files
    ``{scene_id}_{INDEX}.tif`` alongside. Skips indices whose required bands
    are absent for a given scene.
    """
    from pathlib import Path

    import numpy as np
    import rasterio

    if index_names is None:
        index_names = ["ndvi", "evi", "ndwi", "ndmi"]

    out = Path(out_dir)
    scene_band_files: dict[str, dict[str, Path]] = {}
    for tif in sorted(out.glob("*.tif")):
        stem_parts = tif.stem.rsplit("_", 1)
        if len(stem_parts) != 2:
            continue
        scene_id, band = stem_parts
        if band in bands:
            scene_band_files.setdefault(scene_id, {})[band] = tif

    for scene_id, band_files in scene_band_files.items():
        for index_name in index_names:
            req = required_bands([index_name])
            if not req.issubset(band_files.keys()):
                continue
            dest = out / f"{scene_id}_{index_name.upper()}.tif"
            if dest.exists():
                continue
            arrays: dict[str, np.ndarray] = {}
            profile = None
            for b in sorted(req):
                with rasterio.open(band_files[b]) as src:
                    arrays[b] = src.read(1).astype(np.float32)
                    if profile is None:
                        profile = src.profile.copy()
            result_arr = INDEX_REGISTRY[index_name](arrays).astype(np.float32)
            profile.update(dtype="float32", count=1, compress="deflate")
            with rasterio.open(dest, "w", **profile) as dst:
                dst.write(result_arr, 1)


__all__ = [
    "INDEX_REGISTRY",
    "compute_indices",
    "compute_indices_folder",
    "msi",
    "nbr",
    "ndmi",
    "ndre",
    "ndvi",
    "ndwi",
    "evi",
    "required_bands",
    "savi",
    "vv_vh_ratio",
]
