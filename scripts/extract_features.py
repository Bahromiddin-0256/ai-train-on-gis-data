"""Extract statistical features from Sentinel-2 image patches for XGBoost training.

Feature engineering pipeline:
  1. Optionally compute spectral indices (NDVI, EVI, NDWI, NDRE, MSI, NBR, …)
     on-the-fly for each time window and append them as extra channels.
  2. Extract per-channel statistics (mean, std, min, max, percentiles, gradients,
     local variance) from every channel including computed indices.
  3. Extract temporal change features between consecutive time windows.
  4. Compute dedicated NDVI temporal statistics (max, mean, std, p25/p75, range)
     across all time windows — one of the highest-ROI feature groups for crops.

Recommended invocation::

    python scripts/extract_features.py \\
        --data-dir data/processed_regional_mt \\
        --bands B02,B03,B04,B05,B06,B07,B08,B11,B12 \\
        --indices ndvi,evi,ndwi,ndre,msi,nbr \\
        --n-windows 3
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Index augmentation
# ---------------------------------------------------------------------------

def augment_with_indices(
    patch: np.ndarray,
    raw_bands: list[str],
    n_windows: int,
    index_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Compute spectral indices per time window and append as extra channels.

    Parameters
    ----------
    patch:
        ``(n_windows * len(raw_bands), H, W)`` chip array.
    raw_bands:
        Names of the bands already in *patch* (no window suffix).
    n_windows:
        Number of temporal windows stacked along channel axis.
    index_names:
        Lower-case index names to compute (e.g. ``["ndvi", "evi"]``).

    Returns
    -------
    augmented_patch, augmented_band_names
        Patch with extra index channels; updated band list without window suffix.
    """
    from gis_train.data.indices import compute_indices, required_bands as req_bands

    # Only compute indices whose required bands are all available
    computable = [
        n for n in index_names
        if req_bands([n]).issubset(raw_bands)
    ]
    skipped = set(index_names) - set(computable)
    if skipped:
        import warnings
        warnings.warn(
            f"Skipping indices {sorted(skipped)} — required bands not in {raw_bands}",
            stacklevel=2,
        )

    if not computable:
        return patch, raw_bands

    B = len(raw_bands)
    result_windows = []
    for w in range(n_windows):
        window = patch[w * B:(w + 1) * B]                        # (B, H, W)
        idx_arr = compute_indices(window, raw_bands, computable)  # (n_idx, H, W)
        result_windows.append(np.concatenate([window, idx_arr], axis=0))

    augmented = np.concatenate(result_windows, axis=0)
    aug_bands = raw_bands + [n.upper() for n in computable]
    return augmented, aug_bands


# ---------------------------------------------------------------------------
# Temporal NDVI statistics
# ---------------------------------------------------------------------------

def compute_ndvi_temporal_stats(
    patch: np.ndarray,
    all_band_names: list[str],
) -> dict[str, float]:
    """NDVI statistics across time windows (spatial mean per window).

    Returns empty dict when NDVI is not present in *all_band_names*.
    """
    ndvi_channels = [i for i, n in enumerate(all_band_names) if n.startswith("NDVI_w")]
    if not ndvi_channels:
        return {}

    means = np.array([patch[i].mean() for i in ndvi_channels], dtype=np.float32)
    return {
        "ndvi_t_max": float(means.max()),
        "ndvi_t_min": float(means.min()),
        "ndvi_t_mean": float(means.mean()),
        "ndvi_t_std": float(means.std()),
        "ndvi_t_p25": float(np.percentile(means, 25)),
        "ndvi_t_p75": float(np.percentile(means, 75)),
        "ndvi_t_range": float(means.max() - means.min()),
    }


# ---------------------------------------------------------------------------
# Spatial features
# ---------------------------------------------------------------------------

def compute_spatial_gradients(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean absolute Sobel gradient per channel.  patch: (C, H, W)."""
    grad_x = ndimage.sobel(patch, axis=2)
    grad_y = ndimage.sobel(patch, axis=1)
    return np.abs(grad_x).mean(axis=(1, 2)), np.abs(grad_y).mean(axis=(1, 2))


def compute_local_variance(patch: np.ndarray, size: int = 3) -> np.ndarray:
    """Mean local variance (texture) per channel.  patch: (C, H, W)."""
    local_mean = ndimage.uniform_filter(patch, size=size, mode="reflect")
    local_mean_sq = ndimage.uniform_filter(patch ** 2, size=size, mode="reflect")
    return (local_mean_sq - local_mean ** 2).mean(axis=(1, 2))


# ---------------------------------------------------------------------------
# Per-patch feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_patch(
    patch: np.ndarray,
    band_names: list[str] | None = None,
) -> dict[str, float]:
    """Statistics + spatial features from one ``(C, H, W)`` patch."""
    if band_names is None:
        band_names = [f"ch{i}" for i in range(patch.shape[0])]

    features: dict[str, float] = {}
    n_ch = patch.shape[0]

    for i in range(n_ch):
        ch = patch[i].flatten()
        p = band_names[i]
        features[f"{p}_mean"] = float(ch.mean())
        features[f"{p}_std"] = float(ch.std())
        features[f"{p}_min"] = float(ch.min())
        features[f"{p}_max"] = float(ch.max())
        features[f"{p}_p25"] = float(np.percentile(ch, 25))
        features[f"{p}_median"] = float(np.median(ch))
        features[f"{p}_p75"] = float(np.percentile(ch, 75))

    grad_x, grad_y = compute_spatial_gradients(patch)
    local_var = compute_local_variance(patch)
    for i in range(n_ch):
        p = band_names[i]
        features[f"{p}_grad_x"] = float(grad_x[i])
        features[f"{p}_grad_y"] = float(grad_y[i])
        features[f"{p}_var"] = float(local_var[i])

    return features


def extract_temporal_features(
    patch: np.ndarray,
    n_windows: int,
    bands_per_window: int,
    aug_bands: list[str] | None = None,
) -> dict[str, float]:
    """Temporal change features between consecutive time windows."""
    C, H, W = patch.shape
    reshaped = patch.reshape(n_windows, bands_per_window, H, W)
    features: dict[str, float] = {}

    for b in range(bands_per_window):
        means = reshaped[:, b].mean(axis=(1, 2))
        bn = aug_bands[b] if aug_bands else f"band{b}"
        features[f"{bn}_change_w0w1"] = float(means[1] - means[0])
        features[f"{bn}_change_w1w2"] = float(means[2] - means[1]) if n_windows > 2 else 0.0
        features[f"{bn}_total_change"] = float(means[-1] - means[0])
        features[f"{bn}_temporal_std"] = float(means.std())

    return features


# ---------------------------------------------------------------------------
# Band name helpers
# ---------------------------------------------------------------------------

def get_band_names(bands: list[str] | None = None, n_windows: int = 3) -> list[str]:
    """Return window-suffixed channel names, e.g. ``B08_w0``, ``NDVI_w1``."""
    if bands is None:
        bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
    names = []
    for w in range(n_windows):
        for b in bands:
            names.append(f"{b}_w{w}")
    return names


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def extract_features(
    images: np.ndarray,
    raw_bands: list[str] | None = None,
    index_names: list[str] | None = None,
    n_windows: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract full feature matrix from all image patches.

    Parameters
    ----------
    images:
        ``(N, C, H, W)`` array, C = n_windows × len(raw_bands).
    raw_bands:
        Names of the spectral bands stored in *images* (no NDVI or other
        pre-computed indices; those go in *index_names*).
    index_names:
        Spectral indices to compute on-the-fly per time window.
    n_windows:
        Number of temporal windows.

    Returns
    -------
    DataFrame with one row per sample.
    """
    if raw_bands is None:
        raw_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]

    all_features: list[dict[str, float]] = []

    # Run augmentation once on the first patch to get the final band list.
    sample_aug, aug_bands = augment_with_indices(
        images[0], raw_bands, n_windows, index_names or []
    )
    all_band_names = get_band_names(aug_bands, n_windows)

    iterator = tqdm(images, desc="Extracting features") if verbose else images
    for patch in iterator:
        aug_patch, _ = augment_with_indices(patch, raw_bands, n_windows, index_names or [])

        features = extract_features_from_patch(aug_patch, all_band_names)
        temporal = extract_temporal_features(aug_patch, n_windows, len(aug_bands), aug_bands=aug_bands)
        ndvi_stats = compute_ndvi_temporal_stats(aug_patch, all_band_names)

        features.update(temporal)
        features.update(ndvi_stats)
        all_features.append(features)

    return pd.DataFrame(all_features)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing images.npy and labels.npy",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for features CSV (default: data_dir/features.csv)",
)
@click.option(
    "--n-windows",
    type=int,
    default=3,
    help="Number of temporal windows (default: 3)",
)
@click.option(
    "--bands",
    type=str,
    default="B02,B03,B04,B05,B06,B07,B08,B11,B12",
    show_default=True,
    help="Raw spectral bands present in images.npy.",
)
@click.option(
    "--indices",
    type=str,
    default="ndvi,evi,ndwi,ndre,msi,nbr",
    show_default=True,
    help="Spectral indices to compute on-the-fly per time window.",
)
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Only extract features for first N samples (for testing).",
)
def main(
    data_dir: Path,
    output: Path | None,
    n_windows: int,
    bands: str,
    indices: str,
    sample_size: int | None,
) -> None:
    """Extract features from multi-temporal image patches for XGBoost training."""
    images_path = data_dir / "images.npy"
    labels_path = data_dir / "labels.npy"

    if not images_path.exists():
        raise FileNotFoundError(f"Images file not found: {images_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path)

    click.echo(f"Images shape: {images.shape}, dtype: {images.dtype}")
    click.echo(f"Labels shape: {labels.shape}, unique: {np.unique(labels)}")

    if sample_size:
        images = images[:sample_size]
        labels = labels[:sample_size]
        click.echo(f"Using first {sample_size} samples for testing")

    raw_band_list = [b.strip() for b in bands.split(",") if b.strip()]
    index_list = [i.strip() for i in indices.split(",") if i.strip()] if indices else []

    click.echo(f"Raw bands ({len(raw_band_list)}): {raw_band_list}")
    click.echo(f"Computed indices ({len(index_list)}): {index_list}")
    click.echo(f"Temporal windows: {n_windows}")

    click.echo("Extracting features...")
    features_df = extract_features(
        images,
        raw_bands=raw_band_list,
        index_names=index_list,
        n_windows=n_windows,
    )
    features_df["label"] = labels[: len(features_df)]

    if output is None:
        output = data_dir / "features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output, index=False)

    n_feat = len(features_df.columns) - 1  # exclude label
    click.echo(f"Saved {len(features_df)} samples × {n_feat} features → {output}")
    click.echo(f"First 10 features: {list(features_df.columns[:10])}")

    ndvi_cols = [c for c in features_df.columns if c.startswith("ndvi_t_")]
    if ndvi_cols:
        click.echo(f"NDVI temporal features: {ndvi_cols}")


if __name__ == "__main__":
    main()
