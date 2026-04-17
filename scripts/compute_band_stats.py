"""Compute per-channel mean and std from images.npy and update a YAML config.

Usage:
    python scripts/compute_band_stats.py \
        --images data/processed_regional_mt/images.npy \
        --out configs/data/uzbekistan_s2.yaml \
        --n-samples 5000
"""

import click
import numpy as np
from pathlib import Path


@click.command()
@click.option("--images", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True,
              help="YAML config file to update with mean/std.")
@click.option("--n-samples", default=5000, show_default=True,
              help="Number of chips to sample for stats.")
def main(images: Path, out: Path, n_samples: int) -> None:
    """Compute per-channel mean/std and write to YAML config."""
    arr = np.load(images, mmap_mode="r")  # (N, C, H, W)
    n, c, h, w = arr.shape
    click.echo(f"Dataset shape: {arr.shape}")

    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    sample = arr[idx].astype(np.float32) / 10_000.0  # scale to [0, 1]

    mean = sample.mean(axis=(0, 2, 3)).tolist()
    std = sample.std(axis=(0, 2, 3)).tolist()

    click.echo(f"\nComputed stats for {len(idx)} chips, {c} channels:")
    for i, (m, s) in enumerate(zip(mean, std)):
        click.echo(f"  ch{i:02d}: mean={m:.4f}  std={s:.4f}")

    # Update YAML file
    import re
    text = out.read_text()
    mean_line = "mean: [" + ", ".join(f"{v:.4f}" for v in mean) + "]"
    std_line = "std:  [" + ", ".join(f"{v:.4f}" for v in std) + "]"
    text = re.sub(r"^mean:.*$", mean_line, text, flags=re.MULTILINE)
    text = re.sub(r"^std:.*$", std_line, text, flags=re.MULTILINE)
    out.write_text(text)
    click.echo(f"\nUpdated {out}")


if __name__ == "__main__":
    main()
