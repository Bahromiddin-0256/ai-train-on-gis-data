"""Tests for vegetation-index helpers."""

from __future__ import annotations

import numpy as np
import pytest

from gis_train.data.indices import compute_indices, required_bands


def test_ndvi_in_dn_range() -> None:
    bands = ["B04", "B08"]
    chip = np.stack(
        [
            np.full((4, 4), 2000.0, dtype=np.float32),  # B04
            np.full((4, 4), 6000.0, dtype=np.float32),  # B08
        ],
        axis=0,
    )
    out = compute_indices(chip, bands, ["ndvi"])
    assert out.shape == (1, 4, 4)
    # NDVI raw = (6000-2000)/(8000) = 0.5 → rescaled to (0.5 + 1)*5000 = 7500.
    assert np.allclose(out[0], 7500.0, atol=1.0)


def test_compute_indices_stacks_in_order() -> None:
    bands = ["B02", "B03", "B04", "B05", "B08", "B11"]
    chip = np.random.default_rng(0).uniform(100, 9000, size=(len(bands), 3, 3)).astype(np.float32)
    out = compute_indices(chip, bands, ["ndvi", "ndre", "evi"])
    assert out.shape == (3, 3, 3)
    assert (out >= 0).all() and (out <= 10000).all()


def test_unknown_index_raises() -> None:
    with pytest.raises(KeyError):
        compute_indices(np.zeros((2, 2, 2)), ["B04", "B08"], ["nonexistent"])


def test_required_bands_union() -> None:
    assert required_bands(["ndvi"]) == {"B04", "B08"}
    assert required_bands(["ndvi", "ndre"]) == {"B04", "B05", "B08"}
    assert "B11" in required_bands(["ndmi"])
