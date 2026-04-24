"""Score already-extracted Sentinel-2 chips with the XGBoost crop classifier
and (optionally) write the predicted class back to MongoDB as
``crop_type_xgboost``.

This is the XGBoost analogue of scripts/score_uzcosmos_resnet50plus.py:

  1. Load the trained Booster from ``outputs/xgboost/xgboost_model.json``.
  2. For each chip dir under ``--chips-root``:
       a. Read ``images.npy`` (+ optional ``ids.npy``).
       b. Run the same feature-engineering pipeline as scripts/extract_features.py
          (augmented spectral indices → per-channel stats → temporal deltas →
           NDVI-time stats), on-the-fly.
       c. Reindex the resulting DataFrame to the Booster's feature order.
       d. Predict multi:softprob → top-1 class + per-class scores.
       e. Bulk-update Mongo: {crop_type_xgboost, _prob, _scores}.
  3. Write ``predictions_xgboost.csv/.npy`` next to each ``images.npy``.

Usage
-----
    .venv/bin/python scripts/score_uzcosmos_xgboost.py \\
        --model outputs/xgboost/xgboost_model.json \\
        --chips-root data/v6win \\
        --mongo-write
"""

from __future__ import annotations
import sys
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_features import (  # type: ignore[import-not-found]
    augment_with_indices,
    compute_ndvi_temporal_stats,
    extract_features_from_patch,
    extract_temporal_features,
    get_band_names,
)
from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

CLASSES = ("bugdoy", "other", "paxta")
BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12")
INDICES = ("ndvi", "evi", "ndwi", "ndre", "msi", "nbr")
# The chips on disk are 6-window (90-channel). The trained XGBoost model
# (outputs/xgboost_6win/) was fitted on the full 6-window feature stack:
# 967 features = 6 × 15 × 10 stats + 15 × 4 temporal + 7 ndvi_t.
DATA_N_WINDOWS = 6
MODEL_WINDOWS = 6
CHANNELS_PER_WINDOW = len(BANDS) + len(INDICES)  # 15
DATA_CHANNELS = DATA_N_WINDOWS * CHANNELS_PER_WINDOW   # 90
MODEL_CHANNELS = MODEL_WINDOWS * CHANNELS_PER_WINDOW   # 90
CHIP_SIZE = 64


# ---------------------------------------------------------------------------
# ID loading + coercion (mirrors the NN scorer so Mongo writes use the same
# per-row _id-type dispatch logic)
# ---------------------------------------------------------------------------

def _load_ids(chip_dir: Path, manifest: Path | None) -> list | None:
    if manifest is not None:
        return _load_ids_from_file(manifest)
    for name in ("ids.npy", "manifest.csv", "manifest.parquet"):
        p = chip_dir / name
        if p.exists():
            return _load_ids_from_file(p)
    return None


def _load_ids_from_file(path: Path) -> list:
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        raw = arr.tolist()
    elif path.suffix == ".csv":
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return []
        key = next((k for k in ("_id", "mongo_id", "oid", "id") if k in rows[0]), None)
        if key is None:
            raise click.ClickException(
                f"{path} has no _id/mongo_id/oid/id column; columns: {list(rows[0])}"
            )
        raw = [r[key] for r in rows]
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        key = next((k for k in ("_id", "mongo_id", "oid", "id") if k in df.columns), None)
        if key is None:
            raise click.ClickException(
                f"{path} has no _id column; columns: {list(df.columns)}"
            )
        raw = df[key].tolist()
    else:
        raise click.ClickException(f"unsupported manifest format: {path.suffix}")
    return [str(x) if x is not None else "" for x in raw]


def _coerce_id(s: str) -> object:
    """ObjectId / int / str dispatch."""
    from bson import ObjectId
    from bson.errors import InvalidId

    if len(s) == 24:
        try:
            return ObjectId(s)
        except InvalidId:
            pass
    if s.lstrip("-").isdigit():
        try:
            return int(s)
        except ValueError:
            pass
    return s


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parallel feature-extraction worker
# ---------------------------------------------------------------------------

def _extract_features_chunk(
    images_path: str,
    start: int,
    end: int,
    aug_bands: list[str],
    all_band_names: list[str],
) -> list[dict]:
    """Worker: feature-engineer patches[start:end] from a mmap-ed images.npy."""
    import numpy as np  # re-import in worker process
    from extract_features import (  # noqa: PLC0415
        augment_with_indices,
        compute_ndvi_temporal_stats,
        extract_features_from_patch,
        extract_temporal_features,
    )
    images = np.load(images_path, mmap_mode="r")
    bands_per_window = len(aug_bands)
    results = []
    for idx in range(start, end):
        patch = np.ascontiguousarray(images[idx])
        aug_patch, _ = augment_with_indices(
            patch, list(BANDS), MODEL_WINDOWS, list(INDICES),
        )
        feats = extract_features_from_patch(aug_patch, all_band_names)
        feats.update(extract_temporal_features(
            aug_patch, MODEL_WINDOWS, bands_per_window, aug_bands=aug_bands,
        ))
        feats.update(compute_ndvi_temporal_stats(aug_patch, all_band_names))
        results.append(feats)
    return results


def _score_chip_dir(
    chip_dir: Path,
    booster: xgb.Booster,
    feature_names: list[str],
    outer: tqdm,
    n_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Feature-engineer + predict for every chip in a tuman dir."""
    images = np.load(chip_dir / "images.npy", mmap_mode="r")
    n = int(images.shape[0])

    if images.shape[1:] != (DATA_CHANNELS, CHIP_SIZE, CHIP_SIZE):
        raise click.ClickException(
            f"{chip_dir}/images.npy shape {tuple(images.shape)} does not match "
            f"(N, {DATA_CHANNELS}, {CHIP_SIZE}, {CHIP_SIZE})"
        )

    sample_aug, aug_bands = augment_with_indices(
        images[0], list(BANDS), MODEL_WINDOWS, list(INDICES),
    )
    all_band_names = get_band_names(aug_bands, MODEL_WINDOWS)
    bands_per_window = len(aug_bands)
    images_path = str(chip_dir / "images.npy")

    if n_workers > 1:
        n_chunks = min(n, n_workers * 2)
        chunk_size = math.ceil(n / n_chunks)
        chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

        rows: list[dict] = [None] * n  # type: ignore[list-item]
        pbar = tqdm(total=n, desc=f"  {chip_dir.name}", unit="chip",
                    leave=False, position=1)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_range = {
                ex.submit(
                    _extract_features_chunk,
                    images_path, s, e, aug_bands, all_band_names,
                ): (s, e)
                for s, e in chunks
            }
            for fut in as_completed(future_to_range):
                s, e = future_to_range[fut]
                for j, feat in enumerate(fut.result()):
                    rows[s + j] = feat
                pbar.update(e - s)
        pbar.close()
    else:
        rows = []
        pbar = tqdm(total=n, desc=f"  {chip_dir.name}", unit="chip",
                    leave=False, position=1)
        for i in range(n):
            patch = np.ascontiguousarray(images[i])
            aug_patch, _ = augment_with_indices(
                patch, list(BANDS), MODEL_WINDOWS, list(INDICES),
            )
            feats = extract_features_from_patch(aug_patch, all_band_names)
            feats.update(extract_temporal_features(
                aug_patch, MODEL_WINDOWS, bands_per_window, aug_bands=aug_bands,
            ))
            feats.update(compute_ndvi_temporal_stats(aug_patch, all_band_names))
            rows.append(feats)
            pbar.update(1)
        pbar.close()

    df = pd.DataFrame(rows)
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise click.ClickException(
            f"feature extractor missing columns required by model: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )
    df = df[feature_names]

    dmat = xgb.DMatrix(df.values, feature_names=feature_names)
    probs = booster.predict(dmat)
    if probs.ndim == 1:
        probs = probs.reshape(-1, len(CLASSES))
    preds = probs.argmax(axis=1).astype(np.int8)

    outer.update(n)
    return preds, probs.astype(np.float32), n



def _write_predictions(chip_dir: Path, preds: np.ndarray, probs: np.ndarray) -> None:
    np.save(chip_dir / "predictions_xgboost.npy", preds)
    csv_path = chip_dir / "predictions_xgboost.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_idx", "pred", "prob"] + [f"p_{c}" for c in CLASSES])
        for i, (pi, p_row) in enumerate(zip(preds, probs, strict=True)):
            w.writerow(
                [i, CLASSES[int(pi)], f"{float(p_row[int(pi)]):.6f}"]
                + [f"{float(x):.6f}" for x in p_row]
            )


def _mongo_update(
    uri: str,
    db: str,
    collection: str,
    ids: list,
    preds: np.ndarray,
    probs: np.ndarray,
    dry_run: bool,
    preflight: int = 20,
) -> tuple[int, int, int]:
    """Bulk-write XGBoost predictions keyed by _id."""
    from pymongo import MongoClient, UpdateOne

    client = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    col = client[db][collection]

    ops: list[UpdateOne] = []
    skipped_empty = 0
    preflight_ids: list = []
    for i, doc_id in enumerate(ids):
        s = str(doc_id or "")
        if not s or s.lower() in ("none", "nan"):
            skipped_empty += 1
            continue
        pred_idx = int(preds[i])
        scores = {CLASSES[k]: float(probs[i, k]) for k in range(len(CLASSES))}
        key_value = _coerce_id(s)
        if len(preflight_ids) < preflight:
            preflight_ids.append(key_value)
        ops.append(UpdateOne({"_id": key_value}, {"$set": {
            "crop_type_xgboost": CLASSES[pred_idx],
            "crop_type_xgboost_prob": float(probs[i, pred_idx]),
            "crop_type_xgboost_scores": scores,
        }}))

    if not ops:
        client.close()
        return 0, 0, skipped_empty

    if preflight_ids:
        n_found = col.count_documents({"_id": {"$in": preflight_ids}})
        if n_found == 0:
            _log.warning("preflight: 0/%d sampled ids found in %s.%s",
                         len(preflight_ids), db, collection)
        elif n_found < len(preflight_ids):
            _log.warning("preflight: only %d/%d sampled ids found",
                         n_found, len(preflight_ids))

    if dry_run:
        client.close()
        return 0, 0, skipped_empty

    result = col.bulk_write(ops, ordered=False)
    client.close()
    return int(result.matched_count), int(result.modified_count), skipped_empty


def _discover_chip_dirs(root: Path) -> list[Path]:
    if (root / "images.npy").exists():
        return [root]
    return sorted(p.parent for p in root.glob("**/images.npy"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--model", "model_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("outputs/xgboost/xgboost_model.json"),
    show_default=True,
    help="Path to the trained XGBoost Booster (JSON).",
)
@click.option(
    "--chips-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--chips-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
)
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None)
@click.option("--mongo-write", is_flag=True)
@click.option("--skip-no-ids", is_flag=True)
@click.option("--uri", default="mongodb://localhost:27019", show_default=True)
@click.option("--db", default="gis-census", show_default=True)
@click.option("--collection", default="uzcosmos_flats", show_default=True)
@click.option("--dry-run", is_flag=True)
@click.option(
    "--workers",
    type=int,
    default=0,
    show_default=True,
    help="Parallel workers for feature extraction. 0=auto (min(cpu_count,8)), 1=sequential.",
)
def main(
    model_path: Path,
    chips_dir: Path | None,
    chips_root: Path | None,
    manifest: Path | None,
    mongo_write: bool,
    skip_no_ids: bool,
    uri: str,
    db: str,
    collection: str,
    dry_run: bool,
    workers: int,
) -> None:
    """Score chips with the XGBoost model and (optionally) write to Mongo."""
    if (chips_dir is None) == (chips_root is None):
        raise click.UsageError("pass exactly one of --chips-dir or --chips-root")

    if workers == 0:
        workers = min(os.cpu_count() or 1, 8)
    _log.info("using %d feature-extraction workers", workers)

    chip_dirs = _discover_chip_dirs(chips_dir or chips_root)  # type: ignore[arg-type]
    if not chip_dirs:
        raise click.ClickException("no chip dirs found")

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    feature_names = list(booster.feature_names or [])
    if not feature_names:
        raise click.ClickException("booster has no feature_names; retrain with named features")
    _log.info("model loaded — %d features, num_class=%d, objective=%s",
              len(feature_names), booster.num_boosted_rounds(),
              "multi:softprob")

    total = 0
    for d in chip_dirs:
        total += int(np.load(d / "images.npy", mmap_mode="r").shape[0])
    _log.info("total chips to score: %d across %d dirs", total, len(chip_dirs))

    outer = tqdm(total=total, desc="xgb scoring", unit="chip", position=0)
    grand_matched = grand_modified = grand_skipped_empty = 0
    grand_ops = 0
    grand_no_manifest: list[str] = []

    for d in chip_dirs:
        ids = _load_ids(d, manifest if (chips_dir and manifest) else None)
        if chips_root is not None and manifest is not None:
            ids = _load_ids(d, None)

        if skip_no_ids and ids is None:
            grand_no_manifest.append(str(d))
            outer.update(int(np.load(d / "images.npy", mmap_mode="r").shape[0]))
            continue

        preds, probs, n = _score_chip_dir(d, booster, feature_names, outer, workers)
        _write_predictions(d, preds, probs)

        if mongo_write:
            if ids is None:
                grand_no_manifest.append(str(d))
            else:
                matched, modified, skipped = _mongo_update(
                    uri, db, collection, ids, preds, probs, dry_run,
                )
                grand_matched += matched
                grand_modified += modified
                grand_skipped_empty += skipped
                grand_ops += len(ids) - skipped
                outer.set_postfix(
                    matched=grand_matched,
                    modified=grand_modified,
                )

    outer.close()

    click.echo(f"\npredictions written to {len(chip_dirs)} chip dir(s)")
    if mongo_write:
        if dry_run:
            click.echo(f"(dry-run — would have attempted {grand_ops} updates)")
        else:
            click.echo(
                f"mongo: matched={grand_matched}  modified={grand_modified}  "
                f"attempted={grand_ops}  skipped_empty_ids={grand_skipped_empty}"
            )
        if grand_no_manifest:
            click.echo(
                "\nskipped Mongo write for these dirs (no ids.npy / manifest.csv):"
            )
            for d in grand_no_manifest:
                click.echo(f"  - {d}")


if __name__ == "__main__":
    main()
