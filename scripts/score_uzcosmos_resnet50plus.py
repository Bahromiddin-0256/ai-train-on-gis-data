"""Score already-extracted Sentinel-2 chips with the ResNet-50+ TorchScript
model and (optionally) write the predicted class back to MongoDB as
``crop_type_resnet50plus``.

This script does NOT talk to Planetary Computer — it reads chips from disk
(``images.npy`` produced by ``scripts/prepare_labels.py`` /
``scripts/build_dataset.py``). One or more chip dirs can be scored in a
single run.

Layout expected per chip dir::

    <chip_dir>/
      images.npy            (N, 90, 64, 64) float32, raw DN 0..10000
      labels.npy            (N,)            int64   (optional, for sanity stats)
      ids.npy               (N,)            object  (optional; ObjectId strings)
        -- or a CSV/Parquet passed with --manifest, keyed by row order

What it writes
--------------
Always:
    <chip_dir>/predictions_resnet50plus.csv    row_idx, pred, prob, p_bugdoy, p_other, p_paxta
    <chip_dir>/predictions_resnet50plus.npy    (N,) int8 class indices

With --mongo-write (and ids.npy / --manifest present):
    col.bulk_write({_id: doc_id}, $set: {
        crop_type_resnet50plus:        <str>,
        crop_type_resnet50plus_prob:   <float>,
        crop_type_resnet50plus_scores: {class: prob, ...},
    })

Usage
-----
    .venv/bin/python scripts/score_uzcosmos_resnet50plus.py \\
        --model outputs/resnet50_plus/resnet50plus_s2-moco_90ch-6win_focal-tta-mixup_20260423-1601.pt \\
        --chips-root data/v6win \\
        --mongo-write

Single chip dir (no auto-discovery)::

    .venv/bin/python scripts/score_uzcosmos_resnet50plus.py \\
        --model outputs/resnet50_plus/....pt \\
        --chips-dir data/processed_regional_v6win
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

CLASSES: tuple[str, ...] = ("bugdoy", "other", "paxta")
CHIP_SIZE = 64
N_WINDOWS = 6
CHANNELS_PER_WINDOW = 15
IN_CHANNELS = N_WINDOWS * CHANNELS_PER_WINDOW  # 90
DN_SCALE = 10_000.0  # raw Sentinel-2 → reflectance

# Per-channel mean/std from configs/data/uzbekistan_s2.yaml (reflectance scale).
MEAN = np.array([
    0.1705, 0.2007, 0.2155, 0.2437, 0.2916, 0.3087, 0.3192, 0.2940, 0.2572,
    0.5853, 0.5934, 0.5990, 0.5567, 0.2884, 0.5471,
    0.1834, 0.2212, 0.2360, 0.2759, 0.3475, 0.3727, 0.3847, 0.3246, 0.2704,
    0.6088, 0.6236, 0.6208, 0.5716, 0.2687, 0.5783,
    0.1787, 0.2217, 0.2374, 0.2870, 0.3708, 0.3953, 0.4032, 0.3402, 0.2726,
    0.6179, 0.6347, 0.6295, 0.5735, 0.2650, 0.5877,
    0.1624, 0.2030, 0.2052, 0.2689, 0.3947, 0.4302, 0.4471, 0.3219, 0.2434,
    0.6814, 0.7086, 0.6803, 0.6201, 0.2354, 0.6429,
    0.1601, 0.1970, 0.2106, 0.2639, 0.3443, 0.3690, 0.3834, 0.3155, 0.2452,
    0.6350, 0.6474, 0.6463, 0.5843, 0.2550, 0.6015,
    0.1808, 0.2100, 0.2404, 0.2678, 0.2931, 0.3053, 0.3243, 0.3303, 0.2729,
    0.5690, 0.5722, 0.5987, 0.5434, 0.3243, 0.5397,
], dtype=np.float32)[:, None, None]

STD = np.array([
    0.0654, 0.0753, 0.0881, 0.0891, 0.1083, 0.1206, 0.1262, 0.1091, 0.1021,
    0.0875, 0.0995, 0.0759, 0.0669, 0.1095, 0.0740,
    0.0678, 0.0770, 0.0912, 0.0897, 0.1035, 0.1160, 0.1224, 0.1056, 0.0966,
    0.1034, 0.1141, 0.0949, 0.0828, 0.0907, 0.0881,
    0.0673, 0.0783, 0.0974, 0.0997, 0.1222, 0.1326, 0.1384, 0.1244, 0.1029,
    0.0950, 0.1065, 0.0856, 0.0733, 0.0910, 0.0761,
    0.0529, 0.0624, 0.0830, 0.0801, 0.0953, 0.1063, 0.1129, 0.0920, 0.0798,
    0.0841, 0.1002, 0.0665, 0.0586, 0.0712, 0.0693,
    0.0592, 0.0716, 0.0879, 0.0948, 0.1150, 0.1245, 0.1308, 0.1133, 0.0941,
    0.0769, 0.0864, 0.0671, 0.0533, 0.0915, 0.0649,
    0.0473, 0.0562, 0.0698, 0.0743, 0.0810, 0.0855, 0.0938, 0.0958, 0.0789,
    0.0469, 0.0510, 0.0489, 0.0366, 0.0829, 0.0442,
], dtype=np.float32)[:, None, None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_chip_dirs(root: Path) -> list[Path]:
    """Find all chip dirs (each containing images.npy) under *root*."""
    if (root / "images.npy").exists():
        return [root]
    dirs = sorted(p.parent for p in root.glob("**/images.npy"))
    return dirs


def _load_ids(chip_dir: Path, manifest: Path | None) -> list | None:
    """Return the per-row Mongo _id for this chip dir, or None if unavailable.

    Preference: ``--manifest`` > ``<chip_dir>/ids.npy`` > ``<chip_dir>/manifest.csv``.
    """
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
        import pandas as pd

        df = pd.read_parquet(path)
        key = next((k for k in ("_id", "mongo_id", "oid", "id") if k in df.columns), None)
        if key is None:
            raise click.ClickException(
                f"{path} has no _id/mongo_id/oid/id column; columns: {list(df.columns)}"
            )
        raw = df[key].tolist()
    else:
        raise click.ClickException(f"unsupported manifest format: {path.suffix}")

    ids = [str(x) if x is not None else "" for x in raw]
    bad = sum(1 for s in ids if not s or s.lower() in ("none", "nan"))
    if bad:
        _log.warning("%s: %d empty/None ids (out of %d); rows with empty ids will be skipped",
                     path, bad, len(ids))
    dup = len(ids) - len(set(ids))
    if dup:
        _log.warning("%s: %d duplicate ids detected — last write wins per _id", path, dup)
    return ids


def _coerce_id(s: str) -> object:
    """Pick the right Python type for a Mongo _id query.

    24-hex-char → ObjectId; all-digits → int; otherwise str.
    The function is single-dispatch so ids of mixed type coexist in one manifest.
    """
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


def _normalise_batch(batch: np.ndarray) -> np.ndarray:
    """Raw DN (0-10000) → reflectance (0-1) → standardised."""
    batch = batch.astype(np.float32, copy=False) / DN_SCALE
    return (batch - MEAN) / STD


def _score_chip_dir(
    chip_dir: Path,
    model,
    device: torch.device,
    batch: int,
    ids: list | None,
    outer: tqdm,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Score every chip in <chip_dir>/images.npy. Returns (preds, probs, n)."""
    images = np.load(chip_dir / "images.npy", mmap_mode="r")
    n = int(images.shape[0])

    if images.shape[1:] != (IN_CHANNELS, CHIP_SIZE, CHIP_SIZE):
        raise click.ClickException(
            f"{chip_dir}/images.npy shape {tuple(images.shape)} does not match "
            f"(N, {IN_CHANNELS}, {CHIP_SIZE}, {CHIP_SIZE})"
        )
    if ids is not None and len(ids) != n:
        raise click.ClickException(
            f"{chip_dir}: manifest length {len(ids)} != images.npy rows {n}"
        )

    probs = np.empty((n, len(CLASSES)), dtype=np.float32)
    batch_bar = tqdm(
        range(0, n, batch),
        desc=f"  {chip_dir.name}",
        unit="batch",
        leave=False,
        position=1,
    )
    for i in batch_bar:
        chunk = np.ascontiguousarray(images[i : i + batch])
        chunk = _normalise_batch(chunk)
        x = torch.from_numpy(chunk).to(device, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
        probs[i : i + batch] = p
        outer.update(len(chunk))
    batch_bar.close()

    preds = probs.argmax(axis=1).astype(np.int8)
    return preds, probs, n


def _write_predictions(
    chip_dir: Path, preds: np.ndarray, probs: np.ndarray, model_name: str,
) -> None:
    np.save(chip_dir / f"predictions_{model_name}.npy", preds)
    csv_path = chip_dir / f"predictions_{model_name}.csv"
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
    model_name: str,
    preflight: int = 20,
) -> tuple[int, int, int]:
    """Bulk-write predictions keyed by _id, writing fields namespaced by ``model_name``.

    Returns (matched_count, modified_count, skipped_empty).
    """
    from pymongo import MongoClient, UpdateOne

    client = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    col = client[db][collection]

    field_class = f"crop_type_{model_name}"
    field_prob = f"crop_type_{model_name}_prob"
    field_scores = f"crop_type_{model_name}_scores"

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
            field_class: CLASSES[pred_idx],
            field_prob: float(probs[i, pred_idx]),
            field_scores: scores,
        }}))

    if not ops:
        client.close()
        return 0, 0, skipped_empty

    # Pre-flight: confirm a sample of ids actually exists in the collection.
    # Cheap single query; catches stale manifests before the bulk write lands.
    if preflight_ids:
        n_found = col.count_documents({"_id": {"$in": preflight_ids}})
        if n_found == 0:
            _log.warning(
                "preflight: 0/%d sampled ids found in %s.%s — manifest likely "
                "points at a different database/collection (continuing anyway)",
                len(preflight_ids), db, collection,
            )
        elif n_found < len(preflight_ids):
            _log.warning(
                "preflight: only %d/%d sampled ids found in %s.%s — %d docs may "
                "have been deleted or the manifest is partly stale",
                n_found, len(preflight_ids), db, collection,
                len(preflight_ids) - n_found,
            )

    if dry_run:
        client.close()
        return 0, 0, skipped_empty

    result = col.bulk_write(ops, ordered=False)
    client.close()
    return int(result.matched_count), int(result.modified_count), skipped_empty


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--model", "model_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the TorchScript .pt ResNet-50+ model.",
)
@click.option(
    "--chips-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Single chip directory containing images.npy.",
)
@click.option(
    "--chips-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Root directory — every subdir containing images.npy is scored.",
)
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Explicit row-order→_id manifest (.npy / .csv / .parquet). "
         "If omitted, script looks for ids.npy or manifest.csv in each chip dir.",
)
@click.option("--model-name", default="resnet50plus", show_default=True,
              help="Short model id used to namespace outputs: Mongo fields become "
                   "crop_type_<name>, crop_type_<name>_prob, crop_type_<name>_scores; "
                   "per-chip-dir files become predictions_<name>.csv/.npy. "
                   "Use 'tempcnn' for the TempCNN model, etc.")
@click.option("--batch", type=int, default=64, show_default=True,
              help="Model forward batch size.")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]),
              default="auto", show_default=True)
@click.option("--mongo-write", is_flag=True,
              help="Bulk-write crop_type_<model-name> to MongoDB (requires ids.npy / manifest).")
@click.option("--skip-no-ids", is_flag=True,
              help="Skip chip dirs that have no ids.npy (useful while other jobs re-extract them).")
@click.option("--uri", default="mongodb://localhost:27019", show_default=True)
@click.option("--db", default="gis-census", show_default=True)
@click.option("--collection", default="uzcosmos_flats", show_default=True)
@click.option("--dry-run", is_flag=True,
              help="Compute predictions and write CSVs, but do not write to MongoDB.")
def main(
    model_path: Path,
    chips_dir: Path | None,
    chips_root: Path | None,
    manifest: Path | None,
    model_name: str,
    batch: int,
    device: str,
    mongo_write: bool,
    skip_no_ids: bool,
    uri: str,
    db: str,
    collection: str,
    dry_run: bool,
) -> None:
    """Score local chips and (optionally) update Mongo with crop_type_<model-name>."""
    if (chips_dir is None) == (chips_root is None):
        raise click.UsageError("pass exactly one of --chips-dir or --chips-root")

    chip_dirs = _discover_chip_dirs(chips_dir or chips_root)  # type: ignore[arg-type]
    if not chip_dirs:
        raise click.ClickException("no chip dirs found")
    _log.info("found %d chip dir(s)", len(chip_dirs))

    # --- device + model ---------------------------------------------------
    torch_device = torch.device(
        "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    _log.info("loading TorchScript model: %s (device=%s)", model_path, torch_device)
    model = torch.jit.load(str(model_path), map_location=torch_device)
    model.eval()
    with torch.no_grad():
        probe = torch.zeros(1, IN_CHANNELS, CHIP_SIZE, CHIP_SIZE, device=torch_device)
        out = model(probe)
    if out.shape != (1, len(CLASSES)):
        raise click.ClickException(
            f"model output shape {tuple(out.shape)} != (1, {len(CLASSES)}) — "
            f"is this a 90-ch ResNet-50+ checkpoint?"
        )

    # --- total for outer bar ---------------------------------------------
    total = 0
    per_dir_n: list[int] = []
    for d in chip_dirs:
        n = int(np.load(d / "images.npy", mmap_mode="r").shape[0])
        per_dir_n.append(n)
        total += n
    _log.info("total chips to score: %d", total)

    outer = tqdm(total=total, desc="scoring", unit="chip", position=0)
    grand_matched = grand_modified = grand_skipped_empty = 0
    grand_ops = 0
    grand_no_manifest: list[str] = []

    for d in chip_dirs:
        ids = _load_ids(d, manifest if (chips_dir and manifest) else None)
        # When running over a --chips-root with a single --manifest, we cannot
        # sanely share that manifest across multiple dirs; require per-dir files.
        if chips_root is not None and manifest is not None:
            ids = _load_ids(d, None)

        if skip_no_ids and ids is None:
            grand_no_manifest.append(str(d))
            # Skip scoring entirely — useful while another job is mid-write.
            outer.update(int(np.load(d / "images.npy", mmap_mode="r").shape[0]))
            continue

        preds, probs, n = _score_chip_dir(d, model, torch_device, batch, ids, outer)
        _write_predictions(d, preds, probs, model_name)

        if mongo_write:
            if ids is None:
                grand_no_manifest.append(str(d))
            else:
                matched, modified, skipped = _mongo_update(
                    uri, db, collection, ids, preds, probs, dry_run, model_name,
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

    # --- summary ----------------------------------------------------------
    click.echo(f"\npredictions written to {len(chip_dirs)} chip dir(s)")
    if mongo_write:
        if dry_run:
            click.echo(f"(dry-run — would have attempted {grand_ops} updates)")
        else:
            click.echo(
                f"mongo: matched={grand_matched}  modified={grand_modified}  "
                f"attempted={grand_ops}  skipped_empty_ids={grand_skipped_empty}"
            )
            if grand_ops and grand_matched == 0:
                click.echo(
                    "⚠ every update missed — the manifest does not match this "
                    "collection/database. Check --uri/--db/--collection and the "
                    "_id type (ObjectId vs int vs string)."
                )
            elif grand_matched < grand_ops:
                click.echo(
                    f"⚠ {grand_ops - grand_matched} attempted updates did not "
                    "match any document — the manifest may be partly stale."
                )
        if grand_no_manifest:
            click.echo(
                "\nskipped Mongo write for these dirs (no ids.npy / manifest.csv):"
            )
            for d in grand_no_manifest:
                click.echo(f"  - {d}")
            click.echo(
                "\nTo enable Mongo write-back, place a row-aligned ids.npy or "
                "manifest.csv (with an _id column) next to each images.npy."
            )


if __name__ == "__main__":
    main()
