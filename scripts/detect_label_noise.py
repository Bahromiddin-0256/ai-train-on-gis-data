"""Find likely-mislabelled samples in an ``images.npy`` / ``labels.npy`` pair.

Strategy — the standard "self-prediction" recipe for label-noise detection:

    1. Compute compact per-channel summary features from each chip
       (mean / std / percentiles per channel across its 64x64 pixels).
    2. Train an XGBoost classifier with K-fold cross-validation and collect
       *out-of-fold* predicted probabilities — every sample is scored by a
       model that was **not** trained on it.
    3. For each sample, compare its declared label's OOF probability against
       thresholds to assign one of three verdicts:

       * ``accepted`` — declared label is the top OOF class AND its probability
         is above ``--accept-prob``. The classifier agrees confidently.
       * ``rejected`` — declared label is NOT the top OOF class AND the top
         class is above ``--reject-prob``. The classifier confidently picks a
         different class → probable mislabel.
       * ``review``   — everything else (ambiguous / low-confidence).

Outputs a per-sample CSV plus a one-line summary. The schema mirrors
``gis_train.data.worldcereal.PolygonScore`` so both signals (self-prediction +
WorldCereal) can be combined downstream.

Usage::

    python scripts/detect_label_noise.py \\
        --data-dir data/processed_regional_mt \\
        --out-csv data/processed_regional_mt/label_noise_verdicts.csv

The script is deliberately self-contained — it does not depend on
``extract_features.py`` output, so you can run it on any processed dataset
that ships ``images.npy`` + ``labels.npy``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)


def _summary_features(images: np.ndarray) -> np.ndarray:
    """Per-sample compact feature vector: per-channel stats across HxW.

    Input shape ``(N, C, H, W)``; output shape ``(N, C * 6)`` with six
    statistics per channel (mean, std, p10, p50, p90, range).
    """
    n, c, _, _ = images.shape
    flat = images.reshape(n, c, -1).astype(np.float32)

    mean = flat.mean(axis=2)
    std = flat.std(axis=2)
    p10 = np.percentile(flat, 10, axis=2)
    p50 = np.percentile(flat, 50, axis=2)
    p90 = np.percentile(flat, 90, axis=2)
    rng = flat.max(axis=2) - flat.min(axis=2)

    return np.concatenate([mean, std, p10, p50, p90, rng], axis=1)


def _oof_proba(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold predicted probabilities via stratified K-fold XGBoost."""
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold

    oof = np.zeros((len(labels), n_classes), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "verbosity": 0,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels), 1):
        dtrain = xgb.DMatrix(features[train_idx], label=labels[train_idx])
        dval = xgb.DMatrix(features[val_idx], label=labels[val_idx])

        classes, counts = np.unique(labels[train_idx], return_counts=True)
        class_weight = {int(cls): len(train_idx) / (n_classes * cnt) for cls, cnt in zip(classes, counts, strict=True)}
        dtrain.set_weight(np.array([class_weight[int(y)] for y in labels[train_idx]], dtype=np.float32))

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=400,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        oof[val_idx] = booster.predict(dval)
        _log.info(
            "fold %d/%d  val_mlogloss=%.4f",
            fold,
            n_splits,
            booster.best_score,
        )

    return oof


def _verdicts(
    labels: np.ndarray,
    oof: np.ndarray,
    accept_prob: float,
    reject_prob: float,
) -> pd.DataFrame:
    predicted = oof.argmax(axis=1)
    p_declared = oof[np.arange(len(labels)), labels]
    p_predicted = oof[np.arange(len(labels)), predicted]
    margin = p_declared - p_predicted  # negative when declared != predicted

    verdict = np.empty(len(labels), dtype=object)
    reason = np.empty(len(labels), dtype=object)

    accepted_mask = (predicted == labels) & (p_declared >= accept_prob)
    rejected_mask = (predicted != labels) & (p_predicted >= reject_prob)

    verdict[:] = "review"
    reason[:] = "low_confidence_or_near_boundary"
    verdict[accepted_mask] = "accepted"
    reason[accepted_mask] = "self_prediction_agrees"
    verdict[rejected_mask] = "rejected"
    reason[rejected_mask] = "self_prediction_confidently_disagrees"

    return pd.DataFrame(
        {
            "sample_idx": np.arange(len(labels)),
            "declared_label": labels,
            "predicted_label": predicted,
            "p_declared": p_declared.round(4),
            "p_predicted": p_predicted.round(4),
            "margin": margin.round(4),
            "verdict": verdict,
            "reason": reason,
        }
    )


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing images.npy and labels.npy.",
)
@click.option(
    "--out-csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Destination CSV. Default: <data-dir>/label_noise_verdicts.csv.",
)
@click.option("--n-splits", type=int, default=5, show_default=True)
@click.option("--accept-prob", type=float, default=0.65, show_default=True)
@click.option("--reject-prob", type=float, default=0.70, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--limit",
    type=int,
    default=0,
    show_default=True,
    help="0 = no limit. Use a small number for quick smoke tests.",
)
def main(
    data_dir: Path,
    out_csv: Path | None,
    n_splits: int,
    accept_prob: float,
    reject_prob: float,
    seed: int,
    limit: int,
) -> None:
    images_path = data_dir / "images.npy"
    labels_path = data_dir / "labels.npy"
    if not images_path.exists() or not labels_path.exists():
        raise click.ClickException(f"missing images.npy or labels.npy in {data_dir}")

    # mmap the image array so we never hold all 52 GB in RAM; feature
    # reduction reads it sequentially.
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path).astype(np.int64)
    if limit:
        images = images[:limit]
        labels = labels[:limit]

    _log.info("images: %s  labels: %s", images.shape, labels.shape)
    n_classes = int(labels.max()) + 1
    _log.info("extracting compact summary features (per-channel mean/std/percentiles)")
    features = _summary_features(np.asarray(images))
    _log.info("feature matrix: %s", features.shape)

    oof = _oof_proba(features, labels, n_classes=n_classes, n_splits=n_splits, seed=seed)

    df = _verdicts(labels, oof, accept_prob=accept_prob, reject_prob=reject_prob)
    out_csv = out_csv or (data_dir / "label_noise_verdicts.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    counts = df["verdict"].value_counts().to_dict()
    print(f"\nwrote {out_csv}")
    print(f"  total    : {len(df):>7,}")
    for v in ("accepted", "review", "rejected"):
        print(f"  {v:<9}: {counts.get(v, 0):>7,}")

    print("\nper-class breakdown (declared label → verdict counts):")
    crosstab = pd.crosstab(df["declared_label"], df["verdict"])
    print(crosstab.to_string())

    oof_acc = float((oof.argmax(axis=1) == labels).mean())
    print(f"\noverall OOF accuracy: {oof_acc:.4f}")


if __name__ == "__main__":
    main()
