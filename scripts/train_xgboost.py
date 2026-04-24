"""Train XGBoost model on extracted features for crop classification.

This script loads features extracted by extract_features.py, trains an XGBoost
classifier with cross-validation, performs hyperparameter tuning, and saves the
best model along with metrics.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


def prepare_data_random(
    df: pd.DataFrame,
    target_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random stratified train/val/test split."""
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=random_state,
    )
    adjusted_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=adjusted_val_size,
        stratify=train_val_df[target_col], random_state=random_state,
    )
    return train_df, val_df, test_df


def prepare_data_geo(
    df: pd.DataFrame,
    target_col: str = "label",
    tuman_col: str = "tuman_code",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Geographic split: hold out 1 tuman per viloyat for test, 1 for val.

    Viloyat is identified by the first 4 digits of tuman_code.
    Within each viloyat, tumans are ranked by sample count descending.
    The largest tuman → test, second largest → val, rest → train.
    This ensures test/val contain enough samples and come from unseen geography.
    """
    from collections import defaultdict

    by_viloyat: dict[int, list[int]] = defaultdict(list)
    tuman_counts = df[tuman_col].value_counts()
    for tcode, count in tuman_counts.items():
        vil = int(str(int(tcode))[:4])
        by_viloyat[vil].append(int(tcode))

    test_tumans, val_tumans = set(), set()
    for vil, tcodes in by_viloyat.items():
        ranked = sorted(tcodes, key=lambda c: tuman_counts[c], reverse=True)
        test_tumans.add(ranked[0])
        if len(ranked) > 1:
            val_tumans.add(ranked[1])

    test_df  = df[df[tuman_col].isin(test_tumans)]
    val_df   = df[df[tuman_col].isin(val_tumans)]
    train_df = df[~df[tuman_col].isin(test_tumans | val_tumans)]

    click.echo(f"\nGeographic split:")
    click.echo(f"  Test tumans  ({len(test_tumans)}): {sorted(test_tumans)}")
    click.echo(f"  Val tumans   ({len(val_tumans)}): {sorted(val_tumans)}")
    click.echo(f"  Train tumans ({len(by_viloyat) * max(0, max(len(v) for v in by_viloyat.values()) - 2)} approx)")
    return train_df, val_df, test_df


def prepare_data(
    features_path: Path,
    target_col: str = "label",
    tuman_col: str = "tuman_code",
    test_size: float = 0.15,
    val_size: float = 0.15,
    geo_split: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and split data into train/val/test sets."""
    df = pd.read_csv(features_path)

    meta_cols = {target_col, tuman_col}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    if geo_split:
        if tuman_col not in df.columns:
            raise click.ClickException(
                f"--geo-split requires '{tuman_col}' column in features CSV. "
                "Re-run extract_features.py to generate it."
            )
        train_df, val_df, test_df = prepare_data_geo(df, target_col, tuman_col, random_state)
    else:
        train_df, val_df, test_df = prepare_data_random(df, target_col, test_size, val_size, random_state)

    click.echo(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}  Features: {len(feature_cols)}")
    return train_df, val_df, test_df, feature_cols


def select_top_features(
    model: xgb.Booster,
    feature_cols: list[str],
    top_n: int,
) -> list[str]:
    """Return top-N features by gain importance."""
    importance = model.get_score(importance_type="gain")
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in ranked[:top_n] if f in feature_cols]
    click.echo(f"Feature selection: {len(feature_cols)} → {len(selected)} (top-{top_n} by gain)")
    return selected


def get_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights for imbalanced datasets.

    Args:
        y: Target array

    Returns:
        Dictionary mapping class index to weight
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
    return weights


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    params: dict,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 100,
) -> xgb.Booster:
    """Train XGBoost model with early stopping.

    Args:
        train_df: Training data
        val_df: Validation data
        feature_cols: List of feature column names
        target_col: Target column name
        params: XGBoost parameters
        num_boost_round: Maximum number of boosting rounds
        early_stopping_rounds: Rounds to wait for improvement
        verbose_eval: Print evaluation every N rounds

    Returns:
        Trained XGBoost Booster
    """
    # Create DMatrix
    dtrain = xgb.DMatrix(
        train_df[feature_cols].values,
        label=train_df[target_col].values,
        feature_names=feature_cols,
    )
    dval = xgb.DMatrix(
        val_df[feature_cols].values,
        label=val_df[target_col].values,
        feature_names=feature_cols,
    )

    # Compute sample weights for class balance
    class_weights = get_class_weights(train_df[target_col].values)
    sample_weights = np.array([class_weights[y] for y in train_df[target_col].values])
    dtrain.set_weight(sample_weights)

    # Train with early stopping
    evals = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )

    return model


def evaluate_model(
    model: xgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    class_names: list[str] | None = None,
) -> dict:
    """Evaluate model on a dataset.

    Args:
        model: Trained XGBoost model
        df: Data to evaluate on
        feature_cols: Feature column names
        target_col: Target column name
        class_names: Optional class names for reporting

    Returns:
        Dictionary of metrics
    """
    dtest = xgb.DMatrix(df[feature_cols].values, feature_names=feature_cols)
    y_pred_proba = model.predict(dtest)

    if len(y_pred_proba.shape) == 1:
        # Binary classification
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        # Multi-class classification
        y_pred = np.argmax(y_pred_proba, axis=1)

    y_true = df[target_col].values

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    for i, f1 in enumerate(per_class_f1):
        class_name = class_names[i] if class_names else f"class_{i}"
        metrics[f"f1_{class_name}"] = float(f1)

    return metrics, y_pred


def cross_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    params: dict,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Perform cross-validation.

    Args:
        df: Full dataset
        feature_cols: Feature column names
        target_col: Target column name
        params: XGBoost parameters
        n_splits: Number of CV folds
        random_state: Random seed

    Returns:
        Dictionary with CV results
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df[feature_cols], df[target_col])):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        model = train_model(
            train_df,
            val_df,
            feature_cols,
            target_col,
            params,
            num_boost_round=500,
            early_stopping_rounds=30,
            verbose_eval=0,
        )

        metrics, _ = evaluate_model(model, val_df, feature_cols, target_col)
        cv_scores.append(metrics["accuracy"])
        click.echo(f"Fold {fold + 1}: accuracy = {metrics['accuracy']:.4f}")

    return {
        "mean_accuracy": float(np.mean(cv_scores)),
        "std_accuracy": float(np.std(cv_scores)),
        "fold_scores": cv_scores,
    }


@click.command()
@click.option(
    "--features",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to features CSV file",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="outputs/xgboost",
    help="Output directory for model and metrics",
)
@click.option(
    "--test-size",
    type=float,
    default=0.15,
    help="Fraction of data for test set (default: 0.15)",
)
@click.option(
    "--val-size",
    type=float,
    default=0.15,
    help="Fraction of data for validation set (default: 0.15)",
)
@click.option("--max-depth", type=int, default=5, show_default=True,
              help="Maximum tree depth.")
@click.option("--learning-rate", type=float, default=0.05, show_default=True)
@click.option("--n-estimators", type=int, default=1000, show_default=True)
@click.option("--subsample", type=float, default=0.6, show_default=True)
@click.option("--colsample-bytree", type=float, default=0.5, show_default=True)
@click.option("--min-child-weight", type=int, default=10, show_default=True)
@click.option("--gamma", type=float, default=0.1, show_default=True)
@click.option("--reg-alpha", type=float, default=1.0, show_default=True,
              help="L1 regularization.")
@click.option("--reg-lambda", type=float, default=5.0, show_default=True,
              help="L2 regularization.")
@click.option(
    "--num-class",
    type=int,
    default=3,
    help="Number of classes (default: 3)",
)
@click.option(
    "--class-names",
    type=str,
    default="bugdoy,other,paxta",
    help="Comma-separated class names",
)
@click.option("--cv", is_flag=True, help="Run cross-validation before final training.")
@click.option("--cv-folds", type=int, default=5, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--early-stopping", type=int, default=50, show_default=True)
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default="cuda", show_default=True)
@click.option("--geo-split", is_flag=True, default=False,
              help="Geographic split: hold out 1 tuman/viloyat for test, 1 for val.")
@click.option("--top-features", type=int, default=0, show_default=True,
              help="Keep only top-N features by gain importance. 0 = use all.")
def main(
    features: Path,
    output_dir: Path,
    test_size: float,
    val_size: float,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    subsample: float,
    colsample_bytree: float,
    min_child_weight: int,
    gamma: float,
    reg_alpha: float,
    reg_lambda: float,
    num_class: int,
    class_names: str,
    cv: bool,
    cv_folds: int,
    seed: int,
    early_stopping: int,
    device: str,
    geo_split: bool,
    top_features: int,
) -> None:
    """Train XGBoost model on extracted features."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names_list = class_names.split(",")
    click.echo(f"Class names: {class_names_list}")
    click.echo(f"Split strategy: {'geographic (tuman-level)' if geo_split else 'random stratified'}")

    # Load and split data
    click.echo("Loading data...")
    train_df, val_df, test_df, feature_cols = prepare_data(
        features,
        test_size=test_size,
        val_size=val_size,
        geo_split=geo_split,
        random_state=seed,
    )

    # Save test indices so evaluate_xgboost.py can restrict to the held-out set
    test_indices_path = output_dir / "test_indices.npy"
    np.save(str(test_indices_path), test_df.index.values)
    click.echo(f"Saved test indices ({len(test_df)}) → {test_indices_path}")

    # XGBoost parameters
    params = {
        "objective": "multi:softprob" if num_class > 2 else "binary:logistic",
        "num_class": num_class,
        "eval_metric": "mlogloss" if num_class > 2 else "logloss",
        "max_depth": max_depth,
        "eta": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "alpha": reg_alpha,
        "lambda": reg_lambda,
        "tree_method": "hist",
        "device": device,
        "random_state": seed,
        "seed": seed,
    }

    click.echo(f"XGBoost params: {json.dumps(params, indent=2)}")

    # Cross-validation (optional)
    if cv:
        click.echo(f"Running {cv_folds}-fold cross-validation...")
        full_df = pd.concat([train_df, val_df, test_df])
        cv_results = cross_validate(
            full_df,
            feature_cols,
            "label",
            params,
            n_splits=cv_folds,
            random_state=seed,
        )
        click.echo(f"CV Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")

        # Save CV results
        with open(output_dir / "cv_results.json", "w") as f:
            json.dump(cv_results, f, indent=2)

    # Phase 1: train on train-only to get feature importances for selection
    click.echo("\nTraining phase 1 (feature selection)...")
    model_phase1 = train_model(
        train_df, val_df, feature_cols, "label", params,
        num_boost_round=n_estimators,
        early_stopping_rounds=early_stopping,
        verbose_eval=100,
    )
    if top_features > 0:
        feature_cols = select_top_features(model_phase1, feature_cols, top_features)

    # Phase 2: final training with selected features
    click.echo("\nTraining final model...")
    combined_train = pd.concat([train_df, val_df])
    model = train_model(
        combined_train,
        val_df,
        feature_cols,
        "label",
        params,
        num_boost_round=n_estimators,
        early_stopping_rounds=early_stopping,
        verbose_eval=50,
    )

    # Evaluate on all sets
    click.echo("\nEvaluating...")
    train_metrics, _ = evaluate_model(
        model, train_df, feature_cols, "label", class_names_list
    )
    val_metrics, _ = evaluate_model(
        model, val_df, feature_cols, "label", class_names_list
    )
    test_metrics, y_pred = evaluate_model(
        model, test_df, feature_cols, "label", class_names_list
    )

    # Print results
    click.echo(f"\n{'='*50}")
    click.echo("TRAIN METRICS")
    click.echo(f"{'='*50}")
    click.echo(f"Accuracy:  {train_metrics['accuracy']:.4f}")
    click.echo(f"F1 Macro:  {train_metrics['f1_macro']:.4f}")
    click.echo(f"F1 Weighted: {train_metrics['f1_weighted']:.4f}")

    click.echo(f"\n{'='*50}")
    click.echo("VALIDATION METRICS")
    click.echo(f"{'='*50}")
    click.echo(f"Accuracy:  {val_metrics['accuracy']:.4f}")
    click.echo(f"F1 Macro:  {val_metrics['f1_macro']:.4f}")
    click.echo(f"F1 Weighted: {val_metrics['f1_weighted']:.4f}")

    click.echo(f"\n{'='*50}")
    click.echo("TEST METRICS")
    click.echo(f"{'='*50}")
    click.echo(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    click.echo(f"F1 Macro:  {test_metrics['f1_macro']:.4f}")
    click.echo(f"F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    click.echo("\nPer-class F1:")
    for name in class_names_list:
        click.echo(f"  {name}: {test_metrics.get(f'f1_{name}', 'N/A'):.4f}")

    click.echo("\nConfusion Matrix:")
    cm = np.array(test_metrics["confusion_matrix"])
    click.echo(f"True\\Pred | {' | '.join(f'{n:>7}' for n in class_names_list)}")
    click.echo("-" * (50))
    for i, name in enumerate(class_names_list):
        row = " | ".join(f"{cm[i, j]:>7}" for j in range(len(class_names_list)))
        click.echo(f"{name:>9} | {row}")

    # Save model
    model_path = output_dir / "xgboost_model.json"
    model.save_model(str(model_path))
    click.echo(f"\nSaved model to {model_path}")

    # Also save as pickle for convenience
    with open(output_dir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save metrics
    results = {
        "params": params,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "class_names": class_names_list,
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    click.echo(f"Saved metrics to {output_dir / 'metrics.json'}")

    # Save feature importance
    importance = model.get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        [(k, v) for k, v in importance.items()],
        columns=["feature", "importance"],
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    click.echo(f"Saved feature importance to {output_dir / 'feature_importance.csv'}")
    click.echo(f"\nTop 10 features:")
    for _, row in importance_df.head(10).iterrows():
        click.echo(f"  {row['feature']}: {row['importance']:.2f}")


if __name__ == "__main__":
    main()
