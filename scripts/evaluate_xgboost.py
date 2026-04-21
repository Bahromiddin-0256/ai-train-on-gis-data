"""Evaluate a trained XGBoost model and generate detailed reports.

This script loads a saved XGBoost model and evaluates it on test data,
generating detailed metrics, confusion matrix, and prediction files.
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
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def load_model(model_path: Path) -> xgb.Booster:
    """Load XGBoost model from file.

    Args:
        model_path: Path to model file (.json, .pkl, or .model)

    Returns:
        Loaded XGBoost Booster
    """
    model = xgb.Booster()

    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        model.load_model(str(model_path))

    return model


def evaluate_model(
    model: xgb.Booster,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
) -> dict:
    """Evaluate model and compute comprehensive metrics.

    Args:
        model: Trained XGBoost model
        features: Feature array (N, n_features)
        labels: Label array (N,)
        feature_names: List of feature names
        class_names: List of class names

    Returns:
        Dictionary of metrics
    """
    dtest = xgb.DMatrix(features, feature_names=feature_names)
    y_pred_proba = model.predict(dtest)

    num_classes = len(class_names)
    if num_classes == 2:
        y_pred = (y_pred_proba > 0.5).astype(int)
        y_pred_proba_class = y_pred_proba
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_pred_proba_class = y_pred_proba

    y_true = labels

    # Basic metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    for i, name in enumerate(class_names):
        metrics[f"precision_{name}"] = float(precision_per_class[i])
        metrics[f"recall_{name}"] = float(recall_per_class[i])
        metrics[f"f1_{name}"] = float(f1_per_class[i])

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()

    # Classification report
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    return metrics, y_pred, y_pred_proba_class


@click.command()
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained XGBoost model",
)
@click.option(
    "--features",
    "features_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to features CSV file",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for results (default: same as model dir)",
)
@click.option(
    "--class-names",
    type=str,
    default="bugdoy,other,paxta",
    help="Comma-separated class names",
)
@click.option(
    "--target-col",
    type=str,
    default="label",
    help="Target column name (default: label)",
)
@click.option(
    "--test-indices",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to test_indices.npy saved by train_xgboost.py. When provided, "
         "evaluation is restricted to the held-out test set only (recommended).",
)
@click.option(
    "--save-predictions",
    is_flag=True,
    help="Save predictions to CSV",
)
def main(
    model_path: Path,
    features_path: Path,
    output_dir: Path | None,
    class_names: str,
    target_col: str,
    test_indices: Path | None,
    save_predictions: bool,
) -> None:
    """Evaluate XGBoost model on features."""
    if output_dir is None:
        output_dir = model_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names_list = class_names.split(",")
    click.echo(f"Class names: {class_names_list}")

    # Load model
    click.echo(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Load data
    click.echo(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)

    if test_indices is not None:
        idx = np.load(str(test_indices))
        df = df.iloc[idx].reset_index(drop=True)
        click.echo(f"Restricting to test split: {len(df)} samples (test_indices.npy)")
    else:
        click.echo(
            "WARNING: --test-indices not provided. Evaluating on ALL samples "
            "(includes training data — metrics will be inflated)."
        )

    feature_cols = [c for c in df.columns if c != target_col]
    features = df[feature_cols].values
    labels = df[target_col].values

    click.echo(f"Samples: {len(df)}, Features: {len(feature_cols)}")

    # Evaluate
    click.echo("Evaluating model...")
    metrics, y_pred, y_pred_proba = evaluate_model(
        model, features, labels, feature_cols, class_names_list
    )

    # Print results
    click.echo(f"\n{'='*60}")
    click.echo("EVALUATION RESULTS")
    click.echo(f"{'='*60}")
    click.echo(f"\nOverall Metrics:")
    click.echo(f"  Accuracy:           {metrics['accuracy']:.4f}")
    click.echo(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
    click.echo(f"  F1 (weighted):      {metrics['f1_weighted']:.4f}")
    click.echo(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    click.echo(f"  Recall (macro):     {metrics['recall_macro']:.4f}")

    click.echo(f"\nPer-Class Metrics:")
    click.echo(f"  {'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    click.echo(f"  {'-'*42}")
    for name in class_names_list:
        p = metrics.get(f"precision_{name}", 0)
        r = metrics.get(f"recall_{name}", 0)
        f = metrics.get(f"f1_{name}", 0)
        click.echo(f"  {name:<12} {p:<10.4f} {r:<10.4f} {f:<10.4f}")

    click.echo(f"\nConfusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    header = "True\\Pred | " + " | ".join(f"{n:>8}" for n in class_names_list)
    click.echo(header)
    click.echo("-" * (12 + 11 * len(class_names_list)))
    for i, name in enumerate(class_names_list):
        row = " | ".join(f"{cm[i, j]:>8}" for j in range(len(class_names_list)))
        click.echo(f"{name:<9} | {row}")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    click.echo(f"\nSaved detailed metrics to {results_path}")

    # Save predictions
    if save_predictions:
        pred_df = pd.DataFrame({
            "true_label": labels,
            "predicted_label": y_pred,
        })

        # Add probability columns
        if len(y_pred_proba.shape) == 1:
            # Binary
            pred_df["proba_class_1"] = y_pred_proba
        else:
            # Multi-class
            for i, name in enumerate(class_names_list):
                pred_df[f"proba_{name}"] = y_pred_proba[:, i]

        pred_path = output_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        click.echo(f"Saved predictions to {pred_path}")

    # Feature importance
    importance = model.get_score(importance_type="gain")
    if importance:
        importance_df = pd.DataFrame(
            [(k, v) for k, v in importance.items()],
            columns=["feature", "importance"],
        ).sort_values("importance", ascending=False)

        importance_path = output_dir / "feature_importance_eval.csv"
        importance_df.to_csv(importance_path, index=False)
        click.echo(f"Saved feature importance to {importance_path}")

        click.echo(f"\nTop 15 features by importance:")
        for _, row in importance_df.head(15).iterrows():
            click.echo(f"  {row['feature']:<30}: {row['importance']:>12.2f}")


if __name__ == "__main__":
    main()
