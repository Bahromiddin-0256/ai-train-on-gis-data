"""Evaluate a trained checkpoint and produce a confusion matrix PNG.

Usage:
    python scripts/eval_confusion.py \
        --ckpt lightning_logs/version_XX/checkpoints/best.ckpt \
        --data-dir data/processed_regional_mt \
        --out lightning_logs/version_XX/confusion.png
"""

import click
import numpy as np
from pathlib import Path


@click.command()
@click.option("--ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--data-dir", type=click.Path(exists=True, path_type=Path), required=True,
              help="Directory with images.npy + labels.npy (test set).")
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="Output PNG path (default: same dir as ckpt).")
@click.option("--batch-size", default=64, show_default=True)
@click.option("--class-names", default="bugdoy,other,paxta", show_default=True)
def main(ckpt: Path, data_dir: Path, out: Path | None, batch_size: int, class_names: str) -> None:
    """Load checkpoint, run inference on data_dir, save confusion matrix."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torchmetrics.classification import MulticlassConfusionMatrix

    from gis_train.models.classifier import CropClassifier

    names = [n.strip() for n in class_names.split(",")]
    num_classes = len(names)

    model = CropClassifier.load_from_checkpoint(str(ckpt))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    images = torch.from_numpy(np.load(data_dir / "images.npy")).float()
    labels = torch.from_numpy(np.load(data_dir / "labels.npy")).long()

    # Scale reflectance (divide by 10000) — same as val transforms
    images = images / 10_000.0

    ds = TensorDataset(images, labels)
    loader = DataLoader(ds, batch_size=batch_size)

    confmat = MulticlassConfusionMatrix(num_classes=num_classes)
    preds_all, labels_all = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            logits = model(imgs.to(device))
            preds = logits.argmax(dim=1).cpu()
            preds_all.append(preds)
            labels_all.append(lbls)

    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    confmat.update(preds_all, labels_all)
    cm = confmat.compute().numpy()

    acc = (preds_all == labels_all).float().mean().item()
    click.echo(f"Accuracy: {acc:.4f}")
    click.echo(f"\nConfusion matrix (rows=true, cols=pred):")
    click.echo(f"{'':12s}" + "".join(f"{n:>10s}" for n in names))
    for i, row in enumerate(cm):
        click.echo(f"{names[i]:<12s}" + "".join(f"{int(v):>10d}" for v in row))

    # Save PNG
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6, 5))
        cm_pct = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_pct, annot=True, fmt=".2%", xticklabels=names, yticklabels=names,
                    cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix  (acc={acc:.3f})")

        if out is None:
            out = ckpt.parent.parent / "confusion.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        click.echo(f"\nSaved confusion matrix → {out}")
        plt.close(fig)
    except ImportError:
        click.echo("matplotlib/seaborn not installed — skipping PNG")


if __name__ == "__main__":
    main()
