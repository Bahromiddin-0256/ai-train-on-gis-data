"""
scripts/export_model.py
-----------------------
Load the trained CropClassifier Lightning checkpoint, extract the
backbone + head, wrap them in a plain nn.Module, and export as
TorchScript to the sentinelhub ml directory.

Usage (from repo root):
    .venv/bin/python scripts/export_model.py
"""
from __future__ import annotations

import os
import sys

# Make the src package importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn

CKPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "lightning_logs",
    "version_13",
    "checkpoints",
    "best-epoch=18-val_f1=0.0000.ckpt",
)
OUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "sentinelhub",
    "api",
    "ml",
    "crop_classifier.pt",
)


class _CropNet(nn.Module):
    """Thin wrapper combining backbone + head for TorchScript export."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def main() -> None:
    ckpt_path = os.path.abspath(CKPT_PATH)
    out_path = os.path.abspath(OUT_PATH)

    print(f"Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Import the Lightning module — only needed here, NOT in the inference code.
    from gis_train.models.classifier import CropClassifier  # noqa: PLC0415

    lit_model = CropClassifier.load_from_checkpoint(
        ckpt_path, map_location="cpu", weights_only=False
    )
    lit_model.eval()

    net = _CropNet(backbone=lit_model.backbone, head=lit_model.head)
    net.eval()

    # Smoke-test before scripting
    with torch.no_grad():
        dummy = torch.zeros(1, 4, 64, 64)
        out = net(dummy)
        assert out.shape == (1, 3), f"Unexpected output shape: {out.shape}"
    print("Smoke test passed — output shape: (1, 3)")

    scripted = torch.jit.script(net)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scripted.save(out_path)
    print(f"TorchScript model saved to: {out_path}")


if __name__ == "__main__":
    main()
