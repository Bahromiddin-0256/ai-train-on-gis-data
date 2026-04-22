"""
scripts/export_model.py
-----------------------
Load a trained Lightning checkpoint (CropClassifier or TemporalCropClassifier),
auto-detect the model type from its hyper_parameters, wrap the relevant
sub-modules in a plain nn.Module, and export as TorchScript to the
sentinelhub ml directory.

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

CKPT_PATH = os.environ.get(
    "CKPT_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "lightning_logs",
        "version_21",
        "checkpoints",
        "best-epoch=20-val_f1=0.0000.ckpt",
    ),
)
OUT_PATH = os.environ.get(
    "OUT_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "sentinelhub",
        "api",
        "ml",
        "crop_classifier.pt",
    ),
)


# ---------------------------------------------------------------------------
# TorchScript-compatible wrappers (no Lightning dependency at inference time)
# ---------------------------------------------------------------------------

class _CropNet(nn.Module):
    """Thin wrapper combining backbone + head for TorchScript export (CropClassifier)."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class _TemporalCropNet(nn.Module):
    """Thin wrapper for TemporalCropClassifier TorchScript export.

    Replicates the reshape-and-forward logic from TemporalCropClassifier.forward
    without any Lightning/PL dependency so TorchScript can compile it.
    """

    def __init__(
        self,
        spatial: nn.Module,
        temporal: nn.Module,
        head: nn.Module,
        n_windows: int,
        channels_per_window: int,
    ) -> None:
        super().__init__()
        self.spatial = spatial
        self.temporal = temporal
        self.head = head
        self.n_windows = n_windows
        self.channels_per_window = channels_per_window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        xt = x.view(b, self.n_windows, self.channels_per_window, h, w)
        # Apply spatial encoder to each timestep independently.
        feats = self.spatial(xt.reshape(b * self.n_windows, self.channels_per_window, h, w))
        feats = feats.view(b, self.n_windows, -1)
        pooled = self.temporal(feats)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def _is_temporal(hyper_params: dict) -> bool:
    """True if the checkpoint was saved by TemporalCropClassifier."""
    return "n_windows" in hyper_params and "channels_per_window" in hyper_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ckpt_path = os.path.abspath(CKPT_PATH)
    out_path = os.path.abspath(OUT_PATH)

    print(f"Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Peek at hyperparameters to determine the correct Lightning module class.
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams: dict = raw_ckpt.get("hyper_parameters", {})
    print(f"Detected hyper_parameters: {hparams}")

    if _is_temporal(hparams):
        print("Model type: TemporalCropClassifier")
        from gis_train.models.temporal import TemporalCropClassifier  # noqa: PLC0415

        lit_model = TemporalCropClassifier.load_from_checkpoint(
            ckpt_path, map_location="cpu", weights_only=False
        )
        lit_model.eval()

        n_windows: int = int(hparams["n_windows"])
        ch_per_window: int = int(hparams["channels_per_window"])
        in_channels = n_windows * ch_per_window

        net: nn.Module = _TemporalCropNet(
            spatial=lit_model.spatial,
            temporal=lit_model.temporal,
            head=lit_model.head,
            n_windows=n_windows,
            channels_per_window=ch_per_window,
        )
    else:
        print("Model type: CropClassifier")
        from gis_train.models.classifier import CropClassifier  # noqa: PLC0415

        lit_model = CropClassifier.load_from_checkpoint(
            ckpt_path, map_location="cpu", weights_only=False
        )
        lit_model.eval()

        in_channels = int(hparams.get("in_channels", 30))
        net = _CropNet(backbone=lit_model.backbone, head=lit_model.head)

    net.eval()
    num_classes: int = int(hparams.get("num_classes", 3))

    # Smoke-test before scripting
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, 64, 64)
        out = net(dummy)
        expected = (1, num_classes)
        assert out.shape == expected, f"Unexpected output shape: {out.shape}, expected {expected}"
    print(f"Smoke test passed — output shape: {tuple(out.shape)}")

    scripted = torch.jit.script(net)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scripted.save(out_path)
    print(f"TorchScript model saved to: {out_path}")


if __name__ == "__main__":
    main()
