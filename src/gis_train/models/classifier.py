"""LightningModule for crop classification.

The backbone comes from ``torchgeo.models.resnet50`` with the ``SENTINEL2_ALL_MOCO``
self-supervised weights when available. We fall back to a vanilla torchvision
ResNet when torchgeo isn't installed (e.g. in the minimal test environment)
so the unit tests can exercise the training loop offline.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)


# Sentinel-2 ALL-band order used by torchgeo SENTINEL2_ALL_MOCO weights.
# Indices of our 4 working bands (B02, B03, B04, B08) within this ordering.
_S2_ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B10", "B11", "B12"]
_S2_WORKING_BAND_INDICES = [_S2_ALL_BANDS.index(b) for b in ["B02", "B03", "B04", "B08"]]


def _build_backbone(name: str, in_channels: int, pretrained: bool) -> tuple[nn.Module, int]:
    """Return ``(backbone_without_classifier, feature_dim)``.

    Tries torchgeo first (for pretrained Sentinel-2 weights), then falls back
    to torchvision so the test suite works without the torchgeo data extras.

    When ``pretrained=True`` and ``in_channels != 13``, loads the full 13-channel
    SENTINEL2_ALL_MOCO backbone and replaces the first conv with a new layer whose
    weights are initialized from the corresponding pretrained channels.  All 13
    bands are: B01…B12 + B08A; our 4 bands (B02, B03, B04, B08) map to indices
    [1, 2, 3, 7] in that ordering.
    """
    if name == "resnet50":
        try:
            from torchgeo.models import ResNet50_Weights, resnet50  # type: ignore[import-not-found]

            if pretrained:
                # Load full 13-channel pretrained backbone, then slice first conv.
                backbone = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO, in_chans=13)
                feature_dim = backbone.num_features
                backbone.reset_classifier(num_classes=0)

                if in_channels != 13:
                    old_conv = backbone.conv1
                    new_conv = nn.Conv2d(
                        in_channels,
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None,
                    )
                    with torch.no_grad():
                        indices = _S2_WORKING_BAND_INDICES[:in_channels]
                        new_conv.weight.copy_(old_conv.weight[:, indices, :, :])
                    backbone.conv1 = new_conv
                    _log.info(
                        "patched first conv: 13 → %d channels (band indices %s)",
                        in_channels, indices,
                    )
            else:
                backbone = resnet50(weights=None, in_chans=in_channels)
                feature_dim = backbone.num_features
                backbone.reset_classifier(num_classes=0)

            return backbone, feature_dim
        except (ImportError, AttributeError) as exc:
            _log.warning("torchgeo resnet50 unavailable (%s); falling back to torchvision", exc)

    if name in {"resnet50", "resnet50_random"}:
        from torchvision.models import resnet50 as tv_resnet50

        model = tv_resnet50(weights=None)
    elif name == "resnet18":
        from torchvision.models import resnet18 as tv_resnet18

        model = tv_resnet18(weights=None)
    else:
        raise ValueError(f"unknown backbone: {name!r}")

    # Patch the first conv to accept `in_channels` instead of 3.
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feature_dim


class CropClassifier(pl.LightningModule):
    """LightningModule wrapping a ResNet backbone + linear classifier head."""

    def __init__(
        self,
        backbone: str = "resnet50",
        in_channels: int = 4,
        num_classes: int = 2,
        pretrained: bool = True,
        class_weights: list[float] | None = None,
        label_smoothing: float = 0.0,
        optimizer: DictConfig | dict[str, Any] | None = None,
        scheduler: DictConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone, feature_dim = _build_backbone(
            name=backbone, in_channels=in_channels, pretrained=pretrained
        )
        self.head = nn.Linear(feature_dim, num_classes)

        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

        self._optimizer_cfg = optimizer
        self._scheduler_cfg = scheduler
        self._num_classes = num_classes

        metric_kwargs = {"num_classes": num_classes, "average": "macro"}
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=num_classes)

    # ------------------------------------------------------------------
    # Forward / step
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        return loss, logits, labels

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, logits, labels = self._shared_step(batch)
        self.train_acc(logits, labels)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, logits, labels = self._shared_step(batch)
        self.val_acc(logits, labels)
        self.val_f1(logits, labels)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val/f1", self.val_f1, prog_bar=False, on_epoch=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, logits, labels = self._shared_step(batch)
        self.test_acc(logits, labels)
        self.test_f1(logits, labels)
        self.test_confmat(logits, labels)
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", self.test_acc, on_epoch=True)
        self.log("test/f1", self.test_f1, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimizers / schedulers
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Optimizer | dict[str, Any]:
        opt_cfg = self._optimizer_cfg or {
            "_target_": "torch.optim.AdamW",
            "lr": 1.0e-4,
            "weight_decay": 1.0e-4,
        }
        optimizer: Optimizer = instantiate(opt_cfg, params=self.parameters())

        if self._scheduler_cfg is None:
            return optimizer

        scheduler: LRScheduler = instantiate(self._scheduler_cfg, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
