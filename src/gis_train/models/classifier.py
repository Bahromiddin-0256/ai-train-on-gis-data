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
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from gis_train.models.losses import FocalLoss
from gis_train.utils.logging import get_logger

_log = get_logger(__name__)


# Sentinel-2 ALL-band order used by torchgeo SENTINEL2_ALL_MOCO weights (13 bands).
_S2_ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B10", "B11", "B12"]
# 9-band working set: 10m + selected 20m bands (excludes B08A, B09, B10, B01).
_S2_WORKING_9 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
_S2_WORKING_BAND_INDICES = [_S2_ALL_BANDS.index(b) for b in _S2_WORKING_9]
# = [1, 2, 3, 4, 5, 6, 7, 11, 12]
_S2_B08_IDX_IN_13 = _S2_ALL_BANDS.index("B08")  # = 7, used for NDVI channel init


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
                        pretrained_w = old_conv.weight  # (out_ch, 13, kH, kW)

                        if in_channels == len(_S2_WORKING_9):
                            # Single-temporal 9-band: direct index mapping
                            new_conv.weight.copy_(pretrained_w[:, _S2_WORKING_BAND_INDICES, :, :])
                        elif in_channels == len(_S2_WORKING_9) + 1:
                            # Single-temporal 9-band + NDVI (10 channels)
                            new_conv.weight[:, :len(_S2_WORKING_9)] = pretrained_w[:, _S2_WORKING_BAND_INDICES, :, :]
                            new_conv.weight[:, len(_S2_WORKING_9)] = pretrained_w[:, _S2_B08_IDX_IN_13, :, :]
                        else:
                            # Multi-temporal: in_channels = n_windows * n_ch_per_window
                            n_ch_detected = None
                            n_windows_detected = None
                            for n_win in [3, 2, 4, 6, 12]:
                                if in_channels % n_win == 0:
                                    n_ch_detected = in_channels // n_win
                                    n_windows_detected = n_win
                                    break

                            if n_ch_detected is None:
                                _log.warning(
                                    "in_channels=%d: no pretrained mapping found, using random init",
                                    in_channels,
                                )
                                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                            else:
                                scale = 1.0 / (n_windows_detected ** 0.5)
                                for w in range(n_windows_detected):
                                    start = w * n_ch_detected
                                    if n_ch_detected == len(_S2_WORKING_9):
                                        new_conv.weight[:, start:start + n_ch_detected] = (
                                            pretrained_w[:, _S2_WORKING_BAND_INDICES] * scale
                                        )
                                    elif n_ch_detected == len(_S2_WORKING_9) + 1:
                                        new_conv.weight[:, start:start + len(_S2_WORKING_9)] = (
                                            pretrained_w[:, _S2_WORKING_BAND_INDICES] * scale
                                        )
                                        new_conv.weight[:, start + len(_S2_WORKING_9)] = (
                                            pretrained_w[:, _S2_B08_IDX_IN_13] * scale
                                        )
                                    else:
                                        nn.init.kaiming_normal_(
                                            new_conv.weight[:, start:start + n_ch_detected],
                                            mode="fan_out", nonlinearity="relu",
                                        )
                                _log.info(
                                    "patched first conv: 13 → %d channels "
                                    "(%d windows × %d ch/window, scale=%.3f)",
                                    in_channels, n_windows_detected, n_ch_detected, scale,
                                )
                    backbone.conv1 = new_conv
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
    elif name == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)

        # First conv is model.features[0][0]: Conv2d(3, 96, kernel_size=4, stride=4)
        old_conv = model.features[0][0]
        feature_dim = model.classifier[2].in_features  # 768

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            if pretrained and old_conv.weight.shape[1] == 3:
                # ImageNet pretrained: 3 channels. Tile to in_channels.
                # Each tile of 3 channels gets pretrained weights / (in_channels/3).
                reps = in_channels // 3
                remainder = in_channels % 3
                tiled = old_conv.weight.repeat(1, reps + (1 if remainder else 0), 1, 1)
                tiled = tiled[:, :in_channels, :, :]
                scale = 3.0 / in_channels
                new_conv.weight.copy_(tiled * scale)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        model.features[0][0] = new_conv
        model.classifier[2] = nn.Identity()
        return model, feature_dim
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


def _tta_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Average softmax probs over 8 D4 augmentations (flips × 4 rotations)."""
    probs_sum = torch.zeros(x.shape[0], model.head.out_features, device=x.device, dtype=x.dtype)
    n = 0
    for k in range(4):
        xr = torch.rot90(x, k=k, dims=[-2, -1])
        for flip in (False, True):
            xi = torch.flip(xr, dims=[-1]) if flip else xr
            probs_sum = probs_sum + F.softmax(model(xi), dim=1)
            n += 1
    return torch.log(probs_sum / n + 1e-12)


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
        loss: str = "ce",
        focal_gamma: float = 2.0,
        backbone_lr_mult: float = 1.0,
        warmup_epochs: int = 0,
        test_time_augmentation: bool = False,
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
        if loss == "ce":
            self.loss_fn: nn.Module = nn.CrossEntropyLoss(
                weight=weight, label_smoothing=label_smoothing
            )
        elif loss == "focal":
            self.loss_fn = FocalLoss(
                gamma=focal_gamma, weight=weight, label_smoothing=label_smoothing
            )
        else:
            raise ValueError(f"unknown loss: {loss!r}")

        self._optimizer_cfg = optimizer
        self._scheduler_cfg = scheduler
        self._num_classes = num_classes
        self._backbone_lr_mult = backbone_lr_mult
        self._warmup_epochs = warmup_epochs
        self._tta = test_time_augmentation

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

    def _eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._tta:
            return _tta_forward(self, x)
        return self(x)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, labels = batch
        logits = self(images)
        # Mixup/CutMix collate emits soft labels (B, C); cross-entropy handles both.
        if labels.dim() == 2:
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(labels * log_probs).sum(dim=1).mean()
            hard_labels = labels.argmax(dim=1)
        else:
            loss = self.loss_fn(logits, labels)
            hard_labels = labels
        return loss, logits, hard_labels

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, logits, labels = self._shared_step(batch)
        self.train_acc(logits, labels)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, labels = batch
        logits = self._eval_forward(images)
        loss = self.loss_fn(logits, labels) if labels.dim() == 1 else F.cross_entropy(logits, labels)
        self.val_acc(logits, labels if labels.dim() == 1 else labels.argmax(dim=1))
        self.val_f1(logits, labels if labels.dim() == 1 else labels.argmax(dim=1))
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val/f1", self.val_f1, prog_bar=False, on_epoch=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, labels = batch
        logits = self._eval_forward(images)
        loss = self.loss_fn(logits, labels) if labels.dim() == 1 else F.cross_entropy(logits, labels)
        hard = labels if labels.dim() == 1 else labels.argmax(dim=1)
        self.test_acc(logits, hard)
        self.test_f1(logits, hard)
        self.test_confmat(logits, hard)
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", self.test_acc, on_epoch=True)
        self.log("test/f1", self.test_f1, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimizers / schedulers
    # ------------------------------------------------------------------

    def _param_groups(self, base_lr: float) -> list[dict[str, Any]]:
        if self._backbone_lr_mult == 1.0:
            return [{"params": list(self.parameters())}]
        return [
            {"params": list(self.backbone.parameters()), "lr": base_lr * self._backbone_lr_mult},
            {"params": list(self.head.parameters()), "lr": base_lr},
        ]

    def configure_optimizers(self) -> Optimizer | dict[str, Any]:
        opt_cfg = self._optimizer_cfg or {
            "_target_": "torch.optim.AdamW",
            "lr": 1.0e-4,
            "weight_decay": 1.0e-4,
        }
        base_lr = float(opt_cfg.get("lr", 1.0e-4)) if hasattr(opt_cfg, "get") else 1.0e-4
        
        # Instantiate optimizer without Hydra wrapping the params list into ListConfig
        from hydra.utils import get_class
        if isinstance(opt_cfg, dict):
            opt_class = get_class(opt_cfg["_target_"])
            opt_kwargs = {k: v for k, v in opt_cfg.items() if k != "_target_"}
        else:
            opt_class = get_class(opt_cfg._target_)
            opt_kwargs = {k: v for k, v in opt_cfg.items() if k != "_target_"}
            
        optimizer: Optimizer = opt_class(self._param_groups(base_lr), **opt_kwargs)

        if self._scheduler_cfg is None and self._warmup_epochs <= 0:
            return optimizer

        if self._warmup_epochs > 0:
            warmup = self._warmup_epochs
            main_scheduler: LRScheduler | None = None
            if self._scheduler_cfg is not None:
                sched_class = get_class(self._scheduler_cfg._target_) if not isinstance(self._scheduler_cfg, dict) else get_class(self._scheduler_cfg["_target_"])
                sched_kwargs = {k: v for k, v in self._scheduler_cfg.items() if k != "_target_"}
                main_scheduler = sched_class(optimizer, **sched_kwargs)

            def lr_lambda(epoch: int) -> float:
                if epoch < warmup:
                    return (epoch + 1) / warmup
                return 1.0

            warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            if main_scheduler is None:
                return {"optimizer": optimizer, "lr_scheduler": warmup_sched}
            sched = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_scheduler],
                milestones=[warmup],
            )
            return {"optimizer": optimizer, "lr_scheduler": sched}

        sched_class = get_class(self._scheduler_cfg._target_) if not isinstance(self._scheduler_cfg, dict) else get_class(self._scheduler_cfg["_target_"])
        sched_kwargs = {k: v for k, v in self._scheduler_cfg.items() if k != "_target_"}
        scheduler: LRScheduler = sched_class(optimizer, **sched_kwargs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
