"""Temporal crop classifier.

Per-date 2D CNN features + a temporal encoder (TempCNN or GRU) that respects
the time dimension. Input tensor shape ``(B, T*C, H, W)`` is reshaped internally
to ``(B, T, C, H, W)``.

Empirically beats channel-stacked CNNs on multi-temporal S2 crop tasks because
the temporal encoder can learn phenology rather than a single cross-timestamp
linear combination in the first conv.
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


def _make_spatial_encoder(in_channels_per_step: int, feature_dim: int) -> nn.Module:
    """Small per-timestep 2D CNN → global-avg-pool → feature_dim vector."""
    return nn.Sequential(
        nn.Conv2d(in_channels_per_step, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
        nn.BatchNorm2d(128),
        nn.GELU(),
        nn.Conv2d(128, feature_dim, kernel_size=3, padding=1, stride=2, bias=False),
        nn.BatchNorm2d(feature_dim),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
    )


class _TempCNN(nn.Module):
    """1D CNN along the temporal axis (Pelletier et al., 2019)."""

    def __init__(self, feature_dim: int, hidden: int = 256, kernel: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
        )
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T) for conv1d
        return self.net(x.transpose(1, 2))


class _GRUEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 256, layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # (B, T, 2H)
        return out.mean(dim=1)


class TemporalCropClassifier(pl.LightningModule):
    """Per-timestep 2D CNN + temporal encoder (TempCNN or GRU)."""

    def __init__(
        self,
        n_windows: int,
        channels_per_window: int,
        num_classes: int,
        spatial_feature_dim: int = 256,
        temporal_encoder: str = "tempcnn",
        temporal_hidden: int = 256,
        class_weights: list[float] | None = None,
        label_smoothing: float = 0.0,
        loss: str = "ce",
        focal_gamma: float = 2.0,
        test_time_augmentation: bool = False,
        optimizer: DictConfig | dict[str, Any] | None = None,
        scheduler: DictConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.n_windows = n_windows
        self.channels_per_window = channels_per_window
        self.in_channels = n_windows * channels_per_window

        self.spatial = _make_spatial_encoder(channels_per_window, spatial_feature_dim)
        if temporal_encoder == "tempcnn":
            self.temporal: nn.Module = _TempCNN(spatial_feature_dim, hidden=temporal_hidden)
        elif temporal_encoder == "gru":
            self.temporal = _GRUEncoder(spatial_feature_dim, hidden=temporal_hidden)
        else:
            raise ValueError(f"unknown temporal_encoder: {temporal_encoder!r}")

        self.head = nn.Linear(self.temporal.out_dim, num_classes)

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
        self._tta = test_time_augmentation

        metric_kwargs = {"num_classes": num_classes, "average": "macro"}
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=num_classes)

    def _reshape_time(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels "
                f"({self.n_windows} windows × {self.channels_per_window}), got {c}"
            )
        return x.view(b, self.n_windows, self.channels_per_window, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = self._reshape_time(x)  # (B, T, C, H, W)
        b, t, c, h, w = xt.shape
        feats = self.spatial(xt.reshape(b * t, c, h, w))  # (B*T, F)
        feats = feats.view(b, t, -1)
        pooled = self.temporal(feats)  # (B, out_dim)
        return self.head(pooled)

    def _eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._tta:
            return self(x)
        probs_sum = torch.zeros(x.shape[0], self.head.out_features, device=x.device, dtype=x.dtype)
        n = 0
        for k in range(4):
            xr = torch.rot90(x, k=k, dims=[-2, -1])
            for flip in (False, True):
                xi = torch.flip(xr, dims=[-1]) if flip else xr
                probs_sum = probs_sum + F.softmax(self(xi), dim=1)
                n += 1
        return torch.log(probs_sum / n + 1e-12)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if labels.dim() == 2:
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(labels * log_probs).sum(dim=1).mean()
            hard = labels.argmax(dim=1)
        else:
            loss = self.loss_fn(logits, labels)
            hard = labels
        return loss, hard

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss, hard = self._loss(logits, labels)
        self.train_acc(logits, hard)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self._eval_forward(images)
        loss, hard = self._loss(logits, labels)
        self.val_acc(logits, hard)
        self.val_f1(logits, hard)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val/f1", self.val_f1, prog_bar=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self._eval_forward(images)
        loss, hard = self._loss(logits, labels)
        self.test_acc(logits, hard)
        self.test_f1(logits, hard)
        self.test_confmat(logits, hard)
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/acc", self.test_acc, on_epoch=True)
        self.log("test/f1", self.test_f1, on_epoch=True)

    def configure_optimizers(self):
        opt_cfg = self._optimizer_cfg or {
            "_target_": "torch.optim.AdamW",
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
        }
        optimizer: Optimizer = instantiate(opt_cfg, params=self.parameters())
        if self._scheduler_cfg is None:
            return optimizer
        scheduler: LRScheduler = instantiate(self._scheduler_cfg, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


__all__ = ["TemporalCropClassifier"]
