"""Adapter to load Prithvi weights and use in our training pipeline.

This module provides a way to use the Prithvi-EO-1.0-100M foundation model
as a backbone for crop classification, either:
1. As a feature extractor with a custom head
2. As a fine-tuning base with pretrained weights
"""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from gis_train.models.losses import FocalLoss


class PrithviFeatureExtractor(nn.Module):
    """Wrapper to extract features from Prithvi model.

    Prithvi expects: (B, 18, 224, 224) - 3 timesteps × 6 HLS bands
    Bands per timestep: Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2
    """

    # HLS band order (maps to Sentinel-2)
    HLS_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

    def __init__(
        self,
        model_name: str = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification",
        feature_dim: int = 768,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.feature_dim = feature_dim

        # Load the foundation model
        self.backbone = self._load_backbone()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _load_backbone(self) -> nn.Module:
        """Load the Prithvi backbone from HuggingFace."""
        try:
            from terratorch import load_model_from_checkpoint

            model = load_model_from_checkpoint(self.model_name, pretrained=True)
            return model
        except ImportError:
            pass

        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Prithvi model. "
                f"Please install terratorch: pip install terratorch"
            ) from exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input.

        Args:
            x: Input tensor (B, 18, 224, 224)

        Returns:
            Feature tensor
        """
        # Handle different model output formats
        output = self.backbone(x)

        if isinstance(output, dict):
            # Some transformers models return dicts
            return output.get("last_hidden_state", output.get("pooler_output", output))

        return output


class PrithviCropClassifier(pl.LightningModule):
    """LightningModule using Prithvi as a frozen feature extractor.

    This is useful when you have limited data and want to leverage
    the pretrained representations from the Prithvi foundation model.
    """

    def __init__(
        self,
        num_classes: int = 3,
        model_name: str = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification",
        feature_dim: int = 768,
        freeze_backbone: bool = True,
        class_weights: list[float] | None = None,
        label_smoothing: float = 0.0,
        loss: str = "ce",
        focal_gamma: float = 2.0,
        learning_rate: float = 1e-3,
        test_time_augmentation: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Feature extractor
        self.backbone = PrithviFeatureExtractor(
            model_name=model_name,
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_classes),
        )

        # Loss function
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

        self._learning_rate = learning_rate
        self._tta = test_time_augmentation
        self._num_classes = num_classes

        # Metrics
        metric_kwargs = {"num_classes": num_classes, "average": "macro"}
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 18, 224, 224)

        Returns:
            Logits (B, num_classes)
        """
        # Resize to Prithvi's expected input size
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

        features = self.backbone(x)
        return self.head(features)

    def _eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional test-time augmentation."""
        if not self._tta:
            return self(x)

        probs_sum = torch.zeros(x.shape[0], self._num_classes, device=x.device, dtype=x.dtype)
        n = 0
        for k in range(4):
            xr = torch.rot90(x, k=k, dims=[-2, -1])
            for flip in (False, True):
                xi = torch.flip(xr, dims=[-1]) if flip else xr
                probs_sum = probs_sum + F.softmax(self(xi), dim=1)
                n += 1
        return torch.log(probs_sum / n + 1e-12)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared logic for train/val/test steps."""
        images, labels = batch
        logits = self(images)

        if labels.dim() == 2:
            # Soft labels (MixUp/CutMix)
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
        hard = labels if labels.dim() == 1 else labels.argmax(dim=1)
        self.val_acc(logits, hard)
        self.val_f1(logits, hard)
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

    def configure_optimizers(self) -> Any:
        """Configure optimizer - only optimize unfrozen parameters."""
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self._learning_rate, weight_decay=1e-4)
        return optimizer


class Sentinel2ToHLSMapper:
    """Map Sentinel-2 bands to HLS format expected by Prithvi.

    Sentinel-2 bands: B02, B03, B04, B05, B06, B07, B08, B11, B12 (9 bands)
    HLS bands: B02, B03, B04, B08, B11, B12 (6 bands)

    The mapper handles:
    1. Selecting the 6 HLS-compatible bands from Sentinel-2
    2. Handling temporal stacking (3 timesteps)
    3. Resizing to 224×224
    """

    # Sentinel-2 to HLS band indices (in 9-band working set)
    S2_TO_HLS_MAPPING = {
        "B02": 0,  # Blue
        "B03": 1,  # Green
        "B04": 2,  # Red
        "B08": 6,  # NIR (in 9-band set, index 6 is B08)
        "B11": 7,  # SWIR 1
        "B12": 8,  # SWIR 2
    }

    # HLS band order
    HLS_BAND_ORDER = ["B02", "B03", "B04", "B08", "B11", "B12"]

    @classmethod
    def map_sentinel2_to_hls(
        cls,
        sentinel2_data: torch.Tensor,
        num_timesteps: int = 3,
    ) -> torch.Tensor:
        """Map Sentinel-2 9-band data to HLS 6-band format.

        Args:
            sentinel2_data: Input tensor (B, C, H, W) where C is either:
                - 9 bands (single timestep)
                - 27 bands (3 timesteps × 9 bands)
                - 30 bands (3 timesteps × 10 bands with NDVI)
            num_timesteps: Expected number of timesteps (default 3)

        Returns:
            Tensor in HLS format (B, 18, H, W) for 3 timesteps × 6 bands
        """
        batch_size, channels, height, width = sentinel2_data.shape

        # Determine channels per timestep
        if channels == 9:
            # Single timestep - repeat for num_timesteps
            channels_per_t = 9
            has_ndvi = False
        elif channels == 27:
            # 3 timesteps × 9 bands
            channels_per_t = 9
            has_ndvi = False
        elif channels == 30:
            # 3 timesteps × 10 bands (9 + NDVI)
            channels_per_t = 10
            has_ndvi = True
        else:
            raise ValueError(f"Unexpected channel count: {channels}")

        # Extract bands for each timestep
        hls_channels = []
        for t in range(num_timesteps):
            start_idx = t * channels_per_t
            timestep_data = sentinel2_data[:, start_idx:start_idx + channels_per_t, :, :]

            # Map to HLS bands
            hls_timestep = []
            for band in cls.HLS_BAND_ORDER:
                s2_idx = cls.S2_TO_HLS_MAPPING[band]
                if has_ndvi and s2_idx >= 9:
                    # Adjust for NDVI channel - skip it
                    s2_idx = s2_idx - 1 if s2_idx > 9 else s2_idx
                if s2_idx < timestep_data.shape[1]:
                    hls_timestep.append(timestep_data[:, s2_idx:s2_idx + 1, :, :])
                else:
                    # Fill missing bands with zeros
                    hls_timestep.append(
                        torch.zeros(batch_size, 1, height, width, device=sentinel2_data.device)
                    )

            hls_channels.append(torch.cat(hls_timestep, dim=1))

        return torch.cat(hls_channels, dim=1)  # (B, 18, H, W)

    @classmethod
    def prepare_for_prithvi(
        cls,
        sentinel2_data: torch.Tensor,
        target_size: tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """Full preprocessing pipeline for Prithvi model.

        Args:
            sentinel2_data: Sentinel-2 input tensor
            target_size: Target spatial size (H, W)

        Returns:
            Preprocessed tensor ready for Prithvi
        """
        # Map bands
        hls_data = cls.map_sentinel2_to_hls(sentinel2_data)

        # Resize
        if hls_data.shape[-2:] != target_size:
            hls_data = F.interpolate(
                hls_data, size=target_size, mode="bilinear", align_corners=False
            )

        # Normalize to reflectance [0, 1] (assumes input is in DN or reflectance × 10000)
        # If input is already reflectance [0, 1], this may need adjustment
        hls_data = hls_data / 10000.0
        hls_data = torch.clamp(hls_data, 0, 1)

        return hls_data
