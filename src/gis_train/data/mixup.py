"""Mixup / CutMix batch-level augmentation.

Used as a ``collate_fn`` replacement so mixing happens post-batching. Produces
soft labels of shape ``(B, num_classes)``; the LightningModule detects 2-D
targets and uses soft cross-entropy automatically.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()


def _rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator) -> tuple[int, int, int, int]:
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


class MixupCutmixCollate:
    """Stack samples then apply Mixup or CutMix with probability ``p``.

    - ``mixup_alpha`` / ``cutmix_alpha``: Beta distribution shape. Set to 0 to disable.
    - ``switch_prob``: among active modes, chance of choosing CutMix over Mixup.
    - ``p``: per-batch probability of applying any mixing.
    """

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        switch_prob: float = 0.5,
        p: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self._num_classes = num_classes
        self._mixup_alpha = max(mixup_alpha, 0.0)
        self._cutmix_alpha = max(cutmix_alpha, 0.0)
        self._switch_prob = switch_prob
        self._p = p
        self._rng = np.random.default_rng(seed)

    def __call__(self, batch: Sequence[tuple[torch.Tensor, torch.Tensor]]):
        images = torch.stack([b[0] for b in batch], dim=0)
        labels = torch.stack([torch.as_tensor(b[1], dtype=torch.long) for b in batch], dim=0)
        soft = _one_hot(labels, self._num_classes)

        if self._rng.random() >= self._p or images.shape[0] < 2:
            return images, soft
        if self._mixup_alpha == 0.0 and self._cutmix_alpha == 0.0:
            return images, soft

        use_cutmix = (
            self._cutmix_alpha > 0.0
            and (self._mixup_alpha == 0.0 or self._rng.random() < self._switch_prob)
        )
        alpha = self._cutmix_alpha if use_cutmix else self._mixup_alpha
        lam = float(self._rng.beta(alpha, alpha))
        perm = torch.randperm(images.shape[0])

        if use_cutmix:
            _, _, h, w = images.shape
            y1, y2, x1, x2 = _rand_bbox(h, w, lam, self._rng)
            images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
            # Adjust lam to reflect actual cut area.
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(h * w))
        else:
            images = lam * images + (1.0 - lam) * images[perm]

        soft = lam * soft + (1.0 - lam) * soft[perm]
        return images, soft


__all__ = ["MixupCutmixCollate"]
