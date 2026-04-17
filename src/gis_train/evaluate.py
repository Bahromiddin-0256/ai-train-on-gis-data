"""Evaluate a trained checkpoint on the test split.

Usage::

    python -m gis_train.evaluate ckpt=/path/to/epoch=9-step=100.ckpt

The config is layered exactly like ``train.py`` (same defaults), with one extra
required field ``ckpt`` pointing at a Lightning checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gis_train.models.classifier import CropClassifier
from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def evaluate(cfg: DictConfig) -> dict[str, Any]:
    """Load a checkpoint and run ``Trainer.test`` on the configured test split."""
    _log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    ckpt_path = cfg.get("ckpt")
    if not ckpt_path:
        raise ValueError("pass `ckpt=/path/to/checkpoint.ckpt` on the CLI")

    datamodule = instantiate(cfg.data)
    # Rebuild the same architecture, then overwrite with checkpoint weights.
    model: CropClassifier = instantiate(cfg.model)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])

    trainer: pl.Trainer = instantiate(cfg.trainer)
    metrics = trainer.test(model=model, datamodule=datamodule, verbose=False)

    # torchmetrics confusion matrix is available on the model after test().
    confmat = model.test_confmat.compute().int().tolist()

    result = {"metrics": metrics, "confusion_matrix": confmat}
    out_path = Path(cfg.get("output_dir", ".")) / "evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    _log.info("wrote evaluation results to %s", out_path)
    return result


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


def cli() -> None:
    """Console-script entrypoint registered in ``pyproject.toml``."""
    main()


if __name__ == "__main__":  # pragma: no cover
    main()
