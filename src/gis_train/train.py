"""Hydra entrypoint: ``python -m gis_train.train``.

Loads ``configs/config.yaml`` by default, instantiates the DataModule, model,
and Trainer, and kicks off ``fit``/``test``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gis_train.utils.logging import get_logger

_log = get_logger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _set_seed(seed: int) -> None:
    pl.seed_everything(seed, workers=True)


def train(cfg: DictConfig) -> dict[str, Any]:
    """Run training end-to-end and return the Trainer's test metrics.

    Kept importable (not just a ``@hydra.main`` entrypoint) so tests can drive
    it directly with a synthetic config.
    """
    _log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    _set_seed(int(cfg.get("seed", 42)))

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    trainer: pl.Trainer = instantiate(cfg.trainer)

    trainer.fit(model=model, datamodule=datamodule)

    # Only run .test if the DataModule has a test split (e.g. fast_dev_run skips it).
    test_metrics: list[dict[str, float]] = []
    if not getattr(trainer, "fast_dev_run", False):
        test_metrics = trainer.test(model=model, datamodule=datamodule, verbose=False)

    return {"test_metrics": test_metrics}


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


def cli() -> None:
    """Console-script entrypoint registered in ``pyproject.toml``."""
    main()


if __name__ == "__main__":  # pragma: no cover
    main()
