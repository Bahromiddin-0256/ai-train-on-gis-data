"""End-to-end smoke test: drive ``gis_train.train.train`` with the real Hydra config.

This exercises the full pipeline (Hydra compose → DataModule → LightningModule →
Trainer.fit + Trainer.test) on the synthetic source, so no network access or
public datasets are required.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from gis_train.train import train


def test_smoke_train_one_epoch_synthetic(tiny_config_overrides: list[str], tmp_path: Path) -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    overrides = [*tiny_config_overrides, f"output_dir={tmp_path}"]

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)
        result = train(cfg)

    assert isinstance(result, dict)
    assert "test_metrics" in result
    # With synthetic_n=32 + val/test splits, test_dataloader has >=1 batch.
    assert len(result["test_metrics"]) >= 1
    metrics = result["test_metrics"][0]
    assert "test/acc" in metrics
    assert 0.0 <= metrics["test/acc"] <= 1.0
