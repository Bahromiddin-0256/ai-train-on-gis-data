# ai-train-on-gis-data

A PyTorch + [torchgeo](https://torchgeo.readthedocs.io/) starter project for training
**crop classification** models on **Sentinel-2** satellite imagery over
**Uzbekistan / Central Asia**.

The default pipeline loads labeled crop/non-crop samples from the
[CropHarvest](https://github.com/nasaharvest/cropharvest) public dataset (which includes
Uzbekistan) and fine-tunes a ResNet-50 backbone pretrained on Sentinel-2 via
contrastive learning (MoCo weights shipped with torchgeo).

---

## Features

- Config-driven training with **Hydra** — swap datasets, models, and trainer settings from the CLI.
- **PyTorch Lightning** training loop with checkpointing, CSV logging, and rich metrics.
- **torchgeo**-compatible `Dataset` + Lightning `DataModule` for Sentinel-2 tiles.
- **Planetary Computer STAC** download helper for pulling Sentinel-2 L2A scenes over an AOI.
- **CropHarvest** label loader bundled; ESA WorldCereal hook stubbed for extension.
- Test suite with **synthetic GeoTIFF fixtures** — full pipeline smoke-tests run offline.
- GitHub Actions CI: `ruff` lint + `pytest`.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/bahromiddin-0256/ai-train-on-gis-data.git
cd ai-train-on-gis-data
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run the test suite (offline, uses synthetic fixtures)
pytest -q

# 3. Smoke-train for a single dev batch
python -m gis_train.train trainer.fast_dev_run=true

# 4. Real training on CropHarvest Uzbekistan subset
pip install -e ".[cropharvest]"
python scripts/download_data.py --aoi fergana --date-start 2021-04-01 --date-end 2021-10-31
python -m gis_train.train data=uzbekistan_s2 model=resnet50_s2 trainer.max_epochs=20
```

## Project layout

```
├── configs/                Hydra configs (data / model / trainer)
├── src/gis_train/          Importable Python package
│   ├── data/               Dataset, DataModule, download, labels, transforms
│   ├── models/             LightningModule classifier
│   ├── train.py            Hydra entrypoint: `python -m gis_train.train`
│   ├── evaluate.py         Evaluation + confusion matrix
│   └── utils/              Geo + logging helpers
├── scripts/                CLI helpers (download_data, prepare_labels)
├── notebooks/              EDA notebooks
└── tests/                  pytest suite with synthetic fixtures
```

## Data sources

| Source | What it provides | How it's used |
|---|---|---|
| [Sentinel-2 L2A](https://registry.opendata.aws/sentinel-2-l2a-cogs/) via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) | 10m-resolution multispectral imagery | Input features (bands B02, B03, B04, B08 by default) |
| [CropHarvest](https://github.com/nasaharvest/cropharvest) | Global crop/non-crop labels with Sentinel-2 monthly composites | Default training labels (Uzbekistan subset) |
| [ESA WorldCereal](https://esa-worldcereal.org/) | Seasonal cropland + winter-cereals rasters, self-run for 2024-2025 via CDSE openEO | Label cleanup: `scripts/run_worldcereal.py` produces the rasters, then `scripts/validate_uzcosmos_worldcereal.py` (or `gis_train.data.labels.filter_labels_with_worldcereal`) filters uzcosmos polygons to the agreement-accepted subset |

## Configuration

Hydra composes the final run config from three groups:

```bash
python -m gis_train.train \
    data=uzbekistan_s2 \
    model=resnet50_s2 \
    trainer=default \
    trainer.max_epochs=50 \
    data.batch_size=32
```

See `configs/` for all overridable fields.

## Citations

If you use this project, please cite the underlying datasets:

- **CropHarvest**: Tseng, G., Zvonkov, I., Nakalembe, C., Kerner, H. *CropHarvest: A global
  dataset for crop-type classification.* NeurIPS Datasets and Benchmarks, 2021.
- **ESA WorldCereal**: Van Tricht, K., et al. *WorldCereal: a dynamic open-source system
  for global-scale, seasonal agricultural land use mapping and monitoring.* Earth Syst. Sci. Data, 2023.
- **torchgeo**: Stewart, A. J., et al. *TorchGeo: deep learning with geospatial data.*
  SIGSPATIAL, 2022.

## Roadmap / TODO

- [x] WorldCereal label-agreement filter (`scripts/run_worldcereal.py` +
  `scripts/validate_uzcosmos_worldcereal.py` +
  `gis_train.data.labels.filter_labels_with_worldcereal`).
- [ ] Add `TempCNN` time-series model (config stub already present).
- [ ] Weights & Biases / MLflow logger integration.
- [ ] Dockerfile + devcontainer.
- [ ] Hyper-parameter sweep examples (Hydra sweeper).

## License

MIT — see [LICENSE](./LICENSE).
