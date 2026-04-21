# XGBoost Model for Crop Classification

This directory contains scripts for training XGBoost models on Sentinel-2 crop classification data using statistical feature extraction.

## Quick Start

### 1. Run Full Pipeline
```bash
./scripts/run_xgboost_pipeline.sh data/processed_regional_mt outputs/xgboost
```

### 2. Run Steps Manually

#### Extract Features
```bash
python scripts/extract_features.py \
    --data-dir data/processed_regional_mt \
    --output outputs/xgboost/features.csv \
    --n-windows 3 \
    --bands "B02,B03,B04,B05,B06,B07,B08,B11,B12,NDVI"
```

#### Train Model
```bash
python scripts/train_xgboost.py \
    --features outputs/xgboost/features.csv \
    --output-dir outputs/xgboost \
    --class-names "bugdoy,other,paxta" \
    --max-depth 8 \
    --learning-rate 0.05 \
    --n-estimators 1000 \
    --cv  # Optional: run cross-validation
```

#### Evaluate Model
```bash
python scripts/evaluate_xgboost.py \
    --model outputs/xgboost/xgboost_model.json \
    --features outputs/xgboost/features.csv \
    --class-names "bugdoy,other,paxta" \
    --save-predictions
```

## Features Extracted

### Per-Channel Statistics (30 channels)
- Mean, std, min, max
- Percentiles (25, 50, 75)
- Spatial gradients (x, y)
- Local variance (texture)

### Temporal Features (3 windows × 10 bands)
- Change between windows (w0→w1, w1→w2)
- Total change (w0→w2)
- Temporal standard deviation

**Total: ~360 features**

## Model Configuration

Default hyperparameters:
- `max_depth`: 8
- `learning_rate`: 0.05
- `n_estimators`: 1000 (with early stopping)
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `min_child_weight`: 3
- `objective`: multi:softprob

## Output Files

| File | Description |
|------|-------------|
| `xgboost_model.json` | Trained model (XGBoost native format) |
| `xgboost_model.pkl` | Trained model (pickle format) |
| `metrics.json` | All metrics and configuration |
| `feature_importance.csv` | Feature importance by gain |
| `predictions.csv` | Per-sample predictions and probabilities |
| `evaluation_results.json` | Detailed evaluation metrics |

## Testing with Small Sample

To test on a subset:
```bash
python scripts/extract_features.py \
    --data-dir data/processed_regional_mt \
    --sample-size 1000 \
    --output outputs/xgboost_test/features.csv
```

## Hyperparameter Tuning

Run with different parameters:
```bash
python scripts/train_xgboost.py \
    --features outputs/xgboost/features.csv \
    --output-dir outputs/xgboost_tuned \
    --max-depth 10 \
    --learning-rate 0.1 \
    --subsample 0.9 \
    --colsample-bytree 0.9
```

## Expected Results

On the full dataset (~90k samples):
- **Target accuracy**: 80-90%
- **Training time**: 5-15 minutes
- **Feature count**: ~360

## Troubleshooting

### Memory Issues
Use memory-mapped arrays:
```python
images = np.load('images.npy', mmap_mode='r')
```

### Class Imbalance
The training script automatically computes class weights for balanced learning.

### Feature Importance
Top features typically include:
- NDVI statistics (vegetation index)
- NIR (B08) band statistics
- Red edge (B05, B06, B07) statistics
- Temporal change features
