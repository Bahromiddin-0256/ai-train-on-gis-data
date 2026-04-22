# XGBoost Model Training Plan

## Overview
Implement XGBoost training pipeline for crop classification on Sentinel-2 satellite imagery over Uzbekistan. The data consists of multi-temporal (3 windows) Sentinel-2 patches with 3 classes: bugdoy (wheat), other, paxta (cotton).

## Data Summary
- **Input shape**: (90465, 30, 64, 64) - 90k samples, 30 channels, 64×64 patches
- **Channels**: 3 time windows × 10 channels (9 S2 bands + NDVI)
- **Classes**: 3 (bugdoy=0, other=1, paxta=2)
- **Class distribution**: [30830, 32896, 26739] - fairly balanced

## Implementation Approach

### Option 1: Statistical Feature Extraction (Recommended)
Extract per-channel statistics that capture spectral and spatial information:

**Features per channel (10 channels × 3 windows = 30 channels):**
- Mean, std, min, max, percentiles (25, 50, 75)
- Spatial gradients (mean absolute gradient in x, y)
- Texture features (local variance)

**Total features**: ~210 (manageable for XGBoost)

### Option 2: PCA Dimensionality Reduction
Flatten patches and apply PCA to reduce to ~100-500 components.

### Option 3: CNN Feature Extraction
Use pretrained ResNet as feature extractor, then train XGBoost on embeddings.

## Implementation Steps

### Phase 1: Feature Engineering
1. Create `scripts/extract_features.py`
   - Load images.npy and labels.npy
   - Compute statistical features per channel
   - Save features as CSV or NPZ

### Phase 2: XGBoost Training
1. Create `scripts/train_xgboost.py`
   - Load features
   - Split data (train/val/test)
   - Train XGBoost with cross-validation
   - Hyperparameter tuning (Optuna or grid search)
   - Save model and metrics

### Phase 3: Evaluation
1. Create `scripts/evaluate_xgboost.py`
   - Load trained model
   - Compute metrics (accuracy, F1, confusion matrix)
   - Per-class metrics
   - Feature importance analysis

## Technical Details

### Features to Extract
```
Per channel (30 channels):
  - mean, std, min, max
  - percentile_25, median, percentile_75
  - mean_gradient_x, mean_gradient_y
  - local_variance

Across channels per window (3 windows):
  - NDVI statistics (already computed)
  - Band ratios (NIR/Red, NIR/SWIR)

Temporal features:
  - Change detection between windows
  - Temporal statistics per band
```

### XGBoost Configuration
```python
params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'max_depth': 6-10,
    'learning_rate': 0.05-0.1,
    'n_estimators': 500-1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3-5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'tree_method': 'hist',  # fast histogram
    'random_state': 42
}
```

### Expected Outcomes
- Baseline accuracy: ~60-70% (random = 33%)
- Target accuracy: ~80-90%
- Feature importance: NDVI and NIR bands likely most important
- Training time: < 30 minutes for full dataset

## Files to Create
1. `scripts/extract_features.py` - Feature extraction
2. `scripts/train_xgboost.py` - Model training
3. `scripts/evaluate_xgboost.py` - Evaluation
4. `configs/xgboost.yaml` - Hydra config
5. Update `pyproject.toml` - Add xgboost dependency

## Validation
- Run feature extraction on sample
- Train with small dataset first
- Verify metrics match expected ranges
- Compare with CNN baseline
