#!/bin/bash
# XGBoost Training Pipeline
# Usage: ./run_xgboost_pipeline.sh [data_dir] [output_dir]
#
# Feature stack:
#   Raw bands  : B02,B03,B04,B05,B06,B07,B08,B11,B12  (9 Sentinel-2 bands)
#   Indices    : NDVI, EVI, NDWI, NDRE, MSI, NBR       (computed per window)
#   Total ch   : (9 + 6) × 3 windows = 45 channels
#   Stats/ch   : mean, std, min, max, p25, median, p75, grad_x, grad_y, var → 10
#   Temporal   : change_w0w1, change_w1w2, total_change, temporal_std per band
#   NDVI time  : ndvi_t_max/min/mean/std/p25/p75/range  (7 features)

set -e

DATA_DIR="${1:-data/processed_regional_mt}"
OUTPUT_DIR="${2:-outputs/xgboost}"

echo "========================================"
echo "XGBoost Training Pipeline"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Extract features
echo "Step 1: Extracting features..."
#python scripts/extract_features.py \
#    --data-dir "$DATA_DIR" \
#    --output "$OUTPUT_DIR/features.csv" \
#    --n-windows 3 \
#    --bands "B02,B03,B04,B05,B06,B07,B08,B11,B12" \
#    --indices "ndvi,evi,ndwi,ndre,msi,nbr"

# Step 2: Train model
echo ""
echo "Step 2: Training XGBoost model..."
python scripts/train_xgboost.py \
    --features "$OUTPUT_DIR/features.csv" \
    --output-dir "$OUTPUT_DIR" \
    --test-size 0.15 \
    --val-size 0.15 \
    --max-depth 8 \
    --learning-rate 0.05 \
    --n-estimators 1000 \
    --subsample 0.8 \
    --colsample-bytree 0.8 \
    --min-child-weight 3 \
    --class-names "bugdoy,other,paxta" \
    --seed 42 \
    --early-stopping 50 \
    --device cuda

# Step 3: Evaluate
echo ""
echo "Step 3: Generating detailed evaluation..."
python scripts/evaluate_xgboost.py \
    --model "$OUTPUT_DIR/xgboost_model.json" \
    --features "$OUTPUT_DIR/features.csv" \
    --test-indices "$OUTPUT_DIR/test_indices.npy" \
    --output-dir "$OUTPUT_DIR" \
    --class-names "bugdoy,other,paxta" \
    --save-predictions

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
