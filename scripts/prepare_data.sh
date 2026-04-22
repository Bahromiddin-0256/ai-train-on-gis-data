#!/bin/bash
# Full data preparation pipeline
#
# Steps:
#   1. Fetch polygon chips from Sentinel-2 STAC (9 bands + 6 indices per window)
#   2. Combine per-tuman datasets into one (if using build_dataset)
#   3. Recompute per-channel normalization stats → update uzbekistan_s2.yaml
#   4. Extract XGBoost-ready feature CSV
#
# Usage:
#   ./scripts/prepare_data.sh [vectors.geojson] [output_dir] [date_windows]
#
# Args:
#   vectors     Path to GeoJSON with labelled polygons  (default: data/labels/labels.geojson)
#   output_dir  Where to write images.npy + labels.npy  (default: data/processed_regional_mt)
#   date_windows  Comma-separated start:end pairs       (default: 3 standard Uzbekistan windows)
#
# Examples:
#   # Direct chip extraction from one GeoJSON:
#   ./scripts/prepare_data.sh data/labels/fergana.geojson data/processed_fergana
#
#   # Full regional pipeline (reads from MongoDB):
#   ./scripts/prepare_data.sh "" data/processed_regional_mt

set -euo pipefail

VECTORS="${1:-data/labels/labels.geojson}"
OUTPUT_DIR="${2:-data/processed_regional_mt}"
DATE_WINDOWS="${3:-2025-03-15:2025-04-15,2025-04-15:2025-05-31,2025-06-01:2025-06-30,2025-07-15:2025-08-31,2025-09-01:2025-09-30,2025-10-01:2025-11-15}"
BANDS="B02,B03,B04,B05,B06,B07,B08,B11,B12"
INDICES="ndvi,evi,ndwi,ndre,msi,nbr"
DATA_CONFIG="configs/data/uzbekistan_s2.yaml"
N_WINDOWS=6
# Parallelism: how many tumans to process at once, and per-tuman process/thread counts
TUMAN_WORKERS="${TUMAN_WORKERS:-4}"
CHIPS_PROC="${CHIPS_PROC:-2}"
CHIPS_THREADS="${CHIPS_THREADS:-4}"

echo "========================================"
echo "Data Preparation Pipeline"
echo "========================================"
echo "Vectors     : $VECTORS"
echo "Output dir  : $OUTPUT_DIR"
echo "Date windows: $DATE_WINDOWS"
echo "Bands       : $BANDS"
echo "Indices     : $INDICES"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Fetch chips from STAC
# ---------------------------------------------------------------------------
if [ -n "$VECTORS" ] && [ -f "$VECTORS" ]; then
    echo "Step 1: Fetching polygon chips from Sentinel-2 STAC..."
    python scripts/prepare_labels.py \
        --from-stac \
        --vectors "$VECTORS" \
        --bands "$BANDS" \
        --date-windows "$DATE_WINDOWS" \
        --indices "$INDICES" \
        --out "$OUTPUT_DIR" \
        --num-proc 16 \
        --num-threads 4
else
    echo "Step 1: Running full regional pipeline (build_dataset.py)..."
    python scripts/build_dataset.py \
        --bands "$BANDS" \
        --date-windows "$DATE_WINDOWS" \
        --indices "$INDICES" \
        --tuman-workers "$TUMAN_WORKERS" \
        --chips-proc "$CHIPS_PROC" \
        --chips-threads "$CHIPS_THREADS" \
        --out "$OUTPUT_DIR"
fi

echo ""
echo "Dataset shape:"
python -c "
import numpy as np
import sys
arr = np.load('$OUTPUT_DIR/images.npy', mmap_mode='r')
lab = np.load('$OUTPUT_DIR/labels.npy')
import collections
counts = collections.Counter(lab.tolist())
print(f'  images : {arr.shape}  dtype={arr.dtype}')
print(f'  labels : {lab.shape}  classes={dict(sorted(counts.items()))}')
expected_ch = $N_WINDOWS * (len('$BANDS'.split(',')) + len('$INDICES'.split(',')))
if arr.shape[1] != expected_ch:
    print(f'  WARNING: expected {expected_ch} channels, got {arr.shape[1]}')
else:
    print(f'  channels: {arr.shape[1]} = {$N_WINDOWS} windows × {arr.shape[1] // $N_WINDOWS} ch/window  OK')
"

# ---------------------------------------------------------------------------
# Step 2: Compute per-channel normalization stats → update YAML config
# ---------------------------------------------------------------------------
echo ""
echo "Step 2: Computing per-channel normalization stats..."
python scripts/compute_band_stats.py \
    --images "$OUTPUT_DIR/images.npy" \
    --out "$DATA_CONFIG"
echo "Updated mean/std in $DATA_CONFIG"

# ---------------------------------------------------------------------------
# Step 3: Extract XGBoost features
# ---------------------------------------------------------------------------
FEATURES_DIR="${OUTPUT_DIR}/xgboost_features"
mkdir -p "$FEATURES_DIR"

echo ""
echo "Step 3: Extracting XGBoost features..."
python scripts/extract_features.py \
    --data-dir "$OUTPUT_DIR" \
    --output "$FEATURES_DIR/features.csv" \
    --n-windows "$N_WINDOWS" \
    --bands "$BANDS" \
    --indices "$INDICES" \
    --workers 16 \
    --processes 4

echo ""
echo "========================================"
echo "Data preparation complete!"
echo "  Dataset    : $OUTPUT_DIR/images.npy"
echo "  Features   : $FEATURES_DIR/features.csv"
echo "  Band stats : $DATA_CONFIG (updated)"
echo ""
echo "Next steps:"
echo "  DL training : ./scripts/run_resnet50_pipeline.sh $OUTPUT_DIR"
echo "  XGBoost     : ./scripts/run_xgboost_pipeline.sh $OUTPUT_DIR"
echo "========================================"
