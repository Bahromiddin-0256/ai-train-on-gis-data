#!/bin/bash
# ResNet-50 (Sentinel-2 MoCo pretrained) training pipeline
#
# Input : 90-channel multi-temporal chips (9 raw bands + 6 indices × 6 phenological windows)
# Model : ResNet-50 backbone with pretrained Sentinel-2 MoCo weights,
#         first conv patched to accept 90 channels via band-aware warm init.
#
# Usage: ./scripts/run_resnet50_pipeline.sh [data_dir] [output_dir]

set -euo pipefail

DATA_DIR="${1:-data/processed_regional_v6win}"
OUTPUT_DIR="${2:-outputs/resnet50}"
TRAIN_DIR="$OUTPUT_DIR/train"
EXPERIMENT="resnet50"

echo "========================================"
echo "ResNet-50 Pipeline"
echo "========================================"
echo "Data     : $DATA_DIR"
echo "Output   : $OUTPUT_DIR"
echo ""

mkdir -p "$TRAIN_DIR"

# ---------------------------------------------------------------------------
# Step 1: Train
# ---------------------------------------------------------------------------
echo "Step 1: Training..."
python -m gis_train.train \
    model=resnet50_s2 \
    data=uzbekistan_s2 \
    "data.data_dir=$DATA_DIR" \
    "data.source=local" \
    "experiment_name=$EXPERIMENT"

# ---------------------------------------------------------------------------
# Step 2: Locate best checkpoint
# ---------------------------------------------------------------------------
CKPT=$(find "$TRAIN_DIR" -name "best-*.ckpt" 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found in $TRAIN_DIR"
    exit 1
fi
echo ""
echo "Best checkpoint: $CKPT"

# ---------------------------------------------------------------------------
# Step 3: Evaluate on test split
# ---------------------------------------------------------------------------
echo ""
echo "Step 2: Evaluating..."
python -m gis_train.evaluate \
    model=resnet50_s2 \
    data=uzbekistan_s2 \
    "data.data_dir=$DATA_DIR" \
    "data.source=local" \
    "ckpt=$CKPT" \
    "output_dir=$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Step 4: Export TorchScript model
# ---------------------------------------------------------------------------
echo ""
echo "Step 3: Exporting model..."
EXPORT_DATE="$(date +%Y%m%d-%H%M)"
EXPORT_PATH="$OUTPUT_DIR/resnet50_s2-moco_90ch-6win_${EXPORT_DATE}.pt"
CKPT_PATH="$CKPT" OUT_PATH="$EXPORT_PATH" python scripts/export_model.py

echo ""
echo "========================================"
echo "ResNet-50 pipeline complete!"
echo "  Checkpoint  : $CKPT"
echo "  Exported .pt: $EXPORT_PATH"
echo "  Metrics     : $OUTPUT_DIR/evaluation.json"
echo "========================================"
