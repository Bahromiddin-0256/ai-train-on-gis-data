#!/bin/bash
# ResNet-50+ accuracy-upgrade pipeline
#
# Differences from run_resnet50_pipeline.sh:
#   - Focal loss (gamma=2) handles residual class imbalance
#   - Backbone LR = head LR × 0.1 (differential learning rates)
#   - 3-epoch linear warmup then cosine annealing
#   - D4 test-time augmentation at val/test (+1–3% accuracy)
#   - Mixup (alpha=0.2) + CutMix (alpha=1.0) batch augmentation
#   - Temporal dropout (p=0.15) + band dropout (p=0.1)
#
# Usage: ./scripts/run_resnet50_plus_pipeline.sh [data_dir] [output_dir]

set -euo pipefail

DATA_DIR="${1:-data/processed_regional_mt}"
OUTPUT_DIR="${2:-outputs/resnet50_plus}"
TRAIN_DIR="$OUTPUT_DIR/train"
EXPERIMENT="resnet50_plus"

echo "========================================"
echo "ResNet-50+ Pipeline"
echo "========================================"
echo "Data     : $DATA_DIR"
echo "Output   : $OUTPUT_DIR"
echo ""

mkdir -p "$TRAIN_DIR"

# ---------------------------------------------------------------------------
# Step 1: Train
# ---------------------------------------------------------------------------
echo "Step 1: Training (focal loss + TTA + mixup)..."
python -m gis_train.train \
    model=resnet50_s2_plus \
    data=uzbekistan_s2_plus \
    "data.data_dir=$DATA_DIR" \
    "data.source=local" \
    "trainer.default_root_dir=$TRAIN_DIR" \
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
echo "Step 2: Evaluating (with TTA)..."
python -m gis_train.evaluate \
    model=resnet50_s2_plus \
    data=uzbekistan_s2_plus \
    "data.data_dir=$DATA_DIR" \
    "data.source=local" \
    "ckpt=$CKPT" \
    "output_dir=$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Step 4: Export TorchScript model
# ---------------------------------------------------------------------------
echo ""
echo "Step 3: Exporting model..."
CKPT_PATH="$CKPT" python scripts/export_model.py

echo ""
echo "========================================"
echo "ResNet-50+ pipeline complete!"
echo "  Checkpoint  : $CKPT"
echo "  Metrics     : $OUTPUT_DIR/evaluation.json"
echo "========================================"
