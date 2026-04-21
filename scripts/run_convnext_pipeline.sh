#!/bin/bash
# ConvNeXt-Tiny (ImageNet pretrained) training pipeline
#
# Input : 45-channel multi-temporal chips (9 raw bands + 6 indices × 3 windows)
# Model : ConvNeXt-Tiny with ImageNet pretrained weights, first conv replaced
#         to accept 45 channels (tiled init from 3-channel ImageNet weights).
#
# ConvNeXt typically converges more stably than ResNet on small datasets due to
# its depthwise conv structure and stronger regularization defaults.
#
# Usage: ./scripts/run_convnext_pipeline.sh [data_dir] [output_dir]

set -euo pipefail

DATA_DIR="${1:-data/processed_regional_mt}"
OUTPUT_DIR="${2:-outputs/convnext}"
TRAIN_DIR="$OUTPUT_DIR/train"
EXPERIMENT="convnext"

echo "========================================"
echo "ConvNeXt-Tiny Pipeline"
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
    model=convnext_s2 \
    data=uzbekistan_s2 \
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
echo "Step 2: Evaluating..."
python -m gis_train.evaluate \
    model=convnext_s2 \
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
CKPT_PATH="$CKPT" python scripts/export_model.py

echo ""
echo "========================================"
echo "ConvNeXt pipeline complete!"
echo "  Checkpoint  : $CKPT"
echo "  Metrics     : $OUTPUT_DIR/evaluation.json"
echo "========================================"
