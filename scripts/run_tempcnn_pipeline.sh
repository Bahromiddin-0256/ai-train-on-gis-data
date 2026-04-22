#!/bin/bash
# TemporalCNN (per-date spatial encoder + temporal encoder) training pipeline
#
# Input : 45-channel chips reshaped to (3 windows × 15 ch/window)
# Model : Lightweight 2D CNN encodes each date independently → spatial features
#         TempCNN or GRU processes the 3-step time series → class logits.
#
# This architecture explicitly models the temporal trajectory (growth curve),
# which is the highest-signal feature for crop type discrimination.
#
# Usage: ./scripts/run_tempcnn_pipeline.sh [data_dir] [output_dir]

set -euo pipefail

DATA_DIR="${1:-data/processed_regional_mt}"
OUTPUT_DIR="${2:-outputs/tempcnn}"
TRAIN_DIR="$OUTPUT_DIR/train"
EXPERIMENT="tempcnn"

echo "========================================"
echo "TemporalCNN Pipeline"
echo "========================================"
echo "Data     : $DATA_DIR"
echo "Output   : $OUTPUT_DIR"
echo "Layout   : 3 windows × 15 ch/window = 45 channels"
echo ""

mkdir -p "$TRAIN_DIR"

# ---------------------------------------------------------------------------
# Step 1: Train
# ---------------------------------------------------------------------------
echo "Step 1: Training (spatial encoder + TempCNN on 3 time steps)..."
python -m gis_train.train \
    model=tempcnn_s2 \
    data=uzbekistan_s2_plus \
    "data.data_dir=$DATA_DIR" \
    "data.source=local" \
    "data.n_windows=3" \
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
    model=tempcnn_s2 \
    data=uzbekistan_s2_plus \
    "data.data_dir=$DATA_DIR" \
    "data.source=local" \
    "data.n_windows=3" \
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
echo "TemporalCNN pipeline complete!"
echo "  Checkpoint  : $CKPT"
echo "  Metrics     : $OUTPUT_DIR/evaluation.json"
echo "========================================"
