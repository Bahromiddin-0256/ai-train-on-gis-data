#!/bin/bash
# Prithvi-EO-1.0-100M foundation model fine-tuning pipeline
#
# Input : 45-channel chips → auto-mapped to 18 HLS channels (6 bands × 3 timesteps)
#         → resized to 224×224 inside PrithviCropClassifier.forward()
# Model : IBM/NASA Prithvi ViT backbone (frozen) + lightweight classification head.
#
# Requirements:
#   pip install transformers accelerate
#   huggingface-cli login   OR   export HF_TOKEN=hf_...
#
# When to use over ResNet-50:
#   - Limited training data (< 500 polygons/class): frozen Prithvi + head wins
#   - Large data (> 2000 polygons/class): unfreeze backbone (--full-finetune flag)
#
# Usage: ./scripts/run_prithvi_pipeline.sh [data_dir] [output_dir] [--full-finetune]

set -euo pipefail

DATA_DIR="${1:-data/processed_regional_mt}"
OUTPUT_DIR="${2:-outputs/prithvi}"
FULL_FINETUNE="${3:-}"
TRAIN_DIR="$OUTPUT_DIR/train"
EXPERIMENT="prithvi"

# Freeze backbone unless --full-finetune is passed
FREEZE_BACKBONE="true"
if [ "$FULL_FINETUNE" = "--full-finetune" ]; then
    FREEZE_BACKBONE="false"
    echo "Full fine-tuning mode: backbone will be unfrozen (requires more data)"
fi

echo "========================================"
echo "Prithvi-EO Pipeline"
echo "========================================"
echo "Data            : $DATA_DIR"
echo "Output          : $OUTPUT_DIR"
echo "Freeze backbone : $FREEZE_BACKBONE"
echo ""
echo "NOTE: First run downloads ~400MB from HuggingFace. Set HF_TOKEN if needed."
echo ""

mkdir -p "$TRAIN_DIR"

# ---------------------------------------------------------------------------
# Step 1: Train (frozen backbone — only classification head)
# ---------------------------------------------------------------------------
echo "Step 1: Training Prithvi head..."
python -m gis_train.train \
    model=prithvi_s2 \
    data=uzbekistan_s2_plus \
    "data.data_dir=$DATA_DIR" \
    "data.source=local" \
    "model.freeze_backbone=$FREEZE_BACKBONE" \
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
    model=prithvi_s2 \
    data=uzbekistan_s2_plus \
    "data.data_dir=$DATA_DIR" \
    "data.source=local" \
    "model.freeze_backbone=$FREEZE_BACKBONE" \
    "ckpt=$CKPT" \
    "output_dir=$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Step 4: Export to TorchScript (for serving)
# ---------------------------------------------------------------------------
echo ""
echo "Step 3: Exporting Prithvi model..."
python scripts/export_prithvi_model.py

echo ""
echo "========================================"
echo "Prithvi pipeline complete!"
echo "  Checkpoint  : $CKPT"
echo "  Metrics     : $OUTPUT_DIR/evaluation.json"
echo ""
echo "Full fine-tune tip: re-run with --full-finetune once you have"
echo "  > 2000 labelled polygons per class."
echo "========================================"
