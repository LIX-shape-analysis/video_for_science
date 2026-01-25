#!/bin/bash
#
# Full HumanTFM Training Pipeline
# 
# This script handles both Stage 0 (adapter pre-training) and 
# Stage 1/2 (distributed DiT training) in one command.
#
# Usage:
#     ./scripts/run_humanTFM_training.sh \
#         --config configs/humanTFM.yaml \
#         --checkpoint_dir /path/to/checkpoints \
#         --gpus 2,3 \
#         --pretrain_epochs 20
#

set -e  # Exit on error

# Default values
CONFIG="configs/humanTFM.yaml"
CHECKPOINT_DIR="./checkpoints_humanTFM"
GPUS="0,1"
PRETRAIN_EPOCHS=20
PRETRAIN_BATCH_SIZE=4
FORCE_PRETRAIN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --pretrain_epochs)
            PRETRAIN_EPOCHS="$2"
            shift 2
            ;;
        --pretrain_batch_size)
            PRETRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --force_pretrain)
            FORCE_PRETRAIN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config PATH          Config file (default: configs/humanTFM.yaml)"
            echo "  --checkpoint_dir PATH  Checkpoint directory"
            echo "  --gpus IDS             Comma-separated GPU IDs (e.g., 2,3)"
            echo "  --pretrain_epochs N    Stage 0 epochs (default: 20)"
            echo "  --pretrain_batch_size  Stage 0 batch size (default: 4)"
            echo "  --force_pretrain       Force re-run Stage 0 even if adapter exists"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

ADAPTER_PATH="${CHECKPOINT_DIR}/adapter_pretrained.pt"

echo ""
echo "========================================================================"
echo "  HumanTFM Full Training Pipeline"
echo "========================================================================"
echo "  Config: $CONFIG"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  GPUs: $GPUS ($NUM_GPUS total)"
echo "  Adapter path: $ADAPTER_PATH"
echo "========================================================================"
echo ""

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# ============================================================================
# STAGE 0: Pre-train adapter (single process, with VAE in loop)
# ============================================================================
if [ "$FORCE_PRETRAIN" = true ] || [ ! -f "$ADAPTER_PATH" ]; then
    echo ""
    echo "========================================================================"
    echo "  STAGE 0: Pre-training Physics Adapter (with VAE)"
    echo "========================================================================"
    echo "  This trains the adapter to produce VAE-compatible representations."
    echo "  Running on single GPU (no distributed) to avoid race conditions."
    echo "========================================================================"
    echo ""
    
    # Use first GPU only for pretraining
    FIRST_GPU="${GPU_ARRAY[0]}"
    
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" python scripts/pretrain_adapter.py \
        --config "$CONFIG" \
        --output_path "$ADAPTER_PATH" \
        --epochs "$PRETRAIN_EPOCHS" \
        --batch_size "$PRETRAIN_BATCH_SIZE"
    
    echo ""
    echo "========================================================================"
    echo "  Stage 0 complete! Adapter saved to: $ADAPTER_PATH"
    echo "========================================================================"
    echo ""
else
    echo "[Stage 0] Adapter already exists: $ADAPTER_PATH"
    echo "         (use --force_pretrain to re-train)"
    echo ""
fi

# ============================================================================
# STAGE 1/2: Distributed DiT Training
# ============================================================================
echo ""
echo "========================================================================"
echo "  STAGE 1/2: Distributed DiT Training"
echo "========================================================================"
echo "  Using $NUM_GPUS GPUs with torchrun"
echo "========================================================================"
echo ""

CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
    --nproc_per_node="$NUM_GPUS" \
    scripts/train_humanTFM.py \
    --config "$CONFIG" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --pretrained_adapter "$ADAPTER_PATH"

echo ""
echo "========================================================================"
echo "  Training complete!"
echo "========================================================================"
