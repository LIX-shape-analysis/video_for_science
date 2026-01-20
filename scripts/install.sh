#!/bin/bash
# Installation script for wan22-well-finetuning
# Usage: bash scripts/install.sh

set -e

echo "=============================================="
echo "Installing wan22-well-finetuning dependencies"
echo "=============================================="

# Upgrade pip
echo ""
echo "Step 1: Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (critical for other packages)
echo ""
echo "Step 2: Installing PyTorch..."
pip install torch torchvision

# Install base requirements
echo ""
echo "Step 3: Installing base requirements..."
pip install -r requirements.txt

# Install diffusers from source
echo ""
echo "Step 4: Installing diffusers from source (required for Wan2.2)..."
pip install git+https://github.com/huggingface/diffusers

# Optional: flash-attn
echo ""
echo "Step 5: Attempting to install flash-attn (optional, may fail)..."
pip install ninja packaging || true
pip install flash-attn --no-build-isolation || {
    echo "WARNING: flash-attn installation failed. This is optional."
    echo "The code will work without it, but training may be slower."
}

# Optional: logging tools
echo ""
echo "Step 6: Installing optional logging tools..."
pip install wandb tensorboard || true

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Download the dataset:  python scripts/download_data.py --output_dir ./datasets"
echo "  2. Start training:        torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml"
echo ""
