# Wan2.2-I2V Fine-tuning on The Well Physics Dataset

This repository provides tools for fine-tuning the **Wan2.2-I2V-14B** video generative model on physics simulation data from **The Well** dataset, specifically the `turbulent_radiative_layer_2D` subset.

## Overview

The goal is to adapt a state-of-the-art video generation model to predict the temporal evolution of physics simulations. Given initial conditions (density, pressure, velocity fields), the model learns to predict future states of turbulent radiative layer dynamics.

### Key Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using Low-Rank Adaptation
- **Learnable Channel Adapters**: Convert between 4-channel physics data and 3-channel video format
- **Temporal Predictor**: ConvLSTM-based predictor in Wan2.2's VAE latent space for true future prediction
- **Multi-GPU Training**: Supports distributed training on 4xH100 GPUs with FSDP
- **Physics-aware Evaluation**: Includes VRMSE metrics standard in physics ML benchmarks

### Architecture

**With Temporal Predictor (Recommended):**
```
Physics Input   →  Channel   →  Spatial    →   VAE     →  Temporal   →   VAE     →  Spatial   →  Channel   →  Physics
(4 channels)       Encoder       Upscale       Encoder     Predictor      Decoder     Downscale     Decoder      Output
                  (learnable)   (learnable)   (frozen)    (ConvLSTM)    (frozen)    (learnable)   (learnable)  (4 channels)
```

**Legacy Mode (Adapter-only):**
```
Physics Data (4 channels)     →  Channel Encoder  →  Wan2.2-I2V (3 channels)  →  Channel Decoder  →  Physics Prediction (4 channels)
[density, pressure, vx, vy]       (learnable)           (LoRA fine-tuned)           (learnable)        [density, pressure, vx, vy]
```

## Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.4.0
- CUDA >= 12.0 (for H100 support)
- 4x NVIDIA H100 80GB GPUs (or equivalent)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd wan22-well-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch first (required before other packages)
pip install torch torchvision

# Install remaining dependencies
pip install -r requirements.txt

# Install diffusers from source (required for Wan2.2)
pip install git+https://github.com/huggingface/diffusers

# Optional: Install flash-attn for better performance (requires ninja)
pip install ninja packaging
pip install flash-attn --no-build-isolation

# Optional: Install logging tools
pip install wandb tensorboard

# Optional: Install DeepSpeed for distributed training
pip install deepspeed
```

### Troubleshooting Installation

**flash-attn fails to install:**
```bash
# Make sure torch is installed first, then:
pip install ninja packaging
pip install flash-attn --no-build-isolation
```

**If flash-attn still fails**, the code will work without it (using standard attention), though it will be slower and use more memory.

## Data

### The Well Dataset

The [turbulent_radiative_layer_2D](https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_2D/) dataset simulates turbulent radiative layer dynamics with:

- **4 Physical Fields**: density, pressure, velocity_x, velocity_y
- **Spatial Resolution**: 128 × 384
- **Multiple Trajectories**: Training and validation splits

### Download Data

```bash
python scripts/download_data.py --output_dir ./datasets
```

This downloads both `train` and `valid` splits.

## Configuration

The default configuration is in `configs/default.yaml`. Key settings:

```yaml
model:
  name: "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
  channel_adapter:
    hidden_dim: 64
    num_layers: 2
  # Temporal predictor for actual future prediction
  temporal_predictor:
    enabled: true       # Enable temporal prediction in VAE latent space
    type: "convlstm"    # "convlstm" or "simple"
    hidden_channels: 64

lora:
  rank: 32
  alpha: 64
  target_modules: ["to_q", "to_k", "to_v", "to_out.0"]

training:
  batch_size: 1  # Per GPU
  gradient_accumulation_steps: 4
  lr: 1.0e-4
  num_epochs: 50

data:
  n_steps_input: 4   # Input frames
  n_steps_output: 8  # Frames to predict
```

## Training

### Single GPU

```bash
python scripts/train.py --config configs/default.yaml
```

### Multi-GPU (Recommended for 4xH100)

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml \
    --batch_size 1 \
    --gradient_accumulation 4
```

### Training Options

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml \
    --batch_size 2 \
    --lr 5e-5 \
    --lora_rank 64 \
    --num_epochs 100 \
    --wandb_project "my-project"
```

### Resume Training

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml \
    --resume_from checkpoints/epoch_10.pt
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/default.yaml \
    --output_dir ./evaluation_results \
    --num_samples 100 \
    --visualize
```

### Metrics

- **VRMSE**: Variance-scaled Root Mean Squared Error (primary metric from The Well benchmark)
- **MSE**: Mean Squared Error per field
- **Rollout Metrics**: Error accumulation over multiple prediction steps

## Generation / Inference

Generate predictions from initial conditions:

```bash
python scripts/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --input_data ./datasets \
    --output_dir ./predictions \
    --num_frames 16 \
    --create_animation
```

## Project Structure

```
.
├── configs/
│   └── default.yaml          # Default configuration
├── scripts/
│   ├── download_data.py      # Download The Well dataset
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── generate.py           # Generation/inference script
├── src/
│   ├── data/
│   │   ├── dataset.py        # Dataset classes
│   │   └── transforms.py     # Data transformations
│   ├── models/
│   │   ├── channel_adapter.py    # 4ch <-> 3ch adapters
│   │   ├── temporal_predictor.py # ConvLSTM for latent prediction
│   │   └── wan_wrapper.py        # Wan2.2 model wrapper
│   ├── training/
│   │   ├── trainer.py        # Training loop
│   │   ├── optimizer.py      # Optimizer/scheduler utilities
│   │   └── distributed.py    # Distributed training utilities
│   └── evaluation/
│       ├── metrics.py        # Physics evaluation metrics
│       └── evaluator.py      # Evaluation pipeline
├── requirements.txt
└── README.md
```

## Model Architecture Details

### Channel Adapters

Since Wan2.2 expects 3-channel RGB input but our physics data has 4 channels, we use learnable adapters:

**Encoder (4 → 3 channels)**:
```
Conv2d(4, 64) → GroupNorm → SiLU → Conv2d(64, 64) → GroupNorm → SiLU → Conv2d(64, 3)
```

**Decoder (3 → 4 channels)**:
```
Conv2d(3, 64) → GroupNorm → SiLU → Conv2d(64, 64) → GroupNorm → SiLU → Conv2d(64, 4) + Field-specific heads
```

These adapters are trained jointly with the LoRA parameters.

### LoRA Configuration

We apply LoRA to the transformer attention layers:
- Query, Key, Value projections
- Output projections
- Feed-forward layers

Default rank: 32, Alpha: 64

### Training Objective

The model supports two training modes:

#### Mode 1: Adapter-only Training (Default before temporal predictor)
- **Adapter Reconstruction Loss**: Encourages the encode-decode cycle to preserve information
- **Spatial Reconstruction Loss**: Maintains spatial integrity through up/down sampling
- **Cycle Loss**: End-to-end reconstruction consistency

#### Mode 2: Temporal Predictor Training (Recommended for actual prediction)
When `model.temporal_predictor.enabled: true` in config:
- **Temporal Prediction Loss**: Train a ConvLSTM to predict future states in VAE latent space
- **Physics Prediction Loss**: End-to-end prediction loss from input to output physics frames

The temporal predictor approach:
1. Encodes physics frames through adapters to Wan2.2's VAE latent space
2. Uses a ConvLSTM to predict future latent states from past latents
3. Decodes predicted latents back to physics frames

This leverages Wan2.2's powerful VAE representation while training a lightweight temporal model.

## Expected Results

With the default configuration on 4xH100:

| Metric | Expected Value |
|--------|---------------|
| VRMSE (density) | ~0.2-0.3 |
| VRMSE (pressure) | ~0.3-0.5 |
| VRMSE (velocity_x) | ~0.4-0.6 |
| VRMSE (velocity_y) | ~0.4-0.6 |
| Mean VRMSE | ~0.3-0.5 |

*Note: Results depend on training duration and hyperparameters.*

## Hardware Requirements

### Minimum
- 1x GPU with 80GB VRAM (H100, A100)
- 64GB RAM
- 100GB disk space

### Recommended (for full training)
- 4x H100 80GB GPUs
- 256GB RAM
- 500GB SSD storage

### Memory Optimization

For limited GPU memory:
1. Reduce batch size
2. Increase gradient accumulation
3. Enable CPU offloading in FSDP config
4. Use lower LoRA rank

## Citation

If you use this code, please cite:

```bibtex
@article{wan2025,
  title={Wan: Open and Advanced Large-Scale Video Generative Models},
  author={Team Wan and others},
  journal={arXiv preprint arXiv:2503.20314},
  year={2025}
}

@article{thewell2024,
  title={The Well: A Large-Scale Collection of Diverse Physics Simulations for Machine Learning},
  author={The Well Team},
  year={2024}
}
```

## License

This project is released under the Apache 2.0 License.

## Acknowledgments

- [Wan-AI](https://github.com/Wan-Video/Wan2.2) for the Wan2.2 video generation model
- [Polymathic AI](https://polymathic-ai.org/) for The Well dataset
- [Hugging Face](https://huggingface.co/) for the Diffusers library
