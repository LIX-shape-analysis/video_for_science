#!/usr/bin/env python3
"""
HumanTFM Training Script.

One-shot deterministic prediction model with two-stage training.
Based on HumanTFM paper methodology.

Usage:
    # Single GPU
    python scripts/train_humanTFM.py --config configs/humanTFM.yaml
    
    # Multi-GPU (recommended)
    torchrun --nproc_per_node=4 scripts/train_humanTFM.py --config configs/humanTFM.yaml
    
    # With custom checkpoint dir
    torchrun --nproc_per_node=4 scripts/train_humanTFM.py \
        --config configs/humanTFM.yaml \
        --checkpoint_dir /path/to/checkpoints
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.humantfm_model import HumanTFMModel
from src.data import create_dataloaders
from src.training.humantfm_trainer import TwoStageTrainer
from src.training.distributed import setup_distributed, cleanup_distributed


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def update_config_from_args(config: dict, args: argparse.Namespace) -> dict:
    """Update configuration with command line arguments."""
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    
    if args.lr is not None:
        config["training"]["optimizer"]["lr"] = args.lr
    
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
    
    if args.checkpoint_dir is not None:
        config["training"]["checkpoint_dir"] = args.checkpoint_dir
    
    if args.resume_from is not None:
        config["training"]["resume_from"] = args.resume_from
    
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    
    # Two-stage specific
    if args.min_stage1_epochs is not None:
        config["training"]["two_stage"]["min_epochs"] = args.min_stage1_epochs
    
    if args.plateau_patience is not None:
        config["training"]["two_stage"]["plateau_patience"] = args.plateau_patience
    
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_humantfm_model(config: dict) -> HumanTFMModel:
    """Create HumanTFM model from config."""
    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    physics_config = model_config.get("physics_adapter", {})
    train_config = config.get("training", {})
    
    # Determine device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    
    # Parse dtype
    dtype_str = model_config.get("dtype", "bfloat16")
    if dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_str == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    model = HumanTFMModel(
        model_id=model_config.get("name", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"),
        dtype=dtype,
        device=device,
        physics_channels=physics_config.get("physics_channels", 4),
        video_channels=physics_config.get("video_channels", 3),
        adapter_hidden_dim=physics_config.get("hidden_dim", 64),
        physics_size=tuple(physics_config.get("physics_size", [128, 384])),
        video_size=tuple(physics_config.get("video_size", [480, 832])),
        lora_enabled=lora_config.get("enabled", True),
        lora_rank=lora_config.get("rank", 32),
        lora_alpha=lora_config.get("alpha", 64),
        lora_dropout=lora_config.get("dropout", 0.05),
        lora_target_modules=lora_config.get("target_modules"),
        use_residual=physics_config.get("use_residual", True),
        default_text_prompt=train_config.get("text_prompt", "Fluid dynamics simulation"),
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train HumanTFM model on physics simulation data"
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/humanTFM.yaml",
        help="Path to configuration file"
    )
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=None, help="Total epochs")
    
    # Two-stage training
    parser.add_argument("--min_stage1_epochs", type=int, default=None, 
                        help="Minimum epochs in Stage 1")
    parser.add_argument("--plateau_patience", type=int, default=None,
                        help="Epochs without improvement before stage transition")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default=None, 
                        help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="Checkpoint to resume from")
    
    # Other
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--local_rank", type=int, default=-1, 
                        help="Local rank for distributed training")
    parser.add_argument("--pretrained_adapter", type=str, default=None,
                        help="Path to pretrained adapter checkpoint (REQUIRED for good results)")
    
    args = parser.parse_args()
    
    # Load and update config
    config = load_config(args.config)
    config = update_config_from_args(config, args)
    
    # Set seed
    set_seed(config["training"].get("seed", 42))
    
    # Setup distributed training
    rank, world_size, device = setup_distributed()
    
    # Print banner
    if rank == 0:
        print("\n" + "="*70)
        print("  HumanTFM: One-Shot Physics Prediction with Two-Stage Training")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  - Model: {config['model']['name']}")
        print(f"  - Dataset: {config['data']['dataset_name']}")
        print(f"  - Batch size: {config['training']['batch_size']} × {world_size} GPUs")
        print(f"  - Learning rate: {config['training']['optimizer']['lr']}")
        print(f"  - LoRA rank: {config['lora']['rank']}")
        print(f"  - Total epochs: {config['training']['num_epochs']}")
        print(f"\nTwo-Stage Training:")
        two_stage = config['training'].get('two_stage', {})
        print(f"  - Min Stage 1 epochs: {two_stage.get('min_epochs', 3)}")
        print(f"  - Plateau patience: {two_stage.get('plateau_patience', 5)}")
        print(f"  - Stage 1: Latent loss (decoder frozen)")
        print(f"  - Stage 2: Ambient loss (decoder trainable)")
        print("="*70 + "\n")
    
    # Create data loaders
    if rank == 0:
        print("Loading dataset...")
    
    train_loader, val_loader = create_dataloaders(
        config,
        rank=rank,
        world_size=world_size,
    )
    
    if rank == 0:
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    if rank == 0:
        print("\nInitializing HumanTFM model...")
    
    model = create_humantfm_model(config)
    
    # Load pretrained adapter (CRITICAL for good results)
    if args.pretrained_adapter:
        model.load_adapter(args.pretrained_adapter)
        if rank == 0:
            print("  ✓ Loaded pretrained adapter weights")
    else:
        if rank == 0:
            print("\n" + "!"*70)
            print("  WARNING: No pretrained adapter specified!")
            print("  The decoder will be randomly initialized.")
            print("  Use --pretrained_adapter /path/to/adapter.pt")
            print("  Run scripts/pretrain_adapter.py first.")
            print("!"*70 + "\n")
    
    # Create trainer
    if rank == 0:
        print("\nInitializing two-stage trainer...")
    
    trainer = TwoStageTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    # Resume from checkpoint if specified
    if config["training"].get("resume_from"):
        trainer.load_checkpoint(config["training"]["resume_from"])
    
    # Start training
    if rank == 0:
        print("\nStarting training...")
        print("="*70)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user")
            trainer.save_checkpoint("interrupted.pt")
    except Exception as e:
        if rank == 0:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        cleanup_distributed()
    
    if rank == 0:
        print("\n" + "="*70)
        print("Training complete!")
        print(f"Checkpoints saved to: {config['training']['checkpoint_dir']}")
        print("="*70)


if __name__ == "__main__":
    main()
