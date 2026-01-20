#!/usr/bin/env python3
"""
Training script for fine-tuning Wan2.2-I2V on The Well physics dataset.

Usage:
    # Single GPU
    python scripts/train.py --config configs/default.yaml
    
    # Multi-GPU with torchrun (recommended for 4xH100)
    torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml
    
    # With custom settings
    torchrun --nproc_per_node=4 scripts/train.py \
        --config configs/default.yaml \
        --batch_size 2 \
        --lr 5e-5 \
        --num_epochs 100
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_wan22_model
from src.data import create_dataloaders
from src.training import Trainer, setup_distributed, cleanup_distributed


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
    
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    
    if args.data_path is not None:
        config["data"]["base_path"] = args.data_path
    
    if args.checkpoint_dir is not None:
        config["training"]["checkpoint_dir"] = args.checkpoint_dir
    
    if args.resume_from is not None:
        config["training"]["resume_from"] = args.resume_from
    
    if args.lora_rank is not None:
        config["lora"]["rank"] = args.lora_rank
    
    if args.gradient_accumulation is not None:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation
    
    if args.wandb_project is not None:
        config["logging"]["wandb_project"] = args.wandb_project
    
    if args.no_wandb:
        config["logging"]["use_wandb"] = False
    
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    
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


def main():
    parser = argparse.ArgumentParser(
        description="Train Wan2.2-I2V on The Well physics dataset"
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--gradient_accumulation", type=int, default=None, help="Gradient accumulation steps")
    
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank")
    
    # Data
    parser.add_argument("--data_path", type=str, default=None, help="Path to dataset")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to resume from")
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    
    # Other
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Load and update config
    config = load_config(args.config)
    config = update_config_from_args(config, args)
    
    # Set seed
    set_seed(config["training"]["seed"])
    
    # Get local rank from environment if using torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank == -1:
        local_rank = 0
    
    # Setup distributed training
    rank, world_size, device = setup_distributed()
    
    # Print config on main process
    if rank == 0:
        print("=" * 60)
        print("Wan2.2-I2V Fine-tuning on The Well Dataset")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  - Model: {config['model']['name']}")
        print(f"  - Dataset: {config['data']['dataset_name']}")
        print(f"  - Batch size: {config['training']['batch_size']} x {world_size} GPUs")
        print(f"  - Learning rate: {config['training']['optimizer']['lr']}")
        print(f"  - LoRA rank: {config['lora']['rank']}")
        print(f"  - Epochs: {config['training']['num_epochs']}")
        print(f"  - Device: {device}")
        print("=" * 60)
    
    # Create data loaders
    if rank == 0:
        print("\nLoading dataset...")
    
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
        print("\nInitializing model...")
    
    model = create_wan22_model(config)
    
    # Create trainer
    if rank == 0:
        print("\nInitializing trainer...")
    
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    # Start training
    if rank == 0:
        print("\nStarting training...")
        print("=" * 60)
    
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
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Checkpoints saved to: {config['training']['checkpoint_dir']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
