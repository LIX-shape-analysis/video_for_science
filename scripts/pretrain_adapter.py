#!/usr/bin/env python3
"""
Pre-train the Physics Adapter as an autoencoder.

This script trains the PhysicsAdapter (encoder + decoder) to reconstruct
physics fields. This is required BEFORE running HumanTFM training, because
the decoder must know how to map video-space back to physics-space.

Usage:
    python scripts/pretrain_adapter.py \
        --config configs/humanTFM.yaml \
        --output_path /path/to/adapter_pretrained.pt \
        --epochs 10
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.physics_adapter import PhysicsAdapterPair
from src.data.dataset import WellDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train Physics Adapter")
    parser.add_argument("--config", type=str, default="configs/humanTFM.yaml",
                        help="Path to config file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for pretrained adapter weights")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    return parser.parse_args()


def create_dataloader(config: dict, split: str, batch_size: int, num_workers: int):
    """Create dataloader for adapter pre-training."""
    data_config = config.get("data", {})
    
    dataset = WellDataset(
        base_path=data_config.get("base_path", "./datasets/datasets"),
        dataset_name=data_config.get("dataset_name", "turbulent_radiative_layer_2D"),
        n_steps_input=1,  # Only need single frames for autoencoder
        n_steps_output=1,  # Only need single frames
        split=split,
        use_normalization=False,  # Train on raw physics values
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def train_epoch(
    adapter: PhysicsAdapterPair,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch."""
    adapter.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # Get input frames - shape: (B, T, H, W, C) or (B, T, C, H, W)
        frames = batch["input_frames"]  # Use unnormalized frames
        
        # Handle shape - we need (B, C, H, W) for single frames
        if frames.dim() == 5:
            frames = frames[:, 0]  # Take first frame: (B, H, W, C) or (B, C, H, W)
        
        if frames.dim() == 4 and frames.shape[-1] == 4:
            # (B, H, W, C) -> (B, C, H, W)
            frames = frames.permute(0, 3, 1, 2)
        
        frames = frames.to(device).float()
        
        # Forward: encode then decode (autoencoder)
        video_rep = adapter.encode(frames)  # (B, 3, H_vid, W_vid)
        reconstructed = adapter.decode(video_rep)  # (B, 4, H_phys, W_phys)
        
        # Reconstruction loss (MSE)
        loss = F.mse_loss(reconstructed, frames)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    return total_loss / max(num_batches, 1)


def evaluate(
    adapter: PhysicsAdapterPair,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate reconstruction quality."""
    adapter.eval()
    
    total_mse = 0.0
    per_field_mse = [0.0, 0.0, 0.0, 0.0]  # density, pressure, vx, vy
    num_samples = 0
    
    field_names = ["density", "pressure", "velocity_x", "velocity_y"]
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            frames = batch["input_frames"]
            
            if frames.dim() == 5:
                frames = frames[:, 0]
            
            if frames.dim() == 4 and frames.shape[-1] == 4:
                frames = frames.permute(0, 3, 1, 2)
            
            frames = frames.to(device).float()
            B = frames.shape[0]
            
            # Reconstruct
            video_rep = adapter.encode(frames)
            reconstructed = adapter.decode(video_rep)
            
            # Compute MSE
            mse = F.mse_loss(reconstructed, frames, reduction='none')
            total_mse += mse.mean().item() * B
            
            # Per-field MSE
            for c in range(4):
                field_mse = F.mse_loss(reconstructed[:, c], frames[:, c])
                per_field_mse[c] += field_mse.item() * B
            
            num_samples += B
    
    results = {
        "total_mse": total_mse / num_samples,
        "total_rmse": (total_mse / num_samples) ** 0.5,
    }
    
    for c, name in enumerate(field_names):
        mse = per_field_mse[c] / num_samples
        results[f"{name}_mse"] = mse
        results[f"{name}_rmse"] = mse ** 0.5
    
    return results


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("  Physics Adapter Pre-training")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {args.output_path}")
    print(f"{'='*70}\n")
    
    # Create adapter
    adapter_config = config.get("model", {}).get("physics_adapter", {})
    adapter = PhysicsAdapterPair(
        physics_channels=adapter_config.get("physics_channels", 4),
        video_channels=adapter_config.get("video_channels", 3),
        hidden_dim=adapter_config.get("hidden_dim", 64),
        physics_size=tuple(adapter_config.get("physics_size", [128, 384])),
        video_size=tuple(adapter_config.get("video_size", [240, 416])),
        use_residual=False,  # No residual for autoencoder training
    ).to(device)
    
    num_params = sum(p.numel() for p in adapter.parameters())
    print(f"Adapter parameters: {num_params:,}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader = create_dataloader(config, "train", args.batch_size, args.num_workers)
    val_loader = create_dataloader(config, "valid", args.batch_size, args.num_workers)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float("inf")
    
    print("\nStarting training...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(adapter, train_loader, optimizer, device, epoch)
        
        # Evaluate
        val_metrics = evaluate(adapter, val_loader, device)
        val_loss = val_metrics["total_mse"]
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train MSE: {train_loss:.6f}")
        print(f"  Val MSE:   {val_loss:.6f} (RMSE: {val_metrics['total_rmse']:.4f})")
        print(f"  Per-field RMSE:")
        for name in ["density", "pressure", "velocity_x", "velocity_y"]:
            print(f"    {name}: {val_metrics[f'{name}_rmse']:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Create output directory
            output_dir = Path(args.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "adapter_state_dict": adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "config": adapter_config,
            }
            torch.save(checkpoint, args.output_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
    
    # Final summary
    print(f"\n{'='*70}")
    print("  Training Complete!")
    print(f"{'='*70}")
    print(f"  Best validation MSE: {best_val_loss:.6f}")
    print(f"  Best validation RMSE: {best_val_loss**0.5:.4f}")
    print(f"  Saved to: {args.output_path}")
    print(f"{'='*70}\n")
    
    # Final evaluation with more detail
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(args.output_path, map_location=device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    
    final_metrics = evaluate(adapter, val_loader, device)
    print("\nFinal Evaluation:")
    print(f"  Total RMSE: {final_metrics['total_rmse']:.4f}")
    for name in ["density", "pressure", "velocity_x", "velocity_y"]:
        print(f"  {name}: RMSE = {final_metrics[f'{name}_rmse']:.4f}")


if __name__ == "__main__":
    main()
