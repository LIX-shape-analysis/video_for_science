#!/usr/bin/env python3
"""
Pre-train the Physics Adapter WITH VAE in the loop.

This script trains the PhysicsAdapter (encoder + decoder) to reconstruct
physics fields THROUGH the Wan2.2 VAE. This is critical because Stage 1
uses the VAE, so the adapter must learn outputs that survive VAE compression.

Training loop:
    Physics → Adapter.Encode → [VAE.Encode → VAE.Decode] → Adapter.Decode → Physics
                                 ↑ frozen VAE ↑

Usage:
    python scripts/pretrain_adapter.py \
        --config configs/humanTFM.yaml \
        --output_path /path/to/adapter_pretrained.pt \
        --epochs 20
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
from src.data.dataset import WellVideoDataset

try:
    from diffusers import WanImageToVideoPipeline
except ImportError:
    raise ImportError("Please install diffusers: pip install diffusers")


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train Physics Adapter with VAE")
    parser.add_argument("--config", type=str, default="configs/humanTFM.yaml",
                        help="Path to config file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for pretrained adapter weights")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--skip_vae", action="store_true",
                        help="Skip VAE (for debugging only, NOT recommended)")
    return parser.parse_args()


def create_dataloader(config: dict, split: str, batch_size: int, num_workers: int):
    """Create dataloader for adapter pre-training."""
    data_config = config.get("data", {})
    
    dataset = WellVideoDataset(
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


def train_epoch_with_vae(
    adapter: PhysicsAdapterPair,
    vae: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dtype: torch.dtype,
    epoch: int,
    vae_scaling_factor: float = 0.18215,
) -> float:
    """
    Train for one epoch WITH VAE in the loop.
    
    Training path:
        Physics → Adapter.Encode → VAE.Encode → VAE.Decode → Adapter.Decode → Physics
    """
    adapter.train()
    vae.eval()  # VAE is always frozen
    
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
        B = frames.shape[0]
        
        # ============================================================
        # Step 1: Encode Physics -> Video space via Adapter
        # Output is already in [-1, 1] range due to tanh in encoder
        # ============================================================
        video_frames = adapter.encode(frames.to(dtype))  # (B, 3, H_vid, W_vid)
        
        # ============================================================
        # Step 2: Pass through FROZEN VAE (the critical bottleneck!)
        # This forces the adapter to learn VAE-compatible representations
        # ============================================================
        with torch.no_grad():
            # VAE expects (B, C, T, H, W) format
            vae_input = video_frames.unsqueeze(2)  # (B, 3, 1, H, W)
            
            # Encode to latent space
            latent_dist = vae.encode(vae_input)
            if hasattr(latent_dist, 'latent_dist'):
                latents = latent_dist.latent_dist.mode()  # Use mode for deterministic
            else:
                latents = latent_dist.mode() if hasattr(latent_dist, 'mode') else latent_dist
            
            # Scale latents (standard VAE practice)
            latents = latents * vae_scaling_factor
            
            # Decode back to video space
            latents_for_decode = latents / vae_scaling_factor
            decoded = vae.decode(latents_for_decode)
            if hasattr(decoded, 'sample'):
                video_recon = decoded.sample
            else:
                video_recon = decoded
            
            video_recon = video_recon.squeeze(2)  # (B, 3, H_vid, W_vid)
        
        # ============================================================
        # Step 3: Decode Video -> Physics via Adapter (with gradients)
        # ============================================================
        # Note: We detach the VAE output but keep adapter decoder trainable
        # Actually, we want gradients to flow through the adapter decode
        # The VAE output is already no_grad, so adapter.decode will get gradients
        reconstructed = adapter.decode(video_recon.to(dtype))  # (B, 4, H_phys, W_phys)
        
        # ============================================================
        # Step 4: Reconstruction loss
        # ============================================================
        loss = F.mse_loss(reconstructed.float(), frames)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    return total_loss / max(num_batches, 1)


def train_epoch_no_vae(
    adapter: PhysicsAdapterPair,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dtype: torch.dtype,
    epoch: int,
) -> float:
    """Train without VAE (for comparison/debugging only)."""
    adapter.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (no VAE)")
    
    for batch in pbar:
        frames = batch["input_frames"]
        
        if frames.dim() == 5:
            frames = frames[:, 0]
        if frames.dim() == 4 and frames.shape[-1] == 4:
            frames = frames.permute(0, 3, 1, 2)
        
        frames = frames.to(device).float()
        
        # Simple encode -> decode
        video_rep = adapter.encode(frames.to(dtype))
        reconstructed = adapter.decode(video_rep)
        
        loss = F.mse_loss(reconstructed.float(), frames)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    return total_loss / max(num_batches, 1)


def evaluate(
    adapter: PhysicsAdapterPair,
    vae: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    use_vae: bool = True,
    vae_scaling_factor: float = 0.18215,
) -> dict:
    """Evaluate reconstruction quality (with or without VAE)."""
    adapter.eval()
    if vae is not None:
        vae.eval()
    
    total_mse = 0.0
    per_field_mse = [0.0, 0.0, 0.0, 0.0]
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
            
            # Encode
            video_rep = adapter.encode(frames.to(dtype))
            
            # Optional VAE pass
            if use_vae and vae is not None:
                vae_input = video_rep.unsqueeze(2)
                latent_dist = vae.encode(vae_input)
                if hasattr(latent_dist, 'latent_dist'):
                    latents = latent_dist.latent_dist.mode()
                else:
                    latents = latent_dist.mode() if hasattr(latent_dist, 'mode') else latent_dist
                latents = latents * vae_scaling_factor
                latents_for_decode = latents / vae_scaling_factor
                decoded = vae.decode(latents_for_decode)
                video_recon = decoded.sample if hasattr(decoded, 'sample') else decoded
                video_rep = video_recon.squeeze(2)
            
            # Decode
            reconstructed = adapter.decode(video_rep)
            
            # Compute MSE
            mse = F.mse_loss(reconstructed.float(), frames, reduction='none')
            total_mse += mse.mean().item() * B
            
            # Per-field MSE
            for c in range(4):
                field_mse = F.mse_loss(reconstructed[:, c].float(), frames[:, c])
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
    dtype = torch.bfloat16  # Match model training dtype
    
    print(f"\n{'='*70}")
    print("  Physics Adapter Pre-training WITH VAE")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {args.output_path}")
    print(f"  VAE in loop: {not args.skip_vae}")
    print(f"{'='*70}\n")
    
    # ================================================================
    # Load VAE (frozen) - this is the critical addition!
    # ================================================================
    vae = None
    vae_scaling_factor = 0.18215
    
    if not args.skip_vae:
        print("Loading Wan2.2 VAE (this may take a moment)...")
        model_name = config.get("model", {}).get("name", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
        
        # Load just the VAE from the pipeline
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        vae = pipe.vae.to(device)
        vae.requires_grad_(False)  # Freeze VAE
        vae.eval()
        
        if hasattr(vae.config, 'scaling_factor'):
            vae_scaling_factor = vae.config.scaling_factor
        
        print(f"  VAE loaded and frozen")
        print(f"  VAE scaling factor: {vae_scaling_factor}")
        
        # Clean up pipeline to save memory
        del pipe.transformer
        del pipe.text_encoder
        del pipe
        torch.cuda.empty_cache()
    
    # ================================================================
    # Create adapter
    # ================================================================
    adapter_config = config.get("model", {}).get("physics_adapter", {})
    adapter = PhysicsAdapterPair(
        physics_channels=adapter_config.get("physics_channels", 4),
        video_channels=adapter_config.get("video_channels", 3),
        hidden_dim=adapter_config.get("hidden_dim", 64),
        physics_size=tuple(adapter_config.get("physics_size", [128, 384])),
        video_size=tuple(adapter_config.get("video_size", [240, 416])),
        use_residual=False,  # No residual for autoencoder training
    ).to(device).to(dtype)
    
    num_params = sum(p.numel() for p in adapter.parameters())
    print(f"\nAdapter parameters: {num_params:,}")
    
    # ================================================================
    # Create dataloaders
    # ================================================================
    print("\nLoading data...")
    train_loader = create_dataloader(config, "train", args.batch_size, args.num_workers)
    val_loader = create_dataloader(config, "valid", args.batch_size, args.num_workers)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # ================================================================
    # Optimizer & scheduler
    # ================================================================
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # ================================================================
    # Training loop
    # ================================================================
    best_val_loss = float("inf")
    
    print("\nStarting training...")
    if not args.skip_vae:
        print("⚠️  Training WITH VAE - expect higher loss but better Stage 1 compatibility!\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        if args.skip_vae or vae is None:
            train_loss = train_epoch_no_vae(
                adapter, train_loader, optimizer, device, dtype, epoch
            )
        else:
            train_loss = train_epoch_with_vae(
                adapter, vae, train_loader, optimizer, device, dtype, epoch, vae_scaling_factor
            )
        
        # Evaluate (always with VAE if available, to measure true performance)
        val_metrics = evaluate(
            adapter, vae, val_loader, device, dtype,
            use_vae=(not args.skip_vae and vae is not None),
            vae_scaling_factor=vae_scaling_factor,
        )
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
                "trained_with_vae": not args.skip_vae,
            }
            torch.save(checkpoint, args.output_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
    
    # ================================================================
    # Final summary
    # ================================================================
    print(f"\n{'='*70}")
    print("  Training Complete!")
    print(f"{'='*70}")
    print(f"  Best validation MSE: {best_val_loss:.6f}")
    print(f"  Best validation RMSE: {best_val_loss**0.5:.4f}")
    print(f"  Trained with VAE: {not args.skip_vae}")
    print(f"  Saved to: {args.output_path}")
    print(f"{'='*70}\n")
    
    # Final evaluation with more detail
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(args.output_path, map_location=device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    
    final_metrics = evaluate(
        adapter, vae, val_loader, device, dtype,
        use_vae=(not args.skip_vae and vae is not None),
        vae_scaling_factor=vae_scaling_factor,
    )
    print("\nFinal Evaluation (through VAE):")
    print(f"  Total RMSE: {final_metrics['total_rmse']:.4f}")
    for name in ["density", "pressure", "velocity_x", "velocity_y"]:
        print(f"  {name}: RMSE = {final_metrics[f'{name}_rmse']:.4f}")


if __name__ == "__main__":
    main()
