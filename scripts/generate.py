#!/usr/bin/env python3
"""
Generation script for the fine-tuned Wan2.2-I2V model.

This script takes initial physics frames and generates future predictions.

Usage:
    python scripts/generate.py \
        --checkpoint checkpoints/best_model.pt \
        --input_data ./datasets/turbulent_radiative_layer_2D \
        --output_dir ./generated_predictions \
        --num_frames 16
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_wan22_model
from src.data import WellVideoDataset

# Optional visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_animation(
    frames: np.ndarray,
    field_idx: int = 0,
    field_name: str = "density",
    output_path: str = "animation.gif",
    fps: int = 10,
):
    """Create animation of predicted frames."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for animation")
        return
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    vmin = frames[:, field_idx].min()
    vmax = frames[:, field_idx].max()
    
    im = ax.imshow(frames[0, field_idx], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title(f"{field_name} - Frame 0")
    plt.colorbar(im, ax=ax)
    
    def animate(i):
        im.set_array(frames[i, field_idx])
        ax.set_title(f"{field_name} - Frame {i}")
        return [im]
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=1000//fps, blit=True
    )
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Saved animation to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate physics predictions with fine-tuned Wan2.2-I2V"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="./datasets",
        help="Path to input data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_predictions",
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to generate"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of sample to use as initial condition"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    parser.add_argument(
        "--create_animation",
        action="store_true",
        help="Create animation of predictions"
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=40,
        help="Number of diffusion inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for generation"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config["data"]["base_path"] = args.input_data
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Wan2.2-I2V Physics Generation")
    print("=" * 60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Frames to generate: {args.num_frames}")
    print(f"Inference steps: {args.inference_steps}")
    print("=" * 60)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nLoading model...")
    model = create_wan22_model(config)
    model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "channel_adapter" in checkpoint:
        model.channel_adapter.load_state_dict(checkpoint["channel_adapter"])
        if "spatial_encoder" in checkpoint:
            model.spatial_encoder.load_state_dict(checkpoint["spatial_encoder"])
        if "spatial_decoder" in checkpoint:
            model.spatial_decoder.load_state_dict(checkpoint["spatial_decoder"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    # Load dataset for initial conditions
    print("\nLoading dataset for initial conditions...")
    dataset = WellVideoDataset(
        base_path=args.input_data,
        dataset_name=config["data"]["dataset_name"],
        split="valid",
        n_steps_input=config["data"]["n_steps_input"],
        n_steps_output=config["data"]["n_steps_output"],
    )
    
    # Get initial condition
    sample = dataset[args.sample_idx]
    input_frames = sample["input_frames_normalized"].unsqueeze(0).to(device)
    
    print(f"Input shape: {input_frames.shape}")
    print(f"  - {config['data']['n_steps_input']} input frames")
    print(f"  - 4 physics channels (density, pressure, velocity_x, velocity_y)")
    print(f"  - Spatial size: {input_frames.shape[2]} x {input_frames.shape[3]}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    with torch.no_grad():
        predictions = model.generate(
            input_frames=input_frames,
            num_frames=args.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
        )
    
    print(f"Generated {predictions.shape[1]} frames")
    
    # Denormalize predictions
    predictions = dataset.denormalize(predictions)
    
    # Move to CPU and numpy
    predictions_np = predictions.cpu().numpy()[0]  # Remove batch dimension
    input_np = sample["input_frames"].numpy()
    target_np = sample["target_frames"].numpy()
    
    print(f"Prediction shape: {predictions_np.shape}")
    
    # Save predictions
    output_path = os.path.join(args.output_dir, f"prediction_sample_{args.sample_idx}.npz")
    np.savez(
        output_path,
        predictions=predictions_np,
        input_frames=input_np,
        target_frames=target_np,
    )
    print(f"\nSaved predictions to {output_path}")
    
    # Create visualizations
    if MATPLOTLIB_AVAILABLE:
        print("\nCreating visualizations...")
        
        field_names = ["density", "pressure", "velocity_x", "velocity_y"]
        
        # Create comparison figure
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        
        for field_idx, field_name in enumerate(field_names):
            # Last input frame
            axes[field_idx, 0].imshow(input_np[-1, :, :, field_idx], cmap='RdBu_r')
            axes[field_idx, 0].set_title('Input (last)' if field_idx == 0 else '')
            axes[field_idx, 0].set_ylabel(field_name)
            axes[field_idx, 0].axis('off')
            
            # First prediction
            axes[field_idx, 1].imshow(predictions_np[0, :, :, field_idx], cmap='RdBu_r')
            axes[field_idx, 1].set_title('Pred t=1' if field_idx == 0 else '')
            axes[field_idx, 1].axis('off')
            
            # Middle prediction
            mid_idx = len(predictions_np) // 2
            axes[field_idx, 2].imshow(predictions_np[mid_idx, :, :, field_idx], cmap='RdBu_r')
            axes[field_idx, 2].set_title(f'Pred t={mid_idx+1}' if field_idx == 0 else '')
            axes[field_idx, 2].axis('off')
            
            # Last prediction
            axes[field_idx, 3].imshow(predictions_np[-1, :, :, field_idx], cmap='RdBu_r')
            axes[field_idx, 3].set_title(f'Pred t={len(predictions_np)}' if field_idx == 0 else '')
            axes[field_idx, 3].axis('off')
        
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, f"visualization_sample_{args.sample_idx}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved visualization to {fig_path}")
        
        # Create animations if requested
        if args.create_animation:
            for field_idx, field_name in enumerate(field_names):
                anim_path = os.path.join(
                    args.output_dir, 
                    f"animation_{field_name}_sample_{args.sample_idx}.gif"
                )
                # Rearrange predictions for animation: (T, H, W, C) -> (T, C, H, W)
                pred_rearranged = np.transpose(predictions_np, (0, 3, 1, 2))
                create_animation(
                    pred_rearranged,
                    field_idx=field_idx,
                    field_name=field_name,
                    output_path=anim_path,
                )
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
