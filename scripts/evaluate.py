#!/usr/bin/env python3
"""
Evaluation script for the fine-tuned Wan2.2-I2V model on physics data.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --config configs/default.yaml \
        --output_dir ./evaluation_results
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_wan22_model
from src.data import create_dataloaders
from src.evaluation import Evaluator, PhysicsMetrics
from src.evaluation.metrics import compute_vrmse, compute_mse


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def compute_baselines(val_loader, num_samples: int = 100, device: str = "cuda"):
    """
    Compute baseline metrics for comparison.
    
    Baselines:
    1. Repeat Last Frame: Copy the last input frame as prediction
    2. Mean Prediction: Predict the mean value (VRMSE = 1.0 by definition)
    
    Args:
        val_loader: Validation data loader
        num_samples: Number of samples to evaluate
        device: Computation device
        
    Returns:
        Dictionary with baseline metrics
    """
    import numpy as np
    from tqdm import tqdm
    
    repeat_last_mse = []
    repeat_last_vrmse = []
    
    samples_evaluated = 0
    field_names = ["density", "pressure", "velocity_x", "velocity_y"]
    
    print("\nComputing baselines...")
    
    for batch in tqdm(val_loader, desc="Baseline evaluation"):
        if samples_evaluated >= num_samples:
            break
        
        input_frames = batch["input_frames"]  # (B, T_in, H, W, C)
        target_frames = batch["target_frames"]  # (B, T_out, H, W, C)
        
        batch_size = input_frames.shape[0]
        T_out = target_frames.shape[1]
        
        # Baseline: Repeat Last Frame
        last_frame = input_frames[:, -1:, :, :, :]  # (B, 1, H, W, C)
        repeat_pred = last_frame.expand(-1, T_out, -1, -1, -1)  # (B, T_out, H, W, C)
        
        # Compute metrics
        mse = compute_mse(repeat_pred, target_frames).item()
        vrmse = compute_vrmse(repeat_pred, target_frames, field_dim=-1)
        
        repeat_last_mse.append(mse)
        repeat_last_vrmse.append(vrmse.cpu().numpy())
        
        samples_evaluated += batch_size
    
    # Aggregate results
    vrmse_per_field = np.mean(repeat_last_vrmse, axis=0)
    
    baselines = {
        "repeat_last_frame": {
            "mse": float(np.mean(repeat_last_mse)),
            "vrmse_mean": float(np.mean(vrmse_per_field)),
            "vrmse_per_field": {
                name: float(v) for name, v in zip(field_names, vrmse_per_field)
            }
        },
        # Mean prediction baseline: VRMSE = 1.0 by definition
        "mean_prediction": {
            "vrmse_mean": 1.0,
            "note": "VRMSE=1.0 by definition (error equals variance)"
        }
    }
    
    return baselines


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Wan2.2-I2V on physics dataset"
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
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations"
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=10,
        help="Number of rollout steps for temporal evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to dataset (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override data path if specified
    if args.data_path is not None:
        config["data"]["base_path"] = args.data_path
    
    # Update evaluation config
    config["evaluation"]["prediction_dir"] = args.output_dir
    config["evaluation"]["num_samples"] = args.num_samples
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Wan2.2-I2V Physics Evaluation")
    print("=" * 60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples: {args.num_samples}")
    print("=" * 60)
    
    # Create model
    print("\nLoading model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model = create_wan22_model(config)
    model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if "channel_adapter" in checkpoint and "model_state_dict" not in checkpoint:
        # Adapter-only checkpoint (from *_adapters.pt files)
        # Use the model's built-in load method
        model.load_checkpoint(args.checkpoint)
    elif "model_state_dict" in checkpoint:
        # Full checkpoint - extract only the trainable adapter and LoRA weights
        print("Full checkpoint detected - extracting trainable weights only...")
        full_state = checkpoint["model_state_dict"]
        
        # Extract adapter weights
        adapter_state = {}
        spatial_encoder_state = {}
        spatial_decoder_state = {}
        lora_weights = {}
        
        for key, value in full_state.items():
            if key.startswith("channel_adapter."):
                # Remove prefix for loading into the module directly
                new_key = key[len("channel_adapter."):]
                adapter_state[new_key] = value
            elif key.startswith("spatial_encoder."):
                new_key = key[len("spatial_encoder."):]
                spatial_encoder_state[new_key] = value
            elif key.startswith("spatial_decoder."):
                new_key = key[len("spatial_decoder."):]
                spatial_decoder_state[new_key] = value
            elif "lora_" in key and "transformer" in key:
                # LoRA weights in transformer
                lora_weights[key] = value
        
        # Load adapter weights
        if adapter_state:
            model.channel_adapter.load_state_dict(adapter_state)
            print(f"  Loaded channel adapter ({len(adapter_state)} tensors)")
        
        if spatial_encoder_state:
            model.spatial_encoder.load_state_dict(spatial_encoder_state)
            print(f"  Loaded spatial encoder ({len(spatial_encoder_state)} tensors)")
        
        if spatial_decoder_state:
            model.spatial_decoder.load_state_dict(spatial_decoder_state)
            print(f"  Loaded spatial decoder ({len(spatial_decoder_state)} tensors)")
        
        # Load LoRA weights if present
        if lora_weights and model.transformer is not None:
            # Try to load LoRA weights into the transformer
            transformer_state = model.transformer.state_dict()
            loaded_lora = 0
            for key, value in lora_weights.items():
                # Transform key from full model to transformer-only
                if key.startswith("transformer."):
                    transformer_key = key[len("transformer."):]
                    if transformer_key in transformer_state:
                        transformer_state[transformer_key] = value
                        loaded_lora += 1
            
            if loaded_lora > 0:
                model.transformer.load_state_dict(transformer_state)
                print(f"  Loaded LoRA weights ({loaded_lora} tensors)")
    else:
        # Try to use the adapter checkpoint file if it exists
        adapter_path = args.checkpoint.replace(".pt", "_adapters.pt")
        if os.path.exists(adapter_path):
            print(f"Using adapter checkpoint: {adapter_path}")
            model.load_checkpoint(adapter_path)
        else:
            raise ValueError(
                f"Unknown checkpoint format. Expected either:\n"
                f"  1. Adapter checkpoint (*_adapters.pt)\n"
                f"  2. Full checkpoint with 'model_state_dict' key\n"
                f"Hint: Try using the adapters checkpoint: {adapter_path}"
            )
    
    model.eval()
    print("Model loaded successfully")
    
    # Create data loader
    print("\nLoading validation data...")
    _, val_loader = create_dataloaders(config)
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        config=config,
        val_loader=val_loader,
        device=str(device),
    )
    
    # Compute baseline metrics first
    print("\n" + "=" * 60)
    print("Computing baseline metrics...")
    print("=" * 60)
    
    baselines = compute_baselines(
        val_loader, 
        num_samples=args.num_samples,
        device=str(device)
    )
    
    # Run model evaluation
    print("\n" + "=" * 60)
    print("Running model evaluation...")
    print("=" * 60)
    
    metrics = evaluator.evaluate(
        num_samples=args.num_samples,
        save_predictions=True,
    )
    
    # Print comparative results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    # Baseline results
    print("\n--- BASELINES ---")
    print("\n1. Repeat Last Frame:")
    baseline_rl = baselines["repeat_last_frame"]
    print(f"   MSE:  {baseline_rl['mse']:.6f}")
    print(f"   VRMSE (mean): {baseline_rl['vrmse_mean']:.4f}")
    print("   VRMSE per field:")
    for field, value in baseline_rl["vrmse_per_field"].items():
        print(f"      {field}: {value:.4f}")
    
    print("\n2. Mean Prediction:")
    print(f"   VRMSE: 1.0000 (by definition)")
    
    # Model results
    print("\n--- YOUR MODEL ---")
    print("\nPer-field VRMSE:")
    for key, value in metrics.items():
        if key.startswith("vrmse/") and key != "vrmse/mean":
            print(f"  {key}: {value:.4f}")
    
    model_vrmse_mean = metrics.get('vrmse/mean', float('nan'))
    print(f"\nMean VRMSE: {model_vrmse_mean:.4f}")
    
    print("\nPer-field MSE:")
    for key, value in metrics.items():
        if key.startswith("mse/") and key != "mse/mean":
            print(f"  {key}: {value:.6f}")
    
    model_mse_mean = metrics.get('mse/mean', float('nan'))
    print(f"\nMean MSE: {model_mse_mean:.6f}")
    
    # Comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    baseline_mse = baseline_rl['mse']
    baseline_vrmse = baseline_rl['vrmse_mean']
    
    mse_improvement = (baseline_mse - model_mse_mean) / baseline_mse * 100 if baseline_mse > 0 else 0
    vrmse_improvement = (baseline_vrmse - model_vrmse_mean) / baseline_vrmse * 100 if baseline_vrmse > 0 else 0
    
    print(f"\nMSE:   Model={model_mse_mean:.6f} vs Baseline={baseline_mse:.6f}")
    if model_mse_mean < baseline_mse:
        print(f"       Model is {mse_improvement:.1f}% better than baseline")
    else:
        print(f"       Model is {-mse_improvement:.1f}% worse than baseline")
    
    print(f"\nVRMSE: Model={model_vrmse_mean:.4f} vs Baseline={baseline_vrmse:.4f}")
    if model_vrmse_mean < baseline_vrmse:
        print(f"       Model is {vrmse_improvement:.1f}% better than baseline")
    else:
        print(f"       Model is {-vrmse_improvement:.1f}% worse than baseline")
    
    if model_vrmse_mean < 1.0:
        print(f"\n       Model VRMSE < 1.0: Better than mean prediction")
    else:
        print(f"\n       Model VRMSE >= 1.0: Not better than mean prediction")
    
    # Create visualizations
    if args.visualize:
        print("\nCreating visualizations...")
        evaluator.visualize_predictions(num_samples=5)
    
    # Compute rollout metrics
    print("\nComputing rollout metrics...")
    rollout_metrics = evaluator.compute_rollout_metrics(
        num_rollout_steps=args.rollout_steps,
        num_samples=min(20, args.num_samples),
    )
    
    print("\nRollout VRMSE per step:")
    for i, vrmse in enumerate(rollout_metrics["per_step_vrmse"]):
        print(f"  Step {i+1}: {vrmse:.4f}")
    
    # Save all results
    all_results = {
        "model_metrics": metrics,
        "baselines": baselines,
        "rollout_metrics": rollout_metrics,
        "comparison": {
            "model_mse": model_mse_mean,
            "baseline_mse": baseline_mse,
            "mse_improvement_pct": mse_improvement,
            "model_vrmse": model_vrmse_mean,
            "baseline_vrmse": baseline_vrmse,
            "vrmse_improvement_pct": vrmse_improvement,
            "better_than_repeat_last": model_mse_mean < baseline_mse,
            "better_than_mean_pred": model_vrmse_mean < 1.0,
        },
        "config": {
            "checkpoint": args.checkpoint,
            "num_samples": args.num_samples,
            "rollout_steps": args.rollout_steps,
        }
    }
    
    results_path = os.path.join(args.output_dir, "full_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
