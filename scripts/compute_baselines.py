#!/usr/bin/env python3
"""
Compute baseline metrics for physics prediction.

Baselines:
1. Repeat Last Frame: Simply copy the last input frame as prediction
2. Mean Baseline: Predict the mean of the training data
3. Zero Change: Predict that nothing changes (same as repeat last)

These baselines help interpret whether learned models are actually useful.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import WellVideoDataset
from src.evaluation.metrics import compute_vrmse, compute_mse


def compute_baselines(
    data_path: str = "./datasets/datasets",
    dataset_name: str = "turbulent_radiative_layer_2D",
    n_steps_input: int = 4,
    n_steps_output: int = 8,
    num_samples: int = 100,
):
    """
    Compute baseline metrics.
    
    Args:
        data_path: Path to dataset
        dataset_name: Name of the dataset
        n_steps_input: Number of input timesteps
        n_steps_output: Number of output timesteps
        num_samples: Number of samples to evaluate
    """
    print("=" * 60)
    print("Computing Baseline Metrics")
    print("=" * 60)
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = WellVideoDataset(
        base_path=data_path,
        dataset_name=dataset_name,
        split="valid",
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        use_normalization=False,
        compute_stats=True,
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Input steps: {n_steps_input}, Output steps: {n_steps_output}")
    
    # Metrics accumulators
    repeat_last_mse = []
    repeat_last_vrmse = []
    mean_pred_mse = []
    mean_pred_vrmse = []
    
    # Compute mean of training data
    print("\nComputing training data statistics...")
    train_dataset = WellVideoDataset(
        base_path=data_path,
        dataset_name=dataset_name,
        split="train",
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        use_normalization=False,
        compute_stats=True,
    )
    
    # Compute global mean from training data
    train_samples = []
    for i in range(min(500, len(train_dataset))):
        sample = train_dataset[i]
        train_samples.append(sample["input_frames"])
    train_samples = torch.stack(train_samples)
    global_mean = train_samples.mean(dim=(0, 1, 2, 3))  # Mean per channel
    print(f"Global mean per field: {global_mean.numpy()}")
    
    # Evaluate baselines
    print(f"\nEvaluating on {min(num_samples, len(val_dataset))} samples...")
    
    for i in tqdm(range(min(num_samples, len(val_dataset)))):
        sample = val_dataset[i]
        
        input_frames = sample["input_frames"]  # (T_in, H, W, C)
        target_frames = sample["target_frames"]  # (T_out, H, W, C)
        
        T_out = target_frames.shape[0]
        
        # Baseline 1: Repeat Last Frame
        last_frame = input_frames[-1]  # (H, W, C)
        repeat_pred = last_frame.unsqueeze(0).expand(T_out, -1, -1, -1)  # (T_out, H, W, C)
        
        mse = compute_mse(repeat_pred, target_frames).item()
        vrmse = compute_vrmse(repeat_pred, target_frames, field_dim=-1)
        
        repeat_last_mse.append(mse)
        repeat_last_vrmse.append(vrmse.numpy())
        
        # Baseline 2: Predict Global Mean
        mean_pred = global_mean.view(1, 1, 1, -1).expand(T_out, target_frames.shape[1], target_frames.shape[2], -1)
        
        mse = compute_mse(mean_pred, target_frames).item()
        vrmse = compute_vrmse(mean_pred, target_frames, field_dim=-1)
        
        mean_pred_mse.append(mse)
        mean_pred_vrmse.append(vrmse.numpy())
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    
    field_names = ["density", "pressure", "velocity_x", "velocity_y"]
    
    print("\n1. REPEAT LAST FRAME BASELINE")
    print("-" * 40)
    print(f"   MSE: {np.mean(repeat_last_mse):.6f}")
    print(f"   VRMSE (mean): {np.mean(repeat_last_vrmse):.4f}")
    print("   VRMSE per field:")
    vrmse_per_field = np.mean(repeat_last_vrmse, axis=0)
    for name, v in zip(field_names, vrmse_per_field):
        print(f"      {name}: {v:.4f}")
    
    print("\n2. PREDICT GLOBAL MEAN BASELINE")
    print("-" * 40)
    print(f"   MSE: {np.mean(mean_pred_mse):.6f}")
    print(f"   VRMSE (mean): {np.mean(mean_pred_vrmse):.4f}")
    print("   VRMSE per field:")
    vrmse_per_field = np.mean(mean_pred_vrmse, axis=0)
    for name, v in zip(field_names, vrmse_per_field):
        print(f"      {name}: {v:.4f}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
- VRMSE = 1.0 means prediction error equals data variance (same as mean prediction)
- VRMSE < 1.0 means better than predicting the mean
- VRMSE > 1.0 means worse than predicting the mean

For a model to be useful, it should achieve:
- Lower MSE than 'Repeat Last Frame' baseline
- VRMSE < 1.0 (better than mean prediction)

The 'Repeat Last Frame' baseline is often surprisingly strong for 
short-term predictions in physics simulations.
""")
    
    # Return results for programmatic use
    return {
        "repeat_last": {
            "mse": np.mean(repeat_last_mse),
            "vrmse_mean": np.mean(repeat_last_vrmse),
            "vrmse_per_field": np.mean(repeat_last_vrmse, axis=0),
        },
        "mean_prediction": {
            "mse": np.mean(mean_pred_mse),
            "vrmse_mean": np.mean(mean_pred_vrmse),
            "vrmse_per_field": np.mean(mean_pred_vrmse, axis=0),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compute baseline metrics")
    parser.add_argument("--data_path", type=str, default="./datasets/datasets")
    parser.add_argument("--dataset", type=str, default="turbulent_radiative_layer_2D")
    parser.add_argument("--n_steps_input", type=int, default=4)
    parser.add_argument("--n_steps_output", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=100)
    
    args = parser.parse_args()
    
    compute_baselines(
        data_path=args.data_path,
        dataset_name=args.dataset,
        n_steps_input=args.n_steps_input,
        n_steps_output=args.n_steps_output,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
