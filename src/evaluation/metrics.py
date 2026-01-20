"""
Evaluation metrics for physics simulation prediction.

Includes:
- VRMSE: Variance-scaled Root Mean Squared Error
- MSE: Mean Squared Error
- PSNR: Peak Signal-to-Noise Ratio
- Per-field metrics for physics interpretability
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def compute_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Mean Squared Error.
    
    Args:
        prediction: Predicted tensor
        target: Target tensor
        reduction: Reduction method ("mean", "sum", "none")
        
    Returns:
        MSE value(s)
    """
    mse = (prediction - target).pow(2)
    
    if reduction == "mean":
        return mse.mean()
    elif reduction == "sum":
        return mse.sum()
    elif reduction == "none":
        return mse
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Root Mean Squared Error.
    
    Args:
        prediction: Predicted tensor
        target: Target tensor
        reduction: Reduction method
        
    Returns:
        RMSE value(s)
    """
    mse = compute_mse(prediction, target, reduction="none")
    
    if reduction == "mean":
        return mse.mean().sqrt()
    elif reduction == "sum":
        return mse.sum().sqrt()
    elif reduction == "none":
        return mse.sqrt()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_vrmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    field_dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Variance-scaled Root Mean Squared Error (VRMSE).
    
    This is the primary metric used in The Well benchmark.
    VRMSE normalizes the RMSE by the variance of each field,
    making it comparable across fields with different scales.
    
    Args:
        prediction: Predicted tensor (..., F) where F is number of fields
        target: Target tensor (..., F)
        field_dim: Dimension containing the fields
        eps: Small value to prevent division by zero
        
    Returns:
        VRMSE per field (F,)
    """
    # Compute squared error
    squared_error = (prediction - target).pow(2)
    
    # Compute MSE per field
    # Average over all dimensions except the field dimension
    dims_to_reduce = list(range(squared_error.dim()))
    dims_to_reduce.remove(field_dim if field_dim >= 0 else squared_error.dim() + field_dim)
    
    mse_per_field = squared_error.mean(dim=dims_to_reduce)
    
    # Compute variance of target per field
    var_per_field = target.var(dim=dims_to_reduce)
    var_per_field = torch.clamp(var_per_field, min=eps)
    
    # VRMSE = sqrt(MSE / Var)
    vrmse = (mse_per_field / var_per_field).sqrt()
    
    return vrmse


def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        prediction: Predicted tensor
        target: Target tensor
        max_val: Maximum possible value
        eps: Small value to prevent log(0)
        
    Returns:
        PSNR value in dB
    """
    mse = compute_mse(prediction, target, reduction="mean")
    psnr = 10 * torch.log10(max_val ** 2 / (mse + eps))
    return psnr


def compute_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        prediction: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        window_size: Size of the Gaussian window
        sigma: Standard deviation of Gaussian window
        
    Returns:
        SSIM value
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    gauss = torch.exp(
        -torch.arange(window_size).float().sub(window_size // 2).pow(2) / (2 * sigma ** 2)
    )
    gauss = gauss / gauss.sum()
    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(prediction.shape[1], 1, window_size, window_size)
    window = window.to(prediction.device, prediction.dtype)
    
    # Compute means
    mu_x = F.conv2d(prediction, window, padding=window_size // 2, groups=prediction.shape[1])
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=target.shape[1])
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    # Compute variances
    sigma_x_sq = F.conv2d(prediction ** 2, window, padding=window_size // 2, groups=prediction.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=target.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, window, padding=window_size // 2, groups=prediction.shape[1]) - mu_xy
    
    # SSIM formula
    ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    
    return ssim.mean()


class PhysicsMetrics:
    """
    Collection of metrics for evaluating physics simulation predictions.
    
    Handles the specific structure of turbulent_radiative_layer_2D:
    - density (scalar field)
    - pressure (scalar field)
    - velocity_x (vector field component)
    - velocity_y (vector field component)
    """
    
    def __init__(
        self,
        field_names: List[str] = None,
        device: str = "cuda",
    ):
        """
        Args:
            field_names: Names of the physics fields
            device: Computation device
        """
        if field_names is None:
            field_names = ["density", "pressure", "velocity_x", "velocity_y"]
        
        self.field_names = field_names
        self.device = device
        
        # Running statistics
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.mse_sum = {name: 0.0 for name in self.field_names}
        self.vrmse_sum = {name: 0.0 for name in self.field_names}
        self.n_samples = 0
    
    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        Update metrics with a new batch.
        
        Args:
            prediction: Predicted fields (..., F) or (..., H, W, F)
            target: Target fields (..., F) or (..., H, W, F)
        """
        # Ensure on correct device
        prediction = prediction.to(self.device)
        target = target.to(self.device)
        
        # Get batch size
        batch_size = prediction.shape[0]
        
        # Compute per-field metrics
        vrmse = compute_vrmse(prediction, target, field_dim=-1)
        mse = compute_mse(prediction, target, reduction="none")
        
        # Average MSE over spatial/temporal dimensions, keep field dimension
        while mse.dim() > 1:
            mse = mse.mean(dim=0)
        
        # Accumulate
        for i, name in enumerate(self.field_names):
            self.vrmse_sum[name] += vrmse[i].item() * batch_size
            self.mse_sum[name] += mse[i].item() * batch_size
        
        self.n_samples += batch_size
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of metric values
        """
        if self.n_samples == 0:
            return {}
        
        results = {}
        
        # Per-field metrics
        for name in self.field_names:
            results[f"vrmse/{name}"] = self.vrmse_sum[name] / self.n_samples
            results[f"mse/{name}"] = self.mse_sum[name] / self.n_samples
        
        # Averaged metrics
        results["vrmse/mean"] = sum(self.vrmse_sum.values()) / (len(self.field_names) * self.n_samples)
        results["mse/mean"] = sum(self.mse_sum.values()) / (len(self.field_names) * self.n_samples)
        
        # Separate metrics for scalar and vector fields
        scalar_vrmse = (self.vrmse_sum["density"] + self.vrmse_sum["pressure"]) / (2 * self.n_samples)
        vector_vrmse = (self.vrmse_sum["velocity_x"] + self.vrmse_sum["velocity_y"]) / (2 * self.n_samples)
        
        results["vrmse/scalar_fields"] = scalar_vrmse
        results["vrmse/vector_fields"] = vector_vrmse
        
        return results


def evaluate_trajectory(
    model,
    initial_frames: torch.Tensor,
    target_trajectory: torch.Tensor,
    rollout_steps: int,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate model on a full trajectory rollout.
    
    This tests the model's ability to predict multiple steps ahead,
    which is crucial for physics simulation.
    
    Args:
        model: The prediction model
        initial_frames: Initial condition frames (B, T_init, H, W, F)
        target_trajectory: Full target trajectory (B, T_total, H, W, F)
        rollout_steps: Number of steps to roll out
        
    Returns:
        Dictionary with predictions and per-step metrics
    """
    B, T_init, H, W, F = initial_frames.shape
    
    predictions = []
    current_input = initial_frames
    
    for step in range(rollout_steps):
        # Predict next frame(s)
        with torch.no_grad():
            pred = model.generate(
                current_input,
                num_frames=1,
            )
        predictions.append(pred)
        
        # Update input for next step (sliding window)
        current_input = torch.cat([
            current_input[:, 1:],
            pred,
        ], dim=1)
    
    # Stack predictions
    all_predictions = torch.cat(predictions, dim=1)
    
    # Compare with targets
    target_frames = target_trajectory[:, T_init:T_init + rollout_steps]
    
    # Compute metrics per timestep
    per_step_vrmse = []
    for t in range(rollout_steps):
        vrmse = compute_vrmse(all_predictions[:, t], target_frames[:, t])
        per_step_vrmse.append(vrmse)
    
    per_step_vrmse = torch.stack(per_step_vrmse, dim=0)  # (T, F)
    
    return {
        "predictions": all_predictions,
        "targets": target_frames,
        "per_step_vrmse": per_step_vrmse,
        "mean_vrmse": per_step_vrmse.mean(dim=0),
        "final_step_vrmse": per_step_vrmse[-1],
    }
