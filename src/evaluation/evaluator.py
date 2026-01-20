"""
Evaluator for physics simulation prediction models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from pathlib import Path
import json
import numpy as np
from einops import rearrange

from .metrics import PhysicsMetrics, compute_vrmse, compute_mse, compute_psnr

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class Evaluator:
    """
    Evaluator for the fine-tuned Wan2.2 model on physics data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        val_loader: DataLoader,
        device: str = "cuda",
        adapter_only: bool = False,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: The trained model
            config: Configuration dictionary
            val_loader: Validation data loader
            device: Computation device
            adapter_only: If True, use adapter-only prediction (no diffusion pipeline)
        """
        self.model = model
        self.config = config
        self.val_loader = val_loader
        self.device = device
        self.adapter_only = adapter_only
        
        # Setup metrics
        self.metrics = PhysicsMetrics(device=device)
        
        # Output directory
        self.output_dir = Path(config["evaluation"].get("prediction_dir", "./predictions"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def evaluate(
        self,
        num_samples: Optional[int] = None,
        save_predictions: bool = True,
    ) -> Dict[str, float]:
        """
        Run full evaluation.
        
        Args:
            num_samples: Maximum number of samples to evaluate
            save_predictions: Whether to save predictions to disk
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        self.metrics.reset()
        
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        max_samples = num_samples or self.config["evaluation"].get("num_samples", 100)
        samples_evaluated = 0
        
        print(f"Evaluating on {max_samples} samples...")
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            if samples_evaluated >= max_samples:
                break
            
            # Get data
            input_frames = batch["input_frames"].to(self.device)
            target_frames = batch["target_frames"].to(self.device)
            input_normalized = batch["input_frames_normalized"].to(self.device)
            
            # Generate predictions
            batch_size = input_frames.shape[0]
            num_output_frames = target_frames.shape[1]
            
            # Use model to predict future frames
            predictions = self._generate_predictions(
                input_normalized,
                num_frames=num_output_frames,
            )
            
            # Denormalize predictions
            predictions = self._denormalize(predictions, self.val_loader.dataset)
            
            # Update metrics
            self.metrics.update(predictions, target_frames)
            
            # Store for saving
            if save_predictions:
                all_predictions.append(predictions.cpu())
                all_targets.append(target_frames.cpu())
                all_inputs.append(input_frames.cpu())
            
            samples_evaluated += batch_size
        
        # Compute final metrics
        results = self.metrics.compute()
        
        # Add PSNR (normalized data)
        # We'll compute this separately with proper normalization
        
        # Save predictions
        if save_predictions and all_predictions:
            self._save_predictions(
                torch.cat(all_predictions, dim=0),
                torch.cat(all_targets, dim=0),
                torch.cat(all_inputs, dim=0),
            )
        
        # Save metrics
        self._save_metrics(results)
        
        return results
    
    def _generate_predictions(
        self,
        input_frames: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Generate predictions from the model.
        
        Args:
            input_frames: Input frames (B, T_in, H, W, C)
            num_frames: Number of frames to generate
            
        Returns:
            Predicted frames (B, num_frames, H, W, C)
        """
        if self.adapter_only:
            # Use adapter-only prediction (same as training pipeline)
            if hasattr(self.model, 'predict_adapter_only'):
                predictions = self.model.predict_adapter_only(
                    input_frames,
                    num_frames=num_frames,
                )
            else:
                raise ValueError("Model does not support adapter-only prediction")
        else:
            # Use full diffusion pipeline
            if hasattr(self.model, 'generate'):
                predictions = self.model.generate(
                    input_frames,
                    num_frames=num_frames,
                )
            else:
                # Fallback: use forward pass and extract prediction
                outputs = self.model(
                    input_frames=input_frames,
                    target_frames=input_frames,  # Dummy target
                )
                # This would need model-specific handling
                predictions = input_frames[:, -num_frames:]
        
        return predictions
    
    def _denormalize(
        self,
        predictions: torch.Tensor,
        dataset,
    ) -> torch.Tensor:
        """Denormalize predictions using dataset statistics."""
        if hasattr(dataset, 'denormalize'):
            return dataset.denormalize(predictions)
        return predictions
    
    def _save_predictions(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor,
    ):
        """Save predictions to disk."""
        save_path = self.output_dir / "predictions.pt"
        torch.save({
            "predictions": predictions,
            "targets": targets,
            "inputs": inputs,
        }, save_path)
        print(f"Saved predictions to {save_path}")
    
    def _save_metrics(self, metrics: Dict[str, float]):
        """Save metrics to JSON."""
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")
    
    def visualize_predictions(
        self,
        num_samples: int = 5,
        output_path: Optional[str] = None,
    ):
        """
        Create visualizations of predictions vs targets.
        
        Args:
            num_samples: Number of samples to visualize
            output_path: Path to save visualizations
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for visualization")
            return
        
        # Load saved predictions
        pred_path = self.output_dir / "predictions.pt"
        if not pred_path.exists():
            print("No predictions saved. Run evaluate() first.")
            return
        
        data = torch.load(pred_path)
        predictions = data["predictions"]
        targets = data["targets"]
        inputs = data["inputs"]
        
        field_names = ["density", "pressure", "velocity_x", "velocity_y"]
        
        for sample_idx in range(min(num_samples, predictions.shape[0])):
            fig, axes = plt.subplots(
                4, 6, figsize=(18, 12),
                gridspec_kw={'hspace': 0.3, 'wspace': 0.3}
            )
            
            for field_idx, field_name in enumerate(field_names):
                # Input (last frame)
                input_field = inputs[sample_idx, -1, :, :, field_idx].numpy()
                
                # Target (first output frame)
                target_field = targets[sample_idx, 0, :, :, field_idx].numpy()
                
                # Prediction (first output frame)
                pred_field = predictions[sample_idx, 0, :, :, field_idx].numpy()
                
                # Error
                error = np.abs(pred_field - target_field)
                
                # Determine color range
                vmin = min(input_field.min(), target_field.min(), pred_field.min())
                vmax = max(input_field.max(), target_field.max(), pred_field.max())
                
                # Plot input
                axes[field_idx, 0].imshow(input_field, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                axes[field_idx, 0].set_title(f'Input' if field_idx == 0 else '')
                axes[field_idx, 0].set_ylabel(field_name)
                axes[field_idx, 0].set_xticks([])
                axes[field_idx, 0].set_yticks([])
                
                # Plot target
                axes[field_idx, 1].imshow(target_field, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                axes[field_idx, 1].set_title(f'Target' if field_idx == 0 else '')
                axes[field_idx, 1].set_xticks([])
                axes[field_idx, 1].set_yticks([])
                
                # Plot prediction
                axes[field_idx, 2].imshow(pred_field, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                axes[field_idx, 2].set_title(f'Prediction' if field_idx == 0 else '')
                axes[field_idx, 2].set_xticks([])
                axes[field_idx, 2].set_yticks([])
                
                # Plot error
                im = axes[field_idx, 3].imshow(error, cmap='hot')
                axes[field_idx, 3].set_title(f'Abs. Error' if field_idx == 0 else '')
                axes[field_idx, 3].set_xticks([])
                axes[field_idx, 3].set_yticks([])
                plt.colorbar(im, ax=axes[field_idx, 3], fraction=0.046)
                
                # Temporal evolution: show multiple timesteps
                if targets.shape[1] > 1:
                    t_mid = targets.shape[1] // 2
                    
                    # Mid-time prediction
                    axes[field_idx, 4].imshow(
                        predictions[sample_idx, t_mid, :, :, field_idx].numpy(),
                        cmap='RdBu_r', vmin=vmin, vmax=vmax
                    )
                    axes[field_idx, 4].set_title(f'Pred t={t_mid}' if field_idx == 0 else '')
                    axes[field_idx, 4].set_xticks([])
                    axes[field_idx, 4].set_yticks([])
                    
                    # Final prediction
                    axes[field_idx, 5].imshow(
                        predictions[sample_idx, -1, :, :, field_idx].numpy(),
                        cmap='RdBu_r', vmin=vmin, vmax=vmax
                    )
                    axes[field_idx, 5].set_title(f'Pred t={targets.shape[1]-1}' if field_idx == 0 else '')
                    axes[field_idx, 5].set_xticks([])
                    axes[field_idx, 5].set_yticks([])
                else:
                    axes[field_idx, 4].axis('off')
                    axes[field_idx, 5].axis('off')
            
            plt.suptitle(f'Sample {sample_idx}', fontsize=14)
            
            # Save figure
            save_path = output_path or self.output_dir
            if isinstance(save_path, str):
                save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / f'visualization_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Saved visualizations to {save_path}")
    
    def compute_rollout_metrics(
        self,
        num_rollout_steps: int = 10,
        num_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Compute metrics for multi-step rollout.
        
        This evaluates how prediction quality degrades over time.
        
        Args:
            num_rollout_steps: Number of steps to roll out
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with per-step metrics
        """
        self.model.eval()
        
        per_step_errors = {
            'vrmse': [],
            'mse': [],
        }
        
        samples_evaluated = 0
        
        for batch in tqdm(self.val_loader, desc="Rollout evaluation"):
            if samples_evaluated >= num_samples:
                break
            
            input_frames = batch["input_frames_normalized"].to(self.device)
            target_frames = batch["target_frames_normalized"].to(self.device)
            
            # Ensure we have enough target frames
            T_target = target_frames.shape[1]
            if T_target < num_rollout_steps:
                continue
            
            # Initialize current input
            current_input = input_frames
            step_errors_vrmse = []
            step_errors_mse = []
            
            for step in range(min(num_rollout_steps, T_target)):
                # Predict next frame
                with torch.no_grad():
                    pred = self._generate_predictions(current_input, num_frames=1)
                
                # Get target frame
                target = target_frames[:, step:step+1]
                
                # Compute errors
                vrmse = compute_vrmse(pred, target).mean().item()
                mse = compute_mse(pred, target).item()
                
                step_errors_vrmse.append(vrmse)
                step_errors_mse.append(mse)
                
                # Update input (autoregressive)
                current_input = torch.cat([current_input[:, 1:], pred], dim=1)
            
            per_step_errors['vrmse'].append(step_errors_vrmse)
            per_step_errors['mse'].append(step_errors_mse)
            
            samples_evaluated += input_frames.shape[0]
        
        # Average over samples
        results = {
            'per_step_vrmse': np.mean(per_step_errors['vrmse'], axis=0).tolist(),
            'per_step_mse': np.mean(per_step_errors['mse'], axis=0).tolist(),
            'num_steps': num_rollout_steps,
        }
        
        # Save rollout metrics
        rollout_path = self.output_dir / "rollout_metrics.json"
        with open(rollout_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results


def run_evaluation(
    model_path: str,
    config_path: str,
    output_dir: str = "./evaluation_results",
    num_samples: int = 100,
    visualize: bool = True,
) -> Dict[str, float]:
    """
    Run full evaluation pipeline.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        output_dir: Output directory for results
        num_samples: Number of samples to evaluate
        visualize: Whether to create visualizations
        
    Returns:
        Dictionary of evaluation metrics
    """
    import yaml
    from ..models import create_wan22_model
    from ..data import create_dataloaders
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override output directory
    config["evaluation"]["prediction_dir"] = output_dir
    config["evaluation"]["num_samples"] = num_samples
    
    # Create model
    model = create_wan22_model(config)
    model.load_checkpoint(model_path)
    model.eval()
    
    # Create data loaders
    _, val_loader = create_dataloaders(config)
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        config=config,
        val_loader=val_loader,
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(
        num_samples=num_samples,
        save_predictions=True,
    )
    
    # Create visualizations
    if visualize:
        evaluator.visualize_predictions(num_samples=5)
    
    # Compute rollout metrics
    rollout_metrics = evaluator.compute_rollout_metrics(
        num_rollout_steps=10,
        num_samples=min(20, num_samples),
    )
    
    metrics.update(rollout_metrics)
    
    return metrics
