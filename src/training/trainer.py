"""
Main trainer class for Wan2.2 fine-tuning on physics data.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt

from .optimizer import create_optimizer, create_scheduler, WarmupCosineScheduler
from ..evaluation.metrics import compute_vrmse, compute_mse
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    wrap_model_fsdp,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    reduce_loss,
    print_rank0,
    GradientAccumulator,
)

# Optional imports for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Trainer:
    """
    Trainer for fine-tuning Wan2.2 on physics simulation data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: The Wan2.2 model wrapper
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup distributed training
        self.rank, self.world_size, self.device = setup_distributed(
            backend=config["distributed"].get("backend", "nccl")
        )
        
        # Setup model
        self.model = model
        self.model.to(self.device)
        
        # Load pretrained weights
        if hasattr(self.model, 'load_pretrained'):
            self.model.load_pretrained()
        
        # Apply freeze settings for two-stage training
        train_config = config["training"]
        if train_config.get("freeze_adapters", False):
            if hasattr(self.model, 'freeze_adapters'):
                self.model.freeze_adapters()
        if train_config.get("freeze_temporal_predictor", False):
            if hasattr(self.model, 'freeze_temporal_predictor'):
                self.model.freeze_temporal_predictor()
        
        # Set output bounds on channel adapter from dataset statistics
        if hasattr(train_loader.dataset, 'get_field_bounds'):
            field_bounds = train_loader.dataset.get_field_bounds()
            if field_bounds is not None:
                if hasattr(self.model, 'channel_adapter') and hasattr(self.model.channel_adapter, 'set_output_bounds'):
                    self.model.channel_adapter.set_output_bounds(field_bounds.to(self.device))
                    print_rank0(f"Set output bounds on channel adapter: {field_bounds.tolist()}")
        
        # Zero-initialize decoder output for residual prediction mode
        if train_config.get("use_residual_prediction", False):
            if hasattr(self.model, 'channel_adapter') and hasattr(self.model.channel_adapter, 'decoder'):
                decoder = self.model.channel_adapter.decoder
                if hasattr(decoder, 'zero_init_output'):
                    decoder.zero_init_output()
                    print_rank0("Initialized decoder to output zeros (residual mode)")
        
        # Wrap model for distributed training
        if self.world_size > 1:
            if config["distributed"].get("use_fsdp", False):
                self.model = wrap_model_fsdp(self.model, config["distributed"])
            else:
                self.model = wrap_model_ddp(
                    self.model, 
                    self.device,
                    find_unused_parameters=True,
                )
        
        # Get trainable parameters
        if hasattr(self.model, 'module'):
            trainable_params = self.model.module.get_trainable_parameters()
        else:
            trainable_params = self.model.get_trainable_parameters()
        
        # Setup optimizer
        self.optimizer = create_optimizer(trainable_params, config)
        
        # Calculate total training steps
        train_config = config["training"]
        steps_per_epoch = len(train_loader) // train_config["gradient_accumulation_steps"]
        
        if train_config.get("max_steps"):
            self.total_steps = train_config["max_steps"]
            self.num_epochs = self.total_steps // steps_per_epoch + 1
        else:
            self.num_epochs = train_config["num_epochs"]
            self.total_steps = steps_per_epoch * self.num_epochs
        
        # Setup scheduler
        self.scheduler = create_scheduler(self.optimizer, config, self.total_steps)
        
        # Gradient accumulation
        self.gradient_accumulator = GradientAccumulator(
            train_config["gradient_accumulation_steps"],
            self.model,
        )
        
        # Mixed precision
        self.scaler = None
        if train_config["mixed_precision"] == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Checkpointing
        self.checkpoint_dir = Path(train_config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Resume from checkpoint if specified
        if train_config.get("resume_from"):
            self.load_checkpoint(train_config["resume_from"])
    
    def setup_logging(self):
        """Setup logging utilities."""
        self.use_wandb = False
        self.use_tensorboard = False
        self.tb_writer = None
        
        if not is_main_process(self.rank):
            return
        
        logging_config = self.config.get("logging", {})
        
        # WandB
        if logging_config.get("use_wandb", False) and WANDB_AVAILABLE:
            wandb.init(
                project=logging_config.get("wandb_project", "wan22-well"),
                entity=logging_config.get("wandb_entity"),
                config=self.config,
            )
            self.use_wandb = True
        
        # TensorBoard
        if logging_config.get("use_tensorboard", False) and TENSORBOARD_AVAILABLE:
            tb_dir = logging_config.get("tensorboard_dir", "./logs")
            self.tb_writer = SummaryWriter(tb_dir)
            self.use_tensorboard = True
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all configured backends."""
        if not is_main_process(self.rank):
            return
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        if self.use_tensorboard and self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
    
    def train(self):
        """Main training loop."""
        train_config = self.config["training"]
        
        print_rank0(f"Starting training for {self.num_epochs} epochs ({self.total_steps} steps)")
        print_rank0(f"Batch size: {train_config['batch_size']} x {self.world_size} GPUs")
        print_rank0(f"Gradient accumulation: {train_config['gradient_accumulation_steps']}")
        
        # Run initial evaluation at step 0 (before any training)
        if self.val_loader and self.global_step == 0:
            print_rank0("\n=== Initial evaluation at step 0 ===")
            try:
                self.detailed_evaluation(num_samples=10)
            except Exception as e:
                print_rank0(f"Initial evaluation failed: {e}")
            self.model.train()
        
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            print_rank0(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validation
            if self.val_loader and (epoch + 1) % 1 == 0:  # Validate every epoch
                val_loss = self.validate()
                print_rank0(f"Epoch {epoch + 1}/{self.num_epochs} - Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
            
            # Save checkpoint
            if is_main_process(self.rank):
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")
            
            # Check if we've reached max steps
            if train_config.get("max_steps") and self.global_step >= train_config["max_steps"]:
                print_rank0("Reached max steps. Stopping training.")
                break
        
        # Final save
        if is_main_process(self.rank):
            self.save_checkpoint("final_model.pt")
        
        print_rank0("Training complete!")
        
        # Cleanup
        if self.use_wandb:
            wandb.finish()
        if self.tb_writer:
            self.tb_writer.close()
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        train_config = self.config["training"]
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}",
            disable=not is_main_process(self.rank),
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_frames = batch["input_frames_normalized"].to(self.device)
            target_frames = batch["target_frames_normalized"].to(self.device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(
                device_type="cuda",
                enabled=train_config["mixed_precision"] in ["fp16", "bf16"],
                dtype=torch.bfloat16 if train_config["mixed_precision"] == "bf16" else torch.float16,
            ):
                model = self.model.module if hasattr(self.model, 'module') else self.model
                
                # Check which training mode to use
                if train_config.get("use_native_i2v", False):
                    # Native I2V diffusion training: 1 frame -> 8 frames
                    # Extract first input frame as conditioning
                    cond_frame = input_frames[:, :1]  # (B, 1, H, W, C)
                    text_prompt = train_config.get(
                        "text_prompt",
                        "Top-down view of fluid dynamics simulation, evolving turbulence, scientific visualization, accurate physics (turbulent radiative layer 2d)"
                    )
                    outputs = model.forward_i2v_diffusion(
                        cond_frame=cond_frame,
                        target_frames=target_frames,
                        text_prompt=text_prompt,
                    )
                elif hasattr(model, 'use_temporal_predictor') and model.use_temporal_predictor:
                    # Temporal predictor mode
                    outputs = model.forward_with_temporal_predictor(
                        input_frames=input_frames,
                        target_frames=target_frames,
                    )
                else:
                    # Adapter-only mode
                    outputs = self.model(
                        input_frames=input_frames,
                        target_frames=target_frames,
                    )
                loss = outputs["loss"] / train_config["gradient_accumulation_steps"]
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation step
            self.gradient_accumulator.step()
            
            if self.gradient_accumulator.should_step():
                # Gradient clipping
                if train_config.get("max_grad_norm"):
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        train_config["max_grad_norm"],
                    )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % train_config["log_every"] == 0:
                    reduced_loss = reduce_loss(
                        loss * train_config["gradient_accumulation_steps"]
                    ).item()
                    
                    metrics = {
                        "train/loss": reduced_loss,
                        "train/adapter_loss": outputs.get("adapter_loss", torch.tensor(0)).item() if torch.is_tensor(outputs.get("adapter_loss", 0)) else outputs.get("adapter_loss", 0),
                        "train/spatial_loss": outputs.get("spatial_loss", torch.tensor(0)).item() if torch.is_tensor(outputs.get("spatial_loss", 0)) else outputs.get("spatial_loss", 0),
                        "train/vae_recon_loss": outputs.get("vae_recon_loss", torch.tensor(0)).item() if torch.is_tensor(outputs.get("vae_recon_loss", 0)) else outputs.get("vae_recon_loss", 0),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/epoch": self.epoch,
                    }
                    # Add temporal predictor specific losses if present
                    if "temporal_pred_loss" in outputs:
                        metrics["train/temporal_pred_loss"] = outputs["temporal_pred_loss"].item() if torch.is_tensor(outputs["temporal_pred_loss"]) else outputs["temporal_pred_loss"]
                    if "physics_pred_loss" in outputs:
                        metrics["train/physics_pred_loss"] = outputs["physics_pred_loss"].item() if torch.is_tensor(outputs["physics_pred_loss"]) else outputs["physics_pred_loss"]
                    # Legacy losses (for non-temporal predictor mode)
                    if "temporal_loss" in outputs:
                        metrics["train/temporal_loss"] = outputs["temporal_loss"].item() if torch.is_tensor(outputs["temporal_loss"]) else outputs["temporal_loss"]
                    if "cycle_loss" in outputs:
                        metrics["train/cycle_loss"] = outputs["cycle_loss"].item() if torch.is_tensor(outputs["cycle_loss"]) else outputs["cycle_loss"]
                    self.log_metrics(metrics, self.global_step)
                
                # Evaluation
                if (self.val_loader and 
                    train_config.get("eval_every") and 
                    self.global_step % train_config["eval_every"] == 0):
                    val_loss = self.validate()
                    self.log_metrics({"val/loss": val_loss}, self.global_step)
                    self.model.train()
                
                # Detailed evaluation with baseline comparison (every 1000 steps)
                if (self.val_loader and 
                    train_config.get("detailed_eval_every", 1000) and 
                    self.global_step % train_config.get("detailed_eval_every", 1000) == 0):
                    try:
                        # Use fewer samples and steps for fast evaluation during training
                        self.detailed_evaluation(num_samples=10)
                    except Exception as e:
                        print_rank0(f"Detailed evaluation failed: {e}")
                    self.model.train()
                
                # Checkpointing
                if (train_config.get("checkpoint_every") and
                    self.global_step % train_config["checkpoint_every"] == 0):
                    if is_main_process(self.rank):
                        self.save_checkpoint(f"step_{self.global_step}.pt")
            
            # Update progress bar
            total_loss += loss.item() * train_config["gradient_accumulation_steps"]
            num_batches += 1
            progress_bar.set_postfix({
                "loss": total_loss / num_batches,
                "lr": self.optimizer.param_groups[0]["lr"],
            })
            
            # Check max steps
            if train_config.get("max_steps") and self.global_step >= train_config["max_steps"]:
                break
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(
            self.val_loader,
            desc="Validating",
            disable=not is_main_process(self.rank),
        ):
            input_frames = batch["input_frames_normalized"].to(self.device)
            target_frames = batch["target_frames_normalized"].to(self.device)
            
            with torch.amp.autocast(
                device_type="cuda",
                enabled=self.config["training"]["mixed_precision"] in ["fp16", "bf16"],
                dtype=torch.bfloat16 if self.config["training"]["mixed_precision"] == "bf16" else torch.float16,
            ):
                model = self.model.module if hasattr(self.model, 'module') else self.model
                train_config = self.config["training"]
                
                if train_config.get("use_native_i2v", False):
                    # Native I2V mode
                    cond_frame = input_frames[:, :1]
                    text_prompt = train_config.get("text_prompt", "Top-down view of fluid dynamics simulation, evolving turbulence, scientific visualization, accurate physics (turbulent radiative layer 2d)")
                    outputs = model.forward_i2v_diffusion(
                        cond_frame=cond_frame,
                        target_frames=target_frames,
                        text_prompt=text_prompt,
                    )
                elif hasattr(model, 'use_temporal_predictor') and model.use_temporal_predictor:
                    outputs = model.forward_with_temporal_predictor(
                        input_frames=input_frames,
                        target_frames=target_frames,
                    )
                else:
                    outputs = self.model(
                        input_frames=input_frames,
                        target_frames=target_frames,
                    )
            
            total_loss += outputs["loss"].item()
            num_batches += 1
        
        # Reduce loss across all processes
        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        avg_loss = reduce_loss(avg_loss_tensor).item()
        
        return avg_loss
    
    @torch.no_grad()
    def detailed_evaluation(self, num_samples: int = 50):
        """
        Run detailed evaluation with baseline comparison.
        Prints both model and baseline metrics without saving.
        All ranks run evaluation, only rank 0 prints.
        """
        self.model.eval()
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # All ranks will run evaluation, but only rank 0 prints
        should_print = is_main_process(self.rank)
        
        if should_print:
            print_rank0("\n" + "=" * 60)
            print_rank0(f"DETAILED EVALUATION @ Step {self.global_step}")
            print_rank0("=" * 60)
        
        # Accumulators
        model_vrmse_sum = None
        baseline_vrmse_sum = None
        model_mse_sum = 0.0
        baseline_mse_sum = 0.0
        
        # Store samples for visualization (only on main process)
        vis_samples = []  # List of (input, target, prediction, baseline)
        num_vis_samples = 3
        n_evaluated = 0
        
        field_names = ["density", "pressure", "velocity_x", "velocity_y"]
        
        for batch in self.val_loader:
            if n_evaluated >= num_samples:
                break
            
            # Get data (use unnormalized for proper VRMSE calculation)
            input_frames = batch["input_frames"].to(self.device)
            target_frames = batch["target_frames"].to(self.device)
            input_normalized = batch["input_frames_normalized"].to(self.device)
            
            B = input_frames.shape[0]
            T_out = target_frames.shape[1]
            
            # Model prediction
            with torch.amp.autocast(
                device_type="cuda",
                enabled=self.config["training"]["mixed_precision"] in ["fp16", "bf16"],
                dtype=torch.bfloat16 if self.config["training"]["mixed_precision"] == "bf16" else torch.float16,
            ):
                train_config = self.config["training"]
                eval_config = self.config.get("evaluation", {})
                
                if train_config.get("use_native_i2v", False):
                    # Native I2V mode - use first frame as conditioning
                    cond_frame = input_normalized[:, :1]
                    text_prompt = train_config.get("text_prompt", "Top-down view of fluid dynamics simulation, evolving turbulence, scientific visualization, accurate physics (turbulent radiative layer 2d)")
                    
                    # Get reference frame for residual mode
                    reference = None
                    if train_config.get("use_residual_prediction", False):
                        reference = batch.get("reference_normalized")
                        if reference is not None:
                            reference = reference.to(self.device)
                    
                    predictions = model.predict_i2v_diffusion(
                        cond_frame=cond_frame,
                        num_frames=T_out,
                        num_inference_steps=eval_config.get("num_inference_steps", 10),  # Fewer steps for fast eval
                        guidance_scale=eval_config.get("guidance_scale", 5.0),
                        text_prompt=text_prompt,
                        reference_frame=reference,
                    )
                elif hasattr(model, 'predict_with_temporal_predictor'):
                    predictions = model.predict_with_temporal_predictor(
                        input_normalized,
                        num_frames=T_out,
                    )
                else:
                    # Fallback: just return input
                    predictions = input_frames[:, -T_out:]
            
            # Denormalize if needed
            if hasattr(self.val_loader.dataset, 'denormalize'):
                predictions = self.val_loader.dataset.denormalize(predictions)
            
            # Truncate targets/baseline to match predictions (VAE may produce fewer frames)
            T_pred = predictions.shape[1]
            target_frames_eval = target_frames[:, :T_pred]
            
            # Baseline: repeat last frame
            last_frame = input_frames[:, -1:, :, :, :]  # (B, 1, H, W, C)
            baseline_pred = last_frame.expand(-1, T_pred, -1, -1, -1)  # (B, T_pred, H, W, C)
            
            # Compute VRMSE (per field)
            model_vrmse = compute_vrmse(predictions, target_frames_eval, field_dim=-1)
            baseline_vrmse = compute_vrmse(baseline_pred, target_frames_eval, field_dim=-1)
            
            # Accumulate
            if model_vrmse_sum is None:
                model_vrmse_sum = model_vrmse * B
                baseline_vrmse_sum = baseline_vrmse * B
            else:
                model_vrmse_sum += model_vrmse * B
                baseline_vrmse_sum += baseline_vrmse * B
            
            model_mse_sum += compute_mse(predictions, target_frames_eval).item() * B
            baseline_mse_sum += compute_mse(baseline_pred, target_frames_eval).item() * B
            
            # Collect samples for visualization (main process only)
            if should_print and len(vis_samples) < num_vis_samples:
                vis_samples.append({
                    'input': input_frames[0].cpu(),  # First sample in batch
                    'target': target_frames_eval[0].cpu(),
                    'prediction': predictions[0].cpu(),
                    'baseline': baseline_pred[0].cpu(),
                })
            
            n_evaluated += B
        
        # Average
        model_vrmse_avg = model_vrmse_sum / n_evaluated
        baseline_vrmse_avg = baseline_vrmse_sum / n_evaluated
        model_mse_avg = model_mse_sum / n_evaluated
        baseline_mse_avg = baseline_mse_sum / n_evaluated
        
        # Print results
        print_rank0(f"\nSamples evaluated: {n_evaluated}")
        print_rank0("\n{:<12} {:>12} {:>12} {:>10}".format(
            "Field", "Model VRMSE", "Baseline", "Δ%"
        ))
        print_rank0("-" * 48)
        
        for i, name in enumerate(field_names):
            m_v = model_vrmse_avg[i].item()
            b_v = baseline_vrmse_avg[i].item()
            delta_pct = ((m_v - b_v) / b_v) * 100 if b_v > 0 else 0
            win = "✓" if m_v < b_v else ""
            print_rank0(f"{name:<12} {m_v:>12.4f} {b_v:>12.4f} {delta_pct:>+9.1f}% {win}")
        
        # Mean
        model_mean = model_vrmse_avg.mean().item()
        baseline_mean = baseline_vrmse_avg.mean().item()
        delta_pct = ((model_mean - baseline_mean) / baseline_mean) * 100
        print_rank0("-" * 48)
        print_rank0(f"{'Mean':<12} {model_mean:>12.4f} {baseline_mean:>12.4f} {delta_pct:>+9.1f}%")
        
        print_rank0(f"\nMSE: Model={model_mse_avg:.4f}, Baseline={baseline_mse_avg:.4f}")
        
        if model_mean < baseline_mean:
            print_rank0("🎉 Model BEATS baseline!")
        else:
            print_rank0(f"Model needs {(model_mean/baseline_mean - 1)*100:.1f}% improvement to beat baseline")
        
        print_rank0("=" * 60 + "\n")
        
        # Log metrics
        self.log_metrics({
            "detailed_eval/model_vrmse": model_mean,
            "detailed_eval/baseline_vrmse": baseline_mean,
            "detailed_eval/model_mse": model_mse_avg,
            "detailed_eval/baseline_mse": baseline_mse_avg,
        }, self.global_step)
        
        # === Save visualization samples (main process only) ===
        if should_print and vis_samples:
            vis_dir = self.checkpoint_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            for sample_idx, sample in enumerate(vis_samples):
                fig = self._create_comparison_figure(
                    sample['input'], sample['target'], 
                    sample['prediction'], sample['baseline'],
                    field_names
                )
                fig.savefig(
                    vis_dir / f"step_{self.global_step}_sample_{sample_idx}.png",
                    dpi=100, bbox_inches='tight'
                )
                plt.close(fig)
            
            print_rank0(f"Saved {len(vis_samples)} visualization samples to {vis_dir}")
        
        barrier()
    
    def _create_comparison_figure(self, input_frames, target_frames, pred_frames, baseline_frames, field_names):
        """
        Create a comparison figure showing input, target, prediction, and baseline.
        
        Layout (3 rows, each with input + time steps):
        - Row 1: Input (4 fields) | GT targets (4 fields × T steps)
        - Row 2: Input (4 fields) | Baseline (4 fields × T steps)  
        - Row 3: Input (4 fields) | Model pred (4 fields × T steps)
        
        This allows vertical comparison of same time step across approaches.
        """
        T_show = min(target_frames.shape[0], 8)  # Show up to 8 time steps
        n_fields = len(field_names)
        
        # Columns: 4 input fields + (4 fields × T_show time steps)
        n_cols = n_fields + n_fields * T_show
        n_rows = 3  # GT, Baseline, Model
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 3 * n_rows))
        
        row_labels = ['Ground Truth', 'Baseline', 'Model Prediction']
        row_data = [target_frames, baseline_frames, pred_frames]
        row_colors = ['black', 'red', 'blue']
        
        for row_idx, (label, frames, color) in enumerate(zip(row_labels, row_data, row_colors)):
            # First 4 columns: Input (same for all rows)
            for c, field_name in enumerate(field_names):
                ax = axes[row_idx, c]
                im_data = input_frames[-1, :, :, c].numpy()
                
                # Get color scale from target for this field
                all_target = target_frames[:T_show, :, :, c].numpy()
                vmin, vmax = all_target.min(), all_target.max()
                
                ax.imshow(im_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                ax.axis('off')
                
                # Labels
                if row_idx == 0:
                    ax.set_title(f'Input\n{field_name}', fontsize=8)
                if c == 0:
                    ax.set_ylabel(label, fontsize=10, fontweight='bold', color=color)
            
            # Remaining columns: Predictions for each time step
            for t in range(T_show):
                for c, field_name in enumerate(field_names):
                    col_idx = n_fields + t * n_fields + c
                    ax = axes[row_idx, col_idx]
                    
                    im_data = frames[t, :, :, c].numpy()
                    
                    # Use target's color scale for fair comparison
                    all_target = target_frames[:T_show, :, :, c].numpy()
                    vmin, vmax = all_target.min(), all_target.max()
                    
                    ax.imshow(im_data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                    ax.axis('off')
                    
                    # Column header (only first row)
                    if row_idx == 0:
                        ax.set_title(f't={t+1}\n{field_name}', fontsize=7)
        
        fig.suptitle(f'Step {self.global_step}: Input → Predictions Comparison', fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        # Get model state dict (handle DDP wrapper)
        if hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print_rank0(f"Saved checkpoint to {checkpoint_path}")
        
        # Also save adapter weights separately for easy loading
        adapter_path = self.checkpoint_dir / filename.replace(".pt", "_adapters.pt")
        if hasattr(self.model, 'module'):
            self.model.module.save_checkpoint(adapter_path)
        else:
            self.model.save_checkpoint(adapter_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        print_rank0(f"Loading checkpoint from {checkpoint_path}")
        
        # Load to CPU first to avoid OOM when all ranks load simultaneously
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load scaler state
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Restore training state
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        print_rank0(f"Resumed from epoch {self.epoch}, step {self.global_step}")


def create_trainer(
    model: nn.Module,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
) -> Trainer:
    """
    Factory function to create trainer.
    
    Args:
        model: The model to train
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        
    Returns:
        Configured Trainer instance
    """
    return Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
