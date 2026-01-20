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

from .optimizer import create_optimizer, create_scheduler, WarmupCosineScheduler
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
                # Check if model uses temporal predictor
                model = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(model, 'use_temporal_predictor') and model.use_temporal_predictor:
                    outputs = model.forward_with_temporal_predictor(
                        input_frames=input_frames,
                        target_frames=target_frames,
                    )
                else:
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
                # Check if model uses temporal predictor
                model = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(model, 'use_temporal_predictor') and model.use_temporal_predictor:
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
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
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
