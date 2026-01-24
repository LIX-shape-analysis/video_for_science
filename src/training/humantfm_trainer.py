"""
Two-stage trainer for HumanTFM-style physics prediction.

Stage 1: Train in latent space with decoder frozen
Stage 2: Train in ambient (physics) space with decoder unfrozen

Automatic stage switching based on loss plateau detection.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .optimizer import create_optimizer, create_scheduler
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    reduce_loss,
    print_rank0,
    GradientAccumulator,
)
from ..evaluation.metrics import compute_vrmse, compute_mse


class TwoStageTrainer:
    """
    Two-stage trainer for HumanTFM model.
    
    Stage 1: Latent space training
        - Physics decoder frozen
        - Loss = MSE(predicted_latent, target_latent)
        - Fast convergence to learn DiT output mapping
    
    Stage 2: Ambient space training
        - Physics decoder unfrozen
        - Loss = MSE(predicted_physics, target_physics)
        - Fine-grained physics details
    
    Automatic stage switching based on validation loss plateau.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Distributed setup
        self.rank, self.world_size, self.device = setup_distributed(
            backend=config.get("distributed", {}).get("backend", "nccl")
        )
        
        # Model setup
        self.model = model
        self.model.to(self.device)
        
        # Load pretrained weights
        if hasattr(self.model, 'load_pretrained'):
            self.model.load_pretrained()
        
        # Training config
        train_config = config.get("training", {})
        self.num_epochs = train_config.get("num_epochs", 20)
        self.batch_size = train_config.get("batch_size", 1)
        self.grad_accum_steps = train_config.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = train_config.get("max_grad_norm", 1.0)
        self.text_prompt = train_config.get("text_prompt", "Fluid dynamics simulation")
        
        # Two-stage config
        two_stage_config = train_config.get("two_stage", {})
        self.stage = 1  # Start in Stage 1 (latent)
        self.min_stage1_epochs = two_stage_config.get("min_epochs", 3)
        self.plateau_patience = two_stage_config.get("plateau_patience", 5)
        self.plateau_threshold = two_stage_config.get("plateau_threshold", 0.001)  # 0.1% improvement
        
        # Loss tracking for plateau detection
        self.val_loss_history: List[float] = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Freeze decoder for Stage 1
        if hasattr(self.model, 'freeze_decoder'):
            self.model.freeze_decoder()
        
        # Set normalization stats on decoder
        if hasattr(train_loader.dataset, 'mu') and hasattr(self.model, 'set_normalization_stats'):
            mu = train_loader.dataset.mu
            sigma = train_loader.dataset.sigma
            self.model.set_normalization_stats(mu, sigma)
        
        # Wrap for distributed
        if self.world_size > 1:
            self.model = wrap_model_ddp(self.model, device=self.device)
            print_rank0(f"[TwoStageTrainer] Wrapped model in DDP ({self.world_size} GPUs)")
        
        # Get raw model for attribute access
        self.raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Create optimizer (only trainable params)
        trainable_params = self.raw_model.get_trainable_parameters()
        self.optimizer = create_optimizer(trainable_params, config)
        print_rank0(f"[TwoStageTrainer] Trainable params: {sum(p.numel() for p in trainable_params):,}")
        
        # Create scheduler
        total_steps = len(train_loader) * self.num_epochs // self.grad_accum_steps
        self.scheduler = create_scheduler(self.optimizer, config, total_steps)
        
        # Checkpointing
        self.checkpoint_dir = Path(train_config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_every = train_config.get("checkpoint_every", 500)
        if is_main_process():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_every = train_config.get("log_every", 10)
        self.eval_every = train_config.get("eval_every", 250)
        self.detailed_eval_every = train_config.get("detailed_eval_every", 250)
        
        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        
        # Gradient accumulator
        self.grad_accumulator = GradientAccumulator(self.grad_accum_steps, self.model)
        
        print_rank0(f"""
╔════════════════════════════════════════════════════════════════╗
║           HumanTFM Two-Stage Training Configuration            ║
╠════════════════════════════════════════════════════════════════╣
║  Stage 1: Latent space (decoder frozen)                        ║
║    - Min epochs: {self.min_stage1_epochs:3d}                                             ║
║    - Plateau patience: {self.plateau_patience:3d} epochs                                  ║
║  Stage 2: Ambient space (decoder unfrozen)                     ║
║    - Switch when latent loss plateaus                          ║
╠════════════════════════════════════════════════════════════════╣
║  Total epochs: {self.num_epochs:3d}                                                ║
║  Batch size: {self.batch_size} × {self.world_size} GPUs × {self.grad_accum_steps} accum = {self.batch_size * self.world_size * self.grad_accum_steps} effective   ║
╚════════════════════════════════════════════════════════════════╝
        """)
    
    def _check_stage_transition(self, val_loss: float) -> bool:
        """
        Check if we should transition from Stage 1 to Stage 2.
        
        Returns True if transition should happen.
        """
        if self.stage != 1:
            return False
        
        # Must complete minimum epochs first
        if self.current_epoch < self.min_stage1_epochs:
            return False
        
        # Track loss history
        self.val_loss_history.append(val_loss)
        
        # Check for improvement
        if val_loss < self.best_val_loss * (1 - self.plateau_threshold):
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Plateau detection
        if self.epochs_without_improvement >= self.plateau_patience:
            return True
        
        return False
    
    def _transition_to_stage2(self):
        """Transition from Stage 1 to Stage 2."""
        print_rank0("\n" + "="*70)
        print_rank0(" STAGE TRANSITION: Latent → Ambient Space")
        print_rank0("="*70)
        print_rank0(f"  Epoch: {self.current_epoch}")
        print_rank0(f"  Latent loss plateaued for {self.plateau_patience} epochs")
        print_rank0(f"  Best latent loss: {self.best_val_loss:.6f}")
        print_rank0("  Unfreezing physics decoder...")
        print_rank0("="*70 + "\n")
        
        self.stage = 2
        
        # Unfreeze decoder
        if hasattr(self.raw_model, 'unfreeze_decoder'):
            self.raw_model.unfreeze_decoder()
        
        # Update optimizer with new trainable params
        trainable_params = self.raw_model.get_trainable_parameters()
        opt_config = self.config.get("training", {}).get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=opt_config.get("lr", 1e-5),
            weight_decay=opt_config.get("weight_decay", 0.01),
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
        )
        print_rank0(f"  New trainable params: {sum(p.numel() for p in trainable_params):,}")
        
        # Reset loss tracking for Stage 2
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.val_loss_history = []
    
    def train(self):
        """Main training loop."""
        print_rank0(f"\n[TwoStageTrainer] Starting training for {self.num_epochs} epochs")
        start_time = time.time()
        
        # Initial evaluation
        print_rank0("\n=== Initial evaluation at step 0 ===")
        self.detailed_evaluation(num_samples=40)
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validation
            if self.val_loader is not None:
                val_loss = self.validate()
                
                # Check stage transition
                if self._check_stage_transition(val_loss):
                    self._transition_to_stage2()
            
            # Checkpointing
            if is_main_process():
                self.save_checkpoint(f"epoch_{self.current_epoch}.pt")
                if val_loss < self.best_val_loss:
                    self.save_checkpoint("best_model.pt")
        
        elapsed = time.time() - start_time
        print_rank0(f"\n[TwoStageTrainer] Training complete in {elapsed/3600:.1f} hours")
        print_rank0(f"  Final stage: {self.stage}")
        
        # Final checkpoint
        if is_main_process():
            self.save_checkpoint("final_model.pt")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"[Stage {self.stage}] Epoch {self.current_epoch}/{self.num_epochs}",
            disable=not is_main_process(),
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Get data - dataset returns 'input_frames_normalized' and 'target_frames_normalized'
            input_frames = batch['input_frames_normalized'].to(self.device)
            target_frames = batch['target_frames_normalized'].to(self.device)
            
            # Conditioning frame is last input frame
            cond_frame = input_frames[:, -1]  # (B, C, H, W) or (B, H, W, C)
            
            # Forward pass based on stage
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                if self.stage == 1:
                    # Stage 1: Latent loss only
                    outputs = self.raw_model.forward_latent_only(
                        cond_frame=cond_frame,
                        target_frames=target_frames,
                        text_prompt=self.text_prompt,
                    )
                    loss = outputs['latent_loss']
                else:
                    # Stage 2: Ambient (physics) loss
                    outputs = self.raw_model.forward_ambient(
                        cond_frame=cond_frame,
                        target_frames=target_frames,
                        text_prompt=self.text_prompt,
                    )
                    loss = outputs['ambient_loss']
            
            # Gradient accumulation
            loss = loss / self.grad_accum_steps
            loss.backward()
            
            if self.grad_accumulator.should_step():
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.raw_model.get_trainable_parameters(),
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f'{loss.item() * self.grad_accum_steps:.4f}',
                        'lr': f'{lr:.2e}',
                        'stage': self.stage,
                    })
                
                # Detailed evaluation
                if self.global_step % self.detailed_eval_every == 0:
                    self.detailed_evaluation(num_samples=40)
                    self.model.train()
                
                # Checkpointing
                if self.global_step % self.checkpoint_every == 0:
                    if is_main_process():
                        self.save_checkpoint(f"step_{self.global_step}.pt")
            
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            self.grad_accumulator.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        print_rank0(f"[Stage {self.stage}] Epoch {self.current_epoch}/{self.num_epochs} - Train Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", disable=not is_main_process()):
                input_frames = batch['input_frames_normalized'].to(self.device)
                target_frames = batch['target_frames_normalized'].to(self.device)
                cond_frame = input_frames[:, -1]
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    if self.stage == 1:
                        outputs = self.raw_model.forward_latent_only(
                            cond_frame=cond_frame,
                            target_frames=target_frames,
                            text_prompt=self.text_prompt,
                        )
                        loss = outputs['latent_loss']
                    else:
                        outputs = self.raw_model.forward_ambient(
                            cond_frame=cond_frame,
                            target_frames=target_frames,
                            text_prompt=self.text_prompt,
                        )
                        loss = outputs['ambient_loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Reduce across GPUs
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        print_rank0(f"[Stage {self.stage}] Epoch {self.current_epoch}/{self.num_epochs} - Val Loss: {avg_loss:.4f}")
        
        # Check for best model
        if avg_loss < self.best_val_loss:
            print_rank0(f"  New best val loss! (prev: {self.best_val_loss:.4f})")
            if is_main_process():
                self.save_checkpoint("best_model.pt")
        
        return avg_loss
    
    def detailed_evaluation(self, num_samples: int = 50):
        """Run detailed evaluation with baseline comparison."""
        self.model.eval()
        
        field_names = ["density", "pressure", "velocity_x", "velocity_y"]
        
        # Accumulators
        model_vrmse = {f: 0.0 for f in field_names}
        model_mse = {f: 0.0 for f in field_names}
        baseline_vrmse = {f: 0.0 for f in field_names}
        baseline_mse = {f: 0.0 for f in field_names}
        count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= num_samples:
                    break
                
                # Get normalized frames for model input
                input_norm = batch['input_frames_normalized'].to(self.device)
                target_norm = batch['target_frames_normalized'].to(self.device)
                
                # Get unnormalized frames for metrics (dataset returns these)
                input_frames = batch['input_frames'].to(self.device)
                target_frames = batch['target_frames'].to(self.device)
                
                # Convert to (B, T, C, H, W) if in (B, T, H, W, C)
                if input_norm.dim() == 5 and input_norm.shape[-1] == 4:
                    input_norm = input_norm.permute(0, 1, 4, 2, 3)
                if target_norm.dim() == 5 and target_norm.shape[-1] == 4:
                    target_norm = target_norm.permute(0, 1, 4, 2, 3)
                if input_frames.dim() == 5 and input_frames.shape[-1] == 4:
                    input_frames = input_frames.permute(0, 1, 4, 2, 3)
                if target_frames.dim() == 5 and target_frames.shape[-1] == 4:
                    target_frames = target_frames.permute(0, 1, 4, 2, 3)
                
                cond_frame = input_norm[:, -1]
                B, T_out = target_frames.shape[:2]
                
                # Model prediction (using normalized input)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = self.raw_model.predict(cond_frame, num_frames=T_out, text_prompt=self.text_prompt)
                
                # Prediction is already in physics space from decoder
                # If model uses residual, it returns final prediction (reference + delta)
                pred = pred.float()
                
                # Baseline: repeat last input frame (unnormalized)
                baseline = input_frames[:, -1:].repeat(1, T_out, 1, 1, 1)
                
                # Compute metrics per field
                for c, field in enumerate(field_names):
                    pred_field = pred[:, :, c]
                    target_field = target_frames[:, :, c]
                    baseline_field = baseline[:, :, c]
                    
                    model_vrmse[field] += compute_vrmse(pred_field, target_field).item()
                    model_mse[field] += compute_mse(pred_field, target_field).item()
                    baseline_vrmse[field] += compute_vrmse(baseline_field, target_field).item()
                    baseline_mse[field] += compute_mse(baseline_field, target_field).item()
                
                count += 1
        
        # Aggregate across GPUs
        if self.world_size > 1:
            for field in field_names:
                for metrics_dict in [model_vrmse, model_mse, baseline_vrmse, baseline_mse]:
                    t = torch.tensor([metrics_dict[field]], device=self.device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    metrics_dict[field] = t.item()
            
            count_tensor = torch.tensor([count], device=self.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            count = count_tensor.item()
        
        # Average
        for field in field_names:
            model_vrmse[field] /= count
            model_mse[field] /= count
            baseline_vrmse[field] /= count
            baseline_mse[field] /= count
        
        # Print results
        if is_main_process():
            print("\n" + "="*70)
            print(f"DETAILED EVALUATION @ Step {self.global_step} [Stage {self.stage}]")
            print("="*70)
            print(f"\nSamples evaluated: {count}")
            print(f"\n{'Field':<15} {'Model VRMSE':>12} {'Baseline':>12} {'Δ%':>10}")
            print("-"*50)
            
            mean_model = 0.0
            mean_baseline = 0.0
            
            for field in field_names:
                m_vrmse = model_vrmse[field]
                b_vrmse = baseline_vrmse[field]
                delta_pct = ((m_vrmse - b_vrmse) / b_vrmse * 100) if b_vrmse > 0 else 0
                
                print(f"{field:<15} {m_vrmse:>12.4f} {b_vrmse:>12.4f} {delta_pct:>+9.1f}%")
                mean_model += m_vrmse
                mean_baseline += b_vrmse
            
            mean_model /= len(field_names)
            mean_baseline /= len(field_names)
            mean_delta = ((mean_model - mean_baseline) / mean_baseline * 100) if mean_baseline > 0 else 0
            
            print("-"*50)
            print(f"{'Mean':<15} {mean_model:>12.4f} {mean_baseline:>12.4f} {mean_delta:>+9.1f}%")
            print("="*70 + "\n")
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        if not is_main_process():
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'stage': self.stage,
            'model_state_dict': self.raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss_history': self.val_loss_history,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print_rank0(f"[TwoStageTrainer] Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.stage = checkpoint.get('stage', 1)
        self.raw_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.val_loss_history = checkpoint.get('val_loss_history', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print_rank0(f"[TwoStageTrainer] Loaded checkpoint: {path}")
        print_rank0(f"  Epoch: {self.current_epoch}, Step: {self.global_step}, Stage: {self.stage}")


def create_humanTFM_trainer(
    model: nn.Module,
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
) -> TwoStageTrainer:
    """Factory function to create TwoStageTrainer."""
    return TwoStageTrainer(model, config, train_loader, val_loader)
