"""
Optimizer and learning rate scheduler utilities.
"""

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
    ConstantLR,
)
from typing import Dict, Any, List, Optional
import math


def create_optimizer(
    parameters: List[torch.nn.Parameter],
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        parameters: List of parameters to optimize
        config: Optimizer configuration
        
    Returns:
        Configured optimizer
    """
    opt_config = config["training"]["optimizer"]
    name = opt_config["name"].lower()
    
    if name == "adamw":
        optimizer = AdamW(
            parameters,
            lr=opt_config["lr"],
            weight_decay=opt_config.get("weight_decay", 0.01),
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
            eps=opt_config.get("eps", 1e-8),
        )
    elif name == "adam":
        optimizer = Adam(
            parameters,
            lr=opt_config["lr"],
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
            eps=opt_config.get("eps", 1e-8),
        )
    elif name == "sgd":
        optimizer = SGD(
            parameters,
            lr=opt_config["lr"],
            momentum=opt_config.get("momentum", 0.9),
            weight_decay=opt_config.get("weight_decay", 0.01),
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_training_steps: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: The optimizer
        config: Scheduler configuration
        num_training_steps: Total number of training steps
        
    Returns:
        Configured scheduler or None
    """
    sched_config = config["training"]["scheduler"]
    name = sched_config["name"].lower()
    warmup_steps = sched_config.get("warmup_steps", 0)
    
    if name == "cosine":
        # Cosine annealing with warmup
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=sched_config.get("min_lr", 1e-7),
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
                eta_min=sched_config.get("min_lr", 1e-7),
            )
    
    elif name == "cosine_restarts":
        # Cosine annealing with warm restarts
        num_cycles = sched_config.get("num_cycles", 1)
        T_0 = num_training_steps // num_cycles
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=1,
            eta_min=sched_config.get("min_lr", 1e-7),
        )
    
    elif name == "linear":
        # Linear decay with warmup
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=num_training_steps - warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=num_training_steps,
            )
    
    elif name == "constant":
        # Constant learning rate (with optional warmup)
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            constant_scheduler = ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=num_training_steps - warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, constant_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = None
    
    elif name == "none":
        scheduler = None
    
    else:
        raise ValueError(f"Unknown scheduler: {name}")
    
    return scheduler


class WarmupCosineScheduler:
    """
    Custom warmup + cosine annealing scheduler.
    
    This provides more fine-grained control over the learning rate schedule.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
        warmup_init_lr: float = 1e-8,
    ):
        """
        Args:
            optimizer: The optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            warmup_init_lr: Initial learning rate for warmup
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_init_lr = warmup_init_lr
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        self.current_step = 0
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            return [
                self.warmup_init_lr + progress * (base_lr - self.warmup_init_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
                for base_lr in self.base_lrs
            ]
    
    def step(self):
        """Update learning rates."""
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        self.current_step += 1
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'warmup_init_lr': self.warmup_init_lr,
            'base_lrs': self.base_lrs,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']
        self.warmup_init_lr = state_dict['warmup_init_lr']
        self.base_lrs = state_dict['base_lrs']
