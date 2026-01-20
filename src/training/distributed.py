"""
Distributed training utilities for multi-GPU training.

Supports:
- PyTorch DDP (DistributedDataParallel)
- FSDP (Fully Sharded Data Parallel)
- DeepSpeed integration
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Optional, Tuple
import functools


def setup_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: str = "nccl",
) -> Tuple[int, int, torch.device]:
    """
    Initialize distributed training environment.
    
    Args:
        rank: Process rank (auto-detected if None)
        world_size: Total number of processes (auto-detected if None)
        backend: Communication backend ("nccl" for GPU)
        
    Returns:
        Tuple of (rank, world_size, device)
    """
    # Auto-detect from environment variables
    if rank is None:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    
    return rank, world_size, device


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    """Check if this is the main process."""
    return rank == 0


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    All-reduce tensor across all processes.
    
    Args:
        tensor: Input tensor
        op: Reduction operation
        
    Returns:
        Reduced tensor
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Gathered tensor
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> torch.nn.Module:
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: The model to wrap
        device: Target device
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        Wrapped model
    """
    model = model.to(device)
    
    if dist.is_initialized() and get_world_size() > 1:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=find_unused_parameters,
        )
    
    return model


def wrap_model_fsdp(
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> torch.nn.Module:
    """
    Wrap model with Fully Sharded Data Parallel (FSDP).
    
    Args:
        model: The model to wrap
        config: FSDP configuration
        
    Returns:
        Wrapped model
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
    )
    
    # Determine sharding strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = strategy_map.get(
        config.get("fsdp_sharding_strategy", "FULL_SHARD"),
        ShardingStrategy.FULL_SHARD,
    )
    
    # Mixed precision
    if config.get("mixed_precision") == "bf16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif config.get("mixed_precision") == "fp16":
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mp_policy = None
    
    # Auto wrap policy
    # Use size-based wrapping as a simple default
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1e8,  # Wrap modules with > 100M params
    )
    
    # Wrap model
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=config.get("cpu_offload", False)),
        device_id=torch.cuda.current_device(),
    )
    
    return model


class GradientAccumulator:
    """
    Utility class for gradient accumulation with distributed training.
    """
    
    def __init__(
        self,
        accumulation_steps: int,
        model: torch.nn.Module,
    ):
        """
        Args:
            accumulation_steps: Number of steps to accumulate gradients
            model: The model (for sync control)
        """
        self.accumulation_steps = accumulation_steps
        self.model = model
        self.step_count = 0
    
    def should_sync(self) -> bool:
        """Check if gradients should be synchronized."""
        return (self.step_count + 1) % self.accumulation_steps == 0
    
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return self.should_sync()
    
    def sync_context(self):
        """Get context manager for gradient sync control."""
        if isinstance(self.model, DDP):
            if self.should_sync():
                return self.model.no_sync().__enter__, self.model.no_sync().__exit__
            else:
                # Create a context that does nothing
                class NoOpContext:
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                return NoOpContext()
        return None
    
    def step(self):
        """Increment step counter."""
        self.step_count += 1
    
    def reset(self):
        """Reset step counter."""
        self.step_count = 0


def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    """
    Reduce loss across all processes.
    
    Args:
        loss: Loss tensor
        
    Returns:
        Mean loss across all processes
    """
    if dist.is_initialized():
        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        reduced_loss = reduced_loss / get_world_size()
        return reduced_loss
    return loss


def print_rank0(msg: str):
    """Print message only on rank 0."""
    if get_rank() == 0:
        print(msg)
