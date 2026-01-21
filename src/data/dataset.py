"""
Dataset module for loading The Well physics data for video generation.

This module wraps The Well dataset and prepares it for fine-tuning
video generative models like Wan2.2.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from einops import rearrange

try:
    from the_well.data import WellDataset
    from the_well.utils.download import well_download
except ImportError:
    raise ImportError(
        "Please install the_well package: pip install the_well[benchmark]"
    )


class WellVideoDataset(Dataset):
    """
    Dataset wrapper for The Well that prepares data for video generation models.
    
    The turbulent_radiative_layer_2D dataset has:
    - 4 physical fields: density, pressure, velocity_x, velocity_y
    - Spatial dimensions: 128 x 384
    - Multiple trajectories with time evolution
    
    This wrapper prepares the data as:
    - Input: Initial frames (condition for video generation)
    - Output: Future frames (target video to generate)
    """
    
    def __init__(
        self,
        base_path: str,
        dataset_name: str = "turbulent_radiative_layer_2D",
        split: str = "train",
        n_steps_input: int = 4,
        n_steps_output: int = 8,
        use_normalization: bool = False,
        compute_stats: bool = True,
        stats_samples: int = 1000,
        use_residual_targets: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            base_path: Path to the dataset storage
            dataset_name: Name of The Well dataset
            split: Data split ("train" or "valid")
            n_steps_input: Number of input timesteps (initial condition)
            n_steps_output: Number of output timesteps (future prediction)
            use_normalization: Whether to use built-in normalization
            compute_stats: Whether to compute normalization statistics
            stats_samples: Number of samples to use for statistics
            use_residual_targets: If True, targets are deltas (target - last input frame)
        """
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.split = split
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.use_residual_targets = use_residual_targets
        
        # Load the underlying Well dataset
        self.well_dataset = WellDataset(
            well_base_path=base_path,
            well_dataset_name=dataset_name,
            well_split_name=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=use_normalization,
        )
        
        # Get metadata
        self.metadata = self.well_dataset.metadata
        self.n_fields = self.metadata.n_fields  # Should be 4
        
        # Get field names
        self.field_names = [
            name for group in self.metadata.field_names.values() 
            for name in group
        ]
        
        # Compute normalization statistics
        self.mu = None
        self.sigma = None
        self.field_min = None
        self.field_max = None
        if compute_stats:
            self._compute_statistics(stats_samples)
    
    def _compute_statistics(self, n_samples: int = 1000):
        """Compute mean, std, and min/max for normalization and clamping."""
        samples = []
        indices = np.linspace(0, len(self.well_dataset) - 1, 
                             min(n_samples, len(self.well_dataset)), 
                             dtype=int)
        
        for i in indices:
            item = self.well_dataset[i]
            x = item["input_fields"]
            samples.append(x)
        
        samples = torch.stack(samples)
        # Shape: (N, T, H, W, F)
        self.mu = samples.reshape(-1, self.n_fields).mean(dim=0)
        self.sigma = samples.reshape(-1, self.n_fields).std(dim=0)
        
        # Compute min/max per field for output clamping
        self.field_min = samples.reshape(-1, self.n_fields).min(dim=0).values
        self.field_max = samples.reshape(-1, self.n_fields).max(dim=0).values
        
        # Prevent division by zero for sigma
        self.sigma = torch.clamp(self.sigma, min=1e-6)
        
        print(f"Dataset stats computed over {len(indices)} samples:")
        print(f"  Field mins: {self.field_min.tolist()}")
        print(f"  Field maxs: {self.field_max.tolist()}")
    
    def set_statistics(self, mu: torch.Tensor, sigma: torch.Tensor):
        """Set normalization statistics (for validation set)."""
        self.mu = mu
        self.sigma = sigma
    
    def set_field_bounds(self, field_min: torch.Tensor, field_max: torch.Tensor):
        """Set min/max bounds (for validation set)."""
        self.field_min = field_min
        self.field_max = field_max
    
    def get_field_bounds(self) -> torch.Tensor:
        """
        Get per-field min/max bounds for output clamping.
        
        Returns:
            Tensor of shape (n_fields, 2) where [:, 0] = min, [:, 1] = max
        """
        if self.field_min is None or self.field_max is None:
            return None
        return torch.stack([self.field_min, self.field_max], dim=1)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor."""
        if self.mu is None or self.sigma is None:
            return x
        # Move stats to same device as input
        mu = self.mu.to(x.device) if hasattr(self.mu, 'to') else self.mu
        sigma = self.sigma.to(x.device) if hasattr(self.sigma, 'to') else self.sigma
        return (x - mu) / sigma
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize the tensor back to original scale."""
        if self.mu is None or self.sigma is None:
            return x
        # Move stats to same device as input
        mu = self.mu.to(x.device) if hasattr(self.mu, 'to') else self.mu
        sigma = self.sigma.to(x.device) if hasattr(self.sigma, 'to') else self.sigma
        return x * sigma + mu
    
    def __len__(self) -> int:
        return len(self.well_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
            - input_frames: (T_in, H, W, C) input frames
            - target_frames: (T_out, H, W, C) target frames
            - input_frames_normalized: Normalized input frames
            - target_frames_normalized: Normalized target frames
            - metadata: Additional metadata
        """
        item = self.well_dataset[idx]
        
        # Get input and output fields
        # Shape: (T, H, W, F) where F=4 (physics channels)
        input_frames = item["input_fields"]
        target_frames = item["output_fields"]
        
        # Reference frame for residual prediction (last input frame)
        reference_frame = input_frames[-1:]  # (1, H, W, F)
        
        # Normalize
        input_normalized = self.normalize(input_frames)
        target_normalized = self.normalize(target_frames)
        reference_normalized = self.normalize(reference_frame)
        
        # Compute residual targets if enabled
        if self.use_residual_targets:
            # Target becomes delta from reference frame
            # Broadcast reference to all time steps: (T_out, H, W, F) - (1, H, W, F)
            target_normalized = target_normalized - reference_normalized
        
        return {
            "input_frames": input_frames,
            "target_frames": target_frames,  # Original unnormalized targets for eval
            "input_frames_normalized": input_normalized,
            "target_frames_normalized": target_normalized,  # May be deltas if residual mode
            "reference_frame": reference_frame,  # For adding back at inference
            "reference_normalized": reference_normalized,
            "space_grid": item.get("space_grid"),
            "input_time_grid": item.get("input_time_grid"),
            "output_time_grid": item.get("output_time_grid"),
        }


def download_dataset(
    base_path: str,
    dataset_name: str = "turbulent_radiative_layer_2D",
    splits: List[str] = ["train", "valid"],
) -> None:
    """
    Download The Well dataset.
    
    Args:
        base_path: Path to store the dataset
        dataset_name: Name of the dataset to download
        splits: List of splits to download
    """
    for split in splits:
        print(f"Downloading {dataset_name}/{split}...")
        well_download(
            base_path=base_path,
            dataset=dataset_name,
            split=split,
        )
    print("Download complete!")


def create_dataloaders(
    config: Dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        rank: Current process rank (for distributed training)
        world_size: Total number of processes
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    data_config = config["data"]
    training_config = config["training"]
    hardware_config = config["hardware"]
    
    # Check for residual prediction mode
    use_residual = training_config.get("use_residual_prediction", False)
    
    # Create training dataset
    train_dataset = WellVideoDataset(
        base_path=data_config["base_path"],
        dataset_name=data_config["dataset_name"],
        split=data_config["train_split"],
        n_steps_input=data_config["n_steps_input"],
        n_steps_output=data_config["n_steps_output"],
        use_normalization=data_config["use_normalization"],
        use_residual_targets=use_residual,
    )
    
    # Create validation dataset
    val_dataset = WellVideoDataset(
        base_path=data_config["base_path"],
        dataset_name=data_config["dataset_name"],
        split=data_config["val_split"],
        n_steps_input=data_config["n_steps_input"],
        n_steps_output=data_config["n_steps_output"],
        use_normalization=data_config["use_normalization"],
        compute_stats=False,  # Use training stats
        use_residual_targets=use_residual,
    )
    
    # Share normalization statistics and field bounds
    val_dataset.set_statistics(train_dataset.mu, train_dataset.sigma)
    if train_dataset.field_min is not None:
        val_dataset.set_field_bounds(train_dataset.field_min, train_dataset.field_max)
    
    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=hardware_config["num_workers"],
        pin_memory=hardware_config["pin_memory"],
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=hardware_config["num_workers"],
        pin_memory=hardware_config["pin_memory"],
    )
    
    return train_loader, val_loader


class VideoFrameDataset(Dataset):
    """
    A dataset that prepares individual frame sequences for video generation.
    
    This is useful when the video model expects a specific format.
    """
    
    def __init__(
        self,
        well_dataset: WellVideoDataset,
        frame_processor: Optional[callable] = None,
    ):
        """
        Args:
            well_dataset: Underlying WellVideoDataset
            frame_processor: Optional function to process frames
        """
        self.well_dataset = well_dataset
        self.frame_processor = frame_processor
    
    def __len__(self) -> int:
        return len(self.well_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.well_dataset[idx]
        
        # Get normalized frames
        input_frames = item["input_frames_normalized"]
        target_frames = item["target_frames_normalized"]
        
        # Rearrange to video format: (T, H, W, C) -> (T, C, H, W)
        input_frames = rearrange(input_frames, "T H W C -> T C H W")
        target_frames = rearrange(target_frames, "T H W C -> T C H W")
        
        # Apply optional processing
        if self.frame_processor is not None:
            input_frames = self.frame_processor(input_frames)
            target_frames = self.frame_processor(target_frames)
        
        # Concatenate for video sequence
        # First frame of input is the "image" condition
        # Rest of input + all output is the "video" to generate
        condition_frame = input_frames[0]  # (C, H, W)
        video_frames = torch.cat([input_frames[1:], target_frames], dim=0)  # (T-1+T_out, C, H, W)
        
        return {
            "condition_frame": condition_frame,
            "video_frames": video_frames,
            "all_input_frames": input_frames,
            "all_target_frames": target_frames,
            "original_input": item["input_frames"],
            "original_target": item["target_frames"],
        }
