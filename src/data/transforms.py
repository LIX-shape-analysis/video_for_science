"""
Transform utilities for converting between physics channels and video channels.

The Well turbulent_radiative_layer_2D has 4 channels:
- density, pressure, velocity_x, velocity_y

Wan2.2-I2V expects 3 channels (RGB format).

This module provides utilities for data augmentation and format conversion.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class PhysicsToVideoTransform:
    """
    Transform physics simulation data to video-compatible format.
    
    This handles:
    - Channel format conversion
    - Spatial resizing if needed
    - Value range normalization to [0, 1] or [-1, 1]
    """
    
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        value_range: Tuple[float, float] = (-1, 1),
        normalize_per_channel: bool = True,
    ):
        """
        Args:
            target_size: Target spatial size (H, W). None keeps original size.
            value_range: Target value range for the output
            normalize_per_channel: Whether to normalize each channel independently
        """
        self.target_size = target_size
        self.value_range = value_range
        self.normalize_per_channel = normalize_per_channel
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform physics data.
        
        Args:
            x: Input tensor of shape (T, C, H, W) or (C, H, W)
            
        Returns:
            Transformed tensor
        """
        # Handle batch dimension
        squeeze = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
        
        T, C, H, W = x.shape
        
        # Resize if needed
        if self.target_size is not None:
            target_h, target_w = self.target_size
            if (H, W) != (target_h, target_w):
                x = x.view(T, C, H, W)
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Normalize to target range
        if self.normalize_per_channel:
            # Normalize each channel independently
            for c in range(C):
                channel_data = x[:, c]
                min_val = channel_data.min()
                max_val = channel_data.max()
                if max_val - min_val > 1e-8:
                    x[:, c] = (channel_data - min_val) / (max_val - min_val)
                else:
                    x[:, c] = 0.5
        else:
            # Global normalization
            min_val = x.min()
            max_val = x.max()
            if max_val - min_val > 1e-8:
                x = (x - min_val) / (max_val - min_val)
            else:
                x = torch.ones_like(x) * 0.5
        
        # Scale to target range
        low, high = self.value_range
        x = x * (high - low) + low
        
        if squeeze:
            x = x.squeeze(0)
        
        return x


class VideoToPhysicsTransform:
    """
    Transform video model output back to physics format.
    
    This is the inverse of PhysicsToVideoTransform.
    """
    
    def __init__(
        self,
        original_size: Optional[Tuple[int, int]] = None,
        original_range: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            original_size: Original spatial size to restore
            original_range: Original (min, max) values per channel
        """
        self.original_size = original_size
        self.original_range = original_range
    
    def __call__(
        self, 
        x: torch.Tensor,
        original_min: Optional[torch.Tensor] = None,
        original_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Transform video output back to physics format.
        
        Args:
            x: Video tensor of shape (T, C, H, W) or (C, H, W)
            original_min: Per-channel minimum values
            original_max: Per-channel maximum values
            
        Returns:
            Physics tensor in original format
        """
        squeeze = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
        
        T, C, H, W = x.shape
        
        # Resize back if needed
        if self.original_size is not None:
            orig_h, orig_w = self.original_size
            if (H, W) != (orig_h, orig_w):
                x = F.interpolate(x, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        
        # Denormalize if original range is provided
        if original_min is not None and original_max is not None:
            # Assume x is in [0, 1] range
            x = torch.clamp(x, 0, 1)
            for c in range(C):
                x[:, c] = x[:, c] * (original_max[c] - original_min[c]) + original_min[c]
        
        if squeeze:
            x = x.squeeze(0)
        
        return x


class DataAugmentation:
    """
    Data augmentation for physics simulation data.
    
    These augmentations preserve physical meaning:
    - Horizontal/vertical flips (with velocity sign correction)
    - Small spatial shifts
    - Temporal subsampling
    """
    
    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        random_crop: Optional[Tuple[int, int]] = None,
        temporal_subsample: bool = False,
    ):
        """
        Args:
            horizontal_flip: Enable horizontal flipping
            vertical_flip: Enable vertical flipping
            random_crop: Crop size (H, W) if enabled
            temporal_subsample: Enable temporal subsampling
        """
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.random_crop = random_crop
        self.temporal_subsample = temporal_subsample
        
        # Field indices for velocity components
        # In turbulent_radiative_layer_2D: [density, pressure, velocity_x, velocity_y]
        self.velocity_x_idx = 2
        self.velocity_y_idx = 3
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations.
        
        Args:
            x: Input tensor of shape (T, C, H, W)
            
        Returns:
            Augmented tensor
        """
        # Horizontal flip
        if self.horizontal_flip and torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[-1])  # Flip width
            # Flip velocity_x sign
            x[:, self.velocity_x_idx] = -x[:, self.velocity_x_idx]
        
        # Vertical flip
        if self.vertical_flip and torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[-2])  # Flip height
            # Flip velocity_y sign
            x[:, self.velocity_y_idx] = -x[:, self.velocity_y_idx]
        
        # Random crop
        if self.random_crop is not None:
            T, C, H, W = x.shape
            crop_h, crop_w = self.random_crop
            if H > crop_h and W > crop_w:
                start_h = torch.randint(0, H - crop_h + 1, (1,)).item()
                start_w = torch.randint(0, W - crop_w + 1, (1,)).item()
                x = x[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        return x


def prepare_for_wan22(
    frames: torch.Tensor,
    channel_adapter: torch.nn.Module,
    target_size: Tuple[int, int] = (480, 832),
) -> torch.Tensor:
    """
    Prepare physics frames for Wan2.2 input.
    
    Args:
        frames: Physics frames (T, C=4, H, W)
        channel_adapter: Module to convert 4 channels to 3
        target_size: Target spatial size for Wan2.2
        
    Returns:
        Frames ready for Wan2.2 (T, 3, H', W')
    """
    # Apply channel adapter (4 -> 3 channels)
    frames = channel_adapter(frames)  # (T, 3, H, W)
    
    # Resize to Wan2.2 expected size
    T, C, H, W = frames.shape
    if (H, W) != target_size:
        frames = F.interpolate(
            frames, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
    
    return frames


def postprocess_from_wan22(
    frames: torch.Tensor,
    channel_adapter_inverse: torch.nn.Module,
    original_size: Tuple[int, int] = (128, 384),
) -> torch.Tensor:
    """
    Post-process Wan2.2 output back to physics format.
    
    Args:
        frames: Wan2.2 output (T, 3, H, W)
        channel_adapter_inverse: Module to convert 3 channels to 4
        original_size: Original physics spatial size
        
    Returns:
        Physics frames (T, 4, H, W)
    """
    # Resize back to original size first
    T, C, H, W = frames.shape
    if (H, W) != original_size:
        frames = F.interpolate(
            frames,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
    
    # Apply inverse channel adapter (3 -> 4 channels)
    frames = channel_adapter_inverse(frames)  # (T, 4, H, W)
    
    return frames
