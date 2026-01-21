"""
Learnable channel adapter modules for physics-to-video channel mapping.

The Well turbulent_radiative_layer_2D has 4 channels:
- density, pressure, velocity_x, velocity_y

Wan2.2-I2V expects 3 channels (RGB format).

These modules provide learnable transformations between the two representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ChannelAdapter(nn.Module):
    """
    Learnable adapter to convert from physics channels to video channels.
    
    Maps: 4 channels (physics) -> 3 channels (RGB for Wan2.2)
    
    This is designed to be lightweight and learnable, encoding the
    physics fields into a representation suitable for the video model.
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            input_channels: Number of physics channels (4)
            output_channels: Number of video channels (3)
            hidden_dim: Hidden dimension in the adapter
            num_layers: Number of convolutional layers
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        
        # Build the adapter network
        layers = []
        
        # Input projection
        layers.append(nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1))
        if use_layer_norm:
            layers.append(nn.GroupNorm(min(32, hidden_dim), hidden_dim))
        layers.append(nn.SiLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            if use_layer_norm:
                layers.append(nn.GroupNorm(min(32, hidden_dim), hidden_dim))
            layers.append(nn.SiLU())
        
        # Output projection
        layers.append(nn.Conv2d(hidden_dim, output_channels, kernel_size=3, padding=1))
        
        self.network = nn.Sequential(*layers)
        
        # Learnable linear projection as alternative path
        if use_residual and input_channels != output_channels:
            self.residual_proj = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        else:
            self.residual_proj = None
        
        # Learnable scaling parameters
        self.output_scale = nn.Parameter(torch.ones(output_channels, 1, 1))
        self.output_bias = nn.Parameter(torch.zeros(output_channels, 1, 1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W) or (T, C_in, H, W)
               where C_in = input_channels (4)
               
        Returns:
            Output tensor of shape (B, C_out, H, W) or (T, C_out, H, W)
            where C_out = output_channels (3)
        """
        # Main path
        out = self.network(x)
        
        # Residual path
        if self.use_residual and self.residual_proj is not None:
            residual = self.residual_proj(x)
            out = out + residual
        
        # Apply learnable scaling
        out = out * self.output_scale + self.output_bias
        
        return out


class InverseChannelAdapter(nn.Module):
    """
    Learnable adapter to convert from video channels back to physics channels.
    
    Maps: 3 channels (RGB from Wan2.2) -> 4 channels (physics)
    
    This is the inverse of ChannelAdapter, used to decode the video model's
    output back to physics fields.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            input_channels: Number of video channels (3)
            output_channels: Number of physics channels (4)
            hidden_dim: Hidden dimension in the adapter
            num_layers: Number of convolutional layers
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        
        # Build the adapter network
        layers = []
        
        # Input projection
        layers.append(nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1))
        if use_layer_norm:
            layers.append(nn.GroupNorm(min(32, hidden_dim), hidden_dim))
        layers.append(nn.SiLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            if use_layer_norm:
                layers.append(nn.GroupNorm(min(32, hidden_dim), hidden_dim))
            layers.append(nn.SiLU())
        
        # Output projection
        layers.append(nn.Conv2d(hidden_dim, output_channels, kernel_size=3, padding=1))
        
        self.network = nn.Sequential(*layers)
        
        # Residual projection
        if use_residual:
            self.residual_proj = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        else:
            self.residual_proj = None
        
        # Learnable scaling parameters
        self.output_scale = nn.Parameter(torch.ones(output_channels, 1, 1))
        self.output_bias = nn.Parameter(torch.zeros(output_channels, 1, 1))
        
        # Field-specific heads for better physics reconstruction
        self.field_heads = nn.ModuleList([
            nn.Conv2d(output_channels, 1, kernel_size=1)
            for _ in range(output_channels)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C_in, H, W) or (T, C_in, H, W)
               where C_in = input_channels (3)
               
        Returns:
            Output tensor of shape (B, C_out, H, W) or (T, C_out, H, W)
            where C_out = output_channels (4)
        """
        # Main path
        out = self.network(x)
        
        # Residual path
        if self.use_residual and self.residual_proj is not None:
            residual = self.residual_proj(x)
            out = out + residual
        
        # Apply field-specific refinement
        refined_fields = []
        for i, head in enumerate(self.field_heads):
            field = head(out)
            refined_fields.append(field)
        out = torch.cat(refined_fields, dim=1)
        
        # Apply learnable scaling
        out = out * self.output_scale + self.output_bias
        
        return out


class ChannelAdapterPair(nn.Module):
    """
    Combined forward and inverse channel adapters.
    
    This module manages both the physics->video and video->physics transformations,
    ensuring consistency between them.
    """
    
    def __init__(
        self,
        physics_channels: int = 4,
        video_channels: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_residual: bool = True,
    ):
        """
        Args:
            physics_channels: Number of physics channels (4)
            video_channels: Number of video channels (3)
            hidden_dim: Hidden dimension in adapters
            num_layers: Number of convolutional layers
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.physics_channels = physics_channels
        self.video_channels = video_channels
        
        # Forward adapter: physics -> video
        self.encoder = ChannelAdapter(
            input_channels=physics_channels,
            output_channels=video_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_residual=use_residual,
        )
        
        # Inverse adapter: video -> physics
        self.decoder = InverseChannelAdapter(
            input_channels=video_channels,
            output_channels=physics_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_residual=use_residual,
        )
        
        # Output bounds for clamping (None = no clamping)
        # Should be a tensor of shape (physics_channels, 2) where [:, 0] = min, [:, 1] = max
        self.register_buffer('output_bounds', None)
    
    def set_output_bounds(self, bounds: torch.Tensor):
        """
        Set per-field output bounds for clamping predictions.
        
        Args:
            bounds: Tensor of shape (physics_channels, 2) with [min, max] for each field.
                    Fields are: [density, pressure, velocity_x, velocity_y]
        """
        if bounds.shape[0] != self.physics_channels or bounds.shape[1] != 2:
            raise ValueError(f"bounds should be shape ({self.physics_channels}, 2), got {bounds.shape}")
        self.register_buffer('output_bounds', bounds)
        print(f"Output bounds set: {bounds}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Convert physics to video format."""
        return self.encoder(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Convert video back to physics format with optional clamping."""
        output = self.decoder(x)
        
        # Apply per-field clamping if bounds are set
        if self.output_bounds is not None:
            # output shape: (B, C, H, W)
            for c in range(self.physics_channels):
                min_val = self.output_bounds[c, 0]
                max_val = self.output_bounds[c, 1]
                output[:, c] = torch.clamp(output[:, c], min=min_val.item(), max=max_val.item())
        
        return output
    
    def forward(
        self, 
        x: torch.Tensor, 
        direction: str = "encode"
    ) -> torch.Tensor:
        """
        Forward pass in specified direction.
        
        Args:
            x: Input tensor
            direction: "encode" (physics->video) or "decode" (video->physics)
            
        Returns:
            Transformed tensor
        """
        if direction == "encode":
            return self.encode(x)
        elif direction == "decode":
            return self.decode(x)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def get_reconstruction_loss(self, physics_input: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss for adapter training.
        
        This loss encourages the encode-decode cycle to be identity-like.
        
        Args:
            physics_input: Input physics tensor (B, 4, H, W)
            
        Returns:
            Reconstruction loss scalar
        """
        encoded = self.encode(physics_input)
        reconstructed = self.decode(encoded)
        return F.mse_loss(reconstructed, physics_input)


class SpatialAdapter(nn.Module):
    """
    Adapter for spatial resolution changes between physics data and video model.
    
    The Well data is 128x384, while Wan2.2 might expect different sizes.
    This provides a learnable upsampling/downsampling.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (128, 384),
        output_size: Tuple[int, int] = (480, 832),
        channels: int = 3,
        learned: bool = True,
    ):
        """
        Args:
            input_size: Input spatial size (H, W)
            output_size: Output spatial size (H, W)
            channels: Number of channels
            learned: Whether to use learned upsampling vs bilinear
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.learned = learned
        
        if learned:
            # Use a small network for learned upsampling
            self.upsample = nn.Sequential(
                nn.Upsample(size=output_size, mode='bilinear', align_corners=False),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            )
        else:
            self.upsample = nn.Upsample(
                size=output_size, mode='bilinear', align_corners=False
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample input to output size.
        
        Args:
            x: Input tensor (B, C, H_in, W_in)
            
        Returns:
            Upsampled tensor (B, C, H_out, W_out)
        """
        return self.upsample(x)


class InverseSpatialAdapter(nn.Module):
    """
    Inverse spatial adapter for downsampling video output back to physics size.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (480, 832),
        output_size: Tuple[int, int] = (128, 384),
        channels: int = 3,
        learned: bool = True,
    ):
        """
        Args:
            input_size: Input spatial size (H, W)
            output_size: Output spatial size (H, W)
            channels: Number of channels
            learned: Whether to use learned downsampling vs bilinear
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.learned = learned
        
        if learned:
            # Use a small network for learned downsampling
            self.downsample = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.Upsample(size=output_size, mode='bilinear', align_corners=False),
            )
        else:
            self.downsample = nn.Upsample(
                size=output_size, mode='bilinear', align_corners=False
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample input to output size.
        
        Args:
            x: Input tensor (B, C, H_in, W_in)
            
        Returns:
            Downsampled tensor (B, C, H_out, W_out)
        """
        return self.downsample(x)
