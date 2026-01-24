"""
Physics Encoder/Decoder for HumanTFM-style one-shot prediction.

Based on the HumanTFM paper methodology:
- PhysicsEncoder: Maps 4ch physics (128×384) → 3ch video space (480×832)
- PhysicsDecoder: Maps 3ch decoded video (480×832) → 4ch physics (128×384)

These modules replace the previous ChannelAdapter + SpatialAdapter combination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange


class PhysicsEncoder(nn.Module):
    """
    Encode physics fields to video space for VAE input.
    
    Maps: 4ch physics (128, 384) → 3ch video (-1, 1) for VAE (480, 832)
    
    The architecture uses:
    1. Initial conv to expand channels
    2. Learned spatial upsampling via transposed convs
    3. Final projection to 3 channels
    """
    
    def __init__(
        self,
        physics_channels: int = 4,
        video_channels: int = 3,
        hidden_dim: int = 64,
        physics_size: Tuple[int, int] = (128, 384),
        video_size: Tuple[int, int] = (480, 832),
    ):
        super().__init__()
        
        self.physics_channels = physics_channels
        self.video_channels = video_channels
        self.physics_size = physics_size
        self.video_size = video_size
        
        # Calculate upsampling factors
        self.scale_h = video_size[0] / physics_size[0]  # 480/128 = 3.75
        self.scale_w = video_size[1] / physics_size[1]  # 832/384 ≈ 2.17
        
        # === Encoder Network ===
        # Input: (B, 4, 128, 384)
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(physics_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        # Feature extraction (maintain resolution)
        self.feature_layers = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim * 2),
            nn.SiLU(),
        )
        
        # Learnable upsampling - we'll use interpolate followed by conv for flexibility
        self.upsample_refine = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        # Output projection to video channels
        self.output_proj = nn.Conv2d(hidden_dim, video_channels, kernel_size=3, padding=1)
        
        # Output scaling (learnable)
        self.output_scale = nn.Parameter(torch.ones(video_channels, 1, 1))
        self.output_bias = nn.Parameter(torch.zeros(video_channels, 1, 1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Physics data (B, 4, 128, 384) or (B, T, 4, 128, 384)
               Assumed to be normalized in [-1, 1]
            
        Returns:
            Video tensor (B, 3, 480, 832) or (B, T, 3, 480, 832)
        """
        has_time = x.dim() == 5
        if has_time:
            B, T, C, H, W = x.shape
            x = rearrange(x, 'B T C H W -> (B T) C H W')
        
        # Feature extraction at original resolution
        h = self.input_proj(x)
        h = self.feature_layers(h)
        
        # Upsample to video resolution
        h = F.interpolate(h, size=self.video_size, mode='bilinear', align_corners=False)
        
        # Refine after upsampling
        h = self.upsample_refine(h)
        
        # Project to output channels
        out = self.output_proj(h)
        
        # Apply learnable scaling
        out = out * self.output_scale + self.output_bias
        
        # Clamp to [-1, 1] for VAE
        out = torch.tanh(out)
        
        if has_time:
            out = rearrange(out, '(B T) C H W -> B T C H W', B=B, T=T)
        
        return out


class PhysicsDecoder(nn.Module):
    """
    Decode video space back to physics fields.
    
    Maps: 3ch video (480, 832) → 4ch physics (128, 384)
    
    Supports residual prediction: output delta + reference = final
    """
    
    def __init__(
        self,
        video_channels: int = 3,
        physics_channels: int = 4,
        hidden_dim: int = 64,
        video_size: Tuple[int, int] = (480, 832),
        physics_size: Tuple[int, int] = (128, 384),
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.video_channels = video_channels
        self.physics_channels = physics_channels
        self.video_size = video_size
        self.physics_size = physics_size
        self.use_residual = use_residual
        
        # === Decoder Network ===
        # Input: (B, 3, 480, 832)
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(video_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        # Feature extraction at high res (light to save memory)
        self.feature_layers = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim * 2),
            nn.SiLU(),
        )
        
        # Downsample refinement
        self.downsample_refine = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        # Output projection to physics channels
        self.output_proj = nn.Conv2d(hidden_dim, physics_channels, kernel_size=3, padding=1)
        
        # Per-field scaling (learnable)
        self.output_scale = nn.Parameter(torch.ones(physics_channels, 1, 1))
        self.output_bias = nn.Parameter(torch.zeros(physics_channels, 1, 1))
        
        # For storing normalization stats
        self.register_buffer('field_mu', None)
        self.register_buffer('field_sigma', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def zero_init_output(self):
        """
        Zero-initialize output layer for residual prediction.
        
        This ensures that at the start of training:
        - decoder output ≈ 0
        - final prediction = reference + 0 = reference
        """
        with torch.no_grad():
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
            self.output_scale.fill_(1.0)
            self.output_bias.fill_(0.0)
        print("[PhysicsDecoder] Zero-initialized output layer for residual mode")
    
    def set_normalization_stats(self, mu: torch.Tensor, sigma: torch.Tensor):
        """Set normalization statistics for denormalization."""
        self.field_mu = mu
        self.field_sigma = sigma
    
    def forward(
        self,
        x: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        denormalize: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Video data (B, 3, 480, 832) or (B, T, 3, 480, 832)
            reference: Reference frame for residual mode (B, 4, 128, 384)
                       If provided, output = reference + decoded_delta
            denormalize: If True, denormalize output using stored stats
            
        Returns:
            Physics tensor (B, 4, 128, 384) or (B, T, 4, 128, 384)
        """
        has_time = x.dim() == 5
        if has_time:
            B, T, C, H, W = x.shape
            x = rearrange(x, 'B T C H W -> (B T) C H W')
        else:
            B = x.shape[0]
        
        # Feature extraction at high resolution
        h = self.input_proj(x)
        h = self.feature_layers(h)
        
        # Downsample to physics resolution
        h = F.interpolate(h, size=self.physics_size, mode='bilinear', align_corners=False)
        
        # Refine at physics resolution
        h = self.downsample_refine(h)
        
        # Project to physics channels
        delta = self.output_proj(h)
        
        # Apply learnable scaling
        delta = delta * self.output_scale + self.output_bias
        
        if has_time:
            delta = rearrange(delta, '(B T) C H W -> B T C H W', B=B, T=T)
        
        # Residual prediction: add reference
        if self.use_residual and reference is not None:
            # Expand reference to match temporal dimension if needed
            if has_time and reference.dim() == 4:
                reference = reference.unsqueeze(1)  # (B, 1, C, H, W)
            out = reference + delta
        else:
            out = delta
        
        # Denormalize if requested
        if denormalize and self.field_mu is not None and self.field_sigma is not None:
            mu = self.field_mu.view(1, -1, 1, 1) if not has_time else self.field_mu.view(1, 1, -1, 1, 1)
            sigma = self.field_sigma.view(1, -1, 1, 1) if not has_time else self.field_sigma.view(1, 1, -1, 1, 1)
            out = out * sigma.to(out.device) + mu.to(out.device)
        
        return out


class PhysicsAdapterPair(nn.Module):
    """
    Combined physics encoder-decoder pair for HumanTFM.
    
    Provides convenience methods for the full encode-decode cycle.
    """
    
    def __init__(
        self,
        physics_channels: int = 4,
        video_channels: int = 3,
        hidden_dim: int = 64,
        physics_size: Tuple[int, int] = (128, 384),
        video_size: Tuple[int, int] = (480, 832),
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.encoder = PhysicsEncoder(
            physics_channels=physics_channels,
            video_channels=video_channels,
            hidden_dim=hidden_dim,
            physics_size=physics_size,
            video_size=video_size,
        )
        
        self.decoder = PhysicsDecoder(
            video_channels=video_channels,
            physics_channels=physics_channels,
            hidden_dim=hidden_dim,
            video_size=video_size,
            physics_size=physics_size,
            use_residual=use_residual,
        )
    
    def encode(self, physics: torch.Tensor) -> torch.Tensor:
        """Encode physics to video space."""
        return self.encoder(physics)
    
    def decode(
        self,
        video: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        denormalize: bool = False,
    ) -> torch.Tensor:
        """Decode video back to physics space."""
        return self.decoder(video, reference=reference, denormalize=denormalize)
    
    def set_normalization_stats(self, mu: torch.Tensor, sigma: torch.Tensor):
        """Set normalization stats on decoder."""
        self.decoder.set_normalization_stats(mu, sigma)
    
    def freeze_decoder(self):
        """Freeze decoder for Stage 1 training."""
        for param in self.decoder.parameters():
            param.requires_grad = False
        print("[PhysicsAdapterPair] Decoder frozen for Stage 1")
    
    def unfreeze_decoder(self):
        """Unfreeze decoder for Stage 2 training."""
        for param in self.decoder.parameters():
            param.requires_grad = True
        print("[PhysicsAdapterPair] Decoder unfrozen for Stage 2")
    
    def get_trainable_parameters(self):
        """Get all trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
