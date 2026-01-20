"""
Temporal prediction models for latent space prediction.

These models predict future latent states from past latent states,
enabling physics prediction using Wan2.2's VAE latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for spatiotemporal prediction.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Gates: input, forget, output, cell candidate
        self.conv_gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            hidden_state: Tuple of (h, c) each (B, hidden_channels, H, W)
            
        Returns:
            h: Output hidden state
            (h, c): New hidden state tuple
        """
        B, C, H, W = x.shape
        
        if hidden_state is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Compute all gates at once
        gates = self.conv_gates(combined)
        
        # Split into individual gates
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Cell candidate
        
        # Update cell state and hidden state
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM for sequence modeling.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # Create LSTM cells for each layer
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through sequence.
        
        Args:
            x: Input sequence (B, T, C, H, W)
            hidden_states: List of (h, c) tuples for each layer
            
        Returns:
            output: Output sequence (B, T, hidden_channels, H, W)
            hidden_states: Updated hidden states
        """
        B, T, C, H, W = x.shape
        
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
        
        outputs = []
        
        for t in range(T):
            x_t = x[:, t]  # (B, C, H, W)
            
            for layer_idx, cell in enumerate(self.cells):
                x_t, hidden_states[layer_idx] = cell(x_t, hidden_states[layer_idx])
            
            outputs.append(x_t)
        
        output = torch.stack(outputs, dim=1)  # (B, T, hidden_channels, H, W)
        
        return output, hidden_states


class LatentTemporalPredictor(nn.Module):
    """
    Temporal predictor that works in Wan2.2's VAE latent space.
    
    Takes a sequence of latent frames and predicts future latent frames.
    """
    
    def __init__(
        self,
        latent_channels: int = 16,  # Wan2.2 VAE has 16 latent channels
        hidden_channels: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        
        # Input projection
        self.input_proj = nn.Conv2d(
            latent_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        
        # ConvLSTM for temporal modeling
        self.conv_lstm = ConvLSTM(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )
        
        # Output projection back to latent space
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )
        
        # Residual connection scale
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(
        self,
        input_latents: torch.Tensor,
        num_future_frames: int = 1,
        hidden_states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Predict future latent frames.
        
        Args:
            input_latents: Input latent sequence (B, T_in, C, H, W)
            num_future_frames: Number of future frames to predict
            hidden_states: Optional initial hidden states
            
        Returns:
            predicted_latents: Predicted future latents (B, num_future_frames, C, H, W)
            hidden_states: Updated hidden states (for autoregressive generation)
        """
        B, T_in, C, H, W = input_latents.shape
        
        # Project input to hidden dimension
        input_proj = rearrange(input_latents, "B T C H W -> (B T) C H W")
        input_proj = self.input_proj(input_proj)
        input_proj = rearrange(input_proj, "(B T) C H W -> B T C H W", B=B, T=T_in)
        
        # Process input sequence through ConvLSTM
        _, hidden_states = self.conv_lstm(input_proj, hidden_states)
        
        # Autoregressive prediction of future frames
        predictions = []
        last_latent = input_latents[:, -1]  # (B, C, H, W)
        
        for _ in range(num_future_frames):
            # Project last latent
            h = self.input_proj(last_latent).unsqueeze(1)  # (B, 1, hidden, H, W)
            
            # One step through ConvLSTM
            output, hidden_states = self.conv_lstm(h, hidden_states)
            
            # Project back to latent space
            output = output[:, 0]  # (B, hidden, H, W)
            delta = self.output_proj(output)  # (B, C, H, W)
            
            # Residual prediction: predict change from last frame
            predicted = last_latent + self.residual_scale * delta
            predictions.append(predicted)
            
            # Use prediction as next input
            last_latent = predicted
        
        predicted_latents = torch.stack(predictions, dim=1)  # (B, num_future, C, H, W)
        
        return predicted_latents, hidden_states


class SimpleTemporalPredictor(nn.Module):
    """
    Simpler temporal predictor using 3D convolutions.
    
    Faster to train than ConvLSTM, good for initial experiments.
    """
    
    def __init__(
        self,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        num_input_frames: int = 4,
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.num_input_frames = num_input_frames
        
        # 3D conv to process temporal sequence
        self.encoder = nn.Sequential(
            nn.Conv3d(latent_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GELU(),
        )
        
        # Temporal aggregation (reduce time dimension)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )
        
        # Residual scale
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(
        self,
        input_latents: torch.Tensor,
        num_future_frames: int = 1,
    ) -> torch.Tensor:
        """
        Predict future latent frames.
        
        Args:
            input_latents: Input latent sequence (B, T_in, C, H, W)
            num_future_frames: Number of future frames to predict
            
        Returns:
            predicted_latents: Predicted future latents (B, num_future_frames, C, H, W)
        """
        B, T_in, C, H, W = input_latents.shape
        
        # Rearrange for 3D conv: (B, C, T, H, W)
        x = rearrange(input_latents, "B T C H W -> B C T H W")
        
        # Encode temporal sequence
        features = self.encoder(x)  # (B, hidden, T, H, W)
        
        # Pool over time
        features = self.temporal_pool(features)  # (B, hidden, 1, H, W)
        features = features.squeeze(2)  # (B, hidden, H, W)
        
        # Predict delta from last frame
        delta = self.predictor(features)  # (B, C, H, W)
        
        # Generate predictions autoregressively
        predictions = []
        last_latent = input_latents[:, -1]  # (B, C, H, W)
        
        for i in range(num_future_frames):
            # For first prediction, use the computed delta
            # For subsequent predictions, we'd need to re-encode
            # (simplified: just apply same delta scaled by step)
            predicted = last_latent + self.residual_scale * delta * (i + 1)
            predictions.append(predicted)
        
        predicted_latents = torch.stack(predictions, dim=1)
        
        return predicted_latents


def create_temporal_predictor(
    predictor_type: str = "convlstm",
    latent_channels: int = 16,
    hidden_channels: int = 64,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create temporal predictor.
    
    Args:
        predictor_type: "convlstm" or "simple"
        latent_channels: Number of latent channels (16 for Wan2.2)
        hidden_channels: Hidden dimension
        **kwargs: Additional arguments for specific predictor
        
    Returns:
        Temporal predictor module
    """
    if predictor_type == "convlstm":
        return LatentTemporalPredictor(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            num_layers=kwargs.get("num_layers", 2),
            kernel_size=kwargs.get("kernel_size", 3),
        )
    elif predictor_type == "simple":
        return SimpleTemporalPredictor(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            num_input_frames=kwargs.get("num_input_frames", 4),
        )
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
