"""
Wan2.2-I2V Model Wrapper for Physics Simulation Fine-tuning.

This module provides a wrapper around the Wan2.2-I2V model that:
1. Integrates channel adapters for physics data
2. Applies LoRA for efficient fine-tuning
3. Provides training and inference interfaces
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from einops import rearrange
from tqdm import tqdm
import numpy as np

try:
    from diffusers import WanImageToVideoPipeline
except ImportError:
    WanImageToVideoPipeline = None
    print(
        "WARNING: diffusers with Wan2.2 support not found. "
        "Please install from source: pip install git+https://github.com/huggingface/diffusers"
    )

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    raise ImportError("Please install peft: pip install peft")

from .channel_adapter import ChannelAdapterPair, SpatialAdapter, InverseSpatialAdapter
from .temporal_predictor import LatentTemporalPredictor, create_temporal_predictor


class Wan22VideoModel(nn.Module):
    """
    Wan2.2-I2V model wrapper for physics simulation prediction.
    
    This model:
    1. Takes physics simulation frames as input (4 channels)
    2. Converts them to video format (3 channels) via learnable adapter
    3. Processes through the Wan2.2 video model with LoRA
    4. Converts output back to physics format (4 channels)
    
    The key insight is to treat physics simulation as a special kind of video,
    where temporal evolution follows physical laws rather than natural video dynamics.
    """
    
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        # Channel adapter config
        physics_channels: int = 4,
        video_channels: int = 3,
        adapter_hidden_dim: int = 64,
        adapter_num_layers: int = 2,
        # Spatial config
        physics_size: Tuple[int, int] = (128, 384),
        video_size: Tuple[int, int] = (480, 832),
        # LoRA config
        lora_enabled: bool = True,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # Temporal predictor config
        use_temporal_predictor: bool = False,
        temporal_predictor_type: str = "convlstm",
        temporal_hidden_channels: int = 64,
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_id: HuggingFace model ID for Wan2.2
            dtype: Model data type
            device: Device to load model on
            physics_channels: Number of physics channels (4)
            video_channels: Number of video channels (3)
            adapter_hidden_dim: Hidden dimension for channel adapters
            adapter_num_layers: Number of layers in channel adapters
            physics_size: Original physics spatial size (H, W)
            video_size: Target video spatial size (H, W)
            lora_enabled: Whether to use LoRA
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout
            lora_target_modules: Target modules for LoRA
            use_temporal_predictor: Whether to use temporal predictor in VAE latent space
            temporal_predictor_type: Type of temporal predictor ("convlstm" or "simple")
            temporal_hidden_channels: Hidden channels for temporal predictor
        """
        super().__init__()
        
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.physics_size = physics_size
        self.video_size = video_size
        self.lora_enabled = lora_enabled
        self.use_temporal_predictor = use_temporal_predictor
        
        # Initialize channel adapters
        self.channel_adapter = ChannelAdapterPair(
            physics_channels=physics_channels,
            video_channels=video_channels,
            hidden_dim=adapter_hidden_dim,
            num_layers=adapter_num_layers,
        )
        
        # Initialize spatial adapters
        self.spatial_encoder = SpatialAdapter(
            input_size=physics_size,
            output_size=video_size,
            channels=video_channels,
            learned=True,
        )
        
        self.spatial_decoder = InverseSpatialAdapter(
            input_size=video_size,
            output_size=physics_size,
            channels=video_channels,
            learned=True,
        )
        
        # Initialize temporal predictor for VAE latent space prediction
        self.temporal_predictor = None
        if use_temporal_predictor:
            # Wan2.2 VAE has 16 latent channels
            self.temporal_predictor = create_temporal_predictor(
                predictor_type=temporal_predictor_type,
                latent_channels=16,
                hidden_channels=temporal_hidden_channels,
            )
        
        # Placeholder for the main model components (loaded lazily)
        self.pipe = None
        self.transformer = None
        self.vae = None
        
        # LoRA configuration
        if lora_target_modules is None:
            lora_target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",
                "ff.net.0.proj", "ff.net.2"
            ]
        
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
        )
        
        self._model_loaded = False
    
    def load_pretrained(self, local_path: Optional[str] = None):
        """
        Load the pretrained Wan2.2 model.
        
        Args:
            local_path: Optional local path to model weights
        """
        if self._model_loaded:
            return
        
        model_path = local_path if local_path else self.model_id
        
        print(f"Loading Wan2.2 model from {model_path}...")
        
        # Load the pipeline
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
        )
        
        # Extract components
        self.transformer = self.pipe.transformer
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        
        # Apply LoRA to transformer if enabled
        if self.lora_enabled:
            print("Applying LoRA to transformer...")
            self.transformer = get_peft_model(self.transformer, self.lora_config)
            # IMPORTANT: Update the pipeline's transformer reference to use the LoRA-wrapped version
            self.pipe.transformer = self.transformer
            self.transformer.print_trainable_parameters()
        
        # Freeze non-trainable components
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Convert trainable adapters to model dtype for FSDP compatibility
        # FSDP requires all parameters to have uniform dtype
        self.channel_adapter = self.channel_adapter.to(self.dtype)
        self.spatial_encoder = self.spatial_encoder.to(self.dtype)
        self.spatial_decoder = self.spatial_decoder.to(self.dtype)
        if self.temporal_predictor is not None:
            self.temporal_predictor = self.temporal_predictor.to(self.dtype)
        
        # Move to device - both the module and the pipeline
        self.to(self.device)
        self.pipe.to(self.device)
        
        self._model_loaded = True
        print("Model loaded successfully!")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters (only those with requires_grad=True)."""
        params = []
        
        # Channel adapter parameters
        for p in self.channel_adapter.parameters():
            if p.requires_grad:
                params.append(p)
        
        # Spatial adapter parameters
        for p in self.spatial_encoder.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.spatial_decoder.parameters():
            if p.requires_grad:
                params.append(p)
        
        # Temporal predictor parameters (if enabled)
        if self.temporal_predictor is not None:
            for p in self.temporal_predictor.parameters():
                if p.requires_grad:
                    params.append(p)
        
        # LoRA parameters (if enabled)
        if self.lora_enabled and self.transformer is not None:
            for name, param in self.transformer.named_parameters():
                if param.requires_grad:
                    params.append(param)
        
        return params
    
    def freeze_adapters(self):
        """Freeze channel and spatial adapters (for Stage 2 training)."""
        self.channel_adapter.requires_grad_(False)
        self.spatial_encoder.requires_grad_(False)
        self.spatial_decoder.requires_grad_(False)
        print("Adapters frozen")
    
    def unfreeze_adapters(self):
        """Unfreeze channel and spatial adapters."""
        self.channel_adapter.requires_grad_(True)
        self.spatial_encoder.requires_grad_(True)
        self.spatial_decoder.requires_grad_(True)
        print("Adapters unfrozen")
    
    def freeze_temporal_predictor(self):
        """Freeze temporal predictor (for Stage 1 training)."""
        if self.temporal_predictor is not None:
            self.temporal_predictor.requires_grad_(False)
            print("Temporal predictor frozen")
    
    def unfreeze_temporal_predictor(self):
        """Unfreeze temporal predictor."""
        if self.temporal_predictor is not None:
            self.temporal_predictor.requires_grad_(True)
            print("Temporal predictor unfrozen")
    
    def freeze_lora(self):
        """Freeze LoRA parameters."""
        if self.lora_enabled and self.transformer is not None:
            for param in self.transformer.parameters():
                param.requires_grad_(False)
            print("LoRA frozen")
    
    def unfreeze_lora(self):
        """Unfreeze LoRA parameters."""
        if self.lora_enabled and self.transformer is not None:
            # Only unfreeze the LoRA parameters, not base model
            for name, param in self.transformer.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad_(True)
            print("LoRA unfrozen")
    
    def prepare_physics_input(
        self, 
        physics_frames: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare physics simulation frames for the video model.
        
        Args:
            physics_frames: Input frames (B, T, H, W, C) or (B, T, C, H, W)
            normalize: Whether to normalize to [-1, 1]
            
        Returns:
            Tuple of (condition_image, video_latents) ready for the model
        """
        # Ensure channel-first format
        if physics_frames.dim() == 5 and physics_frames.shape[-1] == 4:
            # (B, T, H, W, C) -> (B, T, C, H, W)
            physics_frames = rearrange(physics_frames, "B T H W C -> B T C H W")
        
        B, T, C, H, W = physics_frames.shape
        
        # Reshape for batch processing: (B*T, C, H, W)
        frames_flat = rearrange(physics_frames, "B T C H W -> (B T) C H W")
        
        # Apply channel adapter: 4 -> 3 channels
        video_frames = self.channel_adapter.encode(frames_flat)
        
        # Apply spatial adapter
        video_frames = self.spatial_encoder(video_frames)
        
        # Reshape back: (B, T, 3, H', W')
        video_frames = rearrange(
            video_frames, "(B T) C H W -> B T C H W", B=B, T=T
        )
        
        # Normalize to [-1, 1] for the diffusion model
        if normalize:
            # Assuming input is already normalized, scale to [-1, 1]
            video_frames = video_frames * 2 - 1
            video_frames = torch.clamp(video_frames, -1, 1)
        
        # First frame as condition image
        condition_image = video_frames[:, 0]  # (B, 3, H, W)
        
        # Rest as video to generate (or all if we want to reconstruct)
        target_video = video_frames[:, 1:]  # (B, T-1, 3, H, W)
        
        return condition_image, target_video
    
    def decode_to_physics(
        self,
        video_frames: torch.Tensor,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """
        Decode video model output back to physics format.
        
        Args:
            video_frames: Video output (B, T, 3, H, W)
            denormalize: Whether to denormalize from [-1, 1]
            
        Returns:
            Physics frames (B, T, 4, H, W)
        """
        B, T, C, H, W = video_frames.shape
        
        # Denormalize from [-1, 1] to [0, 1]
        if denormalize:
            video_frames = (video_frames + 1) / 2
            video_frames = torch.clamp(video_frames, 0, 1)
        
        # Reshape for batch processing
        frames_flat = rearrange(video_frames, "B T C H W -> (B T) C H W")
        
        # Apply spatial decoder
        frames_flat = self.spatial_decoder(frames_flat)
        
        # Apply channel adapter inverse: 3 -> 4 channels
        physics_frames = self.channel_adapter.decode(frames_flat)
        
        # Reshape back
        physics_frames = rearrange(
            physics_frames, "(B T) C H W -> B T C H W", B=B, T=T
        )
        
        return physics_frames
    
    def _get_vae_scaling_factor(self) -> float:
        """Get the VAE scaling factor, handling different config formats."""
        if self.vae is None:
            return 1.0
        
        # Try different ways to get the scaling factor
        if hasattr(self.vae, 'config'):
            config = self.vae.config
            # Handle FrozenDict or regular config
            if hasattr(config, 'scaling_factor'):
                return config.scaling_factor
            elif isinstance(config, dict) and 'scaling_factor' in config:
                return config['scaling_factor']
            elif hasattr(config, 'get'):
                sf = config.get('scaling_factor', None)
                if sf is not None:
                    return sf
        
        # Wan2.2 VAE doesn't use a scaling factor (it's 1.0)
        # Based on the official Wan2.2 code, they don't apply scaling
        return 1.0
    
    def encode_to_latent(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames to latent space using VAE.
        
        Args:
            frames: Video frames (B, T, C, H, W) in [-1, 1]
            
        Returns:
            Latent representation
        """
        if self.vae is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")
        
        B, T, C, H, W = frames.shape
        
        # VAE expects (B, C, T, H, W) for video
        frames = rearrange(frames, "B T C H W -> B C T H W")
        
        scaling_factor = self._get_vae_scaling_factor()
        
        with torch.no_grad():
            # Encode to latent space
            latent_dist = self.vae.encode(frames)
            # Handle different return types
            if hasattr(latent_dist, 'latent_dist'):
                latents = latent_dist.latent_dist.sample()
            elif hasattr(latent_dist, 'sample'):
                latents = latent_dist.sample()
            else:
                latents = latent_dist
            latents = latents * scaling_factor
        
        return latents
    
    def decode_from_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to video frames using VAE.
        
        Args:
            latents: Latent representation
            
        Returns:
            Video frames (B, T, C, H, W) in [-1, 1]
        """
        if self.vae is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")
        
        scaling_factor = self._get_vae_scaling_factor()
        latents = latents / scaling_factor
        
        with torch.no_grad():
            # Decode from latent space
            decoded = self.vae.decode(latents)
            # Handle different return types
            if hasattr(decoded, 'sample'):
                frames = decoded.sample
            else:
                frames = decoded
        
        # Rearrange to (B, T, C, H, W)
        frames = rearrange(frames, "B C T H W -> B T C H W")
        
        return frames
    
    def forward(
        self,
        input_frames: torch.Tensor,
        target_frames: torch.Tensor,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Training approach:
        1. Train channel adapters with reconstruction loss (encode->decode cycle)
        2. Train spatial adapters with reconstruction loss
        3. Compute prediction loss between encoded input and target
        
        This approach avoids the complexity of Wan2.2's diffusion training
        while still learning useful mappings for the physics data.
        
        Args:
            input_frames: Input physics frames (B, T_in, H, W, C)
            target_frames: Target physics frames (B, T_out, H, W, C)
            num_inference_steps: Number of diffusion steps (for reference)
            guidance_scale: Guidance scale (for reference)
            
        Returns:
            Dictionary with loss and intermediate outputs
        """
        # Ensure channel-last format: (B, T, H, W, C)
        if input_frames.dim() == 5 and input_frames.shape[2] == 4:
            # Already in (B, T, C, H, W), convert to (B, T, H, W, C)
            input_frames = rearrange(input_frames, "B T C H W -> B T H W C")
        if target_frames.dim() == 5 and target_frames.shape[2] == 4:
            target_frames = rearrange(target_frames, "B T C H W -> B T H W C")
        
        B, T_in, H, W, C = input_frames.shape
        B, T_out, _, _, _ = target_frames.shape
        
        # Flatten batch and time for processing
        input_flat = rearrange(input_frames, "B T H W C -> (B T) C H W")
        target_flat = rearrange(target_frames, "B T H W C -> (B T) C H W")
        
        # === Channel Adapter Training ===
        # Forward pass through channel adapters
        # Physics (4ch) -> Video (3ch)
        encoded_input = self.channel_adapter.encode(input_flat)
        encoded_target = self.channel_adapter.encode(target_flat)
        
        # Video (3ch) -> Physics (4ch)  (reconstruction)
        reconstructed_input = self.channel_adapter.decode(encoded_input)
        reconstructed_target = self.channel_adapter.decode(encoded_target)
        
        # Reconstruction loss for adapters
        adapter_recon_loss = (
            F.mse_loss(reconstructed_input, input_flat) +
            F.mse_loss(reconstructed_target, target_flat)
        ) / 2
        
        # === Spatial Adapter Training ===
        # Upsample to video model size
        encoded_input_upsampled = self.spatial_encoder(encoded_input)
        encoded_target_upsampled = self.spatial_encoder(encoded_target)
        
        # Downsample back
        encoded_input_downsampled = self.spatial_decoder(encoded_input_upsampled)
        encoded_target_downsampled = self.spatial_decoder(encoded_target_upsampled)
        
        # Spatial reconstruction loss
        spatial_recon_loss = (
            F.mse_loss(encoded_input_downsampled, encoded_input) +
            F.mse_loss(encoded_target_downsampled, encoded_target)
        ) / 2
        
        # === Temporal Prediction Loss ===
        # Encourage the model to learn temporal relationships
        # Use the last input frame to predict the first target frame
        last_input_encoded = encoded_input.view(B, T_in, 3, H, W)[:, -1]  # (B, 3, H, W)
        first_target_encoded = encoded_target.view(B, T_out, 3, H, W)[:, 0]  # (B, 3, H, W)
        
        # Simple temporal consistency loss
        temporal_loss = F.mse_loss(last_input_encoded, first_target_encoded)
        
        # === Full Cycle Loss ===
        # Encode physics -> video -> physics and compare with original
        full_cycle_input = self.channel_adapter.decode(
            self.spatial_decoder(
                self.spatial_encoder(
                    self.channel_adapter.encode(input_flat)
                )
            )
        )
        cycle_loss = F.mse_loss(full_cycle_input, input_flat)
        
        # === Total Loss ===
        total_loss = (
            adapter_recon_loss * 1.0 +      # Channel adapter reconstruction
            spatial_recon_loss * 0.5 +       # Spatial adapter reconstruction
            temporal_loss * 0.1 +            # Temporal consistency
            cycle_loss * 0.5                 # Full cycle consistency
        )
        
        return {
            "loss": total_loss,
            "adapter_loss": adapter_recon_loss,
            "spatial_loss": spatial_recon_loss,
            "temporal_loss": temporal_loss,
            "cycle_loss": cycle_loss,
            "encoded_input": encoded_input,
            "encoded_target": encoded_target,
        }
    
    @torch.no_grad()
    def predict_adapter_only(
        self,
        input_frames: torch.Tensor,
        num_frames: int = 8,
    ) -> torch.Tensor:
        """
        Predict future frames using only the adapters (no diffusion pipeline).
        
        This uses the same logic as training:
        1. Encode input through channel adapter (4ch → 3ch)
        2. Pass through spatial adapter cycle
        3. Decode back to physics (3ch → 4ch)
        4. Use the last frame's reconstruction as prediction
        
        This is useful for evaluating adapter quality without the diffusion model.
        
        Args:
            input_frames: Input physics frames (B, T_in, H, W, C)
            num_frames: Number of frames to "predict"
            
        Returns:
            Predicted physics frames (B, num_frames, H, W, C)
        """
        # Ensure channel-last format
        if input_frames.dim() == 5 and input_frames.shape[2] == 4:
            input_frames = rearrange(input_frames, "B T C H W -> B T H W C")
        
        B, T_in, H, W, C = input_frames.shape
        
        # Use the last input frame for prediction
        last_frame = input_frames[:, -1]  # (B, H, W, C)
        last_frame_flat = rearrange(last_frame, "B H W C -> B C H W")
        
        # Full adapter cycle: physics → video → spatial up → spatial down → physics
        encoded = self.channel_adapter.encode(last_frame_flat)  # (B, 3, H, W)
        upsampled = self.spatial_encoder(encoded)  # (B, 3, H', W')
        downsampled = self.spatial_decoder(upsampled)  # (B, 3, H, W)
        reconstructed = self.channel_adapter.decode(downsampled)  # (B, 4, H, W)
        
        # Convert back to (B, H, W, C)
        reconstructed = rearrange(reconstructed, "B C H W -> B H W C")
        
        # Repeat the reconstruction for all output frames
        # This is the "repeat last frame through adapter" baseline
        predictions = reconstructed.unsqueeze(1).expand(-1, num_frames, -1, -1, -1)
        
        return predictions.contiguous()
    
    def encode_physics_to_vae_latent(
        self,
        physics_frames: torch.Tensor,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """
        Encode physics frames all the way to VAE latent space.
        
        Pipeline: physics (4ch) -> adapters -> video (3ch) -> VAE -> latent (16ch)
        
        Args:
            physics_frames: Physics frames (B, T, H, W, C) or (B, T, C, H, W)
            requires_grad: Whether to compute gradients through VAE
            
        Returns:
            VAE latents (B, C_latent, T_latent, H_latent, W_latent)
        """
        if self.vae is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")
        
        # Ensure channel-last format: (B, T, H, W, C)
        if physics_frames.dim() == 5 and physics_frames.shape[2] == 4:
            physics_frames = rearrange(physics_frames, "B T C H W -> B T H W C")
        
        B, T, H, W, C = physics_frames.shape
        
        # Flatten batch and time for adapter processing
        frames_flat = rearrange(physics_frames, "B T H W C -> (B T) C H W")
        
        # Cast to model dtype (channel adapter may be in bfloat16)
        frames_flat = frames_flat.to(self.dtype)
        
        # Apply channel adapter: physics (4ch) -> video (3ch)
        video_frames = self.channel_adapter.encode(frames_flat)
        
        # Apply spatial upsampling to video size
        video_frames = self.spatial_encoder(video_frames)
        
        # Reshape for VAE: (B, T, C, H, W) -> (B, C, T, H, W)
        _, C_vid, H_vid, W_vid = video_frames.shape
        video_frames = rearrange(video_frames, "(B T) C H W -> B C T H W", B=B, T=T)
        
        # Normalize to [-1, 1] for VAE
        video_frames = video_frames * 2 - 1  # Assumes input is in [0, 1]
        
        # Encode to VAE latent space
        scaling_factor = self._get_vae_scaling_factor()
        
        if requires_grad:
            latent_dist = self.vae.encode(video_frames.to(self.dtype))
            if hasattr(latent_dist, 'latent_dist'):
                latents = latent_dist.latent_dist.sample()
            elif hasattr(latent_dist, 'sample'):
                latents = latent_dist.sample()
            else:
                latents = latent_dist
            latents = latents * scaling_factor
        else:
            with torch.no_grad():
                latent_dist = self.vae.encode(video_frames.to(self.dtype))
                if hasattr(latent_dist, 'latent_dist'):
                    latents = latent_dist.latent_dist.sample()
                elif hasattr(latent_dist, 'sample'):
                    latents = latent_dist.sample()
                else:
                    latents = latent_dist
                latents = latents * scaling_factor
        
        return latents.float()
    
    def decode_vae_latent_to_physics(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode VAE latents back to physics frames.
        
        Pipeline: latent (16ch) -> VAE decoder -> video (3ch) -> adapters -> physics (4ch)
        
        Args:
            latents: VAE latents (B, C_latent, T_latent, H_latent, W_latent)
            
        Returns:
            Physics frames (B, T, H, W, C)
        """
        if self.vae is None:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")
        
        scaling_factor = self._get_vae_scaling_factor()
        latents_scaled = latents.to(self.dtype) / scaling_factor
        
        # Decode from VAE latent space
        with torch.no_grad():
            decoded = self.vae.decode(latents_scaled)
            if hasattr(decoded, 'sample'):
                video_frames = decoded.sample
            else:
                video_frames = decoded
        
        video_frames = video_frames.float()
        
        # video_frames is (B, C, T, H, W), convert to (B*T, C, H, W)
        B, C, T, H_vid, W_vid = video_frames.shape
        video_frames = rearrange(video_frames, "B C T H W -> (B T) C H W")
        
        # Denormalize from [-1, 1] to [0, 1]
        video_frames = (video_frames + 1) / 2
        video_frames = torch.clamp(video_frames, 0, 1)
        
        # Cast to model dtype (adapters may be in bfloat16)
        video_frames = video_frames.to(self.dtype)
        
        # Apply spatial decoder to reduce resolution
        video_frames = self.spatial_decoder(video_frames)
        
        # Apply channel adapter: video (3ch) -> physics (4ch)
        physics_frames = self.channel_adapter.decode(video_frames)
        
        # Reshape back to (B, T, H, W, C)
        physics_frames = rearrange(physics_frames, "(B T) C H W -> B T H W C", B=B, T=T)
        
        return physics_frames
    
    def forward_with_temporal_predictor(
        self,
        input_frames: torch.Tensor,
        target_frames: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with temporal prediction in VAE latent space.
        
        This trains:
        1. Channel adapters (physics <-> video)
        2. Spatial adapters (physics resolution <-> video resolution)
        3. Temporal predictor (predict future latents from past latents)
        
        Args:
            input_frames: Input physics frames (B, T_in, H, W, C) or (B, T_in, C, H, W)
            target_frames: Target physics frames (B, T_out, H, W, C) or (B, T_out, C, H, W)
            
        Returns:
            Dictionary with losses and intermediate outputs
        """
        if self.temporal_predictor is None:
            raise RuntimeError("Temporal predictor not initialized. Set use_temporal_predictor=True.")
        
        # Ensure channel-last format: (B, T, H, W, C)
        if input_frames.dim() == 5 and input_frames.shape[2] == 4:
            input_frames = rearrange(input_frames, "B T C H W -> B T H W C")
        if target_frames.dim() == 5 and target_frames.shape[2] == 4:
            target_frames = rearrange(target_frames, "B T C H W -> B T H W C")
        
        B, T_in, H, W, C = input_frames.shape
        _, T_out, _, _, _ = target_frames.shape
        
        # === Step 1: Adapter Training (same as before) ===
        # Flatten for adapter processing
        input_flat = rearrange(input_frames, "B T H W C -> (B T) C H W")
        target_flat = rearrange(target_frames, "B T H W C -> (B T) C H W")
        
        # Channel adapter: physics -> video -> physics
        encoded_input = self.channel_adapter.encode(input_flat)
        encoded_target = self.channel_adapter.encode(target_flat)
        reconstructed_input = self.channel_adapter.decode(encoded_input)
        reconstructed_target = self.channel_adapter.decode(encoded_target)
        
        adapter_recon_loss = (
            F.mse_loss(reconstructed_input, input_flat) +
            F.mse_loss(reconstructed_target, target_flat)
        ) / 2
        
        # Spatial adapter: up -> down cycle
        up_input = self.spatial_encoder(encoded_input)
        up_target = self.spatial_encoder(encoded_target)
        down_input = self.spatial_decoder(up_input)
        down_target = self.spatial_decoder(up_target)
        
        spatial_recon_loss = (
            F.mse_loss(down_input, encoded_input) +
            F.mse_loss(down_target, encoded_target)
        ) / 2
        
        # === Step 2: Encode to VAE latent space ===
        # Note: VAE is frozen, but we still compute latents for temporal prediction
        input_latents = self.encode_physics_to_vae_latent(input_frames, requires_grad=False)
        target_latents = self.encode_physics_to_vae_latent(target_frames, requires_grad=False)
        
        # Debug: print shapes to understand VAE compression
        if not hasattr(self, '_logged_shapes'):
            print(f"\n[DEBUG] VAE temporal compression analysis:")
            print(f"  Input frames: {T_in} → Input latents: {input_latents.shape[2]} (ratio: {T_in/input_latents.shape[2]:.2f}x)")
            print(f"  Target frames: {T_out} → Target latents: {target_latents.shape[2]} (ratio: {T_out/target_latents.shape[2]:.2f}x)")
            print(f"  Latent shape: {input_latents.shape}")
            self._logged_shapes = True
        
        # input_latents: (B, C_lat, T_in_lat, H_lat, W_lat)
        # Rearrange for temporal predictor: (B, T, C, H, W)
        input_latents_seq = rearrange(input_latents, "B C T H W -> B T C H W")
        target_latents_seq = rearrange(target_latents, "B C T H W -> B T C H W")
        
        # === Step 3: Temporal Prediction ===
        # Calculate how many latent frames we need to decode to get T_out physics frames
        # VAE decode formula: T_decoded = (T_latent - 1) * 4 + 1
        # So to get at least T_out frames: T_latent >= (T_out - 1) / 4 + 1
        T_latent_needed = math.ceil((T_out - 1) / 4) + 1
        T_target_lat = target_latents_seq.shape[1]
        
        # Use the max of what we encoded and what we need for decoding
        T_predict = max(T_target_lat, T_latent_needed)
        
        if hasattr(self.temporal_predictor, 'forward'):
            result = self.temporal_predictor(input_latents_seq, num_future_frames=T_predict)
            if isinstance(result, tuple):
                predicted_latents, _ = result
            else:
                predicted_latents = result
        
        # Temporal prediction loss in latent space (only on frames we have targets for)
        temporal_pred_loss = F.mse_loss(
            predicted_latents[:, :T_target_lat], 
            target_latents_seq
        )
        
        # === Step 4: Decode predicted latents and compute physics loss ===
        # Rearrange back for VAE decoder
        predicted_latents_vae = rearrange(predicted_latents, "B T C H W -> B C T H W")
        
        # Decode to physics frames
        predicted_physics = self.decode_vae_latent_to_physics(predicted_latents_vae)
        
        # Debug: check decode output
        if not hasattr(self, '_logged_decode'):
            print(f"  Predicted {T_predict} latent frames → Decoded to {predicted_physics.shape[1]} frames")
            print(f"  Will truncate to {T_out} target frames")
            self._logged_decode = True
        
        # Truncate decoded frames to match target count
        # VAE decode: T_decoded = (T_latent - 1) * 4 + 1, so we may have extra frames
        predicted_physics = predicted_physics[:, :T_out]
        
        # Physics reconstruction loss (now shapes should match)
        physics_pred_loss = F.mse_loss(predicted_physics, target_frames)
        
        # === Total Loss ===
        # Note: temporal_pred_loss in latent space is the primary objective
        total_loss = (
            adapter_recon_loss * 1.0 +      # Channel adapter reconstruction
            spatial_recon_loss * 0.5 +       # Spatial adapter reconstruction
            temporal_pred_loss * 2.0 +       # Temporal prediction in latent space (main objective)
            physics_pred_loss * 1.0          # Physics prediction loss
        )
        
        return {
            "loss": total_loss,
            "adapter_loss": adapter_recon_loss,
            "spatial_loss": spatial_recon_loss,
            "temporal_pred_loss": temporal_pred_loss,
            "physics_pred_loss": physics_pred_loss,
            "predicted_physics": predicted_physics,
            "target_physics": target_frames,
        }
    
    @torch.no_grad()
    def predict_with_temporal_predictor(
        self,
        input_frames: torch.Tensor,
        num_frames: int = 8,
    ) -> torch.Tensor:
        """
        Predict future frames using the temporal predictor in VAE latent space.
        
        Pipeline:
        1. Encode input physics frames to VAE latents
        2. Use temporal predictor to predict future latents
        3. Decode predicted latents back to physics frames
        
        Args:
            input_frames: Input physics frames (B, T_in, H, W, C) or (B, T_in, C, H, W)
            num_frames: Number of future frames to predict
            
        Returns:
            Predicted physics frames (B, num_frames, H, W, C)
        """
        if self.temporal_predictor is None:
            raise RuntimeError("Temporal predictor not initialized. Set use_temporal_predictor=True.")
        
        # Ensure channel-last format
        if input_frames.dim() == 5 and input_frames.shape[2] == 4:
            input_frames = rearrange(input_frames, "B T C H W -> B T H W C")
        
        # Step 1: Encode to VAE latent space
        input_latents = self.encode_physics_to_vae_latent(input_frames, requires_grad=False)
        
        # Rearrange for temporal predictor: (B, T, C, H, W)
        input_latents_seq = rearrange(input_latents, "B C T H W -> B T C H W")
        
        # Cast to model dtype (temporal predictor may be in bfloat16)
        input_latents_seq = input_latents_seq.to(self.dtype)
        
        # Step 2: Calculate how many latent frames needed for num_frames output
        # VAE decode formula: T_decoded = (T_latent - 1) * 4 + 1
        # So to get at least num_frames: T_latent >= (num_frames - 1) / 4 + 1
        T_latent_needed = math.ceil((num_frames - 1) / 4) + 1
        
        # Predict future latents
        result = self.temporal_predictor(input_latents_seq, num_future_frames=T_latent_needed)
        if isinstance(result, tuple):
            predicted_latents, _ = result
        else:
            predicted_latents = result
        
        # Rearrange for VAE decoder: (B, C, T, H, W)
        predicted_latents_vae = rearrange(predicted_latents, "B T C H W -> B C T H W")
        
        # Step 3: Decode to physics frames
        predicted_physics = self.decode_vae_latent_to_physics(predicted_latents_vae)
        
        # Truncate to requested number of frames
        predicted_physics = predicted_physics[:, :num_frames]
        
        return predicted_physics
    
    def _get_empty_prompt_embeds(self, batch_size: int) -> torch.Tensor:
        """Get text embeddings for empty/physics prompt."""
        # Use a simple physics-related prompt
        prompt = "Physics simulation of fluid dynamics"
        
        # Get a sensible max length (avoid overflow from huge model_max_length)
        max_length = min(
            getattr(self.tokenizer, 'model_max_length', 512),
            512  # Reasonable upper bound
        )
        
        inputs = self.tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                inputs.input_ids.to(self.device)
            )[0]
        
        return prompt_embeds
    
    def _encode_condition_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode condition image to latent."""
        # Add time dimension for VAE
        image = image.unsqueeze(2)  # (B, C, 1, H, W)
        
        scaling_factor = self._get_vae_scaling_factor()
        
        with torch.no_grad():
            encoded = self.vae.encode(image)
            # Handle different return types
            if hasattr(encoded, 'latent_dist'):
                latent = encoded.latent_dist.sample()
            elif hasattr(encoded, 'sample'):
                latent = encoded.sample()
            else:
                latent = encoded
            latent = latent * scaling_factor
        
        return latent
    
    @torch.no_grad()
    def generate(
        self,
        input_frames: torch.Tensor,
        num_frames: int = 16,
        num_inference_steps: int = 40,
        guidance_scale: float = 3.5,
    ) -> torch.Tensor:
        """
        Generate future physics frames given initial conditions.
        
        Args:
            input_frames: Input physics frames (B, T_in, H, W, C)
            num_frames: Number of frames to generate
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            
        Returns:
            Generated physics frames (B, num_frames, H, W, C)
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")
        
        # Prepare condition image from first input frame
        condition_image, _ = self.prepare_physics_input(input_frames)
        
        # Use the pipeline for generation
        B = condition_image.shape[0]
        
        # Get dimensions
        height, width = self.video_size
        
        # Generate using pipeline
        # Note: This is a simplified version; full implementation would
        # need proper handling of the pipeline's generate method
        output = self.pipe(
            image=condition_image,
            prompt="Physics simulation of turbulent fluid dynamics",
            negative_prompt="",
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        # Convert output frames to physics format
        generated_video = torch.stack(
            [torch.from_numpy(np.array(f)) for f in output.frames[0]]
        ).to(self.device)
        
        # Normalize and convert channels
        generated_video = generated_video.float() / 255.0
        generated_video = rearrange(generated_video, "T H W C -> 1 T C H W")
        
        # Decode to physics format
        physics_output = self.decode_to_physics(generated_video)
        
        # Rearrange to expected format
        physics_output = rearrange(physics_output, "B T C H W -> B T H W C")
        
        # Slice to match requested number of frames
        # (Wan2.2 may generate more frames due to divisibility requirements)
        if physics_output.shape[1] > num_frames:
            physics_output = physics_output[:, :num_frames]
        
        return physics_output
    
    # =========================================================================
    # Native I2V Diffusion Methods
    # =========================================================================
    
    def _prepare_i2v_conditioning(
        self,
        cond_frame_video: torch.Tensor,
        num_output_frames: int = 8,
    ) -> torch.Tensor:
        """
        Prepare I2V conditioning: mask (4ch) + VAE-encoded conditioning frame (16ch) = 20ch.
        
        Args:
            cond_frame_video: Conditioning frame in video format (B, 3, H, W) in [-1, 1]
            num_output_frames: Number of frames to generate
            
        Returns:
            Conditioning tensor (B, 20, T_latent, H_latent, W_latent)
        """
        B = cond_frame_video.shape[0]
        H_vid, W_vid = cond_frame_video.shape[2:]
        
        # Compute latent dimensions
        # Wan2.2 VAE has stride (4, 8, 8) for (T, H, W)
        vae_stride = (4, 8, 8)
        H_lat = H_vid // vae_stride[1]
        W_lat = W_vid // vae_stride[2]
        T_lat = (num_output_frames - 1) // vae_stride[0] + 1
        
        # Create video tensor: cond_frame + zeros for frames to generate
        # Shape: (B, 3, T, H, W)
        video_for_encoding = torch.zeros(
            B, 3, num_output_frames, H_vid, W_vid,
            device=cond_frame_video.device, dtype=cond_frame_video.dtype
        )
        video_for_encoding[:, :, 0] = cond_frame_video  # First frame = conditioning
        
        # VAE encode
        with torch.no_grad():
            # VAE expects (B, C, T, H, W)
            latent_dist = self.vae.encode(video_for_encoding.to(self.dtype))
            if hasattr(latent_dist, 'latent_dist'):
                cond_latent = latent_dist.latent_dist.sample()
            else:
                cond_latent = latent_dist.sample() if hasattr(latent_dist, 'sample') else latent_dist
        
        # Scale by VAE scaling factor
        scaling_factor = self._get_vae_scaling_factor()
        cond_latent = cond_latent * scaling_factor  # (B, 16, T_lat, H_lat, W_lat)
        
        # Create binary mask: 1 for conditioning frame, 0 for frames to generate
        # Mask needs same shape as latent but with 4 channels
        mask = torch.zeros(B, 4, T_lat, H_lat, W_lat, device=cond_latent.device, dtype=cond_latent.dtype)
        mask[:, :, 0] = 1.0  # First latent frame is the conditioning
        
        # Concatenate: [mask (4ch), cond_latent (16ch)] = 20ch
        y = torch.cat([mask, cond_latent], dim=1)  # (B, 20, T_lat, H_lat, W_lat)
        
        return y
    
    def forward_i2v_diffusion(
        self,
        cond_frame: torch.Tensor,
        target_frames: torch.Tensor,
        text_prompt: str = "Top-down view of fluid dynamics simulation, evolving turbulence, scientific visualization, accurate physics (turbulent radiative layer 2d)",
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass using native I2V diffusion.
        
        Wan 2.2 I2V conditioning works by:
        1. Concatenating [mask (4ch), cond_latent (16ch)] = 20ch conditioning
        2. Concatenating conditioning with noisy latent: [noisy (16ch), cond (20ch)] = 36ch
        3. Passing the 36-channel tensor as hidden_states to transformer
        
        Args:
            cond_frame: Conditioning physics frame (B, 1, H, W, C) or (B, H, W, C)
            target_frames: Target physics frames (B, T, H, W, C)
            text_prompt: Text prompt for conditioning
            
        Returns:
            Dictionary with loss and intermediate outputs
        """
        # Handle input shapes
        if cond_frame.dim() == 5:
            cond_frame = cond_frame[:, 0]  # (B, H, W, C)
        if cond_frame.shape[-1] == 4:  # Channel-last
            cond_frame = rearrange(cond_frame, "B H W C -> B C H W")
        if target_frames.shape[2] == 4:  # Channel-first check for 5D
            pass  # Already (B, T, C, H, W)
        elif target_frames.shape[-1] == 4:  # Channel-last
            target_frames = rearrange(target_frames, "B T H W C -> B T C H W")
        
        B, T_out = target_frames.shape[:2]
        
        # === Step 1: Adapt physics → video ===
        # Conditioning frame
        cond_video = self.channel_adapter.encode(cond_frame.to(self.dtype))  # (B, 3, H, W)
        cond_video = self.spatial_encoder(cond_video)  # Upsample to video size
        
        # Target frames 
        target_flat = rearrange(target_frames, "B T C H W -> (B T) C H W")
        target_video_flat = self.channel_adapter.encode(target_flat.to(self.dtype))
        target_video_flat = self.spatial_encoder(target_video_flat)
        target_video_only = rearrange(target_video_flat, "(B T) C H W -> B C T H W", B=B, T=T_out)
        
        # === CRITICAL FIX: Full video MUST include conditioning at position 0 ===
        # I2V expects: [cond_frame, target_frame0, target_frame1, ..., target_frameN-1]
        # This ensures latent[0] contains conditioning, matching the mask
        cond_video_expanded = cond_video.unsqueeze(2)  # (B, 3, 1, H, W)
        full_video = torch.cat([cond_video_expanded, target_video_only], dim=2)  # (B, 3, 1+T_out, H, W)
        
        # We now have 1 + T_out frames. Use all of them for proper alignment.
        n_video_frames = full_video.shape[2]
        
        # === Step 2: VAE encode the FULL video (cond + targets) ===
        scaling_factor = self._get_vae_scaling_factor()
        with torch.no_grad():
            latent_dist = self.vae.encode(full_video)
            if hasattr(latent_dist, 'latent_dist'):
                target_latent = latent_dist.latent_dist.sample()
            else:
                target_latent = latent_dist.sample() if hasattr(latent_dist, 'sample') else latent_dist
        target_latent = target_latent * scaling_factor  # (B, 16, T_lat, H_lat, W_lat)
        
        # === Step 3: Prepare I2V conditioning (20ch: mask + cond_latent) ===
        # Use n_video_frames to match the full video length
        conditioning = self._prepare_i2v_conditioning(cond_video, num_output_frames=n_video_frames)
        # conditioning shape: (B, 20, T_lat, H_lat, W_lat)
        
        # === Step 4: Sample timestep and add noise ===
        timesteps = torch.randint(
            0, self.pipe.scheduler.config.num_train_timesteps,
            (B,), device=target_latent.device
        )
        noise = torch.randn_like(target_latent)
        noisy_latent = self.pipe.scheduler.add_noise(target_latent, noise, timesteps)
        
        # === Step 5: Concatenate noisy latent with conditioning ===
        # Wan 2.2 I2V: hidden_states = [noisy_latent (16ch), conditioning (20ch)] = 36ch
        hidden_states = torch.cat([noisy_latent, conditioning], dim=1)  # (B, 36, T_lat, H_lat, W_lat)
        
        # === Step 6: Get text embeddings ===
        with torch.no_grad():
            text_inputs = self.pipe.tokenizer(
                [text_prompt] * B,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            text_embeds = self.pipe.text_encoder(**text_inputs)[0]
        
        # === Step 7: Predict noise with transformer ===
        # Diffusers WanTransformer3DModel expects:
        # - hidden_states: (B, C, T, H, W) - the 36-channel input
        # - timestep: (B,) - diffusion timestep
        # - encoder_hidden_states: (B, L, D) - text embeddings
        model_output = self.transformer(
            hidden_states=hidden_states,
            timestep=timesteps,
            encoder_hidden_states=text_embeds,
            return_dict=False,
        )[0]
        
        # === Step 8: Compute loss ===
        # The model outputs noise prediction, loss = MSE(pred, noise)
        diffusion_loss = F.mse_loss(model_output, noise)
        
        # === VAE Reconstruction Loss (CRITICAL for decoder training) ===
        # This trains the decoder to handle VAE-decoded video frames
        # Note: target_latent encodes [cond, target0, target1, ...] so decoded frame 0 = cond
        with torch.no_grad():
            # Decode the target latent through VAE (what happens at inference time)
            vae_decoded_video = self.vae.decode(target_latent / scaling_factor, return_dict=False)[0]
            # Denormalize VAE output from [-1, 1] to [0, 1]
            vae_decoded_video = (vae_decoded_video + 1) / 2
            vae_decoded_video = torch.clamp(vae_decoded_video, 0, 1)
        
        # Now pass through spatial decoder and channel adapter decoder (trainable)
        vae_decoded_flat = rearrange(vae_decoded_video, "B C T H W -> (B T) C H W")
        vae_decoded_flat = vae_decoded_flat.to(self.dtype)
        vae_decoded_spatial = self.spatial_decoder(vae_decoded_flat)
        vae_recon_physics = self.channel_adapter.decode(vae_decoded_spatial)
        T_decoded = vae_decoded_video.shape[2]
        vae_recon_physics = rearrange(vae_recon_physics, "(B T) C H W -> B T C H W", B=B, T=T_decoded)
        
        # Frame alignment: decoded = [cond, target0, target1, ...]
        # frame 0 should match cond_frame, frames 1+ should match target_frames
        # Build the expected physics: [cond_frame, target_frames]
        cond_frame_expanded = cond_frame.unsqueeze(1)  # (B, 1, C, H, W)
        expected_physics = torch.cat([cond_frame_expanded, target_frames], dim=1)  # (B, 1+T_out, C, H, W)
        
        # Handle temporal dimension mismatch (VAE compression may reduce frames)
        T_recon = vae_recon_physics.shape[1]
        T_expected = expected_physics.shape[1]
        if T_recon < T_expected:
            expected_physics = expected_physics[:, :T_recon]
        elif T_recon > T_expected:
            vae_recon_physics = vae_recon_physics[:, :T_expected]
        
        # VAE reconstruction loss - trains decoder on actual inference path
        vae_recon_loss = F.mse_loss(vae_recon_physics, expected_physics.to(self.dtype))
        
        # === Simple adapter loss (as before, for regularization) ===
        cond_recon = self.channel_adapter.decode(
            self.spatial_decoder(cond_video)
        )
        adapter_loss = F.mse_loss(cond_recon, cond_frame.to(self.dtype))
        
        # === Total loss ===
        # vae_recon_loss is the key one for proper decoder training
        total_loss = diffusion_loss + 0.1 * adapter_loss + 0.5 * vae_recon_loss
        
        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "adapter_loss": adapter_loss,
            "vae_recon_loss": vae_recon_loss,
            "timesteps": timesteps,
        }
    
    @torch.no_grad()
    def predict_i2v_diffusion(
        self,
        cond_frame: torch.Tensor,
        num_frames: int = 8,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        text_prompt: str = "Top-down view of fluid dynamics simulation, evolving turbulence, scientific visualization, accurate physics (turbulent radiative layer 2d)",
    ) -> torch.Tensor:
        """
        Generate future physics frames using native I2V diffusion.
        
        Args:
            cond_frame: Conditioning physics frame (B, 1, H, W, C) or (B, H, W, C)
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            text_prompt: Text prompt for conditioning
            
        Returns:
            Predicted physics frames (B, num_frames, H, W, C)
        """
        self.eval()
        
        # Handle input shape
        if cond_frame.dim() == 5:
            cond_frame = cond_frame[:, 0]  # (B, H, W, C)
        if cond_frame.shape[-1] == 4:  # Channel-last
            cond_frame = rearrange(cond_frame, "B H W C -> B C H W")
        
        B = cond_frame.shape[0]
        H_phys, W_phys = cond_frame.shape[2:]
        
        # === Adapt conditioning frame to video ===
        cond_video = self.channel_adapter.encode(cond_frame.to(self.dtype))
        cond_video = self.spatial_encoder(cond_video)  # (B, 3, H_vid, W_vid)
        H_vid, W_vid = cond_video.shape[2:]
        
        # === Initialize random latent ===
        # VAE stride is (4, 8, 8) for (T, H, W)
        # IMPORTANT: I2V model generates [cond, target0, target1, ...] so we need 1 + num_frames total
        # Then we return only the generated frames (skip frame 0 which is conditioning)
        total_video_frames = 1 + num_frames  # cond + targets
        vae_stride_t = 4
        T_lat = (total_video_frames - 1) // vae_stride_t + 1
        # For 9 frames (1 cond + 8 target): (9-1)//4 + 1 = 3 latent frames → decodes to 9 frames
        H_lat = H_vid // 8
        W_lat = W_vid // 8
        
        # === Prepare I2V conditioning (MUST use total_video_frames to match latent T_lat) ===
        conditioning = self._prepare_i2v_conditioning(cond_video, num_output_frames=total_video_frames)
        
        latent = torch.randn(B, 16, T_lat, H_lat, W_lat, device=self.device, dtype=self.dtype)
        
        # === Get text embeddings (with negative prompt for CFG) ===
        text_inputs = self.pipe.tokenizer(
            [text_prompt] * B,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        text_embeds = self.pipe.text_encoder(**text_inputs)[0]
        
        # Negative prompt (empty)
        neg_inputs = self.pipe.tokenizer(
            [""] * B,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        neg_embeds = self.pipe.text_encoder(**neg_inputs)[0]
        
        # === Setup scheduler ===
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        
        # === Denoising loop ===
        for t in tqdm(timesteps, desc="Denoising", leave=False):
            timestep = t.expand(B)
            
            # Concatenate latent with conditioning for transformer input
            hidden_states = torch.cat([latent, conditioning], dim=1)  # (B, 36, T_lat, H_lat, W_lat)
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                # Unconditional forward
                noise_pred_uncond = self.transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=neg_embeds,
                    return_dict=False,
                )[0]
                
                # Conditional forward
                noise_pred_cond = self.transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=text_embeds,
                    return_dict=False,
                )[0]
                
                # CFG combination
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=text_embeds,
                    return_dict=False,
                )[0]
            
            # Scheduler step
            latent = self.pipe.scheduler.step(noise_pred, t, latent, return_dict=False)[0]
        
        # === VAE decode ===
        scaling_factor = self._get_vae_scaling_factor()
        latent = latent / scaling_factor
        video_frames = self.vae.decode(latent, return_dict=False)[0]  # (B, 3, T, H, W)
        
        # Denormalize from [-1, 1] to [0, 1]
        video_frames = (video_frames + 1) / 2
        video_frames = torch.clamp(video_frames, 0, 1)
        
        # === Adapt video → physics ===
        video_frames = rearrange(video_frames, "B C T H W -> (B T) C H W")
        video_frames = video_frames.to(self.dtype)
        video_frames = self.spatial_decoder(video_frames)
        physics_frames = self.channel_adapter.decode(video_frames)
        physics_frames = rearrange(physics_frames, "(B T) C H W -> B T H W C", B=B)
        
        # === CRITICAL: Skip frame 0 (conditioning) and return only generated frames ===
        # Frame 0 = conditioning (should match input)
        # Frames 1+ = generated (what we want to return)
        if physics_frames.shape[1] > 1:
            physics_frames = physics_frames[:, 1:]  # Skip conditioning frame
        
        # Truncate to requested number of frames
        if physics_frames.shape[1] > num_frames:
            physics_frames = physics_frames[:, :num_frames]
        
        return physics_frames
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "channel_adapter": self.channel_adapter.state_dict(),
            "spatial_encoder": self.spatial_encoder.state_dict(),
            "spatial_decoder": self.spatial_decoder.state_dict(),
        }
        
        # Save temporal predictor if enabled
        if self.temporal_predictor is not None:
            checkpoint["temporal_predictor"] = self.temporal_predictor.state_dict()
        
        if self.lora_enabled and self.transformer is not None:
            # Save LoRA weights
            checkpoint["lora_weights"] = {
                name: param.data 
                for name, param in self.transformer.named_parameters() 
                if param.requires_grad
            }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.channel_adapter.load_state_dict(checkpoint["channel_adapter"])
        self.spatial_encoder.load_state_dict(checkpoint["spatial_encoder"])
        self.spatial_decoder.load_state_dict(checkpoint["spatial_decoder"])
        
        # Load temporal predictor if present
        if "temporal_predictor" in checkpoint and self.temporal_predictor is not None:
            self.temporal_predictor.load_state_dict(checkpoint["temporal_predictor"])
            print("Loaded temporal predictor weights")
        
        if "lora_weights" in checkpoint and self.transformer is not None:
            for name, param in self.transformer.named_parameters():
                if name in checkpoint["lora_weights"]:
                    # Use copy_ to ensure tensor stays on correct device
                    param.data.copy_(checkpoint["lora_weights"][name].to(param.device))
        
        # Ensure everything is on the correct device
        self.to(self.device)
        if self.pipe is not None:
            self.pipe.to(self.device)
        
        print(f"Checkpoint loaded from {path}")


def create_wan22_model(config: Dict[str, Any]) -> Wan22VideoModel:
    """
    Factory function to create Wan2.2 model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized Wan22VideoModel
    """
    model_config = config["model"]
    lora_config = config["lora"]
    data_config = config["data"]
    
    # Determine dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(model_config["dtype"], torch.bfloat16)
    
    # Get temporal predictor config if present
    temporal_config = model_config.get("temporal_predictor", {})
    use_temporal_predictor = temporal_config.get("enabled", False)
    temporal_predictor_type = temporal_config.get("type", "convlstm")
    temporal_hidden_channels = temporal_config.get("hidden_channels", 64)
    
    model = Wan22VideoModel(
        model_id=model_config["name"],
        dtype=dtype,
        physics_channels=model_config["channel_adapter"]["input_channels"],
        video_channels=model_config["channel_adapter"]["output_channels"],
        adapter_hidden_dim=model_config["channel_adapter"]["hidden_dim"],
        adapter_num_layers=model_config["channel_adapter"]["num_layers"],
        physics_size=tuple(data_config["spatial_size"]),
        video_size=tuple(data_config.get("target_size", data_config["spatial_size"])),
        lora_enabled=lora_config["enabled"],
        lora_rank=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        lora_target_modules=lora_config["target_modules"],
        use_temporal_predictor=use_temporal_predictor,
        temporal_predictor_type=temporal_predictor_type,
        temporal_hidden_channels=temporal_hidden_channels,
    )
    
    return model
