"""
Wan2.2-I2V Model Wrapper for Physics Simulation Fine-tuning.

This module provides a wrapper around the Wan2.2-I2V model that:
1. Integrates channel adapters for physics data
2. Applies LoRA for efficient fine-tuning
3. Provides training and inference interfaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from einops import rearrange
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
        """
        super().__init__()
        
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.physics_size = physics_size
        self.video_size = video_size
        self.lora_enabled = lora_enabled
        
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
            self.transformer.print_trainable_parameters()
        
        # Freeze non-trainable components
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Move to device
        self.to(self.device)
        
        self._model_loaded = True
        print("Model loaded successfully!")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        params = []
        
        # Channel adapter parameters
        params.extend(self.channel_adapter.parameters())
        
        # Spatial adapter parameters
        params.extend(self.spatial_encoder.parameters())
        params.extend(self.spatial_decoder.parameters())
        
        # LoRA parameters (if enabled)
        if self.lora_enabled and self.transformer is not None:
            for name, param in self.transformer.named_parameters():
                if param.requires_grad:
                    params.append(param)
        
        return params
    
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
                return config.get('scaling_factor', 1.0)
        
        # Default scaling factor for Wan2.2 VAE
        # Wan2.2 uses a scaling factor around 0.18215 (similar to SD)
        return 0.18215
    
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
        
        This computes the diffusion loss between the model's prediction
        and the target frames.
        
        Args:
            input_frames: Input physics frames (B, T_in, H, W, C)
            target_frames: Target physics frames (B, T_out, H, W, C)
            num_inference_steps: Number of diffusion steps (for reference)
            guidance_scale: Guidance scale (for reference)
            
        Returns:
            Dictionary with loss and intermediate outputs
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_pretrained() first.")
        
        # Prepare inputs
        condition_image, input_video = self.prepare_physics_input(input_frames)
        _, target_video = self.prepare_physics_input(target_frames)
        
        # Combine input (after first frame) and target for full video
        full_target_video = torch.cat([input_video, target_video], dim=1)
        
        # Encode to latent space
        target_latents = self.encode_to_latent(full_target_video)
        
        # Get text embeddings (empty prompt for physics)
        prompt_embeds = self._get_empty_prompt_embeds(condition_image.shape[0])
        
        # Sample noise
        noise = torch.randn_like(target_latents)
        
        # Sample timesteps
        # Handle FrozenDict config
        scheduler_config = self.pipe.scheduler.config
        if hasattr(scheduler_config, 'num_train_timesteps'):
            num_train_timesteps = scheduler_config.num_train_timesteps
        elif isinstance(scheduler_config, dict):
            num_train_timesteps = scheduler_config.get('num_train_timesteps', 1000)
        else:
            num_train_timesteps = 1000  # Default fallback
        
        timesteps = torch.randint(
            0, num_train_timesteps,
            (target_latents.shape[0],),
            device=target_latents.device,
        )
        
        # Add noise to latents
        noisy_latents = self.pipe.scheduler.add_noise(
            target_latents, noise, timesteps
        )
        
        # Prepare condition image latent
        condition_latent = self._encode_condition_image(condition_image)
        
        # Predict noise with transformer
        noise_pred = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            # Additional conditioning can be added here
        ).sample
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Compute additional losses
        adapter_loss = self.channel_adapter.get_reconstruction_loss(
            rearrange(input_frames, "B T H W C -> (B T) C H W")
        )
        
        total_loss = loss + 0.1 * adapter_loss
        
        return {
            "loss": total_loss,
            "diffusion_loss": loss,
            "adapter_loss": adapter_loss,
            "noise_pred": noise_pred,
            "target_latents": target_latents,
        }
    
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
        
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        
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
        
        return physics_output
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "channel_adapter": self.channel_adapter.state_dict(),
            "spatial_encoder": self.spatial_encoder.state_dict(),
            "spatial_decoder": self.spatial_decoder.state_dict(),
        }
        
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
        
        if "lora_weights" in checkpoint and self.transformer is not None:
            for name, param in self.transformer.named_parameters():
                if name in checkpoint["lora_weights"]:
                    param.data = checkpoint["lora_weights"][name]
        
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
    )
    
    return model
