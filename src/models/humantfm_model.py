"""
HumanTFM-style one-shot prediction model for physics simulation.

Based on the HumanTFM paper methodology:
- Single forward pass through pretrained DiT (no diffusion loop)
- Fixed timestep t=0 (clean state)
- Rectified flow output: x_pred = x_input - v_pred
- Two-stage training: latent space → ambient space

This replaces the iterative diffusion sampling with deterministic prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from einops import rearrange

try:
    from diffusers import WanImageToVideoPipeline
except ImportError:
    WanImageToVideoPipeline = None

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError("Please install peft: pip install peft")

from .physics_adapter import PhysicsAdapterPair


class HumanTFMModel(nn.Module):
    """
    One-shot deterministic prediction model following HumanTFM methodology.
    
    Key differences from diffusion-based approach:
    1. No noise injection - input is clean latents
    2. Fixed timestep t=0 (end of diffusion = clean state)
    3. Single forward pass through DiT
    4. Rectified flow: output = input - model_prediction
    5. Two-stage training with ambient space supervision
    """
    
    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        # Physics adapter config
        physics_channels: int = 4,
        video_channels: int = 3,
        adapter_hidden_dim: int = 64,
        physics_size: Tuple[int, int] = (128, 384),
        video_size: Tuple[int, int] = (480, 832),
        # LoRA config
        lora_enabled: bool = True,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # Residual prediction
        use_residual: bool = True,
        # Text prompt
        default_text_prompt: str = "Fluid dynamics simulation, turbulent flow, scientific visualization",
    ):
        super().__init__()
        
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.video_size = video_size
        self.physics_size = physics_size
        self.use_residual = use_residual
        self.default_text_prompt = default_text_prompt
        
        # LoRA configuration
        self.lora_enabled = lora_enabled
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
            init_lora_weights=True,
        ) if lora_enabled else None
        
        # Physics adapter pair (encoder + decoder)
        self.physics_adapter = PhysicsAdapterPair(
            physics_channels=physics_channels,
            video_channels=video_channels,
            hidden_dim=adapter_hidden_dim,
            physics_size=physics_size,
            video_size=video_size,
            use_residual=use_residual,
        )
        
        # Placeholders for pipeline components (loaded later)
        self.pipe = None
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        
        # Cached text embeddings
        self._cached_text_embeds = None
        self._cached_prompt = None
        
        self._model_loaded = False
    
    def load_pretrained(self, local_path: Optional[str] = None):
        """Load the pretrained Wan2.2 model."""
        if self._model_loaded:
            return
        
        model_path = local_path if local_path else self.model_id
        print(f"[HumanTFM] Loading Wan2.2 model from {model_path}...")
        
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
            print("[HumanTFM] Applying LoRA to transformer...")
            self.transformer = get_peft_model(self.transformer, self.lora_config)
            self.pipe.transformer = self.transformer
            self.transformer.print_trainable_parameters()
        
        # Freeze VAE and text encoder (always frozen)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Convert physics adapter to model dtype
        self.physics_adapter = self.physics_adapter.to(self.dtype)
        
        # Move to device
        self.to(self.device)
        self.pipe.to(self.device)
        
        self._model_loaded = True
        print("[HumanTFM] Model loaded successfully!")
    
    def _get_vae_scaling_factor(self) -> float:
        """Get VAE scaling factor."""
        if hasattr(self.vae.config, 'scaling_factor'):
            return self.vae.config.scaling_factor
        return 0.18215
    
    def _get_text_embeddings(self, text_prompt: str) -> torch.Tensor:
        """Get text embeddings (cached for efficiency)."""
        if self._cached_prompt == text_prompt and self._cached_text_embeds is not None:
            return self._cached_text_embeds
        
        with torch.no_grad():
            # Use a reasonable max_length - tokenizer.model_max_length can be very large
            max_len = getattr(self.tokenizer, 'model_max_length', 512)
            if max_len is None or max_len > 1000000:  # Handle overflow cases
                max_len = 512
            
            text_inputs = self.tokenizer(
                text_prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=min(max_len, 512),  # Cap at 512 for safety
            )
            text_embeds = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        
        self._cached_prompt = text_prompt
        self._cached_text_embeds = text_embeds
        return text_embeds
    
    def _prepare_conditioning(
        self,
        cond_video: torch.Tensor,
        num_output_frames: int,
    ) -> torch.Tensor:
        """
        Prepare I2V conditioning (mask + conditioning latent).
        
        Args:
            cond_video: Conditioning frame in video space (B, 3, H, W)
            num_output_frames: Number of output video frames
            
        Returns:
            Conditioning tensor (B, 20, T_lat, H_lat, W_lat)
        """
        B = cond_video.shape[0]
        
        # Encode conditioning frame
        cond_video = cond_video.unsqueeze(2)  # (B, 3, 1, H, W) for VAE
        cond_video = rearrange(cond_video, "B C T H W -> B C T H W")
        
        # Repeat to match output frames for encoding (VAE needs full sequence)
        # But we only use the first frame as conditioning
        full_video = cond_video.repeat(1, 1, num_output_frames, 1, 1)
        
        with torch.no_grad():
            latent_dist = self.vae.encode(full_video)
            if hasattr(latent_dist, 'latent_dist'):
                cond_latent = latent_dist.latent_dist.sample()
            else:
                cond_latent = latent_dist.sample() if hasattr(latent_dist, 'sample') else latent_dist
        
        scaling_factor = self._get_vae_scaling_factor()
        cond_latent = cond_latent * scaling_factor  # (B, 16, T_lat, H_lat, W_lat)
        
        _, _, T_lat, H_lat, W_lat = cond_latent.shape
        
        # Create mask: 1 for conditioning frame, 0 for generated
        mask = torch.zeros(B, 4, T_lat, H_lat, W_lat, device=cond_latent.device, dtype=cond_latent.dtype)
        mask[:, :, 0] = 1.0  # First latent is conditioning
        
        # Concatenate: [mask (4ch), cond_latent (16ch)] = 20ch
        conditioning = torch.cat([mask, cond_latent], dim=1)
        
        return conditioning
    
    def forward_latent_only(
        self,
        cond_frame: torch.Tensor,
        target_frames: torch.Tensor,
        text_prompt: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing loss in latent space only.
        
        Used in Stage 1 training (decoder frozen).
        
        Args:
            cond_frame: Conditioning physics frame (B, 1, 4, H, W) or (B, 4, H, W)
            target_frames: Target physics frames (B, T, 4, H, W)
            text_prompt: Optional text prompt
            
        Returns:
            Dict with latent_loss, predicted_latent, target_latent
        """
        text_prompt = text_prompt or self.default_text_prompt
        
        # Handle input shapes
        if cond_frame.dim() == 5:
            cond_frame = cond_frame[:, 0]  # (B, 4, H, W)
        if cond_frame.dim() == 4 and cond_frame.shape[1] != 4:
            cond_frame = rearrange(cond_frame, "B H W C -> B C H W")
        
        if target_frames.dim() == 5 and target_frames.shape[-1] == 4:
            target_frames = rearrange(target_frames, "B T H W C -> B T C H W")
        
        B, T_out = target_frames.shape[:2]
        
        # === Step 1: Encode physics → video ===
        cond_video = self.physics_adapter.encode(cond_frame.to(self.dtype))  # (B, 3, H_vid, W_vid)
        
        target_video = self.physics_adapter.encode(target_frames.to(self.dtype))  # (B, T, 3, H_vid, W_vid)
        
        # Total frames: cond + targets
        n_frames = 1 + T_out
        cond_video_expanded = cond_video.unsqueeze(1)  # (B, 1, 3, H, W)
        full_video = torch.cat([cond_video_expanded, target_video], dim=1)  # (B, 1+T, 3, H, W)
        full_video = rearrange(full_video, "B T C H W -> B C T H W")
        
        # === Step 2: VAE encode to get target latents ===
        scaling_factor = self._get_vae_scaling_factor()
        with torch.no_grad():
            latent_dist = self.vae.encode(full_video)
            if hasattr(latent_dist, 'latent_dist'):
                target_latent = latent_dist.latent_dist.sample()
            else:
                target_latent = latent_dist.sample() if hasattr(latent_dist, 'sample') else latent_dist
        target_latent = target_latent * scaling_factor
        
        # Also encode just input (cond) for the DiT input
        cond_video_for_vae = cond_video.unsqueeze(2)  # (B, 3, 1, H, W)
        cond_video_repeated = cond_video_for_vae.repeat(1, 1, n_frames, 1, 1)
        
        with torch.no_grad():
            latent_dist = self.vae.encode(cond_video_repeated)
            if hasattr(latent_dist, 'latent_dist'):
                input_latent = latent_dist.latent_dist.sample()
            else:
                input_latent = latent_dist.sample() if hasattr(latent_dist, 'sample') else latent_dist
        input_latent = input_latent * scaling_factor
        
        # === Step 3: Prepare conditioning ===
        conditioning = self._prepare_conditioning(cond_video, n_frames)
        
        # === Step 4: DiT forward with t=0 (clean) ===
        text_embeds = self._get_text_embeddings(text_prompt)
        text_embeds = text_embeds.expand(B, -1, -1)
        
        # Concatenate input with conditioning for I2V
        # hidden_states = [input_latent (16ch), conditioning (20ch)] = 36ch
        hidden_states = torch.cat([input_latent, conditioning], dim=1)
        
        # Fixed timestep = 0 (clean state in Wan scheduler)
        timestep = torch.zeros(B, device=self.device, dtype=torch.long)
        
        # DiT forward pass (single step, no loop!)
        v_pred = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            return_dict=False,
        )[0]
        
        # === Step 5: Rectified flow output ===
        # x_pred = x_input - v_pred
        pred_latent = input_latent - v_pred
        
        # === Step 6: Compute latent loss ===
        # Only on the non-conditioning frames (frames 1:)
        latent_loss = F.mse_loss(pred_latent, target_latent)
        
        return {
            "latent_loss": latent_loss,
            "predicted_latent": pred_latent,
            "target_latent": target_latent,
            "input_latent": input_latent,
        }
    
    def forward_ambient(
        self,
        cond_frame: torch.Tensor,
        target_frames: torch.Tensor,
        text_prompt: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with ambient (physics space) loss.
        
        Used in Stage 2 training (decoder unfrozen).
        Gradients flow through VAE decoder → physics decoder.
        
        Args:
            cond_frame: Conditioning physics frame (B, 1, 4, H, W) or (B, 4, H, W)
            target_frames: Target physics frames (B, T, 4, H, W)
            text_prompt: Optional text prompt
            
        Returns:
            Dict with ambient_loss, latent_loss, predictions
        """
        text_prompt = text_prompt or self.default_text_prompt
        
        # Handle input shapes
        if cond_frame.dim() == 5:
            cond_frame = cond_frame[:, 0]  # (B, 4, H, W)
        if cond_frame.dim() == 4 and cond_frame.shape[1] != 4:
            cond_frame = rearrange(cond_frame, "B H W C -> B C H W")
        
        if target_frames.dim() == 5 and target_frames.shape[-1] == 4:
            target_frames = rearrange(target_frames, "B T H W C -> B T C H W")
        
        B, T_out = target_frames.shape[:2]
        
        # === Step 1: Encode physics → video ===
        cond_video = self.physics_adapter.encode(cond_frame.to(self.dtype))
        target_video = self.physics_adapter.encode(target_frames.to(self.dtype))
        
        n_frames = 1 + T_out
        cond_video_expanded = cond_video.unsqueeze(1)
        full_video = torch.cat([cond_video_expanded, target_video], dim=1)
        full_video = rearrange(full_video, "B T C H W -> B C T H W")
        
        # === Step 2: VAE encode ===
        scaling_factor = self._get_vae_scaling_factor()
        with torch.no_grad():
            latent_dist = self.vae.encode(full_video)
            if hasattr(latent_dist, 'latent_dist'):
                target_latent = latent_dist.latent_dist.sample()
            else:
                target_latent = latent_dist.sample() if hasattr(latent_dist, 'sample') else latent_dist
        target_latent = target_latent * scaling_factor
        
        cond_video_for_vae = cond_video.unsqueeze(2)
        cond_video_repeated = cond_video_for_vae.repeat(1, 1, n_frames, 1, 1)
        
        with torch.no_grad():
            latent_dist = self.vae.encode(cond_video_repeated)
            if hasattr(latent_dist, 'latent_dist'):
                input_latent = latent_dist.latent_dist.sample()
            else:
                input_latent = latent_dist.sample() if hasattr(latent_dist, 'sample') else latent_dist
        input_latent = input_latent * scaling_factor
        
        # === Step 3: Prepare conditioning ===
        conditioning = self._prepare_conditioning(cond_video, n_frames)
        
        # === Step 4: DiT forward with t=0 ===
        text_embeds = self._get_text_embeddings(text_prompt)
        text_embeds = text_embeds.expand(B, -1, -1)
        
        hidden_states = torch.cat([input_latent, conditioning], dim=1)
        timestep = torch.zeros(B, device=self.device, dtype=torch.long)
        
        v_pred = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            return_dict=False,
        )[0]
        
        # === Step 5: Rectified flow output ===
        pred_latent = input_latent - v_pred
        
        # === Step 6: VAE decode (with gradients in Stage 2) ===
        # Scale latent for decoder
        pred_latent_for_decode = pred_latent / scaling_factor
        
        # Note: we keep gradients through decoder in stage 2
        decoded = self.vae.decode(pred_latent_for_decode)
        if hasattr(decoded, 'sample'):
            decoded_video = decoded.sample
        else:
            decoded_video = decoded
        
        # Rearrange to (B, T, C, H, W)
        decoded_video = rearrange(decoded_video, "B C T H W -> B T C H W")
        
        # === Step 7: Physics decode ===
        # Skip conditioning frame (use only predicted frames)
        pred_video = decoded_video[:, 1:]  # (B, T_out, 3, H, W)
        
        # Reference for residual
        reference = cond_frame if self.use_residual else None
        
        pred_physics = self.physics_adapter.decode(
            pred_video,
            reference=reference,
        )  # (B, T_out, 4, H_physics, W_physics)
        
        # === Step 8: Compute losses ===
        latent_loss = F.mse_loss(pred_latent, target_latent)
        ambient_loss = F.mse_loss(pred_physics, target_frames.to(pred_physics.dtype))
        
        return {
            "ambient_loss": ambient_loss,
            "latent_loss": latent_loss,
            "predicted_physics": pred_physics,
            "target_physics": target_frames,
            "predicted_latent": pred_latent,
        }
    
    def predict(
        self,
        cond_frame: torch.Tensor,
        num_frames: int = 8,
        text_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Inference: predict future physics frames from conditioning.
        
        Args:
            cond_frame: Conditioning physics frame (B, 4, H, W)
            num_frames: Number of frames to predict
            text_prompt: Optional text prompt
            
        Returns:
            Predicted physics frames (B, T, 4, H, W)
        """
        text_prompt = text_prompt or self.default_text_prompt
        
        if cond_frame.dim() == 5:
            cond_frame = cond_frame[:, 0]
        if cond_frame.dim() == 4 and cond_frame.shape[1] != 4:
            cond_frame = rearrange(cond_frame, "B H W C -> B C H W")
        
        B = cond_frame.shape[0]
        n_frames = 1 + num_frames  # cond + output
        
        # Encode to video
        cond_video = self.physics_adapter.encode(cond_frame.to(self.dtype))
        
        # VAE encode (repeated input)
        scaling_factor = self._get_vae_scaling_factor()
        cond_video_for_vae = cond_video.unsqueeze(2)
        cond_video_repeated = cond_video_for_vae.repeat(1, 1, n_frames, 1, 1)
        
        with torch.no_grad():
            latent_dist = self.vae.encode(cond_video_repeated)
            if hasattr(latent_dist, 'latent_dist'):
                input_latent = latent_dist.latent_dist.sample()
            else:
                input_latent = latent_dist.sample() if hasattr(latent_dist, 'sample') else latent_dist
        input_latent = input_latent * scaling_factor
        
        # Conditioning
        conditioning = self._prepare_conditioning(cond_video, n_frames)
        
        # Text embeddings
        text_embeds = self._get_text_embeddings(text_prompt)
        text_embeds = text_embeds.expand(B, -1, -1)
        
        # DiT forward
        hidden_states = torch.cat([input_latent, conditioning], dim=1)
        timestep = torch.zeros(B, device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            v_pred = self.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=text_embeds,
                return_dict=False,
            )[0]
        
        # Rectified flow
        pred_latent = input_latent - v_pred
        
        # VAE decode
        pred_latent_for_decode = pred_latent / scaling_factor
        with torch.no_grad():
            decoded = self.vae.decode(pred_latent_for_decode)
            if hasattr(decoded, 'sample'):
                decoded_video = decoded.sample
            else:
                decoded_video = decoded
        
        decoded_video = rearrange(decoded_video, "B C T H W -> B T C H W")
        pred_video = decoded_video[:, 1:]  # Skip cond frame
        
        # Physics decode
        reference = cond_frame if self.use_residual else None
        pred_physics = self.physics_adapter.decode(pred_video, reference=reference)
        
        return pred_physics
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        params = []
        
        # Physics adapter parameters
        for p in self.physics_adapter.parameters():
            if p.requires_grad:
                params.append(p)
        
        # LoRA parameters (if enabled)
        if self.lora_enabled and self.transformer is not None:
            for name, p in self.transformer.named_parameters():
                if p.requires_grad and 'lora' in name.lower():
                    params.append(p)
        
        return params
    
    def freeze_decoder(self):
        """Freeze physics decoder for Stage 1."""
        self.physics_adapter.freeze_decoder()
    
    def unfreeze_decoder(self):
        """Unfreeze physics decoder for Stage 2."""
        self.physics_adapter.unfreeze_decoder()
    
    def set_normalization_stats(self, mu: torch.Tensor, sigma: torch.Tensor):
        """Set normalization stats on decoder."""
        self.physics_adapter.set_normalization_stats(mu, sigma)
    
    def load_adapter(self, path: str, strict: bool = True):
        """
        Load pretrained physics adapter weights.
        
        This should be called BEFORE training to initialize the adapter
        with weights from autoencoder pre-training.
        
        Args:
            path: Path to pretrained adapter checkpoint
            strict: Whether to strictly enforce that all keys match
        """
        import os
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        
        print(f"[Rank {rank}] Loading pretrained adapter from {path}...")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Rank {rank}] ERROR: Adapter checkpoint not found at {path}")
        
        checkpoint = torch.load(path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'adapter_state_dict' in checkpoint:
            state_dict = checkpoint['adapter_state_dict']
            print(f"[Rank {rank}]   Found 'adapter_state_dict' key in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"[Rank {rank}]   Found 'state_dict' key in checkpoint")
        else:
            state_dict = checkpoint
            print(f"[Rank {rank}]   Using checkpoint directly as state_dict")
        
        # Log checkpoint keys for debugging
        print(f"[Rank {rank}]   State dict keys: {list(state_dict.keys())[:5]}... ({len(state_dict)} total)")
        
        # Load weights
        missing, unexpected = self.physics_adapter.load_state_dict(state_dict, strict=strict)
        
        if missing:
            print(f"[Rank {rank}]   ⚠️  Missing keys: {missing}")
        if unexpected:
            print(f"[Rank {rank}]   ⚠️  Unexpected keys: {unexpected}")
        
        # ========== VERIFICATION: Log weight statistics ==========
        print(f"[Rank {rank}] ✓ Adapter loaded! Verifying weights...")
        
        # Encoder verification
        enc_weight = self.physics_adapter.encoder.conv_in.weight
        enc_sum = enc_weight.sum().item()
        enc_mean = enc_weight.mean().item()
        enc_std = enc_weight.std().item()
        print(f"[Rank {rank}]   Encoder conv_in: sum={enc_sum:.4f}, mean={enc_mean:.6f}, std={enc_std:.6f}")
        
        # Decoder verification
        dec_weight = self.physics_adapter.decoder.conv_out.weight
        dec_sum = dec_weight.sum().item()
        dec_mean = dec_weight.mean().item()
        dec_std = dec_weight.std().item()
        print(f"[Rank {rank}]   Decoder conv_out: sum={dec_sum:.4f}, mean={dec_mean:.6f}, std={dec_std:.6f}")
        
        # Total parameter count verification
        encoder_params = sum(p.numel() for p in self.physics_adapter.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.physics_adapter.decoder.parameters())
        print(f"[Rank {rank}]   Encoder params: {encoder_params:,}, Decoder params: {decoder_params:,}")
        
        # Report validation loss from checkpoint if available
        if 'val_loss' in checkpoint:
            print(f"[Rank {rank}]   Pretrain reconstruction MSE: {checkpoint['val_loss']:.6f}")
        if 'val_metrics' in checkpoint:
            metrics = checkpoint['val_metrics']
            if 'total_rmse' in metrics:
                print(f"[Rank {rank}]   Pretrain reconstruction RMSE: {metrics['total_rmse']:.4f}")
        
        if 'epoch' in checkpoint:
            print(f"[Rank {rank}]   Pretrained at epoch: {checkpoint['epoch']}")
        
        print(f"[Rank {rank}] ✅ Adapter verification complete!")

