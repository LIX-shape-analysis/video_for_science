"""Model definitions for Wan2.2 fine-tuning."""

from .channel_adapter import ChannelAdapter, InverseChannelAdapter, ChannelAdapterPair
from .wan_wrapper import Wan22VideoModel, create_wan22_model

__all__ = [
    "ChannelAdapter",
    "InverseChannelAdapter", 
    "ChannelAdapterPair",
    "Wan22VideoModel",
    "create_wan22_model",
]
