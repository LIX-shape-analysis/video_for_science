"""Model definitions for Wan2.2 fine-tuning."""

from .channel_adapter import ChannelAdapter, InverseChannelAdapter, ChannelAdapterPair
from .wan_wrapper import Wan22VideoModel, create_wan22_model
from .temporal_predictor import (
    LatentTemporalPredictor,
    SimpleTemporalPredictor,
    ConvLSTM,
    create_temporal_predictor,
)

__all__ = [
    "ChannelAdapter",
    "InverseChannelAdapter", 
    "ChannelAdapterPair",
    "Wan22VideoModel",
    "create_wan22_model",
    "LatentTemporalPredictor",
    "SimpleTemporalPredictor",
    "ConvLSTM",
    "create_temporal_predictor",
]
