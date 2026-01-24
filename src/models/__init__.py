"""Model definitions for Wan2.2 fine-tuning."""

from .channel_adapter import ChannelAdapter, InverseChannelAdapter, ChannelAdapterPair
from .wan_wrapper import Wan22VideoModel, create_wan22_model
from .temporal_predictor import (
    LatentTemporalPredictor,
    SimpleTemporalPredictor,
    ConvLSTM,
    create_temporal_predictor,
)
from .physics_adapter import PhysicsEncoder, PhysicsDecoder, PhysicsAdapterPair
from .humantfm_model import HumanTFMModel

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
    "PhysicsEncoder",
    "PhysicsDecoder",
    "PhysicsAdapterPair",
    "HumanTFMModel",
]

