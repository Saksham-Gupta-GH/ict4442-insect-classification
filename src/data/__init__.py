"""
Data module for Insect Acoustic Classification
Provides feature extraction, data augmentation, and dataset loaders.
"""

from src.data.features import LogMelExtractor, HandcraftedFeatureExtractor
from src.data.augmentation import (
    WaveformAugmentation,
    SpecAugment,
    AudioAugmentationPipeline
)
from src.data.dataset import InsectAudioDataset, SyntheticInsectDataset, create_dataloaders

__all__ = [
    "LogMelExtractor",
    "HandcraftedFeatureExtractor",
    "WaveformAugmentation",
    "SpecAugment",
    "AudioAugmentationPipeline",
    "InsectAudioDataset",
    "SyntheticInsectDataset",
    "create_dataloaders"
]
