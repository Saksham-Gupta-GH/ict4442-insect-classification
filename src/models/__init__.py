"""
Models package for Insect Acoustic Classification.
Provides architectures spanning 4 distinct families:
1. CNN (VGG-style Log-Mel CNN and EfficientNetV2-S Transfer Learning) - Owned by Saurabh Tiwari
2. CRNN (Convolutional Recurrent Neural Network) - Owned by Saksham Gupta
3. Classical MLP Baseline (Handcrafted Acoustic Features) - Owned by Abhishek Patro
4. AST (Audio Spectrogram Transformer) - Owned by Harsh Kumar Roy
"""

from src.models.vgg_cnn import LogMelCNN
from src.models.efficientnet import InsectEfficientNetV2
from src.models.crnn import InsectCRNN
from src.models.mlp_baseline import InsectMLP
from src.models.ast_model import InsectAST

__all__ = [
    "LogMelCNN",
    "InsectEfficientNetV2",
    "InsectCRNN",
    "InsectMLP",
    "InsectAST"
]
