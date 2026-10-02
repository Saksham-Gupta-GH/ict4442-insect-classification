"""
VGG-Style Convolutional Baseline Architecture for Insect Bioacoustics.
Owned by: Saurabh Tiwari (230911238)

Key Features:
1. Deep convolutional stack with Batch Normalization & Dropout for local time-frequency representation learning.
2. Global Average Pooling for variable time-length robustness.
3. Optional Auxiliary Temperature Conditioning branch for multimodal fusion (Dolbear's Law investigation).
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two-layer 3x3 Convolutional Block with BatchNorm, ReLU, MaxPool, and Dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LogMelCNN(nn.Module):
    """
    VGG-style Deep Convolutional Neural Network for Log-Mel Spectrograms.
    
    Parameters
    ----------
    num_classes : int
        Number of insect species classes.
    in_channels : int
        Number of input channels (1 for grayscale log-Mel, 3 for RGB).
    base_channels : int
        Base channel width (default: 32).
    use_temperature : bool
        Whether to condition classification on auxiliary ambient temperature.
    dropout_rate : float
        Dropout probability in classification head.
    """

    def __init__(
        self,
        num_classes: int = 40,
        in_channels: int = 1,
        base_channels: int = 32,
        use_temperature: bool = False,
        dropout_rate: float = 0.4
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.use_temperature = use_temperature

        # Convolutional Backbone
        self.block1 = ConvBlock(in_channels, base_channels, dropout_rate=0.2)
        self.block2 = ConvBlock(base_channels, base_channels * 2, dropout_rate=0.2)
        self.block3 = ConvBlock(base_channels * 2, base_channels * 4, dropout_rate=0.3)
        self.block4 = ConvBlock(base_channels * 4, base_channels * 8, dropout_rate=0.3)

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.acoustic_feat_dim = base_channels * 8  # 256 for base_channels=32

        # Temperature Conditioning Branch (Auxiliary Late Fusion)
        if self.use_temperature:
            self.temp_embedding = nn.Sequential(
                nn.Linear(1, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            )
            classifier_in_dim = self.acoustic_feat_dim + 64
        else:
            self.temp_embedding = None
            classifier_in_dim = self.acoustic_feat_dim

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts acoustic representation vector before classifier head."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(
        self,
        spectrogram: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        spectrogram : torch.Tensor
            Batch of spectrograms of shape (B, C, H, W).
        temperature : Optional[torch.Tensor]
            Batch of scalar temperatures of shape (B, 1) or (B,).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        # 1. Extract acoustic representation
        feat = self.extract_features(spectrogram)

        # 2. Auxiliary temperature fusion (if enabled)
        if self.use_temperature and temperature is not None and self.temp_embedding is not None:
            if temperature.ndim == 1:
                temperature = temperature.unsqueeze(1)
            temp_feat = self.temp_embedding(temperature)
            feat = torch.cat([feat, temp_feat], dim=1)

        # 3. Class logits
        logits = self.classifier(feat)
        return logits
