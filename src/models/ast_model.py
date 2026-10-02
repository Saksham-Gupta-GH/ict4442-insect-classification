"""
Audio Spectrogram Transformer (AST) Architecture.
Owned by: Harsh Kumar Roy (230911296)

Applies self-attention across 2D spectrogram patches.
"""

from typing import Optional
import torch
import torch.nn as nn
import timm


class InsectAST(nn.Module):
    """
    Audio Spectrogram Transformer (AST) for bioacoustic species identification.
    Uses Vision Transformer (ViT) patch embedding adapted for 128-band Mel spectrograms.
    """

    def __init__(
        self,
        num_classes: int = 40,
        in_channels: int = 1,
        pretrained: bool = False,
        use_temperature: bool = False,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_temperature = use_temperature

        # Load ViT backbone via timm
        try:
            self.vit = timm.create_model(
                "vit_tiny_patch16_224",
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=0  # Remove head to get embedding
            )
            embed_dim = self.vit.num_features
        except Exception:
            # Fallback if timm download not available
            self.vit = nn.Sequential(
                nn.Conv2d(in_channels, 192, kernel_size=16, stride=16),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            embed_dim = 192

        # Temperature Conditioning Branch
        if self.use_temperature:
            self.temp_embedding = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 64),
                nn.ReLU(inplace=True)
            )
            head_in_dim = embed_dim + 64
        else:
            self.temp_embedding = None
            head_in_dim = embed_dim

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_in_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(
        self,
        spectrogram: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Interpolate spectrogram to ViT input size if necessary (e.g., 224x224)
        if spectrogram.shape[-2:] != (224, 224):
            spectrogram = torch.nn.functional.interpolate(
                spectrogram, size=(224, 224), mode="bilinear", align_corners=False
            )

        feat = self.vit(spectrogram)

        if self.use_temperature and temperature is not None and self.temp_embedding is not None:
            if temperature.ndim == 1:
                temperature = temperature.unsqueeze(1)
            temp_feat = self.temp_embedding(temperature)
            feat = torch.cat([feat, temp_feat], dim=1)

        logits = self.classifier(feat)
        return logits
