"""
EfficientNetV2-S Transfer Learning Model for Insect Bioacoustics.
Owned by: Saurabh Tiwari (230911238)

Key Features:
1. Pretrained EfficientNetV2-S backbone adapted for bioacoustic log-Mel spectrograms.
2. Serves as direct comparison against the state-of-the-art benchmark (InsectEffNet by Faiß et al.).
3. Auxiliary Temperature Conditioning branch for multimodal fusion.
4. Layer-wise fine-tuning support with separate backbone/classifier learning rates.
"""

from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_V2_S_Weights


class InsectEfficientNetV2(nn.Module):
    """
    EfficientNetV2-S transfer learning model for bioacoustic classification.
    
    Parameters
    ----------
    num_classes : int
        Number of target insect species.
    in_channels : int
        Number of input channels (1 for grayscale log-Mel, 3 for repeated RGB).
    pretrained : bool
        Whether to load ImageNet pre-trained weights (default: True).
    use_temperature : bool
        Whether to enable auxiliary temperature conditioning late-fusion.
    dropout_rate : float
        Classifier dropout probability (default: 0.3).
    """

    def __init__(
        self,
        num_classes: int = 40,
        in_channels: int = 1,
        pretrained: bool = True,
        use_temperature: bool = False,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.use_temperature = use_temperature

        # Load EfficientNetV2-S backbone
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        try:
            base_model = models.efficientnet_v2_s(weights=weights)
        except Exception:
            # Fallback if download offline
            base_model = models.efficientnet_v2_s(weights=None)

        # Adapt input conv if in_channels == 1
        if in_channels == 1:
            orig_conv = base_model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False
            )
            if pretrained and orig_conv.weight is not None:
                # Average RGB weights across single channel
                with torch.no_grad():
                    new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
            base_model.features[0][0] = new_conv

        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Backbone output feature dimension (EfficientNetV2-S outputs 1280 features)
        self.backbone_out_dim = 1280

        # Temperature Conditioning Branch
        if self.use_temperature:
            self.temp_embedding = nn.Sequential(
                nn.Linear(1, 32),
                nn.BatchNorm1d(32),
                nn.SiLU(inplace=True),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(inplace=True)
            )
            classifier_in_dim = self.backbone_out_dim + 64
        else:
            self.temp_embedding = None
            classifier_in_dim = self.backbone_out_dim

        # Custom bioacoustic classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(classifier_in_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def freeze_backbone(self) -> None:
        """Freezes feature extractor weights for warm-up classifier training."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes all layers for end-to-end fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr_backbone: float = 1e-4, lr_head: float = 1e-3) -> List[Dict]:
        """Returns parameter groups with differentiated learning rates for fine-tuning."""
        return [
            {"params": self.features.parameters(), "lr": lr_backbone},
            {"params": self.classifier.parameters(), "lr": lr_head},
            *(
                [{"params": self.temp_embedding.parameters(), "lr": lr_head}]
                if self.temp_embedding is not None
                else []
            )
        ]

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
            Spectrogram tensor of shape (B, 1, H, W) or (B, 3, H, W).
        temperature : Optional[torch.Tensor]
            Auxiliary ambient temperature tensor of shape (B, 1) or (B,).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        # If in_channels is 3 and 1-channel passed, repeat
        if self.in_channels == 3 and spectrogram.size(1) == 1:
            spectrogram = spectrogram.repeat(1, 3, 1, 1)

        feat = self.features(spectrogram)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)

        # Auxiliary Temperature Fusion
        if self.use_temperature and temperature is not None and self.temp_embedding is not None:
            if temperature.ndim == 1:
                temperature = temperature.unsqueeze(1)
            temp_feat = self.temp_embedding(temperature)
            feat = torch.cat([feat, temp_feat], dim=1)

        logits = self.classifier(feat)
        return logits
