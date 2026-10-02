"""
MLP Baseline on Handcrafted Bioacoustic Features.
Owned by: Abhishek Patro (230911148)

Serves as the non-deep classical reference point.
"""

from typing import Optional
import torch
import torch.nn as nn


class InsectMLP(nn.Module):
    """
    3-Layer Multilayer Perceptron trained on handcrafted acoustic features
    (MFCCs, spectral centroid, spectral bandwidth, rolloff, zero-crossing rate).
    """

    def __init__(
        self,
        in_features: int = 49,
        num_classes: int = 40,
        hidden_dims: tuple = (256, 128),
        dropout_rate: float = 0.3,
        use_temperature: bool = False
    ):
        super().__init__()
        self.use_temperature = use_temperature

        # MLP Layers
        layers = []
        curr_dim = in_features + (1 if use_temperature else 0)

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate)
            ])
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_temperature and temperature is not None:
            if temperature.ndim == 1:
                temperature = temperature.unsqueeze(1)
            features = torch.cat([features, temperature], dim=1)

        return self.net(features)
