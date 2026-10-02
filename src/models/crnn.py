"""
CRNN Architecture (Convolutional Front-End with Bidirectional LSTM).
Owned by: Saksham Gupta (230911186)

Models the temporal sequence of insect stridulation pulse trains.
"""

from typing import Optional
import torch
import torch.nn as nn


class InsectCRNN(nn.Module):
    """
    CRNN architecture combining 2D Convolutions for local spectral texture
    and BiLSTM for sequence modeling over stridulation pulse trains.
    """

    def __init__(
        self,
        num_classes: int = 40,
        in_channels: int = 1,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        dropout_rate: float = 0.3,
        use_temperature: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_temperature = use_temperature

        # Conv front-end
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Mel dim -> n_mels/2
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Mel dim -> n_mels/4
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Keep time resolution, compress frequency
            nn.Dropout2d(0.2)
        )

        # Recurrent Layer (BiLSTM)
        # Input to BiLSTM will be (B, time_steps, feature_dim)
        # With 128 mel bins -> after pool (2, 2, 2) freq is 16; 128 channels * 16 = 2048
        self.lstm_in_dim = 128 * 16
        self.lstm = nn.LSTM(
            input_size=self.lstm_in_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if rnn_num_layers > 1 else 0.0
        )

        self.acoustic_feat_dim = rnn_hidden_size * 2  # Bidirectional

        # Temperature Conditioning Branch
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

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        spectrogram: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Conv front-end: (B, C, F, T) -> (B, 128, F', T')
        x = self.conv(spectrogram)
        b, c, f, t = x.shape

        # Reshape for sequence model: (B, T', C*F')
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # BiLSTM over time steps
        lstm_out, _ = self.lstm(x)  # (B, T', 2 * hidden_size)

        # Temporal Average Pooling
        feat = torch.mean(lstm_out, dim=1)  # (B, 2 * hidden_size)

        # Temperature Conditioning Fusion
        if self.use_temperature and temperature is not None and self.temp_embedding is not None:
            if temperature.ndim == 1:
                temperature = temperature.unsqueeze(1)
            temp_feat = self.temp_embedding(temperature)
            feat = torch.cat([feat, temp_feat], dim=1)

        logits = self.classifier(feat)
        return logits
