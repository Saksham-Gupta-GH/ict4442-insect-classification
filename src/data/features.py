"""
Acoustic Feature Extraction Module for Insect Bioacoustics.
Owned by: Saurabh Tiwari (230911238)

Implements:
1. PyTorch-native Log-Mel Spectrogram Extractor (128 Mel bands, dB scaling)
2. Handcrafted Acoustic Feature Extractor (MFCCs + Deltas, Spectral Centroid, Bandwidth, Rolloff, ZCR)
"""

import math
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import librosa


class LogMelExtractor(nn.Module):
    """
    Computes 128-band Log-Mel Spectrograms from raw audio waveforms.
    
    Parameters
    ----------
    sample_rate : int
        Audio sampling rate in Hz (default: 22050).
    n_fft : int
        FFT window size (default: 1024).
    win_length : Optional[int]
        Window length for STFT (default: same as n_fft).
    hop_length : int
        Hop length between STFT frames (default: 512).
    n_mels : int
        Number of Mel frequency bands (default: 128).
    f_min : float
        Minimum frequency in Hz (default: 0.0).
    f_max : Optional[float]
        Maximum frequency in Hz (default: sample_rate / 2).
    power : float
        Exponent for the magnitude spectrogram (default: 2.0 for power spectrogram).
    top_db : float
        Dynamic range for decibel conversion (default: 80.0 dB).
    normalize : bool
        Whether to standardize the output spectrogram to zero mean and unit variance.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        top_db: float = 80.0,
        normalize: bool = True
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.power = power
        self.top_db = top_db
        self.normalize = normalize

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=self.power,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney"
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=self.top_db)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extracts log-Mel spectrogram tensor.

        Parameters
        ----------
        waveform : torch.Tensor
            Input audio tensor of shape (batch_size, num_samples) or (batch_size, 1, num_samples) or (num_samples,).

        Returns
        -------
        torch.Tensor
            Log-Mel spectrogram tensor of shape (batch_size, 1, n_mels, time_steps) or (1, n_mels, time_steps).
        """
        # Ensure 2D (batch_size, time) or 3D (batch_size, 1, time)
        original_ndim = waveform.ndim
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, time)
        elif waveform.ndim == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)  # (batch_size, time)

        # Compute power Mel spectrogram
        mel_spec = self.mel_transform(waveform)  # (batch_size, n_mels, time_steps)

        # Convert to decibels (log scale)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        # Add channel dimension: (batch_size, 1, n_mels, time_steps)
        if log_mel_spec.ndim == 3:
            log_mel_spec = log_mel_spec.unsqueeze(1)

        # Normalize to standard zero-mean, unit-variance if enabled
        if self.normalize:
            mean = log_mel_spec.mean(dim=(-2, -1), keepdim=True)
            std = log_mel_spec.std(dim=(-2, -1), keepdim=True) + 1e-6
            log_mel_spec = (log_mel_spec - mean) / std

        if original_ndim == 1:
            return log_mel_spec.squeeze(0)  # (1, n_mels, time_steps)

        return log_mel_spec


class HandcraftedFeatureExtractor:
    """
    Extracts classical bioacoustic features from audio waveforms (for MLP Baseline).
    Features extracted:
    - MFCCs (13 coefficients)
    - Delta MFCCs (13 coefficients)
    - Delta-Delta MFCCs (13 coefficients)
    - Spectral Centroid (mean & std)
    - Spectral Bandwidth (mean & std)
    - Spectral Rolloff (mean & std)
    - Zero Crossing Rate (mean & std)
    - Root Mean Square Energy (mean & std)
    Total feature vector dimension: 49 features.
    """

    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13, n_fft: int = 1024, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts summary statistics of handcrafted acoustic features.

        Parameters
        ----------
        audio : np.ndarray
            1D audio signal array.

        Returns
        -------
        np.ndarray
            1D feature vector of dimension 49.
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()

        # Ensure floating point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # 1. MFCCs + Deltas
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)

        # 2. Spectral Features
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length)
        zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=self.hop_length)
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)

        spectral_stats = np.array([
            np.mean(centroid), np.std(centroid),
            np.mean(bandwidth), np.std(bandwidth),
            np.mean(rolloff), np.std(rolloff),
            np.mean(zcr), np.std(zcr),
            np.mean(rms), np.std(rms)
        ], dtype=np.float32)

        # Concatenate into single 1D vector
        features = np.concatenate([
            mfcc_mean,
            mfcc_delta_mean,
            mfcc_delta2_mean,
            spectral_stats
        ]).astype(np.float32)

        return features
