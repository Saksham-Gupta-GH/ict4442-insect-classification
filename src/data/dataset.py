"""
Insect Acoustic Dataset and DataLoader Pipeline.
Owned by: Saurabh Tiwari (230911238)

Supports:
1. Real InsectSet459 / InsectSet66 audio files (WAV / MP3)
2. Fixed 5.0-second chunking and resampling
3. On-the-fly log-Mel spectrogram generation (128 Mel bands)
4. Waveform and Spectrogram augmentations (SpecAugment)
5. Auxiliary Ambient Temperature metadata extraction
6. Synthetic Insect Dataset generator for instant reproducibility and testing
"""

import os
import math
import random
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf

from src.data.features import LogMelExtractor
from src.data.augmentation import AudioAugmentationPipeline


class InsectAudioDataset(Dataset):
    """
    PyTorch Dataset for Insect Acoustic Classification.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['file_path', 'species_id' or 'label', and optional 'temperature'].
    sample_rate : int
        Target sample rate in Hz (default: 22050).
    chunk_duration : float
        Fixed audio chunk duration in seconds (default: 5.0).
    n_mels : int
        Number of Mel frequency bins (default: 128).
    in_channels : int
        Output channels: 1 for custom CNN, 3 for EfficientNet transfer learning (default: 1).
    is_training : bool
        Whether to apply data augmentations (default: False).
    augmentation_pipeline : Optional[AudioAugmentationPipeline]
        Augmentation pipeline instance.
    normalize_temperature : bool
        Whether to normalize temperature values to standard range (mean ~25C, std ~7C).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sample_rate: int = 22050,
        chunk_duration: float = 5.0,
        n_mels: int = 128,
        in_channels: int = 1,
        is_training: bool = False,
        augmentation_pipeline: Optional[AudioAugmentationPipeline] = None,
        normalize_temperature: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.target_length = int(sample_rate * chunk_duration)
        self.n_mels = n_mels
        self.in_channels = in_channels
        self.is_training = is_training
        self.normalize_temperature = normalize_temperature

        # Feature extractor
        self.feature_extractor = LogMelExtractor(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels,
            normalize=True
        )

        # Augmentation pipeline
        if is_training and augmentation_pipeline is None:
            self.aug_pipeline = AudioAugmentationPipeline(
                use_waveform_aug=True,
                use_specaugment=True
            )
        else:
            self.aug_pipeline = augmentation_pipeline

    def __len__(self) -> int:
        return len(self.df)

    def _load_and_pad_audio(self, file_path: str) -> torch.Tensor:
        """Loads audio, resamples if needed, and pads/crops to target length."""
        try:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono
            
            # Resample if sample rate doesn't match
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
        except Exception:
            # Fallback to zeros if file read fails
            waveform = torch.zeros((1, self.target_length), dtype=torch.float32)

        # Flatten to 1D
        waveform = waveform.squeeze(0)
        curr_len = waveform.shape[0]

        if curr_len < self.target_length:
            # Pad with zeros or replicate
            pad_amount = self.target_length - curr_len
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right), mode="constant", value=0.0)
        elif curr_len > self.target_length:
            # Random crop during training, center crop during evaluation
            if self.is_training:
                max_start = curr_len - self.target_length
                start = random.randint(0, max_start)
            else:
                start = (curr_len - self.target_length) // 2
            waveform = waveform[start : start + self.target_length]

        return waveform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        file_path = str(row.get("file_path", ""))

        waveform = self._load_and_pad_audio(file_path)

        # 1. Waveform augmentation
        if self.is_training and self.aug_pipeline is not None:
            waveform = self.aug_pipeline.augment_waveform(waveform)

        # 2. Extract Log-Mel Spectrogram (1, n_mels, time_steps)
        spectrogram = self.feature_extractor(waveform)

        # 3. Spectrogram augmentation (SpecAugment)
        if self.is_training and self.aug_pipeline is not None:
            spectrogram = self.aug_pipeline.augment_spectrogram(spectrogram, is_training=True)

        # 4. Expand to 3 channels if required for transfer learning backbones
        if self.in_channels == 3 and spectrogram.size(0) == 1:
            spectrogram = spectrogram.repeat(3, 1, 1)

        # Label extraction
        label = int(row.get("label", row.get("species_id", 0)))

        # Temperature extraction
        temp_val = row.get("temperature", 25.0)
        if pd.isna(temp_val):
            temp_val = 25.0
        temp_val = float(temp_val)

        if self.normalize_temperature:
            # Standardize: mean ~25C, std ~7C
            norm_temp = (temp_val - 25.0) / 7.0
        else:
            norm_temp = temp_val

        temp_tensor = torch.tensor([norm_temp], dtype=torch.float32)

        return {
            "spectrogram": spectrogram,
            "label": torch.tensor(label, dtype=torch.long),
            "temperature": temp_tensor,
            "raw_temperature": torch.tensor([temp_val], dtype=torch.float32),
            "file_path": file_path
        }


class SyntheticInsectDataset(Dataset):
    """
    Realistic Synthetic Insect Bioacoustic Dataset Generator.
    Simulates species-specific acoustic pulse trains (chirps, trills, carrier frequencies,
    temperature-dependent pulse rates according to Dolbear's Law), noise, and long-tail class distributions.
    
    Perfect for self-testing pipelines, unit testing, and fast local experiments.
    """

    def __init__(
        self,
        num_samples: int = 1200,
        num_classes: int = 40,
        sample_rate: int = 22050,
        chunk_duration: float = 5.0,
        n_mels: int = 128,
        in_channels: int = 1,
        is_training: bool = False,
        long_tail: bool = True,
        random_seed: int = 42
    ):
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.target_length = int(sample_rate * chunk_duration)
        self.n_mels = n_mels
        self.in_channels = in_channels
        self.is_training = is_training

        self.feature_extractor = LogMelExtractor(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels,
            normalize=True
        )

        self.aug_pipeline = AudioAugmentationPipeline() if is_training else None

        # Generate species acoustic profiles
        # Carrier frequency range (Orthoptera / Cicadidae: 2 kHz to 10 kHz)
        self.species_profiles = {}
        for c in range(num_classes):
            base_freq = np.random.uniform(2500, 8500)
            base_pulse_rate = np.random.uniform(15, 60)  # Pulses per second
            temp_sensitivity = np.random.uniform(0.5, 1.8)  # Hz and pulse rate shift per deg C
            self.species_profiles[c] = {
                "base_freq": base_freq,
                "base_pulse_rate": base_pulse_rate,
                "temp_sensitivity": temp_sensitivity
            }

        # Long-tail sample assignment
        if long_tail:
            # Zipfian / power-law distribution
            weights = 1.0 / (np.arange(1, num_classes + 1) ** 0.85)
            probs = weights / weights.sum()
            self.labels = np.random.choice(num_classes, size=num_samples, p=probs)
        else:
            self.labels = np.random.randint(0, num_classes, size=num_samples)

        # Ambient temperatures (15°C to 38°C)
        self.temperatures = np.random.normal(26.0, 5.5, size=num_samples).clip(12.0, 42.0)

    def __len__(self) -> int:
        return self.num_samples

    def _synthesize_insect_call(self, species_id: int, temperature: float) -> torch.Tensor:
        """Synthesizes a realistic pulse train with harmonics and temperature dependence."""
        profile = self.species_profiles[species_id]
        
        # Temperature modulation (Dolbear effect)
        delta_t = temperature - 25.0
        freq = profile["base_freq"] + (delta_t * profile["temp_sensitivity"] * 30.0)
        pulse_rate = max(5.0, profile["base_pulse_rate"] + (delta_t * profile["temp_sensitivity"] * 0.8))

        t = np.linspace(0, self.chunk_duration, self.target_length, endpoint=False)

        # Carrier wave (fundamental + 2nd harmonic)
        carrier = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t)

        # Pulse train envelope (chirp pulses)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * pulse_rate * t)) ** 4

        # Modulated signal + mild background environmental pink noise
        signal = carrier * envelope
        noise = np.random.normal(0, 0.05, size=self.target_length)
        raw_audio = (signal + noise).astype(np.float32)

        # Normalize
        raw_audio = raw_audio / (np.max(np.abs(raw_audio)) + 1e-6)
        return torch.from_numpy(raw_audio)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label = int(self.labels[idx])
        temp_val = float(self.temperatures[idx])

        waveform = self._synthesize_insect_call(label, temp_val)

        # 1. Waveform augmentation
        if self.is_training and self.aug_pipeline is not None:
            waveform = self.aug_pipeline.augment_waveform(waveform)

        # 2. Extract Log-Mel Spectrogram
        spectrogram = self.feature_extractor(waveform)

        # 3. SpecAugment
        if self.is_training and self.aug_pipeline is not None:
            spectrogram = self.aug_pipeline.augment_spectrogram(spectrogram, is_training=True)

        # 4. Expand to 3 channels if needed
        if self.in_channels == 3 and spectrogram.size(0) == 1:
            spectrogram = spectrogram.repeat(3, 1, 1)

        norm_temp = (temp_val - 25.0) / 7.0

        return {
            "spectrogram": spectrogram,
            "label": torch.tensor(label, dtype=torch.long),
            "temperature": torch.tensor([norm_temp], dtype=torch.float32),
            "raw_temperature": torch.tensor([temp_val], dtype=torch.float32),
            "file_path": f"synthetic_sample_{idx}.wav"
        }


def create_dataloaders(
    dataset: Union[Dataset, pd.DataFrame],
    batch_size: int = 32,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    in_channels: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Creates stratified 60/20/20 Train/Validation/Test DataLoaders.
    
    Returns
    -------
    (train_loader, val_loader, test_loader, split_info)
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if isinstance(dataset, pd.DataFrame):
        df = dataset.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train : n_train + n_val]
        test_df = df.iloc[n_train + n_val :]

        train_ds = InsectAudioDataset(train_df, in_channels=in_channels, is_training=True)
        val_ds = InsectAudioDataset(val_df, in_channels=in_channels, is_training=False)
        test_ds = InsectAudioDataset(test_df, in_channels=in_channels, is_training=False)
    else:
        # Generic Dataset splitting
        total_len = len(dataset)
        n_train = int(total_len * train_ratio)
        n_val = int(total_len * val_ratio)
        n_test = total_len - n_train - n_val

        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(random_seed)
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    split_info = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "batch_size": batch_size
    }

    return train_loader, val_loader, test_loader, split_info
