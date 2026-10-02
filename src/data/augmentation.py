"""
Bioacoustic Data Augmentation Pipeline.
Owned by: Saurabh Tiwari (230911238)

Implements:
1. Waveform-level augmentations (Time Shift, Gaussian Noise, Gain, Pitch Perturbation)
2. Spectrogram-level augmentations (SpecAugment: Frequency Masking & Time Masking)
3. Mixup augmentation for spectrograms
4. Composable AudioAugmentationPipeline
"""

import random
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T


class WaveformAugmentation:
    """
    Applies audio waveform level augmentations to improve generalization on field recordings.
    """

    def __init__(
        self,
        time_shift_max_ratio: float = 0.2,
        noise_level: float = 0.005,
        gain_min: float = 0.8,
        gain_max: float = 1.2,
        p_time_shift: float = 0.5,
        p_noise: float = 0.5,
        p_gain: float = 0.5
    ):
        self.time_shift_max_ratio = time_shift_max_ratio
        self.noise_level = noise_level
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.p_time_shift = p_time_shift
        self.p_noise = p_noise
        self.p_gain = p_gain

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : torch.Tensor
            Audio waveform tensor of shape (num_samples,) or (1, num_samples).

        Returns
        -------
        torch.Tensor
            Augmented waveform tensor.
        """
        audio = waveform.clone()

        # 1. Random Time Shift (Circular roll)
        if random.random() < self.p_time_shift:
            num_samples = audio.shape[-1]
            max_shift = int(num_samples * self.time_shift_max_ratio)
            shift = random.randint(-max_shift, max_shift)
            audio = torch.roll(audio, shifts=shift, dims=-1)

        # 2. Additive Gaussian Noise (simulates environmental background noise)
        if random.random() < self.p_noise:
            noise = torch.randn_like(audio) * self.noise_level
            audio = audio + noise

        # 3. Random Gain / Volume scaling
        if random.random() < self.p_gain:
            gain = random.uniform(self.gain_min, self.gain_max)
            audio = audio * gain

        # Clamp to valid [-1.0, 1.0] audio range
        audio = torch.clamp(audio, -1.0, 1.0)
        return audio


class SpecAugment(nn.Module):
    """
    SpecAugment: Frequency and Time Masking for Log-Mel Spectrograms.
    Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR", Interspeech 2019.
    
    Parameters
    ----------
    freq_mask_param : int
        Maximum frequency channels to mask (F).
    time_mask_param : int
        Maximum time steps to mask (T).
    num_freq_masks : int
        Number of frequency masks to apply.
    num_time_masks : int
        Number of time masks to apply.
    mask_value : float
        Value to fill in the masked region (default: 0.0 for standardized spectrograms).
    """

    def __init__(
        self,
        freq_mask_param: int = 16,
        time_mask_param: int = 32,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        mask_value: float = 0.0
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spec : torch.Tensor
            Spectrogram tensor of shape (..., n_mels, time_steps).

        Returns
        -------
        torch.Tensor
            Masked spectrogram tensor.
        """
        if not self.training:
            return spec

        out = spec.clone()
        *_, n_mels, n_steps = out.shape

        # Apply Frequency Masks
        for _ in range(self.num_freq_masks):
            f_len = random.randint(0, min(self.freq_mask_param, n_mels))
            f_start = random.randint(0, n_mels - f_len)
            out[..., f_start : f_start + f_len, :] = self.mask_value

        # Apply Time Masks
        for _ in range(self.num_time_masks):
            t_len = random.randint(0, min(self.time_mask_param, n_steps))
            t_start = random.randint(0, n_steps - t_len)
            out[..., :, t_start : t_start + t_len] = self.mask_value

        return out


def apply_mixup(
    spectrograms: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies Mixup data augmentation to a batch of spectrograms and class targets.
    
    Parameters
    ----------
    spectrograms : torch.Tensor
        Batch of spectrograms (B, C, H, W).
    targets : torch.Tensor
        Batch of integer class labels (B,).
    alpha : float
        Beta distribution parameter.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
        (mixed_spectrograms, targets_a, targets_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = spectrograms.size(0)
    index = torch.randperm(batch_size).to(spectrograms.device)

    mixed_spectrograms = lam * spectrograms + (1 - lam) * spectrograms[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed_spectrograms, targets_a, targets_b, lam


class AudioAugmentationPipeline:
    """
    End-to-end unified augmentation pipeline coordinating waveform and spectrogram transforms.
    """

    def __init__(
        self,
        use_waveform_aug: bool = True,
        use_specaugment: bool = True,
        time_shift_max_ratio: float = 0.2,
        noise_level: float = 0.005,
        freq_mask_param: int = 16,
        time_mask_param: int = 32,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        self.use_waveform_aug = use_waveform_aug
        self.use_specaugment = use_specaugment

        if self.use_waveform_aug:
            self.waveform_aug = WaveformAugmentation(
                time_shift_max_ratio=time_shift_max_ratio,
                noise_level=noise_level
            )
        else:
            self.waveform_aug = None

        if self.use_specaugment:
            self.spec_aug = SpecAugment(
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
                num_freq_masks=num_freq_masks,
                num_time_masks=num_time_masks
            )
        else:
            self.spec_aug = None

    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.use_waveform_aug and self.waveform_aug is not None:
            return self.waveform_aug(waveform)
        return waveform

    def augment_spectrogram(self, spectrogram: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        if self.use_specaugment and self.spec_aug is not None and is_training:
            self.spec_aug.train(True)
            return self.spec_aug(spectrogram)
        return spectrogram
