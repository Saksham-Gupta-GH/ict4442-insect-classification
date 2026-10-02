"""
Pipeline Verification and Smoke Test Demo.
Runs comprehensive sanity checks across:
1. Feature Extraction (128-band log-Mel Spectrogram & Handcrafted features)
2. Augmentation Pipeline (Waveform perturbations + SpecAugment)
3. VGG-Style CNN (LogMelCNN)
4. EfficientNetV2-S Transfer Learning (InsectEfficientNetV2)
5. Training loop, checkpointing, and metrics evaluation
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import pandas as pd

from src.data.features import LogMelExtractor, HandcraftedFeatureExtractor
from src.data.augmentation import AudioAugmentationPipeline, SpecAugment, WaveformAugmentation
from src.data.dataset import SyntheticInsectDataset, create_dataloaders
from src.models.vgg_cnn import LogMelCNN
from src.models.efficientnet import InsectEfficientNetV2
from src.models.crnn import InsectCRNN
from src.models.mlp_baseline import InsectMLP
from src.models.ast_model import InsectAST
from src.utils.trainer import ModelTrainer
from src.utils.metrics import calculate_metrics, analyze_long_tail_performance


def test_feature_extraction():
    print("\n[1/5] Testing Feature Extraction...")
    sample_rate = 22050
    duration = 5.0
    num_samples = int(sample_rate * duration)
    dummy_waveform = torch.randn(1, num_samples)

    # Log-Mel
    extractor = LogMelExtractor(sample_rate=sample_rate, n_mels=128, n_fft=1024, hop_length=512)
    spec = extractor(dummy_waveform)
    print(f" -> Log-Mel Spectrogram shape: {spec.shape} (Expected: (1, 1, 128, ~216))")
    assert spec.ndim == 4 and spec.shape[2] == 128, "Log-Mel Spectrogram shape mismatch!"

    # Handcrafted Features
    hc_extractor = HandcraftedFeatureExtractor(sample_rate=sample_rate)
    hc_feats = hc_extractor.extract(dummy_waveform.squeeze().numpy())
    print(f" -> Handcrafted Feature vector shape: {hc_feats.shape} (Expected: (49,))")
    assert hc_feats.shape == (49,), "Handcrafted feature shape mismatch!"
    print(" -> Feature Extraction tests PASSED.")


def test_augmentations():
    print("\n[2/5] Testing Data Augmentations...")
    dummy_waveform = torch.sin(torch.linspace(0, 100, 110250))
    pipeline = AudioAugmentationPipeline()

    aug_wave = pipeline.augment_waveform(dummy_waveform)
    assert aug_wave.shape == dummy_waveform.shape, "Waveform shape changed after augmentation!"

    dummy_spec = torch.randn(1, 1, 128, 216)
    aug_spec = pipeline.augment_spectrogram(dummy_spec, is_training=True)
    assert aug_spec.shape == dummy_spec.shape, "Spectrogram shape changed after SpecAugment!"
    print(" -> Augmentation tests PASSED.")


def test_models_forward_backward():
    print("\n[3/5] Testing Model Architectures (Forward & Backward passes)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b, c, h, w = 4, 1, 128, 216
    x = torch.randn(b, c, h, w, device=device)
    temp = torch.randn(b, 1, device=device)
    targets = torch.randint(0, 40, (b,), device=device)

    # 1. LogMelCNN
    cnn = LogMelCNN(num_classes=40, in_channels=1, use_temperature=True).to(device)
    out_cnn = cnn(x, temperature=temp)
    assert out_cnn.shape == (b, 40), f"LogMelCNN output shape mismatch: {out_cnn.shape}"
    loss_cnn = torch.nn.functional.cross_entropy(out_cnn, targets)
    loss_cnn.backward()
    print(" -> LogMelCNN (VGG-style + Temp conditioning) PASSED.")

    # 2. EfficientNetV2-S
    effnet = InsectEfficientNetV2(num_classes=40, in_channels=1, pretrained=False, use_temperature=True).to(device)
    out_eff = effnet(x, temperature=temp)
    assert out_eff.shape == (b, 40), f"InsectEfficientNetV2 output shape mismatch: {out_eff.shape}"
    loss_eff = torch.nn.functional.cross_entropy(out_eff, targets)
    loss_eff.backward()
    print(" -> InsectEfficientNetV2 (Transfer Learning + Temp conditioning) PASSED.")

    # 3. CRNN
    crnn = InsectCRNN(num_classes=40, in_channels=1, use_temperature=True).to(device)
    out_crnn = crnn(x, temperature=temp)
    assert out_crnn.shape == (b, 40), f"InsectCRNN output shape mismatch: {out_crnn.shape}"
    print(" -> InsectCRNN (BiLSTM) PASSED.")

    # 4. AST
    ast_m = InsectAST(num_classes=40, in_channels=1, pretrained=False, use_temperature=True).to(device)
    out_ast = ast_m(x, temperature=temp)
    assert out_ast.shape == (b, 40), f"InsectAST output shape mismatch: {out_ast.shape}"
    print(" -> InsectAST PASSED.")


def test_training_and_evaluation():
    print("\n[4/5] Testing End-to-End Training & Evaluation (2 Epochs)...")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results/demo", exist_ok=True)

    dataset = SyntheticInsectDataset(
        num_samples=300,
        num_classes=10,
        chunk_duration=5.0,
        n_mels=128,
        in_channels=1,
        is_training=True,
        long_tail=True,
        random_seed=42
    )
    train_loader, val_loader, test_loader, split_info = create_dataloaders(
        dataset, batch_size=16, in_channels=1
    )

    model = LogMelCNN(num_classes=10, in_channels=1, use_temperature=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir="checkpoints",
        model_name="demo_vgg_cnn",
        use_temperature=True
    )

    history = trainer.fit(train_loader, val_loader, epochs=2, early_stopping_patience=2)
    test_loss, test_metrics, y_true, y_pred = trainer.evaluate(test_loader)

    print(f" -> Demo Test Accuracy: {test_metrics['accuracy']*100:.2f}%, Macro-F1: {test_metrics['macro_f1']*100:.2f}%")
    assert "accuracy" in test_metrics and "macro_f1" in test_metrics
    print(" -> End-to-End Training & Evaluation PASSED.")


def main():
    print("=" * 60)
    print(" RUNNING SYSTEM INTEGRATION & VERIFICATION TESTS")
    print(" Owned by: Saurabh Tiwari (230911238)")
    print("=" * 60)
    test_feature_extraction()
    test_augmentations()
    test_models_forward_backward()
    test_training_and_evaluation()

    print("\n" + "=" * 60)
    print(" ALL PIPELINE SANITY TESTS SUCCESSFULLY PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
