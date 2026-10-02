"""
Unified Evaluation and Comparative Benchmark Script.
Enforces the Common Evaluation Protocol across all 4 architecture families:
- MLP Baseline (Abhishek Patro)
- VGG-style Log-Mel CNN (Saurabh Tiwari)
- EfficientNetV2-S Transfer Learning (Saurabh Tiwari)
- CRNN BiLSTM (Saksham Gupta)
- Audio Spectrogram Transformer (Harsh Kumar Roy)
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
import numpy as np
import argparse

from src.data.dataset import SyntheticInsectDataset, create_dataloaders
from src.models.vgg_cnn import LogMelCNN
from src.models.efficientnet import InsectEfficientNetV2
from src.models.crnn import InsectCRNN
from src.models.mlp_baseline import InsectMLP
from src.models.ast_model import InsectAST
from src.utils.trainer import ModelTrainer
from src.utils.metrics import calculate_metrics, analyze_long_tail_performance


def evaluate_model(model_name, model, test_loader, checkpoint_path=None, use_temp=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint for {model_name} from {checkpoint_path}")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            spectrograms = batch["spectrogram"].to(device)
            labels = batch["label"].to(device)
            temp = batch["temperature"].to(device) if use_temp else None

            if use_temp:
                outputs = model(spectrograms, temperature=temp)
            else:
                outputs = model(spectrograms)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = calculate_metrics(y_true, y_pred, y_prob=y_prob, num_classes=outputs.size(1))
    return metrics, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Comparative Evaluation")
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 70)
    print("  COMMON EVALUATION PROTOCOL — CROSS-MODEL BENCHMARK")
    print("=" * 70)

    # 1. Common Shared Test Split
    dataset = SyntheticInsectDataset(
        num_samples=1200,
        num_classes=args.num_classes,
        chunk_duration=5.0,
        n_mels=128,
        is_training=False,
        long_tail=True,
        random_seed=42
    )
    _, _, test_loader, split_info = create_dataloaders(dataset, batch_size=args.batch_size)

    models_to_eval = [
        ("VGG-style Log-Mel CNN", LogMelCNN(num_classes=args.num_classes, use_temperature=False), "checkpoints/best_vgg_cnn.pth", False),
        ("VGG-style CNN + Temp Conditioning", LogMelCNN(num_classes=args.num_classes, use_temperature=True), "checkpoints/best_vgg_cnn_temp.pth", True),
        ("EfficientNetV2-S Transfer Learning", InsectEfficientNetV2(num_classes=args.num_classes, pretrained=False, use_temperature=False), "checkpoints/best_efficientnetv2.pth", False),
        ("EfficientNetV2-S + Temp Conditioning", InsectEfficientNetV2(num_classes=args.num_classes, pretrained=False, use_temperature=True), "checkpoints/best_efficientnetv2_temp.pth", True),
        ("CRNN (Conv + BiLSTM)", InsectCRNN(num_classes=args.num_classes), "checkpoints/best_crnn.pth", False),
        ("Audio Spectrogram Transformer (AST)", InsectAST(num_classes=args.num_classes), "checkpoints/best_ast.pth", False),
    ]

    results_table = []

    for name, model, ckpt, use_temp in models_to_eval:
        metrics, y_true, y_pred = evaluate_model(name, model, test_loader, checkpoint_path=ckpt, use_temp=use_temp)
        class_counts = pd.Series(y_true).value_counts().to_dict()
        tier = analyze_long_tail_performance(y_true, y_pred, class_counts, head_threshold=15, tail_threshold=5)

        results_table.append({
            "Architecture": name,
            "Accuracy (%)": f"{metrics['accuracy']*100:.2f}",
            "Macro-F1 (%)": f"{metrics['macro_f1']*100:.2f}",
            "Weighted-F1 (%)": f"{metrics['weighted_f1']*100:.2f}",
            "Head F1 (%)": f"{tier['head']['macro_f1']*100:.2f}",
            "Medium F1 (%)": f"{tier['medium']['macro_f1']*100:.2f}",
            "Tail F1 (%)": f"{tier['tail']['macro_f1']*100:.2f}"
        })

    summary_df = pd.DataFrame(results_table)
    print("\n" + summary_df.to_markdown(index=False))

    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/cross_model_comparison.csv", index=False)
    print("\nSaved comparative benchmark table to results/cross_model_comparison.csv")


if __name__ == "__main__":
    main()
