"""
Training Script for EfficientNetV2-S Transfer Learning on Log-Mel Spectrograms.
Owned by: Saurabh Tiwari (230911238)

Usage:
  python scripts/train_efficientnet.py --epochs 10 --batch-size 16 --use-temp
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import argparse
import yaml
import pandas as pd

from src.data.dataset import SyntheticInsectDataset, InsectAudioDataset, create_dataloaders
from src.models.efficientnet import InsectEfficientNetV2
from src.utils.trainer import ModelTrainer
from src.utils.metrics import (
    calculate_metrics,
    analyze_long_tail_performance,
    plot_training_curves,
    plot_confusion_matrix_heatmap
)


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-S Transfer Learning for Insect Bioacoustics")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Backbone learning rate")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Classifier head learning rate")
    parser.add_argument("--num-classes", type=int, default=40, help="Number of species classes")
    parser.add_argument("--use-temp", action="store_true", default=True, help="Enable temperature conditioning")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretraining")
    parser.add_argument("--data-csv", type=str, default=None, help="Path to metadata CSV if using real audio")
    parser.add_argument("--output-dir", type=str, default="results/efficientnet", help="Directory for logs and plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    print("=" * 60)
    print("  INSECT SPECIES IDENTIFICATION — EfficientNetV2-S")
    print("  Author: Saurabh Tiwari (230911238)")
    print("=" * 60)

    # 1. Dataset Loading
    if args.data_csv and os.path.exists(args.data_csv):
        print(f"Loading dataset from: {args.data_csv}")
        df = pd.read_csv(args.data_csv)
        train_loader, val_loader, test_loader, split_info = create_dataloaders(
            df, batch_size=args.batch_size, in_channels=1
        )
    else:
        print("Using Synthetic Insect Bioacoustic Dataset (Dolbear-calibrated chirps & trills)...")
        dataset = SyntheticInsectDataset(
            num_samples=1200,
            num_classes=args.num_classes,
            chunk_duration=5.0,
            n_mels=128,
            in_channels=1,
            is_training=True,
            long_tail=True
        )
        train_loader, val_loader, test_loader, split_info = create_dataloaders(
            dataset, batch_size=args.batch_size, in_channels=1
        )

    print(f"Dataset split: Train={split_info['train_samples']}, Val={split_info['val_samples']}, Test={split_info['test_samples']}")

    # 2. Model Initialization
    model = InsectEfficientNetV2(
        num_classes=args.num_classes,
        in_channels=1,
        pretrained=not args.no_pretrained,
        use_temperature=args.use_temp,
        dropout_rate=0.3
    )
    print(f"Model: InsectEfficientNetV2 (Pretrained: {not args.no_pretrained}, Temp Conditioning: {args.use_temp})")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")

    # 3. Differentiated Learning Rates for Backbone and Head
    param_groups = model.get_param_groups(lr_backbone=args.lr_backbone, lr_head=args.lr_head)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # 4. Trainer
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        checkpoint_dir="checkpoints",
        model_name="efficientnetv2",
        use_temperature=args.use_temp
    )

    # 5. Fit Model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=6
    )

    # 6. Final Evaluation on Test Set
    trainer.load_checkpoint(os.path.join("checkpoints", "best_efficientnetv2.pth"))
    test_loss, test_metrics, y_true, y_pred = trainer.evaluate(test_loader)

    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION RESULTS — EfficientNetV2-S")
    print("=" * 60)
    print(f"Test Loss:        {test_loss:.4f}")
    print(f"Top-1 Accuracy:   {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro-F1 Score:   {test_metrics['macro_f1']*100:.2f}%")
    print(f"Weighted-F1:      {test_metrics['weighted_f1']*100:.2f}%")
    if "top5_accuracy" in test_metrics:
        print(f"Top-5 Accuracy:   {test_metrics['top5_accuracy']*100:.2f}%")
    print("=" * 60)

    # 7. Long-Tail Tier Analysis
    class_counts = pd.Series(y_true).value_counts().to_dict()
    tier_results = analyze_long_tail_performance(y_true, y_pred, class_counts, head_threshold=15, tail_threshold=5)
    print("\nLong-Tail Performance Breakdown:")
    for tier, res in tier_results.items():
        print(f" - {tier.upper()} Tier (Classes: {res['num_classes']}, Samples: {res['num_samples']}): Accuracy = {res['accuracy']*100:.2f}%, Macro-F1 = {res['macro_f1']*100:.2f}%")

    # 8. Plot Curves & Confusion Matrix
    curves_path = os.path.join(args.output_dir, "efficientnet_training_curves.png")
    plot_training_curves(history, save_path=curves_path)
    print(f"\nSaved training curves to: {curves_path}")

    cm_path = os.path.join(args.output_dir, "efficientnet_confusion_matrix.png")
    plot_confusion_matrix_heatmap(y_true, y_pred, save_path=cm_path, max_classes_to_show=15)
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()
