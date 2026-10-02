"""
Evaluation Metrics and Bioacoustic Benchmark Analysis.
Owned by: Saurabh Tiwari (230911238)

Implements:
1. Macro-F1, Weighted-F1, Micro-F1, Top-1 and Top-5 Accuracy
2. Long-Tail Tier Analysis (Head / Medium / Tail species breakdown)
3. Confusion Matrix and Training Curve Plotting Utilities
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Computes standard bioacoustic classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth integer labels (N,).
    y_pred : np.ndarray
        Predicted integer class labels (N,).
    y_prob : Optional[np.ndarray]
        Predicted probability distributions (N, C).
    num_classes : Optional[int]
        Total number of classes.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, macro_f1, weighted_f1, micro_f1, etc.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weight)
    }

    if y_prob is not None and num_classes is not None and num_classes >= 5:
        try:
            top5 = top_k_accuracy_score(
                y_true, y_prob, k=5, labels=np.arange(num_classes)
            )
            metrics["top5_accuracy"] = float(top5)
        except Exception:
            metrics["top5_accuracy"] = 0.0

    return metrics


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Generates per-species precision, recall, f1-score, and support DataFrame."""
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report_dict).transpose()
    return df


def analyze_long_tail_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_counts: Dict[int, int],
    head_threshold: int = 100,
    tail_threshold: int = 40
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates model performance across long-tail tiers:
    - Head Tier: species with > head_threshold recordings
    - Medium Tier: species with [tail_threshold, head_threshold] recordings
    - Tail Tier: species with < tail_threshold recordings
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    head_classes = {c for c, count in class_counts.items() if count > head_threshold}
    med_classes = {c for c, count in class_counts.items() if tail_threshold <= count <= head_threshold}
    tail_classes = {c for c, count in class_counts.items() if count < tail_threshold}

    results = {}
    for tier_name, tier_set in [("head", head_classes), ("medium", med_classes), ("tail", tail_classes)]:
        mask = np.isin(y_true, list(tier_set))
        if np.sum(mask) > 0:
            tier_acc = accuracy_score(y_true[mask], y_pred[mask])
            _, _, tier_f1, _ = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average="macro", zero_division=0
            )
            results[tier_name] = {
                "num_classes": len(tier_set),
                "num_samples": int(np.sum(mask)),
                "accuracy": float(tier_acc),
                "macro_f1": float(tier_f1)
            }
        else:
            results[tier_name] = {
                "num_classes": len(tier_set),
                "num_samples": 0,
                "accuracy": 0.0,
                "macro_f1": 0.0
            }

    return results


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots training and validation loss and Macro-F1 curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Curve
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    axes[0].set_title("Cross-Entropy Loss vs Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # 2. Macro-F1 Curve
    if "val_macro_f1" in history:
        axes[1].plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="#2ca02c", lw=2)
    if "val_acc" in history:
        axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="#d62728", linestyle="--", lw=2)
    axes[1].set_title("Validation Macro-F1 & Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Metric Score", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    max_classes_to_show: int = 20
) -> plt.Figure:
    """Plots a normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] > max_classes_to_show:
        cm = cm[:max_classes_to_show, :max_classes_to_show]
        if class_names is not None:
            class_names = class_names[:max_classes_to_show]

    # Normalize by row
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-6)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
        ax=ax
    )
    ax.set_title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Species", fontsize=11)
    ax.set_ylabel("True Species", fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
