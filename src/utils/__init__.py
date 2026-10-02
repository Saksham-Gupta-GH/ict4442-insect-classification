"""
Utilities package for insect acoustic classification.
Provides metric calculation, long-tail tier analysis, visualization, and training engine.
"""

from src.utils.metrics import (
    calculate_metrics,
    per_class_report,
    analyze_long_tail_performance,
    plot_training_curves,
    plot_confusion_matrix_heatmap
)
from src.utils.trainer import ModelTrainer

__all__ = [
    "calculate_metrics",
    "per_class_report",
    "analyze_long_tail_performance",
    "plot_training_curves",
    "plot_confusion_matrix_heatmap",
    "ModelTrainer"
]
