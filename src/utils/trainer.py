"""
PyTorch Training & Evaluation Engine for Bioacoustic Models.
Owned by: Saurabh Tiwari (230911238)

Implements:
1. Training loop with learning rate scheduling and gradient clipping.
2. Validation loop with Top-1 Accuracy, Top-5 Accuracy, and Macro-F1 evaluation.
3. Early stopping and Best Model Checkpoint persistence.
4. Comprehensive history logging.
"""

import os
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import calculate_metrics


class ModelTrainer:
    """
    Standardized trainer for all bioacoustic architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "vgg_cnn",
        use_temperature: bool = False
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.use_temperature = use_temperature

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": []
        }
        self.best_macro_f1 = 0.0
        self.best_epoch = 0

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Runs one full training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            spectrograms = batch["spectrogram"].to(self.device)
            labels = batch["label"].to(self.device)
            temp = batch["temperature"].to(self.device) if self.use_temperature else None

            self.optimizer.zero_grad()

            if self.use_temperature:
                outputs = self.model(spectrograms, temperature=temp)
            else:
                outputs = self.model(spectrograms)

            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """Runs evaluation over validation or test set."""
        self.model.eval()
        total_loss = 0.0
        total = 0

        all_preds = []
        all_labels = []
        all_probs = []

        for batch in val_loader:
            spectrograms = batch["spectrogram"].to(self.device)
            labels = batch["label"].to(self.device)
            temp = batch["temperature"].to(self.device) if self.use_temperature else None

            if self.use_temperature:
                outputs = self.model(spectrograms, temperature=temp)
            else:
                outputs = self.model(spectrograms)

            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            total += len(labels)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

        avg_loss = total_loss / total
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = calculate_metrics(y_true, y_pred, y_prob=y_prob, num_classes=outputs.size(1))
        return avg_loss, metrics, y_true, y_pred

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        early_stopping_patience: int = 6
    ) -> Dict[str, List[float]]:
        """
        Executes full training and validation pipeline with early stopping and checkpointing.
        """
        print(f"Starting training for {self.model_name} on device: {self.device}")
        epochs_without_improvement = 0

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_metrics, _, _ = self.evaluate(val_loader)

            val_acc = val_metrics["accuracy"]
            val_macro_f1 = val_metrics["macro_f1"]

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_macro_f1)

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                f"Val Macro-F1: {val_macro_f1*100:.2f}%"
            )

            # Checkpoint saving on best Macro-F1
            if val_macro_f1 > self.best_macro_f1:
                self.best_macro_f1 = val_macro_f1
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
                print(f" -> Best model saved with Macro-F1: {val_macro_f1*100:.2f}%")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch} (no improvement for {early_stopping_patience} epochs).")
                    break

        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed/60:.2f} mins. Best Val Macro-F1: {self.best_macro_f1*100:.2f}% at epoch {self.best_epoch}.")
        return self.history

    def save_checkpoint(self, is_best: bool = True, filename: Optional[str] = None) -> str:
        """Saves model weights and training metadata."""
        if filename is None:
            filename = f"best_{self.model_name}.pth" if is_best else f"checkpoint_{self.model_name}.pth"
        path = os.path.join(self.checkpoint_dir, filename)

        torch.save({
            "epoch": self.best_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_macro_f1": self.best_macro_f1,
            "model_name": self.model_name
        }, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Loads model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_macro_f1 = checkpoint.get("best_macro_f1", 0.0)
        print(f"Loaded checkpoint from {path} (Best Macro-F1: {self.best_macro_f1*100:.2f}%)")
