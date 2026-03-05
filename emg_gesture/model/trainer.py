"""
Training Module for EMG Gesture Classification.

Handles model training, validation, checkpointing, and logging.
"""

import csv
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

from .bilstm_cnn import BiLSTMCNN, create_model_from_config


class Trainer:
    """
    Trainer for the BiLSTM-CNN gesture classification model.

    Handles the training loop, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: BiLSTMCNN,
        config_path: str,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer.

        Args:
            model: BiLSTM-CNN model to train.
            config_path: Path to config.yaml file.
            device: Device to train on (auto-detected if None).
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = model
        self.config_path = config_path

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model.to(self.device)
        print(f"[Trainer] Using device: {self.device}")

        # Training parameters
        train_config = self.config['training']
        self.epochs = train_config['epochs']
        self.learning_rate = train_config['learning_rate']
        self.weight_decay = train_config['weight_decay']
        self.patience = train_config['early_stopping_patience']

        # Setup paths
        self.checkpoints_dir = Path(self.config['paths']['checkpoints_dir'])
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoints_dir / 'best_model.pt'
        self.log_path = Path(self.config['paths']['training_log'])

        # Initialize optimizer and scheduler (set up in train())
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[CosineAnnealingLR] = None
        self.criterion: Optional[nn.CrossEntropyLoss] = None

        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.training_history: List[Dict] = []

        # Class names for confusion matrix
        self.gesture_classes = self.config['gesture_classes']

    def setup_training(
        self,
        train_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Set up optimizer, scheduler, and loss function.

        Args:
            train_loader: Training data loader.
            class_weights: Optional class weights for imbalanced data.
        """
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Cosine annealing scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.learning_rate * 0.01
        )

        # Loss function with optional class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (windows, features, labels) in enumerate(train_loader):
            windows = windows.to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(windows, features)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float, np.ndarray]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Tuple of (average_loss, accuracy, confusion_matrix).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        n_classes = len(self.gesture_classes)
        confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

        with torch.no_grad():
            for windows, features, labels in val_loader:
                windows = windows.to(self.device)
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(windows, features)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update confusion matrix
                for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                    confusion[t, p] += 1

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy, confusion

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            is_best: If True, this is the best model so far.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config_path': str(self.config_path),
            'training_history': self.training_history
        }
        torch.save(checkpoint, path)

        if is_best:
            print(f"[Trainer] Saved best model to {path}")

    def load_checkpoint(self, path: Path) -> bool:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            True if checkpoint loaded successfully, False otherwise.
        """
        if not path.exists():
            print(f"[Trainer] No checkpoint found at {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.training_history = checkpoint.get('training_history', [])

        print(f"[Trainer] Loaded checkpoint from {path} (epoch {self.current_epoch})")
        return True

    def print_confusion_matrix(self, confusion: np.ndarray):
        """
        Print a formatted confusion matrix.

        Args:
            confusion: Confusion matrix of shape (n_classes, n_classes).
        """
        n_classes = len(self.gesture_classes)

        # Find max label length for formatting
        max_len = max(len(g) for g in self.gesture_classes)

        # Header
        header = " " * (max_len + 2) + " | "
        header += " | ".join(f"{g[:8]:>8}" for g in self.gesture_classes)
        print("\n" + "=" * len(header))
        print("CONFUSION MATRIX")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        # Rows
        for i, gesture in enumerate(self.gesture_classes):
            row = f"{gesture:>{max_len}} | "
            row += " | ".join(f"{confusion[i, j]:>8}" for j in range(n_classes))
            print(row)

        print("=" * len(header))

        # Per-class metrics
        print("\nPer-class metrics:")
        for i, gesture in enumerate(self.gesture_classes):
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"  {gesture}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    def save_training_log(self):
        """Save training history to CSV."""
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
            for record in self.training_history:
                writer.writerow([
                    record['epoch'],
                    record['train_loss'],
                    record['train_acc'],
                    record['val_loss'],
                    record['val_acc'],
                    record['lr']
                ])
        print(f"[Trainer] Saved training log to {self.log_path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        resume: bool = False
    ) -> Dict:
        """
        Full training loop with validation and early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            class_weights: Optional class weights for imbalanced data.
            resume: If True, resume from latest checkpoint.

        Returns:
            Dictionary with final training metrics.
        """
        # Setup training components
        self.setup_training(train_loader, class_weights)

        # Resume from checkpoint if requested
        if resume:
            self.load_checkpoint(self.best_model_path)

        start_epoch = self.current_epoch

        print(f"\n[Trainer] Starting training from epoch {start_epoch + 1}")
        print(f"[Trainer] Training samples: {len(train_loader.dataset)}")
        print(f"[Trainer] Validation samples: {len(val_loader.dataset)}")
        print(f"[Trainer] Early stopping patience: {self.patience}")
        print("-" * 60)

        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, confusion = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Record history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': current_lr
            })

            # Print progress
            print(f"Epoch {epoch + 1:3d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self.save_checkpoint(self.best_model_path, is_best=True)
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n[Trainer] Early stopping triggered after {epoch + 1} epochs")
                break

        # Load best model for final evaluation
        self.load_checkpoint(self.best_model_path)

        # Final validation
        final_loss, final_acc, final_confusion = self.validate(val_loader)

        # Print final results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")

        self.print_confusion_matrix(final_confusion)

        # Save training log
        self.save_training_log()

        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'final_confusion': final_confusion,
            'epochs_trained': self.current_epoch + 1
        }


def main():
    """Test the trainer with mock data."""
    import argparse
    from torch.utils.data import TensorDataset

    parser = argparse.ArgumentParser(description='Trainer Test')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Create model
    model = create_model_from_config(args.config)

    # Create trainer
    trainer = Trainer(model, args.config)

    # Create mock data
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    n_samples = 100
    n_channels = config['signal']['n_channels']
    window_size = int(config['signal']['window_size_ms'] * config['signal']['sample_rate'] / 1000)
    n_features = n_channels * 5
    n_classes = len(config['gesture_classes'])

    windows = torch.randn(n_samples, n_channels, window_size)
    features = torch.randn(n_samples, n_features)
    labels = torch.randint(0, n_classes, (n_samples,))

    dataset = TensorDataset(windows, features, labels)
    train_size = int(0.8 * n_samples)
    val_size = n_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Train for a few epochs
    trainer.epochs = 5
    results = trainer.train(train_loader, val_loader)

    print(f"\nTraining results: {results}")


if __name__ == '__main__':
    main()
