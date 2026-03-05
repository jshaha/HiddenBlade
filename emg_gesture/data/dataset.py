"""
PyTorch Dataset for EMG Gesture Recognition.

Loads recorded EMG sessions, applies preprocessing, and provides
train/validation splits for model training.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import yaml

from .preprocessor import EMGPreprocessor


class EMGGestureDataset(Dataset):
    """
    PyTorch Dataset for EMG gesture classification.

    Loads .npz recordings, applies preprocessing pipeline, and returns
    (window_tensor, feature_tensor, label) tuples.
    """

    def __init__(
        self,
        config_path: str,
        recordings_dir: Optional[str] = None,
        preprocessor: Optional[EMGPreprocessor] = None,
        compute_stats: bool = False
    ):
        """
        Initialize the dataset.

        Args:
            config_path: Path to config.yaml file.
            recordings_dir: Directory containing .npz recordings.
                           If None, uses path from config.
            preprocessor: Optional preprocessor instance to reuse.
            compute_stats: If True, compute normalization stats from data.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.gesture_classes = self.config['gesture_classes']
        self.label_to_idx = {g: i for i, g in enumerate(self.gesture_classes)}
        self.idx_to_label = {i: g for i, g in enumerate(self.gesture_classes)}

        # Setup recordings directory
        if recordings_dir is None:
            recordings_dir = self.config['paths']['recordings_dir']
        self.recordings_dir = Path(recordings_dir)

        # Setup preprocessor
        if preprocessor is None:
            self.preprocessor = EMGPreprocessor(config_path)
        else:
            self.preprocessor = preprocessor

        # Load all recordings
        self.samples: List[Dict] = []
        self._load_recordings()

        # Compute or load normalization stats
        if compute_stats and len(self.samples) > 0:
            self._compute_normalization_stats()
        else:
            self.preprocessor.load_normalization_stats()

    def _load_recordings(self):
        """Load all .npz recordings from the recordings directory."""
        npz_files = list(self.recordings_dir.glob('*.npz'))

        if len(npz_files) == 0:
            print(f"[Dataset] No recordings found in {self.recordings_dir}")
            return

        for npz_path in npz_files:
            try:
                recording = np.load(npz_path, allow_pickle=True)
                data = recording['data']
                label = str(recording['label'])

                # Skip if label not in gesture classes
                if label not in self.label_to_idx:
                    print(f"[Dataset] Skipping {npz_path.name}: unknown label '{label}'")
                    continue

                # Create windows from recording
                windows, features = self.preprocessor.preprocess_recording(
                    data, normalize=False  # Normalize later after computing stats
                )

                label_idx = self.label_to_idx[label]

                for i in range(len(windows)):
                    self.samples.append({
                        'window': windows[i],
                        'features': features[i],
                        'label': label_idx,
                        'source': npz_path.name
                    })

                print(f"[Dataset] Loaded {npz_path.name}: {len(windows)} windows, label='{label}'")

            except Exception as e:
                print(f"[Dataset] Error loading {npz_path}: {e}")

        print(f"[Dataset] Total samples: {len(self.samples)}")

    def _compute_normalization_stats(self):
        """Compute normalization statistics from all loaded data."""
        if len(self.samples) == 0:
            return

        # Concatenate all windows
        all_data = np.concatenate([s['window'] for s in self.samples], axis=0)
        self.preprocessor.compute_normalization_stats(all_data, save=True)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (window_tensor, feature_tensor, label):
                - window_tensor: Shape (channels, time_steps) for CNN
                - feature_tensor: Shape (n_features,) for LSTM
                - label: Integer class label
        """
        sample = self.samples[idx]

        # Get window and apply normalization
        window = sample['window'].copy()
        features = sample['features'].copy()

        if self.preprocessor.norm_stats is not None:
            window = self.preprocessor.normalize(window)

        # Transpose window to (channels, time_steps) for Conv1D
        window = window.T  # (time, channels) -> (channels, time)

        # Convert to tensors
        window_tensor = torch.from_numpy(window).float()
        feature_tensor = torch.from_numpy(features).float()
        label = sample['label']

        return window_tensor, feature_tensor, label

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of samples across classes.

        Returns:
            Dictionary mapping class names to sample counts.
        """
        distribution = {g: 0 for g in self.gesture_classes}
        for sample in self.samples:
            label_name = self.idx_to_label[sample['label']]
            distribution[label_name] += 1
        return distribution

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced data.

        Returns:
            Tensor of class weights (inverse frequency).
        """
        distribution = self.get_class_distribution()
        total = sum(distribution.values())

        if total == 0:
            return torch.ones(len(self.gesture_classes))

        weights = []
        for gesture in self.gesture_classes:
            count = distribution[gesture]
            if count > 0:
                weights.append(total / (len(self.gesture_classes) * count))
            else:
                weights.append(1.0)

        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    config_path: str,
    recordings_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    val_split: Optional[float] = None,
    num_workers: int = 0,
    compute_stats: bool = True
) -> Tuple[DataLoader, DataLoader, EMGGestureDataset]:
    """
    Create train and validation DataLoaders.

    Args:
        config_path: Path to config.yaml file.
        recordings_dir: Directory containing recordings.
        batch_size: Batch size (uses config default if None).
        val_split: Validation split ratio (uses config default if None).
        num_workers: Number of data loading workers.
        compute_stats: If True, compute normalization stats from training data.

    Returns:
        Tuple of (train_loader, val_loader, full_dataset).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if batch_size is None:
        batch_size = config['training']['batch_size']
    if val_split is None:
        val_split = config['training']['val_split']

    # Create full dataset
    dataset = EMGGestureDataset(
        config_path=config_path,
        recordings_dir=recordings_dir,
        compute_stats=compute_stats
    )

    if len(dataset) == 0:
        print("[Dataset] Warning: Empty dataset, returning empty loaders")
        return None, None, dataset

    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"[Dataset] Train: {train_size}, Validation: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, dataset


def generate_mock_recordings(
    config_path: str,
    n_samples_per_class: int = 1000,
    output_dir: Optional[str] = None
):
    """
    Generate mock recordings for testing the training pipeline.

    Args:
        config_path: Path to config.yaml file.
        n_samples_per_class: Number of samples per gesture class.
        output_dir: Output directory for recordings.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if output_dir is None:
        output_dir = config['paths']['recordings_dir']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_channels = config['signal']['n_channels']
    sample_rate = config['signal']['sample_rate']
    gesture_classes = config['gesture_classes']

    for gesture_idx, gesture in enumerate(gesture_classes):
        # Generate synthetic EMG data with class-specific patterns
        data = np.random.randn(n_samples_per_class, n_channels) * 100

        # Add class-specific patterns
        t = np.arange(n_samples_per_class) / sample_rate
        for ch in range(n_channels):
            # Different frequency components per class
            base_freq = 30 + gesture_idx * 20  # 30Hz, 50Hz, 70Hz
            data[:, ch] += 50 * np.sin(2 * np.pi * base_freq * t + ch * 0.5)

            # Different amplitude patterns
            amplitude_mod = 1.0 + 0.5 * gesture_idx
            data[:, ch] *= amplitude_mod

        # Save recording
        output_path = output_dir / f"{gesture}_mock.npz"
        np.savez(
            output_path,
            data=data,
            label=gesture,
            sample_rate=sample_rate,
            n_channels=n_channels
        )
        print(f"[Dataset] Generated mock recording: {output_path}")


def main():
    """Test the dataset with mock data."""
    import argparse

    parser = argparse.ArgumentParser(description='EMG Dataset Test')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--generate-mock', action='store_true',
                        help='Generate mock recordings')
    parser.add_argument('--test-loader', action='store_true',
                        help='Test data loader')

    args = parser.parse_args()

    if args.generate_mock:
        generate_mock_recordings(args.config)

    if args.test_loader:
        train_loader, val_loader, dataset = create_dataloaders(args.config)

        if train_loader is None:
            print("No data available")
            return

        print(f"\nClass distribution: {dataset.get_class_distribution()}")
        print(f"Class weights: {dataset.get_class_weights()}")

        # Test iteration
        for batch_idx, (windows, features, labels) in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Windows shape: {windows.shape}")
            print(f"  Features shape: {features.shape}")
            print(f"  Labels: {labels}")
            if batch_idx >= 2:
                break


if __name__ == '__main__':
    main()
