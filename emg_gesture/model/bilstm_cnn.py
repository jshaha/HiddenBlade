"""
BiLSTM-CNN Hybrid Model for EMG Gesture Classification.

Combines a CNN branch for spatial feature extraction with a
BiLSTM branch for temporal feature extraction.
"""

from typing import Tuple

import torch
import torch.nn as nn
import yaml


class CNNBranch(nn.Module):
    """
    CNN branch for spatial feature extraction from raw EMG signals.

    Uses 1D convolutions to extract spatial patterns across time.
    """

    def __init__(
        self,
        n_channels: int,
        window_size: int,
        filters: list = [32, 64, 128],
        kernel_size: int = 3,
        pool_size: int = 2
    ):
        """
        Initialize the CNN branch.

        Args:
            n_channels: Number of EMG channels (input channels).
            window_size: Number of time steps in the input window.
            filters: List of filter counts for each conv layer.
            kernel_size: Kernel size for conv layers.
            pool_size: Pool size for max pooling.
        """
        super().__init__()

        self.n_channels = n_channels
        self.filters = filters

        layers = []
        in_channels = n_channels

        current_size = window_size

        for i, out_channels in enumerate(filters):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(pool_size))

            in_channels = out_channels
            current_size = current_size // pool_size

        self.conv_layers = nn.Sequential(*layers)

        # Calculate output size
        self.output_size = filters[-1] * max(1, current_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time_steps).

        Returns:
            Flattened feature vector of shape (batch, output_size).
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x


class BiLSTMBranch(nn.Module):
    """
    Bidirectional LSTM branch for temporal feature extraction.

    Processes sequences of feature vectors to capture temporal dynamics.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize the BiLSTM branch.

        Args:
            input_size: Size of input feature vectors.
            hidden_size: LSTM hidden state size.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate between LSTM layers.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output size is 2 * hidden_size (bidirectional)
        self.output_size = hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size) or
               (batch, input_size) for single feature vectors.

        Returns:
            Final hidden state of shape (batch, hidden_size * 2).
        """
        # Handle single feature vectors by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_size)

        # LSTM forward pass
        output, (h_n, c_n) = self.lstm(x)

        # Concatenate final hidden states from both directions
        # h_n shape: (num_layers * 2, batch, hidden_size)
        forward_hidden = h_n[-2, :, :]  # Last forward layer
        backward_hidden = h_n[-1, :, :]  # Last backward layer
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        return combined


class BiLSTMCNN(nn.Module):
    """
    Hybrid BiLSTM-CNN model for EMG gesture classification.

    Combines spatial features from CNN and temporal features from BiLSTM.
    """

    def __init__(
        self,
        n_channels: int,
        window_size: int,
        n_features: int,
        n_classes: int,
        cnn_filters: list = [32, 64, 128],
        cnn_kernel_size: int = 3,
        cnn_pool_size: int = 2,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_hidden_size: int = 256,
        fc_dropout: float = 0.5
    ):
        """
        Initialize the hybrid model.

        Args:
            n_channels: Number of EMG channels.
            window_size: Number of time steps in input window.
            n_features: Size of extracted feature vectors for LSTM.
            n_classes: Number of gesture classes.
            cnn_filters: List of filter counts for CNN layers.
            cnn_kernel_size: Kernel size for CNN layers.
            cnn_pool_size: Pool size for CNN max pooling.
            lstm_hidden_size: LSTM hidden state size.
            lstm_num_layers: Number of LSTM layers.
            lstm_dropout: Dropout rate for LSTM.
            fc_hidden_size: Hidden size for fusion FC layers.
            fc_dropout: Dropout rate for fusion layers.
        """
        super().__init__()

        self.n_classes = n_classes

        # CNN branch for spatial features
        self.cnn_branch = CNNBranch(
            n_channels=n_channels,
            window_size=window_size,
            filters=cnn_filters,
            kernel_size=cnn_kernel_size,
            pool_size=cnn_pool_size
        )

        # BiLSTM branch for temporal features
        self.lstm_branch = BiLSTMBranch(
            input_size=n_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout
        )

        # Fusion layers
        fusion_input_size = self.cnn_branch.output_size + self.lstm_branch.output_size

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fc_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, fc_hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size // 2, n_classes)
        )

    def forward(
        self,
        window: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the hybrid model.

        Args:
            window: Raw EMG window of shape (batch, channels, time_steps).
            features: Extracted features of shape (batch, n_features).

        Returns:
            Class logits of shape (batch, n_classes).
        """
        # CNN branch
        cnn_out = self.cnn_branch(window)

        # BiLSTM branch
        lstm_out = self.lstm_branch(features)

        # Concatenate and fuse
        combined = torch.cat([cnn_out, lstm_out], dim=1)
        logits = self.fusion(combined)

        return logits

    def predict(
        self,
        window: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with confidence scores.

        Args:
            window: Raw EMG window of shape (batch, channels, time_steps).
            features: Extracted features of shape (batch, n_features).

        Returns:
            Tuple of (predictions, confidences):
                - predictions: Predicted class indices of shape (batch,)
                - confidences: Confidence scores of shape (batch,)
        """
        logits = self.forward(window, features)
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        return predictions, confidences


def create_model_from_config(config_path: str) -> BiLSTMCNN:
    """
    Create a BiLSTM-CNN model from configuration file.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Initialized BiLSTMCNN model.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    signal_config = config['signal']
    model_config = config['model']
    gesture_classes = config['gesture_classes']

    # Calculate window size in samples
    window_size = int(signal_config['window_size_ms'] * signal_config['sample_rate'] / 1000)

    # Calculate number of features (5 features per channel: MAV, RMS, WL, ZC, SSC)
    n_features = signal_config['n_channels'] * 5

    model = BiLSTMCNN(
        n_channels=signal_config['n_channels'],
        window_size=window_size,
        n_features=n_features,
        n_classes=len(gesture_classes),
        cnn_filters=model_config['cnn_filters'],
        cnn_kernel_size=model_config['cnn_kernel_size'],
        cnn_pool_size=model_config['cnn_pool_size'],
        lstm_hidden_size=model_config['lstm_hidden_size'],
        lstm_num_layers=model_config['lstm_num_layers'],
        lstm_dropout=model_config['lstm_dropout'],
        fc_hidden_size=model_config['fc_hidden_size'],
        fc_dropout=model_config['fc_dropout']
    )

    return model


def main():
    """Test the model architecture."""
    import argparse

    parser = argparse.ArgumentParser(description='BiLSTM-CNN Model Test')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Create model
    model = create_model_from_config(args.config)
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with mock data
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = 4
    n_channels = config['signal']['n_channels']
    window_size = int(config['signal']['window_size_ms'] * config['signal']['sample_rate'] / 1000)
    n_features = n_channels * 5

    # Mock inputs
    window = torch.randn(batch_size, n_channels, window_size)
    features = torch.randn(batch_size, n_features)

    print(f"\nInput shapes:")
    print(f"  Window: {window.shape}")
    print(f"  Features: {features.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(window, features)
        predictions, confidences = model.predict(window, features)

    print(f"\nOutput shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Confidences: {confidences.shape}")
    print(f"\nPredictions: {predictions}")
    print(f"Confidences: {confidences}")


if __name__ == '__main__':
    main()
