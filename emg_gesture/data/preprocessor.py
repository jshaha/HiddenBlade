"""
EMG Signal Preprocessing Module.

Handles filtering, normalization, and feature extraction for EMG signals.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
from scipy import signal
import yaml


class EMGPreprocessor:
    """
    Preprocesses raw EMG signals for gesture classification.

    Applies bandpass filtering, notch filtering, normalization,
    and extracts time-domain features.
    """

    def __init__(self, config_path: str):
        """
        Initialize the preprocessor with configuration.

        Args:
            config_path: Path to config.yaml file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sample_rate = self.config['signal']['sample_rate']
        self.n_channels = self.config['signal']['n_channels']

        # Preprocessing parameters
        prep_config = self.config['preprocessing']
        self.bandpass_low = prep_config['bandpass_low']
        self.bandpass_high = prep_config['bandpass_high']
        self.bandpass_order = prep_config['bandpass_order']
        self.notch_freq = prep_config['notch_freq']
        self.notch_q = prep_config['notch_q']

        # Design filters
        self._design_filters()

        # Normalization statistics
        self.norm_stats: Optional[Dict[str, np.ndarray]] = None
        self.stats_path = Path(self.config['paths']['normalization_stats'])

    def _design_filters(self):
        """Design bandpass and notch filters."""
        nyquist = self.sample_rate / 2

        # Bandpass filter (Butterworth)
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist

        # Clamp to valid range
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))

        self.bp_b, self.bp_a = signal.butter(
            self.bandpass_order,
            [low, high],
            btype='band'
        )

        # Notch filter for power line noise
        notch_freq_normalized = self.notch_freq / nyquist
        if notch_freq_normalized < 1.0:
            self.notch_b, self.notch_a = signal.iirnotch(
                notch_freq_normalized,
                self.notch_q
            )
        else:
            # If notch freq is above Nyquist, skip notch filter
            self.notch_b, self.notch_a = np.array([1.0]), np.array([1.0])

    def apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to EMG data.

        Args:
            data: Input data of shape (samples, channels) or (samples,).

        Returns:
            Filtered data with same shape as input.
        """
        if data.ndim == 1:
            return signal.filtfilt(self.bp_b, self.bp_a, data)
        else:
            filtered = np.zeros_like(data)
            for ch in range(data.shape[1]):
                filtered[:, ch] = signal.filtfilt(self.bp_b, self.bp_a, data[:, ch])
            return filtered

    def apply_notch(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove power line noise.

        Args:
            data: Input data of shape (samples, channels) or (samples,).

        Returns:
            Filtered data with same shape as input.
        """
        if data.ndim == 1:
            return signal.filtfilt(self.notch_b, self.notch_a, data)
        else:
            filtered = np.zeros_like(data)
            for ch in range(data.shape[1]):
                filtered[:, ch] = signal.filtfilt(self.notch_b, self.notch_a, data[:, ch])
            return filtered

    def filter_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Apply full filtering pipeline (bandpass + notch).

        Args:
            data: Raw EMG data of shape (samples, channels).

        Returns:
            Filtered data with same shape.
        """
        # Apply bandpass filter
        filtered = self.apply_bandpass(data)
        # Apply notch filter
        filtered = self.apply_notch(filtered)
        return filtered

    def compute_normalization_stats(self, data: np.ndarray, save: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute mean and std for normalization from training data.

        Args:
            data: Training data of shape (samples, channels).
            save: If True, save stats to disk.

        Returns:
            Dictionary with 'mean' and 'std' arrays.
        """
        self.norm_stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0) + 1e-8  # Add epsilon to avoid division by zero
        }

        if save:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.stats_path, **self.norm_stats)
            print(f"[Preprocessor] Saved normalization stats to {self.stats_path}")

        return self.norm_stats

    def load_normalization_stats(self) -> bool:
        """
        Load normalization statistics from disk.

        Returns:
            True if stats loaded successfully, False otherwise.
        """
        if self.stats_path.exists():
            stats = np.load(self.stats_path)
            self.norm_stats = {
                'mean': stats['mean'],
                'std': stats['std']
            }
            print(f"[Preprocessor] Loaded normalization stats from {self.stats_path}")
            return True
        return False

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to zero mean, unit variance per channel.

        Args:
            data: Input data of shape (samples, channels) or (window, samples, channels).

        Returns:
            Normalized data with same shape.

        Raises:
            ValueError: If normalization stats not available.
        """
        if self.norm_stats is None:
            raise ValueError("Normalization stats not available. "
                             "Call compute_normalization_stats() or load_normalization_stats() first.")

        return (data - self.norm_stats['mean']) / self.norm_stats['std']

    def extract_mav(self, window: np.ndarray) -> np.ndarray:
        """
        Mean Absolute Value - average of absolute signal values.

        Args:
            window: EMG window of shape (samples, channels).

        Returns:
            MAV feature array of shape (channels,).
        """
        return np.mean(np.abs(window), axis=0)

    def extract_rms(self, window: np.ndarray) -> np.ndarray:
        """
        Root Mean Square - square root of mean squared signal.

        Args:
            window: EMG window of shape (samples, channels).

        Returns:
            RMS feature array of shape (channels,).
        """
        return np.sqrt(np.mean(window ** 2, axis=0))

    def extract_wl(self, window: np.ndarray) -> np.ndarray:
        """
        Waveform Length - cumulative length of waveform.

        Args:
            window: EMG window of shape (samples, channels).

        Returns:
            WL feature array of shape (channels,).
        """
        return np.sum(np.abs(np.diff(window, axis=0)), axis=0)

    def extract_zc(self, window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Zero Crossings - number of times signal crosses zero.

        Args:
            window: EMG window of shape (samples, channels).
            threshold: Minimum amplitude change to count as crossing.

        Returns:
            ZC feature array of shape (channels,).
        """
        zc = np.zeros(window.shape[1])
        for ch in range(window.shape[1]):
            sign_changes = np.diff(np.sign(window[:, ch]))
            # Count crossings with sufficient amplitude change
            if threshold > 0:
                amp_changes = np.abs(np.diff(window[:, ch]))
                zc[ch] = np.sum((sign_changes != 0) & (amp_changes > threshold))
            else:
                zc[ch] = np.sum(sign_changes != 0)
        return zc

    def extract_ssc(self, window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Slope Sign Changes - number of times slope direction changes.

        Args:
            window: EMG window of shape (samples, channels).
            threshold: Minimum slope change to count.

        Returns:
            SSC feature array of shape (channels,).
        """
        ssc = np.zeros(window.shape[1])
        for ch in range(window.shape[1]):
            diff1 = np.diff(window[:, ch])
            diff2 = np.diff(diff1)
            sign_changes = np.diff(np.sign(diff1))
            if threshold > 0:
                ssc[ch] = np.sum((sign_changes != 0) & (np.abs(diff2) > threshold))
            else:
                ssc[ch] = np.sum(sign_changes != 0)
        return ssc

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract all time-domain features from a window.

        Args:
            window: EMG window of shape (samples, channels).

        Returns:
            Feature vector of shape (n_channels * 5,) containing
            MAV, RMS, WL, ZC, SSC for each channel.
        """
        mav = self.extract_mav(window)
        rms = self.extract_rms(window)
        wl = self.extract_wl(window)
        zc = self.extract_zc(window)
        ssc = self.extract_ssc(window)

        # Concatenate all features
        features = np.concatenate([mav, rms, wl, zc, ssc])
        return features

    def preprocess_window(self, window: np.ndarray, normalize: bool = True
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline for a single window.

        Args:
            window: Raw EMG window of shape (samples, channels).
            normalize: If True, apply normalization.

        Returns:
            Tuple of (filtered_window, features):
                - filtered_window: Shape (samples, channels), for CNN input
                - features: Shape (n_features,), for LSTM input
        """
        # Apply filtering
        filtered = self.filter_signal(window)

        # Normalize if stats available
        if normalize and self.norm_stats is not None:
            filtered = self.normalize(filtered)

        # Extract features
        features = self.extract_features(filtered)

        return filtered, features

    def preprocess_recording(self, data: np.ndarray, normalize: bool = True
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess an entire recording, creating overlapping windows.

        Args:
            data: Raw recording of shape (total_samples, channels).
            normalize: If True, apply normalization.

        Returns:
            Tuple of (windows, features):
                - windows: Shape (n_windows, window_size, channels)
                - features: Shape (n_windows, n_features)
        """
        window_size = int(self.config['signal']['window_size_ms'] * self.sample_rate / 1000)
        hop_size = int(self.config['signal']['hop_size_ms'] * self.sample_rate / 1000)

        # Calculate number of windows
        n_windows = (len(data) - window_size) // hop_size + 1

        windows = []
        features = []

        for i in range(n_windows):
            start = i * hop_size
            end = start + window_size
            window = data[start:end]

            w, f = self.preprocess_window(window, normalize=normalize)
            windows.append(w)
            features.append(f)

        return np.array(windows), np.array(features)


def main():
    """Test the preprocessor with mock data."""
    import argparse

    parser = argparse.ArgumentParser(description='EMG Preprocessor Test')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    preprocessor = EMGPreprocessor(args.config)

    # Generate mock data
    n_samples = 1000
    n_channels = preprocessor.n_channels
    mock_data = np.random.randn(n_samples, n_channels) * 100

    # Add some sinusoidal components
    t = np.arange(n_samples) / preprocessor.sample_rate
    for ch in range(n_channels):
        mock_data[:, ch] += 50 * np.sin(2 * np.pi * 50 * t + ch)  # 50Hz component
        mock_data[:, ch] += 30 * np.sin(2 * np.pi * 60 * t)  # 60Hz power line noise

    print(f"Input shape: {mock_data.shape}")
    print(f"Input mean: {mock_data.mean():.2f}, std: {mock_data.std():.2f}")

    # Compute normalization stats
    preprocessor.compute_normalization_stats(mock_data, save=False)

    # Preprocess
    filtered = preprocessor.filter_signal(mock_data)
    print(f"\nFiltered shape: {filtered.shape}")
    print(f"Filtered mean: {filtered.mean():.2f}, std: {filtered.std():.2f}")

    # Extract features from a window
    window = filtered[:200]  # 200ms window at 1000Hz
    features = preprocessor.extract_features(window)
    print(f"\nFeatures shape: {features.shape}")
    print(f"Features: MAV={features[:n_channels].mean():.4f}, "
          f"RMS={features[n_channels:2*n_channels].mean():.4f}")

    # Full preprocessing
    windows, all_features = preprocessor.preprocess_recording(mock_data)
    print(f"\nWindows shape: {windows.shape}")
    print(f"Features shape: {all_features.shape}")


if __name__ == '__main__':
    main()
