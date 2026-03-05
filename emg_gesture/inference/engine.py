"""
Real-Time Inference Engine for EMG Gesture Classification.

Handles sliding window inference with prediction smoothing.
"""

import time
from collections import deque
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import numpy as np
import torch
import yaml

from model.bilstm_cnn import create_model_from_config
from data.preprocessor import EMGPreprocessor


class InferenceEngine:
    """
    Real-time inference engine for EMG gesture classification.

    Maintains a rolling buffer, runs inference on each hop,
    and applies majority voting for smooth predictions.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the inference engine.

        Args:
            config_path: Path to config.yaml file.
            checkpoint_path: Path to model checkpoint. Uses default if None.
            device: Device to run inference on (auto-detected if None).
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.gesture_classes = self.config['gesture_classes']
        self.idx_to_label = {i: g for i, g in enumerate(self.gesture_classes)}

        # Signal parameters
        signal_config = self.config['signal']
        self.n_channels = signal_config['n_channels']
        self.sample_rate = signal_config['sample_rate']
        self.window_size = int(signal_config['window_size_ms'] * self.sample_rate / 1000)
        self.hop_size = int(signal_config['hop_size_ms'] * self.sample_rate / 1000)

        # Inference parameters
        inference_config = self.config['inference']
        self.vote_window_size = inference_config['vote_window_size']
        self.min_stable_duration_ms = inference_config['min_stable_duration_ms']
        self.confidence_threshold = inference_config['confidence_threshold']

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"[InferenceEngine] Using device: {self.device}")

        # Load model
        self.model = self._load_model(checkpoint_path)

        # Setup preprocessor
        self.preprocessor = EMGPreprocessor(config_path)
        if not self.preprocessor.load_normalization_stats():
            print("[InferenceEngine] Warning: No normalization stats found")

        # Prediction history for smoothing
        self.prediction_history: deque = deque(maxlen=self.vote_window_size)
        self.confidence_history: deque = deque(maxlen=self.vote_window_size)

        # State for stable prediction tracking
        self.current_stable_prediction: Optional[str] = None
        self.stable_prediction_start: Optional[float] = None
        self.last_emitted_gesture: Optional[str] = None

        # Callbacks
        self.on_gesture_callback: Optional[Callable[[str, float], None]] = None

    def _load_model(self, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
        """
        Load the model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint. Uses default if None.

        Returns:
            Loaded model in eval mode.
        """
        if checkpoint_path is None:
            checkpoint_path = Path(self.config['paths']['checkpoints_dir']) / 'best_model.pt'
        else:
            checkpoint_path = Path(checkpoint_path)

        model = create_model_from_config(self.config_path)

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[InferenceEngine] Loaded model from {checkpoint_path}")
        else:
            print(f"[InferenceEngine] Warning: No checkpoint found at {checkpoint_path}")
            print("[InferenceEngine] Using randomly initialized model")

        model.to(self.device)
        model.eval()

        return model

    def set_gesture_callback(self, callback: Callable[[str, float], None]):
        """
        Set callback for confirmed gesture events.

        Args:
            callback: Function called with (gesture_label, confidence) when
                     a stable gesture is confirmed.
        """
        self.on_gesture_callback = callback

    def reset(self):
        """Reset the prediction state."""
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.current_stable_prediction = None
        self.stable_prediction_start = None
        self.last_emitted_gesture = None

    def _majority_vote(self) -> Tuple[Optional[str], float]:
        """
        Apply majority voting over recent predictions.

        Returns:
            Tuple of (majority_label, average_confidence) or (None, 0) if no predictions.
        """
        if len(self.prediction_history) == 0:
            return None, 0.0

        # Count votes
        vote_counts = {}
        confidence_sums = {}

        for pred, conf in zip(self.prediction_history, self.confidence_history):
            if pred not in vote_counts:
                vote_counts[pred] = 0
                confidence_sums[pred] = 0.0
            vote_counts[pred] += 1
            confidence_sums[pred] += conf

        # Find majority
        majority_pred = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        avg_confidence = confidence_sums[majority_pred] / vote_counts[majority_pred]

        return majority_pred, avg_confidence

    def infer_window(self, window: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Run inference on a single window.

        Args:
            window: EMG window of shape (window_size, n_channels).

        Returns:
            Tuple of (prediction_label, confidence, all_probabilities).
        """
        # Preprocess
        filtered, features = self.preprocessor.preprocess_window(window)

        # Prepare tensors
        # Window: (channels, time) for Conv1D
        window_tensor = torch.from_numpy(filtered.T).float().unsqueeze(0)
        feature_tensor = torch.from_numpy(features).float().unsqueeze(0)

        # Move to device
        window_tensor = window_tensor.to(self.device)
        feature_tensor = feature_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(window_tensor, feature_tensor)
            probs = torch.softmax(logits, dim=1)

        probs_np = probs.cpu().numpy()[0]
        pred_idx = np.argmax(probs_np)
        confidence = probs_np[pred_idx]
        label = self.idx_to_label[pred_idx]

        return label, float(confidence), probs_np

    def process_window(self, window: np.ndarray) -> Tuple[str, float, bool]:
        """
        Process a window and update prediction state.

        Applies majority voting and checks for stable predictions.

        Args:
            window: EMG window of shape (window_size, n_channels).

        Returns:
            Tuple of (smoothed_prediction, confidence, is_new_gesture):
                - smoothed_prediction: Majority vote label
                - confidence: Average confidence for majority prediction
                - is_new_gesture: True if this is a newly confirmed gesture
        """
        # Run inference
        raw_pred, raw_conf, probs = self.infer_window(window)

        # Add to history
        self.prediction_history.append(raw_pred)
        self.confidence_history.append(raw_conf)

        # Get smoothed prediction
        smoothed_pred, avg_conf = self._majority_vote()

        # Check confidence threshold
        if avg_conf < self.confidence_threshold:
            smoothed_pred = "rest"  # Default to rest if confidence too low

        # Track stable predictions
        current_time = time.time()
        is_new_gesture = False

        if smoothed_pred != self.current_stable_prediction:
            # Prediction changed, start new stability tracking
            self.current_stable_prediction = smoothed_pred
            self.stable_prediction_start = current_time
        else:
            # Same prediction, check if stable long enough
            stable_duration_ms = (current_time - self.stable_prediction_start) * 1000

            if stable_duration_ms >= self.min_stable_duration_ms:
                # Stable prediction confirmed
                if smoothed_pred != self.last_emitted_gesture:
                    is_new_gesture = True
                    self.last_emitted_gesture = smoothed_pred

                    # Call callback if set
                    if self.on_gesture_callback:
                        self.on_gesture_callback(smoothed_pred, avg_conf)

        return smoothed_pred, avg_conf, is_new_gesture

    def get_current_prediction(self) -> Tuple[Optional[str], float]:
        """
        Get the current smoothed prediction without processing new data.

        Returns:
            Tuple of (current_prediction, confidence) or (None, 0) if no history.
        """
        return self._majority_vote()


class MockInferenceEngine(InferenceEngine):
    """
    Mock inference engine for testing without a trained model.

    Returns random predictions cycling through gesture classes.
    """

    def __init__(self, config_path: str, **kwargs):
        """
        Initialize mock inference engine.

        Args:
            config_path: Path to config.yaml file.
            **kwargs: Additional arguments (ignored).
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.gesture_classes = self.config['gesture_classes']
        self.idx_to_label = {i: g for i, g in enumerate(self.gesture_classes)}

        # Inference parameters
        inference_config = self.config['inference']
        self.vote_window_size = inference_config['vote_window_size']
        self.min_stable_duration_ms = inference_config['min_stable_duration_ms']
        self.confidence_threshold = inference_config['confidence_threshold']

        # Prediction history for smoothing
        self.prediction_history: deque = deque(maxlen=self.vote_window_size)
        self.confidence_history: deque = deque(maxlen=self.vote_window_size)

        # State tracking
        self.current_stable_prediction: Optional[str] = None
        self.stable_prediction_start: Optional[float] = None
        self.last_emitted_gesture: Optional[str] = None
        self.on_gesture_callback: Optional[Callable[[str, float], None]] = None

        # Mock state
        self._mock_counter = 0
        self._mock_cycle_length = 50  # Change gesture every N windows

        print("[MockInferenceEngine] Initialized (no model loaded)")

    def infer_window(self, window: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Return mock predictions cycling through gestures.

        Args:
            window: EMG window (ignored).

        Returns:
            Tuple of (prediction_label, confidence, all_probabilities).
        """
        # Cycle through gestures
        gesture_idx = (self._mock_counter // self._mock_cycle_length) % len(self.gesture_classes)
        self._mock_counter += 1

        # Generate mock probabilities
        probs = np.random.dirichlet(np.ones(len(self.gesture_classes)) * 0.5)
        probs[gesture_idx] += 2.0  # Bias towards current gesture
        probs = probs / probs.sum()

        label = self.idx_to_label[gesture_idx]
        confidence = probs[gesture_idx]

        return label, float(confidence), probs


def main():
    """Test the inference engine."""
    import argparse

    parser = argparse.ArgumentParser(description='Inference Engine Test')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock inference engine')

    args = parser.parse_args()

    # Create engine
    if args.mock:
        engine = MockInferenceEngine(args.config)
    else:
        engine = InferenceEngine(args.config)

    # Set callback
    def on_gesture(gesture: str, confidence: float):
        print(f"\n>>> GESTURE CONFIRMED: {gesture} (confidence: {confidence:.3f})")

    engine.set_gesture_callback(on_gesture)

    # Test with mock data
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    window_size = int(config['signal']['window_size_ms'] * config['signal']['sample_rate'] / 1000)
    n_channels = config['signal']['n_channels']

    print("\nProcessing mock windows...")
    for i in range(30):
        window = np.random.randn(window_size, n_channels) * 100
        pred, conf, is_new = engine.process_window(window)
        status = "NEW!" if is_new else ""
        print(f"Window {i:3d}: {pred:12s} (conf: {conf:.3f}) {status}")
        time.sleep(0.05)


if __name__ == '__main__':
    main()
